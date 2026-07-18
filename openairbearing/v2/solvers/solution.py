"""Solver result record plus the load/stiffness/flow reductions.

The reductions are pure functions of (problem, pressure stack) so they can
be verified independently of the solver that produced the pressure.
``Solution`` freezes its arrays and can ``verify()`` its own consistency.

Flow-rate conventions match v1: per-edge-node volumetric flow at the
ambient reference, Q = −6e4·ρ·h³·∂ₙ(p²)·dA_face/(12·μ·pₐ·dn) for cartesian
grids; polar 1-D integrates over the circumference (π·r form), and polar
2-D uses the per-node r·dθ/2 face element so it sums to the same total.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from openairbearing.v2.geometry.pads import JournalPad
from openairbearing.v2.primitives.constants import LPM_PER_M3S
from openairbearing.v2.primitives.types import SampleScalar, StackScalar
from openairbearing.v2.problem import BearingProblem

if TYPE_CHECKING:
  import plotly.graph_objects as go

__all__ = [
  "Solution",
  "SolveMethod",
  "compute_flows",
  "compute_load",
  "compute_stiffness",
]

type SolveMethod = Literal["analytic", "numeric", "numeric2d"]

_SOLVE_METHODS: tuple[str, ...] = ("analytic", "numeric", "numeric2d")


@dataclass(frozen=True, slots=True)
class Solution:
  """Pressure field stack and derived performance curves over the sample sweep."""

  problem: BearingProblem
  method: SolveMethod
  p: StackScalar
  load: SampleScalar
  stiffness: SampleScalar
  q_supply: SampleScalar
  q_ambient: SampleScalar
  q_chamber: SampleScalar

  def __post_init__(self) -> None:
    if self.method not in _SOLVE_METHODS:
      raise ValueError(f"method must be one of {_SOLVE_METHODS}, got {self.method!r}")
    nh = self.problem.n_samples
    expected = (nh, *self.problem.geom.shape)
    if self.p.shape != expected:
      raise ValueError(f"p shape {self.p.shape}, expected {expected}")
    for name in ("load", "stiffness", "q_supply", "q_ambient", "q_chamber"):
      arr = getattr(self, name)
      if arr.shape != (nh,):
        raise ValueError(f"{name} shape {arr.shape}, expected ({nh},)")
    for arr in (
      self.p,
      self.load,
      self.stiffness,
      self.q_supply,
      self.q_ambient,
      self.q_chamber,
    ):
      arr.setflags(write=False)

  @property
  def samples(self) -> SampleScalar:
    """Sample axis (gap [m] for thrust pads, eccentricity [m] for journal)."""
    return self.problem.samples

  @property
  def peak_index(self) -> int:
    """Sample index of maximum static stiffness."""
    return int(np.argmax(self.stiffness))

  @property
  def peak_sample(self) -> float:
    """Sample value at maximum static stiffness."""
    return float(self.samples[self.peak_index])

  @property
  def peak_stiffness(self) -> float:
    """Maximum static stiffness [N/m]."""
    return float(self.stiffness[self.peak_index])

  def verify(self, rtol: float = 1e-9) -> None:
    """Check finiteness, Dirichlet edges, and the derived curves."""
    if not np.all(np.isfinite(self.p)):
      raise AssertionError("p contains non-finite values")
    state = self.problem.state
    bc = self.problem.boundaries
    if self.p.ndim == 2:
      edges = (
        (bc.x_lo, self.p[:, 0], "x_lo"),
        (bc.x_hi, self.p[:, -1], "x_hi"),
      )
    else:
      edges = ()
    for edge, values, name in edges:
      if edge.kind == "dirichlet":
        expected = edge.pressure(state)
        if not np.allclose(values, expected, rtol=1e-9):
          raise AssertionError(f"dirichlet edge {name} deviates from {expected}")
    if not np.allclose(self.load, compute_load(self.problem, self.p), rtol=rtol):
      raise AssertionError("load does not match the pressure stack")
    if not np.allclose(self.stiffness, compute_stiffness(self.problem, self.load), rtol=rtol):
      raise AssertionError("stiffness does not match the load curve")

  def plot_pressure(self, *, slider: bool = True) -> go.Figure:
    """Plot the pressure distribution (delegates to postprocess.plots)."""
    from openairbearing.v2.postprocess import plots

    return plots.plot_pressure_distribution(self, slider=slider)

  def plot_key_results(self) -> tuple[go.Figure, ...]:
    """Plot the standard result figure set (delegates to postprocess.plots)."""
    from openairbearing.v2.postprocess import plots

    return plots.plot_key_results(self)


# ── Reductions ────────────────────────────────────────────────────────────────


def compute_load(problem: BearingProblem, p: StackScalar) -> SampleScalar:
  """Integrate gauge pressure over the pad face into load [N]."""
  p_rel = p - problem.state.p_ambient
  if p.ndim == 2:
    return p_rel @ problem.load_weights
  return np.einsum("kij,ij->k", p_rel, problem.load_weights)


def compute_stiffness(problem: BearingProblem, load: SampleScalar) -> SampleScalar:
  """Static stiffness [N/m]: ∓dw/ds over the sample axis (sign per pad)."""
  return problem.stiffness_sign * np.gradient(load, problem.samples)


def compute_flows(
  problem: BearingProblem, p: StackScalar, method: SolveMethod
) -> tuple[SampleScalar, SampleScalar, SampleScalar]:
  """Volumetric flow rates [L/min]: (supply, ambient, chamber)."""
  if p.ndim == 2:
    return _flows_1d(problem, p, method)
  return _flows_2d(problem, p)


def _flows_1d(
  problem: BearingProblem, p: StackScalar, method: SolveMethod
) -> tuple[SampleScalar, SampleScalar, SampleScalar]:
  state = problem.state
  gas = problem.gas
  h3 = problem.samples[:, None] ** 3 if method == "analytic" else problem.gaps**3
  coeff = -LPM_PER_M3S * gas.rho / (12.0 * gas.mu * state.p_ambient)
  q = coeff * h3 * np.gradient(p**2, axis=1) / problem.dx[None, :]
  if problem.polar:
    q = q * np.pi * problem.x[None, :]
  qa = q[:, -1]
  qc = q[:, 1]
  return qa - qc, qa, qc


def _flows_2d(
  problem: BearingProblem, p: StackScalar
) -> tuple[SampleScalar, SampleScalar, SampleScalar]:
  state = problem.state
  gas = problem.gas
  h3 = problem.gaps**3
  coeff = -LPM_PER_M3S * gas.rho / (12.0 * gas.mu * state.p_ambient)
  grad_x = np.gradient(p**2, axis=1)
  grad_y = np.gradient(p**2, axis=2)
  dx = problem.dx
  dy = problem.dy
  if problem.polar:
    # per-node radial flow; sums over θ to the polar 1-D total
    face = (problem.x[:, None] * dy[None, :] / 2.0)[None, :, :]
    qr = coeff * h3 * grad_x / dx[None, :, None] * face
    qa = np.sum(qr[:, -1, :], axis=1)
    qc = np.sum(qr[:, 0, :], axis=1)
    return qa - qc, qa, qc
  if isinstance(problem.pad, JournalPad):
    qy = coeff * h3 * grad_y * problem.pad.r * dx[0] / dy[None, None, :]
    qa = np.sum(qy[:, :, -1], axis=1)
    qc = np.sum(qy[:, :, 0], axis=1)
    return qa - qc, qa, qc
  qx = coeff * h3 * grad_x * dy[None, None, :] / dx[None, :, None]
  qy = coeff * h3 * grad_y * dx[None, :, None] / dy[None, None, :]
  qa = np.sum(np.abs(qx[:, 0, :]) + np.abs(qx[:, -1, :]), axis=1) + np.sum(
    np.abs(qy[:, :, 0]) + np.abs(qy[:, :, -1]), axis=1
  )
  qc = np.zeros(problem.n_samples)
  return qa - qc, qa, qc
