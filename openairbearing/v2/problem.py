"""Declarative bearing specs and the compiled film problem.

The data spine of v2: a ``BearingSpec`` is pure declarative data (pad,
restrictor, reservoir state, error profile, grid). ``build_problem``
compiles it once into a ``BearingProblem`` — immutable grid axes, film-gap
stack, integration weights, and boundary statements with clean array
conventions:

- 1-D pads: fields are ``(nx,)``, sample stacks are ``(nh, nx)``.
- 2-D pads: fields are ``(nx, ny)``, sample stacks are ``(nh, nx, ny)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from openairbearing.v2.geometry.grid import GridSpec
from openairbearing.v2.geometry.pads import (
  AnnularPad,
  CircularPad,
  JournalPad,
  LinearPad,
  Pad,
  RectangularPad,
)
from openairbearing.v2.geometry.profiles import SurfaceError, eval_surface_error
from openairbearing.v2.operating.restrictor import PorousRestrictor
from openairbearing.v2.operating.state import OperatingState
from openairbearing.v2.primitives.constants import AIR, Gas
from openairbearing.v2.primitives.types import F64, AxisScalar, FieldScalar, SampleScalar

__all__ = [
  "BC_DIRICHLET",
  "BC_NEUMANN",
  "BC_PERIODIC",
  "BCKind",
  "BearingProblem",
  "BearingSpec",
  "BoundarySpec",
  "EdgeBC",
  "PressureSlot",
  "build_problem",
]

# ── Boundary statements ───────────────────────────────────────────────────────

BC_DIRICHLET: int = 0
BC_NEUMANN: int = 1
BC_PERIODIC: int = 2

type BCKind = Literal["dirichlet", "neumann", "periodic"]
type PressureSlot = Literal["supply", "chamber", "ambient", ""]

_BC_CODES: dict[str, int] = {
  "dirichlet": BC_DIRICHLET,
  "neumann": BC_NEUMANN,
  "periodic": BC_PERIODIC,
}

_PRESSURE_SLOTS: tuple[str, ...] = ("supply", "chamber", "ambient", "")


@dataclass(frozen=True, slots=True)
class EdgeBC:
  """One grid-edge boundary statement; ``slot`` names the reservoir pressure."""

  kind: BCKind
  slot: PressureSlot = ""

  def __post_init__(self) -> None:
    if self.kind not in _BC_CODES:
      raise ValueError(f"EdgeBC.kind must be one of {tuple(_BC_CODES)}, got {self.kind!r}")
    if self.slot not in _PRESSURE_SLOTS:
      raise ValueError(f"EdgeBC.slot must be a pressure slot, got {self.slot!r}")
    if self.kind == "dirichlet" and not self.slot:
      raise ValueError("dirichlet EdgeBC requires a pressure slot")
    if self.kind != "dirichlet" and self.slot:
      raise ValueError(f"{self.kind} EdgeBC takes no pressure slot")

  @property
  def code(self) -> int:
    """Integer kind code passed to numba kernels."""
    return _BC_CODES[self.kind]

  def pressure(self, state: OperatingState) -> float:
    """Resolve the slot to an absolute pressure; 0 for value-free kinds."""
    if self.slot == "supply":
      return state.p_supply
    if self.slot == "chamber":
      return state.p_chamber
    if self.slot == "ambient":
      return state.p_ambient
    return 0.0


@dataclass(frozen=True, slots=True)
class BoundarySpec:
  """Boundary statements on the four grid edges (y edges unused in 1-D)."""

  x_lo: EdgeBC
  x_hi: EdgeBC
  y_lo: EdgeBC = field(default_factory=lambda: EdgeBC("periodic"))
  y_hi: EdgeBC = field(default_factory=lambda: EdgeBC("periodic"))


# ── Spec and compiled problem ─────────────────────────────────────────────────

_PAD_TYPES: tuple[type, ...] = (CircularPad, AnnularPad, LinearPad, RectangularPad, JournalPad)


@dataclass(frozen=True, slots=True)
class BearingSpec:
  """Declarative bearing definition: pad, restrictor, state, error, grid."""

  pad: Pad
  restrictor: PorousRestrictor
  state: OperatingState = field(default_factory=OperatingState)
  error: SurfaceError = field(default_factory=SurfaceError)
  grid: GridSpec = field(default_factory=GridSpec)
  gas: Gas = AIR

  def __post_init__(self) -> None:
    if not isinstance(self.pad, _PAD_TYPES):
      raise TypeError(f"pad must be a v2 pad spec, got {self.pad!r}")
    if not isinstance(self.restrictor, PorousRestrictor):
      raise TypeError(f"restrictor must be a PorousRestrictor, got {self.restrictor!r}")
    if not isinstance(self.state, OperatingState):
      raise TypeError(f"state must be an OperatingState, got {self.state!r}")
    if not isinstance(self.error, SurfaceError):
      raise TypeError(f"error must be a SurfaceError, got {self.error!r}")
    if not isinstance(self.grid, GridSpec):
      raise TypeError(f"grid must be a GridSpec, got {self.grid!r}")
    if not isinstance(self.gas, Gas):
      raise TypeError(f"gas must be a Gas, got {self.gas!r}")
    if self.pad.DIM == 2 and self.grid.ny < 2:
      raise ValueError(f"{type(self.pad).__name__} requires grid.ny >= 2")


@dataclass(frozen=True, slots=True, init=False)
class BearingProblem:
  """Compiler-produced grid arrays, film stack, weights, and boundaries."""

  spec: BearingSpec
  x: AxisScalar
  y: AxisScalar
  dx: AxisScalar
  dy: AxisScalar
  samples: SampleScalar
  geom: FieldScalar
  gaps: F64
  dA: FieldScalar
  load_weights: FieldScalar
  stiffness_sign: float
  beta: SampleScalar
  source: float
  boundaries: BoundarySpec

  def __init__(self) -> None:
    raise TypeError("BearingProblem is compiler-produced; use build_problem")

  @classmethod
  def _from_compiled(
    cls,
    *,
    spec: BearingSpec,
    x: AxisScalar,
    y: AxisScalar,
    samples: SampleScalar,
    geom: FieldScalar,
    gaps: F64,
    dA: FieldScalar,
    load_weights: FieldScalar,
    stiffness_sign: float,
    boundaries: BoundarySpec,
  ) -> BearingProblem:
    """Join arrays emitted by the one canonical problem compiler."""
    out = object.__new__(cls)
    object.__setattr__(out, "spec", spec)
    object.__setattr__(out, "x", _readonly(x))
    object.__setattr__(out, "y", _readonly(y))
    object.__setattr__(out, "dx", _readonly(np.gradient(x)))
    object.__setattr__(out, "dy", _readonly(np.gradient(y) if y.size > 1 else np.ones(1)))
    object.__setattr__(out, "samples", _readonly(samples))
    object.__setattr__(out, "geom", _readonly(geom))
    object.__setattr__(out, "gaps", _readonly(gaps))
    object.__setattr__(out, "dA", _readonly(dA))
    object.__setattr__(out, "load_weights", _readonly(load_weights))
    object.__setattr__(out, "stiffness_sign", float(stiffness_sign))
    object.__setattr__(
      out, "beta", _readonly(spec.restrictor.feeding_parameter(spec.pad.extent, samples))
    )
    object.__setattr__(out, "source", spec.restrictor.source_coeff(spec.gas))
    object.__setattr__(out, "boundaries", boundaries)
    return out

  @property
  def pad(self) -> Pad:
    return self.spec.pad

  @property
  def restrictor(self) -> PorousRestrictor:
    return self.spec.restrictor

  @property
  def state(self) -> OperatingState:
    return self.spec.state

  @property
  def gas(self) -> Gas:
    return self.spec.gas

  @property
  def area(self) -> float:
    return self.spec.pad.area

  @property
  def dim(self) -> int:
    """Spatial dimension of the compiled grid (1 or 2)."""
    return 1 if self.y.size <= 1 else 2

  @property
  def polar(self) -> bool:
    return self.pad.CSYS == "polar"

  @property
  def journal(self) -> bool:
    return isinstance(self.pad, JournalPad)

  @property
  def n_samples(self) -> int:
    return int(self.samples.size)


def build_problem(spec: BearingSpec) -> BearingProblem:
  """Compile one declarative spec into the immutable film problem."""
  if not isinstance(spec, BearingSpec):
    raise TypeError(f"spec must be a BearingSpec, got {spec!r}")
  x, y = _grid_axes(spec)
  samples = np.linspace(spec.grid.sample_min, spec.grid.sample_max, spec.grid.n_samples)
  geom = _error_field(spec, x, y)
  gaps = _film_stack(spec, x, samples, geom)
  dA = _area_weights(spec, x, y)
  load_weights = _load_weights(spec, x, y, dA)
  return BearingProblem._from_compiled(
    spec=spec,
    x=x,
    y=y,
    samples=samples,
    geom=geom,
    gaps=gaps,
    dA=dA,
    load_weights=load_weights,
    stiffness_sign=1.0 if isinstance(spec.pad, JournalPad) else -1.0,
    boundaries=_boundary_spec(spec.pad),
  )


# ── Compiler internals ────────────────────────────────────────────────────────


def _readonly(arr: F64) -> F64:
  out = np.ascontiguousarray(arr, dtype=np.float64)
  out.setflags(write=False)
  return out


def _grid_axes(spec: BearingSpec) -> tuple[AxisScalar, AxisScalar]:
  """Node coordinates per pad; the y axis is a single placeholder in 1-D."""
  pad = spec.pad
  nx, ny = spec.grid.nx, spec.grid.ny
  if isinstance(pad, CircularPad):
    return np.linspace(pad.r_center, pad.r, nx), _theta_or_placeholder(pad, ny)
  if isinstance(pad, AnnularPad):
    return np.linspace(pad.r_inner, pad.r, nx), _theta_or_placeholder(pad, ny)
  if isinstance(pad, LinearPad):
    return np.linspace(0.0, pad.length, nx), np.zeros(1)
  if isinstance(pad, RectangularPad):
    return np.linspace(-pad.lx / 2, pad.lx / 2, nx), np.linspace(-pad.ly / 2, pad.ly / 2, ny)
  if isinstance(pad, JournalPad):
    theta = np.linspace(-np.pi, np.pi, nx, endpoint=False)
    return theta, np.linspace(-pad.length / 2, pad.length / 2, ny)
  raise TypeError(f"unsupported pad {pad!r}")


def _theta_or_placeholder(pad: Pad, ny: int) -> AxisScalar:
  """Periodic θ axis for polar pads solved on a 2-D grid."""
  if ny <= 1:
    return np.zeros(1)
  return np.linspace(0.0, 2.0 * np.pi, ny, endpoint=False)


def _error_field(spec: BearingSpec, x: AxisScalar, y: AxisScalar) -> FieldScalar:
  pad = spec.pad
  if isinstance(pad, JournalPad):
    return eval_surface_error(
      spec.error, x=x, y=y, x_extent=2.0 * np.pi, y_extent=pad.length, layout="cartesian"
    )
  if isinstance(pad, RectangularPad):
    return eval_surface_error(
      spec.error, x=x, y=y, x_extent=pad.lx, y_extent=pad.ly, layout="cartesian"
    )
  layout = "polar" if pad.CSYS == "polar" else "cartesian"
  return eval_surface_error(
    spec.error, x=x, y=y, x_extent=pad.extent, y_extent=pad.extent, layout=layout
  )


def _film_stack(spec: BearingSpec, x: AxisScalar, samples: SampleScalar, geom: FieldScalar) -> F64:
  """Film-gap field per sample: nominal gap plus the error profile.

  Journal pads sweep eccentricity instead: the nominal film is the
  eccentric clearance field h(e, θ), matching the v1 formula.
  """
  pad = spec.pad
  if isinstance(pad, JournalPad):
    r = pad.r
    c = pad.clearance
    e = samples[:, None]
    theta = x[None, :]
    clearance = r - np.sqrt((r - c / 2) ** 2 + e**2 + 2.0 * e * (r - c / 2) * np.cos(theta))
    return clearance[:, :, None] + geom[None, :, :]
  if geom.ndim == 1:
    return samples[:, None] + geom[None, :]
  return samples[:, None, None] + geom[None, :, :]


def _area_weights(spec: BearingSpec, x: AxisScalar, y: AxisScalar) -> FieldScalar:
  """Load-integration weights per grid node (trapezoidal at 1-D edges)."""
  pad = spec.pad
  if y.size <= 1:
    if pad.CSYS == "polar":
      dA = np.pi * np.gradient(x**2)
    else:
      dA = np.gradient(x).copy()
    dA[[0, -1]] /= 2.0
    return dA
  if isinstance(pad, JournalPad):
    dtheta = 2.0 * np.pi / x.size
    dy = pad.length / (y.size - 1)
    wy = np.ones(y.size)
    wy[[0, -1]] = 0.5
    return pad.r * dtheta * dy * wy[None, :] * np.ones((x.size, y.size))
  if pad.CSYS == "polar":
    r0 = x[0]
    dx2 = x**2 - np.insert(x[:-1], 0, r0) ** 2
    dtheta = 2.0 * np.pi / y.size
    return 0.5 * dx2[:, None] * dtheta * np.ones((x.size, y.size))
  dx = float(x[1] - x[0])
  dy = float(y[1] - y[0])
  wx = np.ones(x.size)
  wx[[0, -1]] = 0.5
  wy = np.ones(y.size)
  wy[[0, -1]] = 0.5
  return dx * dy * wx[:, None] * wy[None, :]


def _load_weights(spec: BearingSpec, x: AxisScalar, y: AxisScalar, dA: FieldScalar) -> FieldScalar:
  """Journal pads project pressure onto cos θ along the eccentricity direction."""
  if isinstance(spec.pad, JournalPad):
    return np.cos(x)[:, None] * dA
  return dA


def _boundary_spec(pad: Pad) -> BoundarySpec:
  """Canonical boundary statements per pad type."""
  ambient = EdgeBC("dirichlet", "ambient")
  chamber = EdgeBC("dirichlet", "chamber")
  neumann = EdgeBC("neumann")
  periodic = EdgeBC("periodic")
  if isinstance(pad, CircularPad):
    return BoundarySpec(x_lo=neumann, x_hi=ambient, y_lo=periodic, y_hi=periodic)
  if isinstance(pad, AnnularPad):
    return BoundarySpec(x_lo=chamber, x_hi=ambient, y_lo=periodic, y_hi=periodic)
  if isinstance(pad, LinearPad):
    return BoundarySpec(x_lo=chamber, x_hi=ambient)
  if isinstance(pad, RectangularPad):
    return BoundarySpec(x_lo=ambient, x_hi=ambient, y_lo=ambient, y_hi=ambient)
  if isinstance(pad, JournalPad):
    return BoundarySpec(x_lo=periodic, x_hi=periodic, y_lo=ambient, y_hi=ambient)
  raise TypeError(f"unsupported pad {pad!r}")
