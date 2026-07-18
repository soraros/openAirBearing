"""Analytic pressure solutions for 1-D pads (pure functions of problem data).

All solutions solve the 1-D isothermal Reynolds equation with porous feed
in the squared pressure ψ = p² and return p with shape ``(nh, nx)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import i0, k0

from openairbearing.v2.geometry.pads import AnnularPad, CircularPad, LinearPad
from openairbearing.v2.primitives.types import StackScalar

if TYPE_CHECKING:
  from openairbearing.v2.problem import BearingProblem

__all__ = ["pressure_analytic"]


def pressure_analytic(problem: BearingProblem) -> StackScalar:
  """Analytic pressure stack for pads with a closed-form 1-D solution."""
  pad = problem.pad
  if isinstance(pad, CircularPad):
    return _circular(problem, pad)
  if isinstance(pad, AnnularPad):
    return _annular(problem, pad)
  if isinstance(pad, LinearPad):
    return _linear(problem, pad)
  raise ValueError(f"no analytic solution for {type(pad).__name__}")


def _wave(problem: BearingProblem) -> StackScalar:
  """Porous feeding wave number f = √(2β) per sample."""
  return np.sqrt(2.0 * problem.beta)


def _ratios(problem: BearingProblem) -> tuple[float, float]:
  """Nondimensional supply and chamber pressures (Pₐ = 1)."""
  state = problem.state
  return state.p_supply / state.p_ambient, state.p_chamber / state.p_ambient


def _circular(problem: BearingProblem, pad: CircularPad) -> StackScalar:
  """Bessel solution for the circular thrust bearing."""
  state = problem.state
  ps, pa = state.p_supply, state.p_ambient
  f = _wave(problem)  # (nh,)
  rr = problem.x / pad.r  # (nx,)
  p = ps * np.sqrt(1.0 - (1.0 - pa**2 / ps**2) * i0(np.outer(f, rr)) / i0(f)[:, None])
  return p


def _annular(problem: BearingProblem, pad: AnnularPad) -> StackScalar:
  """Bessel-function solution for annular bearings and seals."""
  f = _wave(problem)  # (nh,)
  ps, pc = _ratios(problem)

  # nondimensionals
  r = problem.x / pad.r  # (nx,)
  rc = pad.r_inner / pad.r

  numer1 = (1.0 - ps**2) * k0(f * rc) + (ps**2 - pc**2) * k0(f)
  numer2 = (1.0 - ps**2) * i0(f * rc) + (ps**2 - pc**2) * i0(f)
  denom = i0(f * rc) * k0(f) - i0(f) * k0(f * rc)

  c1 = numer1 / denom
  c2 = numer2 / denom

  p = problem.state.p_ambient * np.sqrt(
    ps**2 - c1[:, None] * i0(np.outer(f, r)) + c2[:, None] * k0(np.outer(f, r))
  )
  return p


def _linear(problem: BearingProblem, pad: LinearPad) -> StackScalar:
  """Exponential solution for infinitely wide linear bearings and seals."""
  state = problem.state
  f = _wave(problem)  # (nh,)
  ps, pc = _ratios(problem)
  slip = np.sqrt(1.0 + state.slip)

  # nondimensionals
  r = problem.x / pad.length  # (nx,)

  exp_f = np.exp(f / slip)

  numer1 = -(pc**2) + ps**2 + exp_f * (1.0 - ps**2)
  numer2 = exp_f * (-1.0 + ps**2 + exp_f * (pc**2 - ps**2))
  denom = -1.0 + np.exp((2.0 * f) / slip)

  c1 = numer1 / denom
  c2 = numer2 / denom

  p = state.p_ambient * np.sqrt(
    ps**2
    + c1[:, None] * np.exp(np.outer(f, r) / slip)
    + c2[:, None] * np.exp(-np.outer(f, r) / slip)
  )
  return p
