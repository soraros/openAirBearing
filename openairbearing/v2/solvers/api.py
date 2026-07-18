"""Solve entry points: method dispatch over a compiled bearing problem."""

from __future__ import annotations

from openairbearing.v2.primitives.types import StackScalar
from openairbearing.v2.problem import BearingProblem, BearingSpec, build_problem
from openairbearing.v2.solvers.analytic import pressure_analytic
from openairbearing.v2.solvers.fdm import solve_fdm_1d, solve_fdm_2d
from openairbearing.v2.solvers.solution import (
  Solution,
  SolveMethod,
  compute_flows,
  compute_load,
  compute_stiffness,
)

__all__ = ["solve_bearing", "solve_pressure"]


def solve_bearing(problem: BearingProblem | BearingSpec, method: SolveMethod) -> Solution:
  """Solve one bearing spec/problem over its full sample sweep.

  Accepts a declarative ``BearingSpec`` (compiled on the spot) or a
  pre-compiled ``BearingProblem`` for repeated solves.
  """
  if isinstance(problem, BearingSpec):
    problem = build_problem(problem)
  if not isinstance(problem, BearingProblem):
    raise TypeError(f"expected a BearingProblem or BearingSpec, got {problem!r}")
  p = solve_pressure(problem, method)
  load = compute_load(problem, p)
  stiffness = compute_stiffness(problem, load)
  qs, qa, qc = compute_flows(problem, p, method)
  return Solution(
    problem=problem,
    method=method,
    p=p,
    load=load,
    stiffness=stiffness,
    q_supply=qs,
    q_ambient=qa,
    q_chamber=qc,
  )


def solve_pressure(problem: BearingProblem, method: SolveMethod) -> StackScalar:
  """Pressure stack for one compiled problem and solution method."""
  match method:
    case "analytic":
      if not problem.pad.ANALYTIC:
        raise ValueError(f"no analytic solution for {type(problem.pad).__name__}")
      if problem.dim != 1:
        raise ValueError("analytic requires a 1-D problem (grid.ny == 1)")
      return pressure_analytic(problem)
    case "numeric":
      if problem.dim != 1:
        raise ValueError("numeric requires a 1-D problem (grid.ny == 1); use numeric2d")
      return solve_fdm_1d(problem)
    case "numeric2d":
      if problem.dim != 2:
        raise ValueError("numeric2d requires a 2-D problem (grid.ny >= 2)")
      return solve_fdm_2d(problem)
    case _:
      raise ValueError(f"invalid solution method {method!r}")
