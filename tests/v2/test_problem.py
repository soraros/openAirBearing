"""Tests for the problem compiler: axes, film stack, weights, boundaries."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from openairbearing.v2 import catalog
from openairbearing.v2.geometry.grid import GridSpec
from openairbearing.v2.geometry.pads import CircularPad
from openairbearing.v2.geometry.profiles import SurfaceError
from openairbearing.v2.operating.state import OperatingState
from openairbearing.v2.primitives.constants import P_ATM
from openairbearing.v2.problem import BearingSpec, build_problem

# ── Compiled axes and stacks ──────────────────────────────────────────────────


def test_circular_problem_arrays(circular_spec):
  p = build_problem(circular_spec)
  assert p.x.shape == (30,)
  assert p.y.shape == (1,)
  assert p.samples.shape == (20,)
  assert p.geom.shape == (30,)
  assert p.gaps.shape == (20, 30)
  assert p.dim == 1 and p.polar and not p.journal
  np.testing.assert_allclose(p.samples, np.linspace(1e-6, 20e-6, 20), rtol=1e-12)
  np.testing.assert_allclose(p.gaps[0], p.samples[0] + p.geom, rtol=1e-12)


def test_rectangular_problem_arrays(rectangular_spec):
  p = build_problem(rectangular_spec)
  assert p.x.shape == (40,) and p.y.shape == (20,)
  assert p.gaps.shape == (20, 40, 20)
  assert p.dim == 2 and not p.polar
  np.testing.assert_allclose(p.x[0], -40e-3, rtol=1e-12)
  np.testing.assert_allclose(p.y[0], -20e-3, rtol=1e-12)


def test_problem_arrays_are_readonly(circular_spec):
  p = build_problem(circular_spec)
  for arr in (p.x, p.y, p.samples, p.geom, p.gaps, p.dA, p.load_weights):
    assert not arr.flags.writeable


def test_spec_is_frozen(circular_spec):
  with pytest.raises(FrozenInstanceError):
    circular_spec.grid = GridSpec()


def test_spec_type_validation():
  with pytest.raises(TypeError, match="pad"):
    BearingSpec(pad="circle", restrictor=catalog.circular().restrictor)
  with pytest.raises(ValueError, match="ny"):
    replace(catalog.rectangular(), grid=GridSpec(ny=1))


# ── Integration weights ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
  "spec_fn, rtol",
  [(catalog.circular, 0.02), (catalog.annular, 0.02), (catalog.linear, 0.02)],
)
def test_dA_sums_to_area_1d(spec_fn, rtol):
  """1-D trapezoidal weights integrate to the pad area."""
  p = build_problem(spec_fn())
  np.testing.assert_allclose(np.sum(p.dA), p.area, rtol=rtol)


def test_dA_rectangular_exact(rectangular_spec):
  """2-D cartesian weights integrate to exactly lx·ly."""
  p = build_problem(rectangular_spec)
  np.testing.assert_allclose(np.sum(p.dA), p.area, rtol=1e-12)


def test_dA_polar_2d_exact(annular_spec):
  """2-D polar weights integrate to exactly π(r²−rᵢ²)."""
  spec = replace(annular_spec, grid=replace(annular_spec.grid, ny=24))
  p = build_problem(spec)
  np.testing.assert_allclose(np.sum(p.dA), p.area, rtol=1e-12)


def test_journal_load_weights(journal_spec):
  """Journal load projects pressure onto cos θ along the eccentricity direction."""
  p = build_problem(journal_spec)
  assert p.load_weights.shape == (p.x.size, p.y.size)
  # weights are signed through cos θ, unlike the positive area weights
  assert np.any(p.load_weights < 0.0)
  assert np.all(p.dA > 0.0)
  assert p.stiffness_sign == 1.0


def test_stiffness_sign_thrust(circular_spec):
  assert build_problem(circular_spec).stiffness_sign == -1.0


# ── Journal film ──────────────────────────────────────────────────────────────


def test_journal_clearance_at_zero_eccentricity(journal_spec):
  """At e → 0 the journal film is the uniform radial clearance c/2."""
  p = build_problem(journal_spec)
  pad = journal_spec.pad
  np.testing.assert_allclose(p.gaps[0], pad.clearance / 2, atol=5e-8)


def test_journal_clearance_pinches(journal_spec):
  """At e = c/2 the film pinches to zero at θ = 0 (v1 clearance formula)."""
  pad = journal_spec.pad
  e_max = pad.clearance / 2
  spec = replace(
    journal_spec,
    grid=replace(journal_spec.grid, sample_min=e_max - 1e-9, sample_max=e_max),
  )
  q = build_problem(spec)
  gaps = q.gaps[-1, :, 0]
  i_min = int(np.argmin(gaps))
  assert gaps[i_min] < 1e-6
  assert abs(q.x[i_min]) < 2 * np.pi / q.x.size  # pinch at θ = 0


# ── Boundaries ────────────────────────────────────────────────────────────────


def test_boundary_spec_per_pad():
  cases = {
    "circular": ("neumann", "dirichlet", "periodic", "periodic"),
    "annular": ("dirichlet", "dirichlet", "periodic", "periodic"),
    "linear": ("dirichlet", "dirichlet", "periodic", "periodic"),
    "rectangular": ("dirichlet", "dirichlet", "dirichlet", "dirichlet"),
    "journal": ("periodic", "periodic", "dirichlet", "dirichlet"),
  }
  for name, expected in cases.items():
    bc = build_problem(getattr(catalog, name)()).boundaries
    got = (bc.x_lo.kind, bc.x_hi.kind, bc.y_lo.kind, bc.y_hi.kind)
    assert got == expected, f"{name}: {got} != {expected}"


def test_edge_pressure_resolution(circular_spec):
  state = OperatingState(p_supply=0.7e6 + P_ATM)
  p = build_problem(replace(circular_spec, state=state))
  assert p.boundaries.x_hi.pressure(p.state) == pytest.approx(P_ATM)
  ann = build_problem(catalog.annular())
  assert ann.boundaries.x_lo.pressure(ann.state) == pytest.approx(P_ATM)


def test_error_profile_enters_geom(circular_spec):
  spec = replace(circular_spec, error=SurfaceError(kind="quadratic", amplitude=2e-6))
  p = build_problem(spec)
  assert p.geom.max() == pytest.approx(2e-6)
  assert p.geom.min() == pytest.approx(0.0)
  np.testing.assert_allclose(p.gaps[0], p.samples[0] + p.geom, rtol=1e-12)
