"""Tests for the v2 solvers: pinned regression, BCs, dispatch, and physics."""

from dataclasses import replace

import numpy as np
import pytest

from openairbearing.v2 import catalog
from openairbearing.v2.problem import BearingSpec, build_problem
from openairbearing.v2.solvers.api import solve_bearing, solve_pressure
from openairbearing.v2.solvers.fdm import assemble_1d, thomas_1d
from openairbearing.v2.solvers.solution import Solution

from .conftest import RTOL_NUMERIC, RTOL_NUMERIC_FINE

# ── Pinned regression: circular ───────────────────────────────────────────────


def test_circular_analytic_pinned(circular_spec):
  r = solve_bearing(circular_spec, "analytic")
  np.testing.assert_allclose(
    r.load[:3], [613.1564843027, 575.1381364761, 524.7198067878], rtol=1e-8
  )
  np.testing.assert_allclose(r.load[-3:], [55.3395027047, 48.6962918432, 43.0191338173], rtol=1e-8)
  np.testing.assert_allclose(r.peak_stiffness, 5.818180e07, rtol=1e-5)
  assert r.peak_index == 4
  np.testing.assert_allclose(r.q_supply[:3], [0.1086958706, 0.4279860451, 0.8359538857], rtol=1e-6)
  np.testing.assert_allclose(r.p[0, 0], 701325.0, rtol=1e-8)
  np.testing.assert_allclose(r.p[0, -1], 101325.0, rtol=1e-8)
  np.testing.assert_allclose(r.p[-1, 0], 174617.674906, rtol=1e-6)


def test_circular_numeric_pinned(circular_spec):
  r = solve_bearing(circular_spec, "numeric")
  np.testing.assert_allclose(
    r.load[:3], [611.9922249607, 574.6195309504, 524.4380040429], rtol=1e-8
  )
  np.testing.assert_allclose(r.load[-3:], [55.3142161213, 48.6735347152, 42.9986111883], rtol=1e-8)
  np.testing.assert_allclose(r.peak_stiffness, 5.814531e07, rtol=1e-5)
  assert r.peak_index == 4
  np.testing.assert_allclose(r.q_supply[:3], [0.1053694531, 0.4249758298, 0.8338864913], rtol=1e-6)


# ── Pinned regression: annular ────────────────────────────────────────────────


def test_annular_analytic_pinned(annular_spec):
  r = solve_bearing(annular_spec, "analytic")
  np.testing.assert_allclose(
    r.load[:3], [1203.8196088681, 1075.763879856, 906.1418098859], rtol=1e-8
  )
  np.testing.assert_allclose(r.load[-3:], [39.0196227054, 33.6684444883, 29.2248619522], rtol=1e-8)
  np.testing.assert_allclose(r.peak_stiffness, 1.756458e08, rtol=1e-5)
  assert r.peak_index == 2
  np.testing.assert_allclose(r.q_supply[:3], [0.1990002184, 0.7547944893, 1.4716759818], rtol=1e-6)
  np.testing.assert_allclose(np.max(r.p), 701321.436745, rtol=1e-6)


def test_annular_numeric_pinned(annular_spec):
  r = solve_bearing(annular_spec, "numeric")
  np.testing.assert_allclose(
    r.load[:3], [1201.9842048094, 1075.0158415095, 905.7346714336], rtol=1e-8
  )
  np.testing.assert_allclose(r.peak_stiffness, 1.753877e08, rtol=1e-5)
  assert r.peak_index == 2
  np.testing.assert_allclose(r.q_supply[:3], [0.1958182685, 0.7524980383, 1.4700342347], rtol=1e-6)


# ── Pinned regression: linear ─────────────────────────────────────────────────


def test_linear_analytic_pinned(linear_spec):
  r = solve_bearing(linear_spec, "analytic")
  np.testing.assert_allclose(
    r.load[:3], [11790.6630921399, 11179.6271388015, 10304.3147379028], rtol=1e-8
  )
  np.testing.assert_allclose(
    r.load[-3:], [874.0382965003, 758.6235378853, 661.7999123329], rtol=1e-8
  )
  np.testing.assert_allclose(r.peak_stiffness, 1.178164e09, rtol=1e-5)
  assert r.peak_index == 4
  np.testing.assert_allclose(r.q_supply[:3], [0.5093843898, 2.4876036691, 5.4158921441], rtol=1e-6)


def test_linear_numeric_pinned(linear_spec):
  r = solve_bearing(linear_spec, "numeric")
  np.testing.assert_allclose(
    r.load[:3], [11763.1788468648, 11165.4038427634, 10296.1308186835], rtol=1e-8
  )
  np.testing.assert_allclose(r.peak_stiffness, 1.176717e09, rtol=1e-5)
  assert r.peak_index == 4
  np.testing.assert_allclose(r.q_supply[:3], [0.494976933, 2.4654554736, 5.3980032778], rtol=1e-6)


# ── Pinned regression: rectangular / journal 2-D ──────────────────────────────


@pytest.fixture
def rect_spec_small(rectangular_spec) -> BearingSpec:
  return replace(rectangular_spec, grid=replace(rectangular_spec.grid, nx=15, ny=10, n_samples=3))


def test_rectangular_numeric2d_pinned(rect_spec_small):
  r = solve_bearing(rect_spec_small, "numeric2d")
  np.testing.assert_allclose(r.load, [809.2846658841, 174.6694572388, 36.0271999004], rtol=1e-6)
  np.testing.assert_allclose(
    r.stiffness,
    [66801600.91003207, 40697761.36756157, 14593921.825091075],
    rtol=1e-6,
  )
  np.testing.assert_allclose(np.max(r.p), 409999.7616550458, rtol=1e-6)


def test_journal_numeric2d_pinned(journal_spec):
  spec = replace(journal_spec, grid=replace(journal_spec.grid, nx=40, ny=25, n_samples=5))
  r = solve_bearing(spec, "numeric2d")
  np.testing.assert_allclose(
    r.load,
    [2.3010499555e-01, 1.1290531617e02, 2.4985298128e02, 4.5432188236e02, 7.3171807208e02],
    rtol=1e-6,
  )
  # flow out of the two axial ends is symmetric at small eccentricity
  np.testing.assert_allclose(r.q_ambient[0], -r.q_chamber[0], rtol=1e-8)


# ── Boundary conditions ───────────────────────────────────────────────────────


def test_analytic_edges_at_ambient(circular_spec):
  r = solve_bearing(circular_spec, "analytic")
  pa = circular_spec.state.p_ambient
  np.testing.assert_allclose(r.p[:, -1], pa, rtol=1e-6)


def test_annular_numeric_edges(annular_spec):
  """Annular seal: inner edge at pc, outer at pa."""
  r = solve_bearing(annular_spec, "numeric")
  np.testing.assert_allclose(r.p[:, 0], annular_spec.state.p_chamber, rtol=1e-6)
  np.testing.assert_allclose(r.p[:, -1], annular_spec.state.p_ambient, rtol=1e-6)


def test_circular_numeric_neumann_center(circular_spec):
  """Neumann at the center: the gradient at r_center is small."""
  r = solve_bearing(circular_spec, "numeric")
  dp = np.abs(r.p[:, 1] - r.p[:, 0])
  assert np.all(dp < 100.0)


def test_rectangular_edges_at_ambient(rect_spec_small):
  r = solve_bearing(rect_spec_small, "numeric2d")
  pa = rect_spec_small.state.p_ambient
  np.testing.assert_allclose(r.p[:, 0, :], pa, rtol=1e-9)
  np.testing.assert_allclose(r.p[:, -1, :], pa, rtol=1e-9)
  np.testing.assert_allclose(r.p[:, :, 0], pa, rtol=1e-9)
  np.testing.assert_allclose(r.p[:, :, -1], pa, rtol=1e-9)


# ── Kernel unit checks ────────────────────────────────────────────────────────


def test_assemble_1d_uniform_main_diagonal():
  """Uniform gap on a cartesian grid: interior main diagonal is -2ε/dx² + s."""
  nh, nx = 2, 10
  gaps = np.full((nh, nx), 1e-5)
  x = np.linspace(0.0, 1.0, nx)
  dx = np.gradient(x)
  source, slip, mu = -1.0, 0.0, 1.85e-5
  lower, main, upper, rhs = assemble_1d(gaps, x, dx, source, slip, mu, False, 0, 0, 1.0, 1.0, 4.0)
  eps = 1e-5**3 / (24 * mu)
  expected = -2.0 * eps / dx[1] ** 2 + source
  np.testing.assert_allclose(main[:, 2:-2], expected, rtol=1e-10)
  np.testing.assert_allclose(rhs[:, 1:-1], 4.0 * source, rtol=1e-12)
  # dirichlet rows: identity with the prescribed psi value
  np.testing.assert_allclose(rhs[:, 0], 1.0, rtol=1e-12)
  np.testing.assert_allclose(rhs[:, -1], 1.0, rtol=1e-12)


def test_thomas_solves_tridiagonal():
  """Thomas kernel inverts a known tridiagonal system."""
  nh, nx = 1, 8
  main = np.full((nh, nx), 2.0)
  lower = np.full((nh, nx), -0.5)
  upper = np.full((nh, nx), -0.5)
  lower[0, 0] = 0.0
  upper[0, -1] = 0.0
  x_true = np.arange(1.0, nx + 1.0)
  rhs = (main * x_true).copy()
  rhs[0, 1:] -= 0.5 * x_true[:-1]
  rhs[0, :-1] -= 0.5 * x_true[1:]
  psi = thomas_1d(lower, main, upper, rhs)
  np.testing.assert_allclose(psi[0], x_true, rtol=1e-10)


# ── Analytic vs numeric cross-validation ──────────────────────────────────────


@pytest.mark.parametrize("name", ["circular", "annular", "linear"])
def test_analytic_vs_numeric_fine(name):
  """Fine-grid analytic and numeric solutions agree within tolerance."""
  spec = getattr(catalog, name)()
  spec = replace(
    spec,
    grid=replace(spec.grid, nx=100, n_samples=200, sample_min=1e-6, sample_max=100e-6),
  )
  ra = solve_bearing(spec, "analytic")
  rn = solve_bearing(spec, "numeric")
  np.testing.assert_allclose(ra.p, rn.p, rtol=RTOL_NUMERIC_FINE)
  np.testing.assert_allclose(ra.load, rn.load, rtol=RTOL_NUMERIC_FINE)
  np.testing.assert_allclose(ra.stiffness, rn.stiffness, rtol=RTOL_NUMERIC)
  np.testing.assert_allclose(ra.q_supply, rn.q_supply, rtol=RTOL_NUMERIC)


# ── Dispatch ──────────────────────────────────────────────────────────────────


def test_invalid_method(circular_spec):
  with pytest.raises(ValueError, match="invalid solution method"):
    solve_bearing(circular_spec, "magic")


def test_analytic_unsupported_pad(rectangular_spec):
  with pytest.raises(ValueError, match="no analytic solution"):
    solve_bearing(rectangular_spec, "analytic")


def test_numeric_requires_1d(rectangular_spec):
  with pytest.raises(ValueError, match="1-D"):
    solve_bearing(rectangular_spec, "numeric")


def test_numeric2d_requires_2d(circular_spec):
  with pytest.raises(ValueError, match="2-D"):
    solve_bearing(circular_spec, "numeric2d")


def test_solve_accepts_spec_or_problem(circular_spec):
  """solve_bearing compiles a spec on the spot or reuses a problem."""
  from_spec = solve_bearing(circular_spec, "analytic")
  from_problem = solve_bearing(build_problem(circular_spec), "analytic")
  np.testing.assert_allclose(from_spec.load, from_problem.load, rtol=1e-12)


# ── Physical sanity ───────────────────────────────────────────────────────────


def test_load_decreases_with_gap(circular_fine):
  r = solve_bearing(circular_fine, "analytic")
  n = r.load.size
  assert np.mean(r.load[: n // 4]) > np.mean(r.load[-n // 4 :])


def test_stiffness_peak_is_interior(circular_fine):
  r = solve_bearing(circular_fine, "analytic")
  assert 0 < r.peak_index < r.stiffness.size - 1


def test_load_bounded_by_supply_pressure(rect_spec_small):
  """Load cannot exceed (p_supply - p_ambient) · area."""
  r = solve_bearing(rect_spec_small, "numeric2d")
  state = rect_spec_small.state
  bound = (state.p_supply - state.p_ambient) * rect_spec_small.pad.area
  assert np.all(r.load < bound)
  assert np.all(r.load > 0.0)


def test_journal_load_increases_with_eccentricity(journal_spec):
  spec = replace(journal_spec, grid=replace(journal_spec.grid, nx=40, ny=25, n_samples=5))
  r = solve_bearing(spec, "numeric2d")
  assert np.all(np.diff(r.load) > 0.0)


def test_solution_verify(circular_spec):
  solve_bearing(circular_spec, "analytic").verify()
  solve_bearing(circular_spec, "numeric").verify()


def test_solution_verify_detects_bad_load(circular_spec):
  r = solve_bearing(circular_spec, "analytic")
  bad = Solution(
    problem=r.problem,
    method=r.method,
    p=r.p.copy(),
    load=r.load.copy() * 2.0,
    stiffness=r.stiffness.copy(),
    q_supply=r.q_supply.copy(),
    q_ambient=r.q_ambient.copy(),
    q_chamber=r.q_chamber.copy(),
  )
  with pytest.raises(AssertionError, match="load"):
    bad.verify()
