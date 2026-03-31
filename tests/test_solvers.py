import numpy as np
import pytest

from openairbearing.bearings import (
  AnnularBearing,
  CircularBearing,
  InfiniteLinearBearing,
  RectangularBearing,
)
from openairbearing.solvers import (
  build_diff_matrix,
  get_pressure_analytic_annular,
  get_pressure_analytic_circular,
  get_pressure_analytic_infinite,
  get_pressure_numeric,
  get_pressure_2d_numeric,
  solve_bearing,
)

from conftest import FINE_GRID, RTOL_NUMERIC, RTOL_NUMERIC_FINE


# ── Pinned regression: circular analytic ─────────────────────────────


def test_circular_analytic_load_pinned():
  """Pinned load capacity for default CircularBearing, analytic solver."""
  b = CircularBearing()
  r = solve_bearing(b, "analytic")
  np.testing.assert_allclose(
    r.w[:3], [613.1538509633, 575.1295023429, 524.7043952184], rtol=1e-8
  )
  np.testing.assert_allclose(
    r.w[-3:], [55.3272623859, 48.6852739463, 43.0092005762], rtol=1e-8
  )


def test_circular_analytic_stiffness_pinned():
  """Peak stiffness value and location for default CircularBearing, analytic."""
  b = CircularBearing()
  r = solve_bearing(b, "analytic")
  np.testing.assert_allclose(np.max(r.k), 5.818647e07, rtol=1e-5)
  assert np.argmax(r.k) == 4


def test_circular_analytic_flow_pinned():
  """Pinned supply flow for default CircularBearing, analytic."""
  b = CircularBearing()
  r = solve_bearing(b, "analytic")
  np.testing.assert_allclose(
    r.qs[:3], [0.1086880988, 0.4279355669, 0.8358394598], rtol=1e-6
  )


def test_circular_analytic_pressure_pinned():
  """Pinned pressure extremes for default CircularBearing, analytic."""
  b = CircularBearing()
  r = solve_bearing(b, "analytic")
  np.testing.assert_allclose(r.p[0, 0], 701325.0, rtol=1e-8)
  np.testing.assert_allclose(r.p[0, -1], 174601.760788, rtol=1e-6)
  np.testing.assert_allclose(r.p[-1, 0], 101325.0, rtol=1e-8)
  np.testing.assert_allclose(np.max(r.p), 701325.0, rtol=1e-8)
  np.testing.assert_allclose(np.min(r.p), 101325.0, rtol=1e-8)


# ── Pinned regression: circular numeric ──────────────────────────────


def test_circular_numeric_load_pinned():
  """Pinned load capacity for default CircularBearing, numeric solver."""
  b = CircularBearing()
  r = solve_bearing(b, "numeric")
  np.testing.assert_allclose(
    r.w[:3], [611.9896762261, 574.6109668406, 524.4226340486], rtol=1e-8
  )
  np.testing.assert_allclose(
    r.w[-3:], [55.3019803849, 48.6625210871, 42.9886819207], rtol=1e-8
  )


def test_circular_numeric_stiffness_pinned():
  """Peak stiffness for default CircularBearing, numeric."""
  b = CircularBearing()
  r = solve_bearing(b, "numeric")
  np.testing.assert_allclose(np.max(r.k), 5.814999e07, rtol=1e-5)
  assert np.argmax(r.k) == 4


def test_circular_numeric_flow_pinned():
  """Pinned supply flow for default CircularBearing, numeric."""
  b = CircularBearing()
  r = solve_bearing(b, "numeric")
  np.testing.assert_allclose(
    r.qs[:3], [0.1053624036, 0.4249264169, 0.8337728646], rtol=1e-6
  )


def test_circular_numeric_pressure_pinned():
  """Pinned pressure for default CircularBearing, numeric."""
  b = CircularBearing()
  r = solve_bearing(b, "numeric")
  np.testing.assert_allclose(r.p[0, 0], 701325.0, rtol=1e-8)
  np.testing.assert_allclose(r.p[0, -1], 174419.622238, rtol=1e-6)
  np.testing.assert_allclose(r.p[-1, 0], 101325.0, rtol=1e-8)


# ── Pinned regression: annular analytic ──────────────────────────────


def test_annular_analytic_load_pinned():
  """Pinned load capacity for default AnnularBearing, analytic."""
  b = AnnularBearing()
  r = solve_bearing(b, "analytic")
  np.testing.assert_allclose(
    r.w[:3], [1203.8107900863, 1075.7364470627, 906.0928107924], rtol=1e-8
  )
  np.testing.assert_allclose(
    r.w[-3:], [39.0101178633, 33.6601416387, 29.2175785559], rtol=1e-8
  )


def test_annular_analytic_stiffness_pinned():
  """Peak stiffness for default AnnularBearing, analytic."""
  b = AnnularBearing()
  r = solve_bearing(b, "analytic")
  np.testing.assert_allclose(np.max(r.k), 1.756636e08, rtol=1e-5)
  assert np.argmax(r.k) == 2


def test_annular_analytic_flow_pinned():
  """Pinned supply flow for default AnnularBearing, analytic."""
  b = AnnularBearing()
  r = solve_bearing(b, "analytic")
  np.testing.assert_allclose(
    r.qs[:3], [0.1989846647, 0.7547099378, 1.4714780924], rtol=1e-6
  )


def test_annular_analytic_pressure_pinned():
  """Pinned pressure for default AnnularBearing, analytic."""
  b = AnnularBearing()
  r = solve_bearing(b, "analytic")
  # Inner edge is at pc (= pa for default annular)
  np.testing.assert_allclose(r.p[0, 0], 101325.0, rtol=1e-8)
  np.testing.assert_allclose(r.p[-1, 0], 101325.0, rtol=1e-8)
  np.testing.assert_allclose(np.max(r.p), 701321.430943, rtol=1e-6)
  np.testing.assert_allclose(np.min(r.p), 101325.0, rtol=1e-8)


# ── Pinned regression: annular numeric ───────────────────────────────


def test_annular_numeric_load_pinned():
  """Pinned load capacity for default AnnularBearing, numeric."""
  b = AnnularBearing()
  r = solve_bearing(b, "numeric")
  np.testing.assert_allclose(
    r.w[:3], [1201.9755671151, 1074.988503184, 905.6857340172], rtol=1e-8
  )
  np.testing.assert_allclose(
    r.w[-3:], [39.0070794633, 33.6575335206, 29.2153233731], rtol=1e-8
  )


def test_annular_numeric_stiffness_pinned():
  """Peak stiffness for default AnnularBearing, numeric."""
  b = AnnularBearing()
  r = solve_bearing(b, "numeric")
  np.testing.assert_allclose(np.max(r.k), 1.754056e08, rtol=1e-5)
  assert np.argmax(r.k) == 2


def test_annular_numeric_flow_pinned():
  """Pinned supply flow for default AnnularBearing, numeric."""
  b = AnnularBearing()
  r = solve_bearing(b, "numeric")
  np.testing.assert_allclose(
    r.qs[:3], [0.1958035403, 0.7524142807, 1.4698368971], rtol=1e-6
  )


def test_annular_numeric_pressure_pinned():
  """Pinned pressure for default AnnularBearing, numeric."""
  b = AnnularBearing()
  r = solve_bearing(b, "numeric")
  np.testing.assert_allclose(np.max(r.p), 701320.042464, rtol=1e-6)
  np.testing.assert_allclose(np.min(r.p), 101325.0, rtol=1e-8)


# ── Pinned regression: infinite linear analytic ──────────────────────


def test_infinite_analytic_load_pinned():
  """Pinned load for default InfiniteLinearBearing, analytic."""
  b = InfiniteLinearBearing()
  r = solve_bearing(b, "analytic")
  np.testing.assert_allclose(
    r.w[:3], [11790.6472132695, 11179.5683061811, 10304.2022766731], rtol=1e-8
  )
  np.testing.assert_allclose(
    r.w[-3:], [873.9510239384, 758.5464563405, 661.7316569177], rtol=1e-8
  )


def test_infinite_analytic_stiffness_pinned():
  """Peak stiffness for default InfiniteLinearBearing, analytic."""
  b = InfiniteLinearBearing()
  r = solve_bearing(b, "analytic")
  np.testing.assert_allclose(np.max(r.k), 1.178211e09, rtol=1e-5)
  assert np.argmax(r.k) == 4


def test_infinite_analytic_flow_pinned():
  """Pinned supply flow for default InfiniteLinearBearing, analytic."""
  b = InfiniteLinearBearing()
  r = solve_bearing(b, "analytic")
  np.testing.assert_allclose(
    r.qs[:3], [0.5093758138, 2.4875112243, 5.4156455242], rtol=1e-6
  )


def test_infinite_analytic_pressure_pinned():
  """Pinned pressure for default InfiniteLinearBearing, analytic."""
  b = InfiniteLinearBearing()
  r = solve_bearing(b, "analytic")
  np.testing.assert_allclose(np.max(r.p), 409999.99998, rtol=1e-6)
  np.testing.assert_allclose(np.min(r.p), 101325.0, rtol=1e-8)


# ── Pinned regression: infinite linear numeric ───────────────────────


def test_infinite_numeric_load_pinned():
  """Pinned load for default InfiniteLinearBearing, numeric."""
  b = InfiniteLinearBearing()
  r = solve_bearing(b, "numeric")
  np.testing.assert_allclose(
    r.w[:3], [11763.1634228165, 11165.3457113158, 10296.0188058093], rtol=1e-8
  )
  np.testing.assert_allclose(
    r.w[-3:], [873.9214192273, 758.5242235932, 661.7147776736], rtol=1e-8
  )


def test_infinite_numeric_stiffness_pinned():
  """Peak stiffness for default InfiniteLinearBearing, numeric."""
  b = InfiniteLinearBearing()
  r = solve_bearing(b, "numeric")
  np.testing.assert_allclose(np.max(r.k), 1.176764e09, rtol=1e-5)
  assert np.argmax(r.k) == 4


def test_infinite_numeric_flow_pinned():
  """Pinned supply flow for default InfiniteLinearBearing, numeric."""
  b = InfiniteLinearBearing()
  r = solve_bearing(b, "numeric")
  np.testing.assert_allclose(
    r.qs[:3], [0.4949690852, 2.4653657739, 5.3977592457], rtol=1e-6
  )


def test_infinite_numeric_pressure_pinned():
  """Pinned pressure for default InfiniteLinearBearing, numeric."""
  b = InfiniteLinearBearing()
  r = solve_bearing(b, "numeric")
  np.testing.assert_allclose(np.max(r.p), 409999.999841, rtol=1e-6)
  np.testing.assert_allclose(np.min(r.p), 101325.0, rtol=1e-8)


# ── Pinned regression: rectangular numeric2d ─────────────────────────


def test_rectangular_numeric2d_load_pinned():
  """Pinned load for RectangularBearing(nx=15, ny=10, nh=3), numeric2d."""
  b = RectangularBearing(nx=15, ny=10, nh=3)
  r = solve_bearing(b, "numeric2d")
  np.testing.assert_allclose(
    r.w, [577.5366826483, 95.6577244962, 18.255097039], rtol=1e-6
  )


def test_rectangular_numeric2d_stiffness_pinned():
  """Pinned stiffness for RectangularBearing(nx=15, ny=10, nh=3), numeric2d."""
  b = RectangularBearing(nx=15, ny=10, nh=3)
  r = solve_bearing(b, "numeric2d")
  np.testing.assert_allclose(
    r.k,
    [50724100.85811495, 29435872.92680276, 8147644.995490576],
    rtol=1e-6,
  )


def test_rectangular_numeric2d_pressure_pinned():
  """Pinned pressure at center for RectangularBearing(nx=15, ny=10, nh=3)."""
  b = RectangularBearing(nx=15, ny=10, nh=3)
  r = solve_bearing(b, "numeric2d")
  np.testing.assert_allclose(np.max(r.p), 409998.933128, rtol=1e-6)
  # center node pressure at each ha step
  np.testing.assert_allclose(
    r.p[5, 7, :],
    [409998.9331279043, 177517.0287101212, 116861.9840851069],
    rtol=1e-6,
  )


# ── Analytic pressure boundary conditions ────────────────────────────


def test_pressure_analytic_circular_boundary_at_edge(circular):
  """Pressure at outer edge equals ambient."""
  p = get_pressure_analytic_circular(circular)
  np.testing.assert_allclose(p[-1, :], circular.pa, rtol=1e-3)


def test_pressure_analytic_annular_boundary_values(annular):
  """Annular BC: p(xc) = pc, p(xa) = pa."""
  p = get_pressure_analytic_annular(annular)
  np.testing.assert_allclose(p[0, :], annular.pc, rtol=1e-6)
  np.testing.assert_allclose(p[-1, :], annular.pa, rtol=1e-6)


# ── Numeric boundary conditions ──────────────────────────────────────


def test_pressure_numeric_boundary_circular(circular):
  """Numeric circular: Neumann at center, Dirichlet at edge."""
  p = get_pressure_numeric(circular)
  np.testing.assert_allclose(p[-1, :], circular.pa, rtol=1e-6)
  dp_center = abs(p[1, :] - p[0, :])
  assert np.all(dp_center < 100), "Neumann BC: small gradient at center"


def test_pressure_numeric_boundary_annular(annular):
  """Numeric annular: Dirichlet at both ends."""
  p = get_pressure_numeric(annular)
  np.testing.assert_allclose(p[0, :], annular.pc, rtol=1e-4)
  np.testing.assert_allclose(p[-1, :], annular.pa, rtol=1e-6)


def test_pressure_2d_rectangular_boundary_ambient(rectangular):
  """All edges at ambient for rectangular bearing."""
  p = get_pressure_2d_numeric(rectangular)
  np.testing.assert_allclose(p[0, :, :], rectangular.pa, rtol=1e-3)
  np.testing.assert_allclose(p[-1, :, :], rectangular.pa, rtol=1e-3)
  np.testing.assert_allclose(p[:, 0, :], rectangular.pa, rtol=1e-3)
  np.testing.assert_allclose(p[:, -1, :], rectangular.pa, rtol=1e-3)


# ── build_diff_matrix ────────────────────────────────────────────────


def test_build_diff_matrix_shape():
  """Output matrix has correct dimensions N×N."""
  N = 10
  eps = np.ones(N)
  dr = np.ones(N) * 0.1
  mat = build_diff_matrix(1.0, eps, dr)
  assert mat.shape == (N, N)


def test_build_diff_matrix_tridiagonal():
  """Matrix is tridiagonal (only diags -1, 0, +1 are nonzero)."""
  N = 10
  eps = np.ones(N)
  dr = np.ones(N) * 0.1
  mat = build_diff_matrix(1.0, eps, dr).toarray()
  for i in range(N):
    for j in range(N):
      if abs(i - j) > 1:
        assert mat[i, j] == 0, f"nonzero at ({i},{j})"


def test_build_diff_matrix_uniform_coefficient():
  """With uniform eps and dr, interior main diagonal is -2*eps/dr²."""
  N = 20
  eps = np.ones(N) * 3.0
  dr = np.ones(N) * 0.5
  mat = build_diff_matrix(1.0, eps, dr).toarray()
  expected_main = -2 * 3.0 / 0.5**2
  for i in range(2, N - 2):
    assert mat[i, i] == pytest.approx(expected_main, rel=1e-10)


# ── Analytic vs numeric cross-validation ─────────────────────────────


@pytest.mark.parametrize(
  "BearingCls", [CircularBearing, AnnularBearing, InfiniteLinearBearing]
)
def test_analytic_vs_numeric_pressure(BearingCls):
  """Fine-grid analytic and numeric pressures agree within tolerance."""
  b = BearingCls(**FINE_GRID)
  r_a = solve_bearing(b, "analytic")
  r_n = solve_bearing(b, "numeric")
  np.testing.assert_allclose(r_a.p, r_n.p, rtol=RTOL_NUMERIC_FINE)


@pytest.mark.parametrize(
  "BearingCls", [CircularBearing, AnnularBearing, InfiniteLinearBearing]
)
def test_analytic_vs_numeric_load(BearingCls):
  """Fine-grid analytic and numeric load capacities agree."""
  b = BearingCls(**FINE_GRID)
  r_a = solve_bearing(b, "analytic")
  r_n = solve_bearing(b, "numeric")
  np.testing.assert_allclose(r_a.w, r_n.w, rtol=RTOL_NUMERIC_FINE)


@pytest.mark.parametrize(
  "BearingCls", [CircularBearing, AnnularBearing, InfiniteLinearBearing]
)
def test_analytic_vs_numeric_stiffness(BearingCls):
  """Fine-grid analytic and numeric stiffnesses agree."""
  b = BearingCls(**FINE_GRID)
  r_a = solve_bearing(b, "analytic")
  r_n = solve_bearing(b, "numeric")
  np.testing.assert_allclose(r_a.k, r_n.k, rtol=RTOL_NUMERIC)


@pytest.mark.parametrize(
  "BearingCls", [CircularBearing, AnnularBearing, InfiniteLinearBearing]
)
def test_analytic_vs_numeric_flow(BearingCls):
  """Fine-grid analytic and numeric supply flows agree."""
  b = BearingCls(**FINE_GRID)
  r_a = solve_bearing(b, "analytic")
  r_n = solve_bearing(b, "numeric")
  np.testing.assert_allclose(r_a.qs, r_n.qs, rtol=RTOL_NUMERIC)


# ── solve_bearing dispatch ───────────────────────────────────────────


def test_solve_bearing_invalid_soltype(circular):
  with pytest.raises(ValueError, match="Invalid solution type"):
    solve_bearing(circular, "magic")


def test_solve_bearing_unsupported_case_returns_empty():
  """Analytic solver for rectangular → no analytic solution → empty Result."""
  b = RectangularBearing(nh=5)
  r = solve_bearing(b, "analytic")
  assert r.name == "none"
  assert r.p.size == 0


# ── Physical sanity checks ───────────────────────────────────────────


def test_load_decreases_with_gap(circular_fine):
  """Load capacity decreases as air gap increases."""
  r = solve_bearing(circular_fine, "analytic")
  n = len(r.w)
  assert np.mean(r.w[: n // 4]) > np.mean(r.w[-n // 4 :])


def test_stiffness_has_maximum(circular_fine):
  """Stiffness peaks at an intermediate air gap, not at boundary."""
  r = solve_bearing(circular_fine, "analytic")
  peak_idx = np.argmax(r.k)
  assert 0 < peak_idx < len(r.k) - 1, "stiffness peak is interior"
