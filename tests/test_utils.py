import numpy as np
import pytest

from openairbearing.bearings import (
  AnnularBearing,
  BaseBearing,
  CircularBearing,
  InfiniteLinearBearing,
  JournalBearing,
  RectangularBearing,
)
from openairbearing.utils import (
  Result,
  get_area,
  get_beta,
  get_dA,
  get_geom,
  get_kappa,
  get_load_capacity,
  get_Qsc,
  get_stiffness,
  get_volumetric_flow,
  round_to_sig_dig,
)

from conftest import RTOL_EXACT


# ── Result dataclass ──────────────────────────────────────────────────


def test_result_fields():
  """Result stores solver outputs without transformation."""
  p = np.ones((10, 5))
  r = Result(
    name="analytic",
    p=p,
    w=np.array([1.0]),
    k=np.array([2.0]),
    qs=np.array([3.0]),
    qa=np.array([4.0]),
    qc=np.array([5.0]),
  )
  assert r.name == "analytic"
  np.testing.assert_array_equal(r.p, p)


# ── get_area ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
  "case, xa, xc, ya, expected",
  [
    ("circular", 10, 0, 0, np.pi * 100),
    ("annular", 10, 5, 0, np.pi * 75),
    ("infinite", 10, 0, 0, 10),
    ("rectangular", 10, 0, 5, 50),
  ],
)
def test_area_formula(case, xa, xc, ya, expected):
  """Area matches closed-form for each geometry."""
  b = BaseBearing(case=case, xa=xa, xc=xc, ya=ya)
  assert get_area(b) == pytest.approx(expected)


def test_area_journal():
  """Journal area = 2πrL."""
  b = JournalBearing()
  assert get_area(b) == pytest.approx(2 * np.pi * b.xa * b.ya)


def test_area_unknown_case():
  with pytest.raises(ValueError, match="Unknown case"):
    BaseBearing(case="unknown")


# ── get_geom ──────────────────────────────────────────────────────────


def test_geom_none_is_zero(circular):
  """No error → flat geometry."""
  geom = get_geom(circular)
  np.testing.assert_array_equal(geom, 0)


def test_geom_1d_linear():
  """1D linear error has correct shape and minimum pinned to zero."""
  b = BaseBearing(case="circular", csys="polar", nx=20, error_type="linear", error=5e-6)
  geom = get_geom(b)
  assert geom.shape == (20,)
  assert geom.min() == pytest.approx(0), "geom shifted to min=0"
  assert geom.max() > 0


def test_geom_1d_quadratic():
  """1D quadratic error has correct shape and minimum pinned to zero."""
  b = BaseBearing(
    case="circular", csys="polar", nx=20, error_type="quadratic", error=5e-6
  )
  geom = get_geom(b)
  assert geom.shape == (20,)
  assert geom.min() == pytest.approx(0)


@pytest.mark.parametrize(
  "error_type", ["none", "linear", "quadratic", "tiltx", "tilty"]
)
def test_geom_2d_cartesian_shapes(error_type):
  """2D Cartesian geometry arrays have (nx, ny) shape."""
  b = BaseBearing(
    case="rectangular",
    csys="cartesian",
    nx=15,
    ny=10,
    xa=10,
    ya=5,
    error_type=error_type,
    error=5e-6,
  )
  geom = get_geom(b)
  assert geom.shape == (15, 10)
  assert geom.min() == pytest.approx(0)


@pytest.mark.parametrize("error_type", ["none", "linear", "quadratic"])
def test_geom_2d_polar_shapes(error_type):
  """2D polar geometry arrays have (nx, ny) shape."""
  b = BaseBearing(
    case="circular",
    csys="polar",
    nx=15,
    ny=10,
    xa=10,
    ya=5,
    error_type=error_type,
    error=5e-6,
  )
  geom = get_geom(b)
  assert geom.shape == (15, 10)


def test_geom_invalid_error_type_1d():
  with pytest.raises(ValueError, match="Unknown error type"):
    BaseBearing(case="circular", csys="polar", nx=10, error_type="invalid")


def test_geom_invalid_error_type_2d():
  with pytest.raises(ValueError, match="Unknown error type"):
    BaseBearing(
      case="rectangular",
      csys="cartesian",
      nx=10,
      ny=5,
      xa=10,
      ya=5,
      error_type="invalid",
    )


def test_geom_invalid_csys_2d():
  with pytest.raises(ValueError, match="Unknown coordinate system"):
    BaseBearing(case="circular", csys="spherical", nx=10, ny=5, xa=10, ya=5)


# ── get_beta ──────────────────────────────────────────────────────────


def test_beta_formula(circular):
  """β = 6κxa²/(hp·ha³), vectorized over ha."""
  b = circular
  beta = get_beta(b)
  expected = 6 * b.kappa * b.xa**2 / (b.hp * b.ha**3)
  np.testing.assert_allclose(beta, expected, rtol=RTOL_EXACT)
  assert beta.shape == b.ha.shape


def test_beta_pinned_endpoints():
  """Pinned beta endpoints for default CircularBearing."""
  b = CircularBearing()
  np.testing.assert_allclose(b.beta[0], 693.626667, rtol=1e-4)
  np.testing.assert_allclose(b.beta[-1], 0.086703, rtol=1e-4)


# ── get_kappa / get_Qsc roundtrip ────────────────────────────────────


@pytest.mark.parametrize(
  "BearingCls, expected_kappa",
  [
    (CircularBearing, 1.5200e-15),
    (AnnularBearing, 8.1400e-16),
    (InfiniteLinearBearing, 5.4000e-16),
  ],
)
def test_kappa_pinned(BearingCls, expected_kappa):
  """Pinned kappa for each default bearing."""
  b = BearingCls()
  np.testing.assert_allclose(b.kappa, expected_kappa, rtol=1e-4)


@pytest.mark.parametrize(
  "BearingCls, expected_Qsc",
  [
    (CircularBearing, 2.8),
    (AnnularBearing, 3.0),
    (InfiniteLinearBearing, 19.3),
  ],
)
def test_kappa_Qsc_roundtrip(BearingCls, expected_Qsc):
  """kappa → Qsc → kappa is an identity; Qsc matches expected value."""
  b = BearingCls()
  kappa_orig = b.kappa
  Qsc_recovered = get_Qsc(b)
  np.testing.assert_allclose(Qsc_recovered, expected_Qsc, rtol=0.01)
  b.Qsc = Qsc_recovered
  kappa_recovered = get_kappa(b)
  np.testing.assert_allclose(kappa_recovered, kappa_orig, rtol=0.01)


# ── round_to_sig_dig ─────────────────────────────────────────────────


@pytest.mark.parametrize(
  "number, digits, expected",
  [
    (1234, 3, 1230),
    (0.001234, 3, 0.00123),
    (9.999, 2, 10.0),
    (1e-15, 3, 1e-15),
  ],
)
def test_round_to_sig_dig(number, digits, expected):
  assert round_to_sig_dig(number, digits) == pytest.approx(expected)


# ── get_dA ────────────────────────────────────────────────────────────


def test_dA_polar_1d_integrates_to_area(circular):
  """∑dA equals bearing area for circular bearing."""
  dA = get_dA(circular)
  assert dA.shape == (circular.nx,)
  np.testing.assert_allclose(np.sum(dA), circular.A, rtol=0.02)


@pytest.mark.parametrize(
  "BearingCls, expected_area",
  [
    (CircularBearing, 0.0010752101),
    (AnnularBearing, 0.0021512056),
    (InfiniteLinearBearing, 0.04),
  ],
)
def test_dA_sum_pinned(BearingCls, expected_area):
  """Pinned: ∑dA = A for each default bearing."""
  b = BearingCls()
  np.testing.assert_allclose(np.sum(get_dA(b)), expected_area, rtol=1e-4)


def test_dA_cartesian_1d_shape():
  b = BaseBearing(case="infinite", csys="cartesian", nx=10, ny=1, xa=10)
  dA = get_dA(b)
  assert dA.shape == (10,)


def test_dA_cartesian_2d(rectangular):
  """2D Cartesian dA is scalar dx*dy."""
  dA = get_dA(rectangular)
  expected = rectangular.dx * rectangular.dy
  np.testing.assert_allclose(dA, expected, rtol=RTOL_EXACT)


def test_dA_invalid_csys():
  b = BaseBearing(case="circular", csys="spherical", nx=10, ny=1, xa=10)
  with pytest.raises(ValueError, match="invalid csys"):
    get_dA(b)


# ── get_load_capacity ─────────────────────────────────────────────────


def test_load_capacity_zero_at_ambient(circular):
  """Uniform ambient pressure → zero load."""
  p = np.full((circular.nx, circular.nh), circular.pa)
  w = get_load_capacity(circular, p)
  np.testing.assert_allclose(w, 0, atol=1e-10)


def test_load_capacity_positive_above_ambient(circular):
  """Pressure above ambient everywhere → positive load for bearing type."""
  p = np.full((circular.nx, circular.nh), circular.ps)
  w = get_load_capacity(circular, p)
  assert np.all(w > 0), "load must be positive when p > pa"


def test_load_capacity_uniform_pressure(circular):
  """Uniform gauge pressure p-pa over area → w = (p-pa)*A."""
  p_gauge = 1e5  # 100 kPa gauge
  p = np.full((circular.nx, circular.nh), circular.pa + p_gauge)
  w = get_load_capacity(circular, p)
  expected = p_gauge * circular.A
  np.testing.assert_allclose(w, expected, rtol=0.02)


# ── get_stiffness ─────────────────────────────────────────────────────


def test_stiffness_sign_for_bearing(circular):
  """For a bearing (not journal), stiffness = -dw/dha, so positive when w decreases with ha."""
  # synthetic w that decreases with ha: w = 1/ha
  w = 1.0 / circular.ha.flatten()
  k = get_stiffness(circular, w)
  # dw/dha < 0 → k = -dw/dha > 0
  assert np.all(k[1:-1] > 0), "stiffness positive for decreasing w"


def test_stiffness_formula(circular):
  """Matches raw -np.gradient(w, ha)."""
  w = np.exp(-circular.ha.flatten() * 1e5)
  k = get_stiffness(circular, w)
  expected = -np.gradient(w, circular.ha.flatten())
  np.testing.assert_allclose(k, expected, rtol=RTOL_EXACT)


# ── get_volumetric_flow ──────────────────────────────────────────────


def test_volumetric_flow_returns_three_arrays(circular):
  """Solver integration: flow computation doesn't crash and returns 3 arrays."""
  from openairbearing.solvers import get_pressure_analytic_circular

  p = get_pressure_analytic_circular(circular)
  qs, qa, qc = get_volumetric_flow(circular, p, "analytic")
  assert qs.shape == (circular.nh,)
  assert qa.shape == (circular.nh,)
  assert qc.shape == (circular.nh,)


def test_supply_flow_is_ambient_minus_chamber(circular):
  """Mass conservation: qs = qa - qc."""
  from openairbearing.solvers import get_pressure_analytic_circular

  p = get_pressure_analytic_circular(circular)
  qs, qa, qc = get_volumetric_flow(circular, p, "analytic")
  np.testing.assert_allclose(qs, qa - qc, rtol=RTOL_EXACT)


def test_supply_flow_pinned_circular():
  """Pinned supply flow for default circular analytic."""
  from openairbearing.solvers import get_pressure_analytic_circular

  b = CircularBearing()
  p = get_pressure_analytic_circular(b)
  qs, qa, qc = get_volumetric_flow(b, p, "analytic")
  np.testing.assert_allclose(
    qs[:3], [0.1086880988, 0.4279355669, 0.8358394598], rtol=1e-6
  )
