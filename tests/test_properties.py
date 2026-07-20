"""Property-based tests: invariants that must hold for any valid config.

Slow solver-level properties run few examples on small grids; arithmetic
invariants run more. Nothing here pins absolute values — those live in
the regression tests.
"""

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from openairbearing.bearings import (
  AnnularBearing,
  CircularBearing,
  InfiniteLinearBearing,
  RectangularBearing,
)
from openairbearing.solution_analytic import solve_bearing_analytic
from openairbearing.solution_fem import solve_bearing_fem_1d
from openairbearing.utils import (
  get_dA,
  get_geom_1d,
  get_geom_2d,
  get_Qsc,
  get_supply_flow,
  get_volumetric_flow,
)

# ── strategies ────────────────────────────────────────────────────────────────

RADIUS = st.floats(min_value=5e-3, max_value=100e-3)
LENGTH = st.floats(min_value=10e-3, max_value=200e-3)
QSC = st.floats(min_value=0.1, max_value=100.0)
NX = st.integers(min_value=12, max_value=40)
AMP = st.floats(min_value=-10e-6, max_value=10e-6)

ERROR_KINDS_2D = ["none", "linear", "quadratic", "saddle", "tiltx", "tilty"]

THRUST_CLASSES = [CircularBearing, AnnularBearing, InfiniteLinearBearing]


# ── arithmetic invariants ─────────────────────────────────────────────────────


@given(cls=st.sampled_from(THRUST_CLASSES), qsc=QSC)
@settings(max_examples=25)
def test_qsc_round_trips_through_kappa(cls, qsc):
  """get_Qsc returns the configured flow rate (kappa calibration is consistent)."""
  b = cls(Qsc=qsc)
  assert get_Qsc(b) == pytest.approx(qsc, rel=0.01)


@given(
  xa=RADIUS,
  xc_frac=st.floats(min_value=0.05, max_value=0.9),
  nx=st.integers(min_value=10, max_value=60),
)
@settings(max_examples=30)
def test_da_weights_sum_to_area_1d(xa, xc_frac, nx):
  """1-D dA trapezoid weights integrate to the pad area."""
  from openairbearing.bearings import BaseBearing
  from openairbearing.utils import get_area

  annular = BaseBearing(case="annular", csys="polar", xa=xa, xc=xc_frac * xa, nx=nx)
  assert np.sum(get_dA(annular)) == pytest.approx(get_area(annular), rel=0.05)
  circular = BaseBearing(case="circular", csys="polar", xa=xa, nx=nx)
  assert np.sum(get_dA(circular)) == pytest.approx(get_area(circular), rel=0.05)
  infinite = BaseBearing(case="infinite", csys="cartesian", xa=xa, nx=nx)
  assert np.sum(get_dA(infinite)) == pytest.approx(get_area(infinite), rel=0.05)


@given(nx=NX, ny=st.integers(min_value=6, max_value=30), amp=AMP, kind=st.sampled_from(ERROR_KINDS_2D))
@settings(max_examples=30)
def test_geom_profiles_shift_to_zero_min_and_stay_bounded(nx, ny, amp, kind):
  """Every 2-D error profile has min == 0 and max <= |amplitude|."""
  b = RectangularBearing(nx=nx, ny=ny, error_type=kind, error=amp)
  geom = get_geom_2d(b, x=b.fem_2d.basis.doflocs[0], y=b.fem_2d.basis.doflocs[1])
  assert np.min(geom) == pytest.approx(0.0, abs=1e-15)
  assert np.max(geom) <= abs(amp) + 1e-12


@given(xa=RADIUS, nx=NX, amp=AMP)
@settings(max_examples=30)
def test_geom_1d_bounded(xa, nx, amp):
  """1-D profiles also have min == 0 and max <= |amplitude|."""
  kind = np.random.choice(["linear", "quadratic"])
  b = CircularBearing(xa=xa, nx=nx, error_type=kind, error=amp)
  geom = get_geom_1d(b, x=b.x)
  assert np.min(geom) == pytest.approx(0.0, abs=1e-15)
  assert np.max(geom) <= abs(amp) + 1e-12


# ── solver invariants ─────────────────────────────────────────────────────────


@given(
  cls=st.sampled_from(THRUST_CLASSES),
  size=RADIUS,
  nh=st.integers(min_value=3, max_value=8),
)
@settings(max_examples=15, deadline=None, suppress_health_check=list(HealthCheck))
def test_pressure_stays_within_envelope(cls, size, nh):
  """pa <= p <= ps for analytic and FEM 1-D results (small tolerance)."""
  if cls is AnnularBearing:
    b = cls(xa=size * 1.5, xc=size, nh=nh)
  else:
    b = cls(xa=size, nh=nh)
  tol = 1.0
  p_a = solve_bearing_analytic(b).p
  assert np.all(p_a >= b.pa - tol)
  assert np.all(p_a <= b.ps + tol)
  for snap in solve_bearing_fem_1d(b).p_1d:
    assert np.all(snap >= b.pa - tol)
    assert np.all(snap <= b.ps + tol)


@given(cls=st.sampled_from(THRUST_CLASSES), xa=RADIUS)
@settings(max_examples=15, deadline=None)
def test_load_decreases_with_gap(cls, xa):
  """Thrust load is strictly decreasing in the film gap."""
  if cls is AnnularBearing:
    b = cls(xa=xa, xc=xa / 3, nh=2)
  else:
    b = cls(xa=xa, nh=2)
  w = solve_bearing_analytic(b).w
  assert w[0] > w[-1]


@given(cls=st.sampled_from(THRUST_CLASSES), xa=RADIUS)
@settings(max_examples=15, deadline=None)
def test_load_bounded_by_supply_pressure(cls, xa):
  """Load cannot exceed (ps - pa) * area (with integration tolerance)."""
  if cls is AnnularBearing:
    b = cls(xa=xa, xc=xa / 3)
  else:
    b = cls(xa=xa)
  w = solve_bearing_analytic(b).w
  assert np.all(w <= (b.ps - b.pa) * b.A * 1.02 + 1e-9)
  assert np.all(w >= 0.0)


@given(xa=RADIUS, nx=st.integers(min_value=60, max_value=120), nh=st.integers(min_value=3, max_value=6))
@settings(max_examples=10, deadline=None)
def test_supply_flow_matches_edge_outflow(xa, nx, nh):
  """Steady state: porous inflow ~ edge outflow for polar pads.

  Checked away from the thinnest gaps, where the boundary layer is
  narrower than the grid and one-sided edge gradients are under-resolved.
  """
  b = CircularBearing(xa=xa, nx=nx, nh=nh, ha_min=5e-6, ha_max=20e-6)
  r = solve_bearing_analytic(b)
  qs, qa, qc = get_volumetric_flow(b, r.p)
  src = get_supply_flow(b, r.p)
  np.testing.assert_allclose(src, qa - qc, rtol=0.2)
  np.testing.assert_allclose(r.qs, src, rtol=1e-9)


# ── analytic vs FEM agreement ─────────────────────────────────────────────────


@given(
  cls=st.sampled_from(THRUST_CLASSES),
  xa=st.floats(min_value=10e-3, max_value=60e-3),
  nx=st.integers(min_value=40, max_value=80),
)
@settings(max_examples=8, deadline=None)
def test_fem_1d_matches_analytic_on_resolved_grid(cls, xa, nx):
  """FEM 1-D load and supply flow track the analytic solution."""
  if cls is AnnularBearing:
    b = cls(xa=xa, xc=xa / 3, nx=nx, nh=4)
  else:
    b = cls(xa=xa, nx=nx, nh=4)
  ra = solve_bearing_analytic(b)
  rf = solve_bearing_fem_1d(b)
  w_err = np.mean(np.abs(ra.w - rf.w) / np.maximum(np.abs(ra.w), 1e-9))
  q_err = np.mean(np.abs(ra.qs - rf.qs) / np.maximum(np.abs(ra.qs), 1e-9))
  assert w_err < 0.05
  assert q_err < 0.10


# ── journal properties ────────────────────────────────────────────────────────

from openairbearing.bearings import JournalBearing
from openairbearing.solution_fem import solve_bearing_fem_2d


@given(
  e1=st.floats(min_value=0.5e-6, max_value=5e-6),
  e2=st.floats(min_value=5.5e-6, max_value=13e-6),
  nx=st.integers(min_value=12, max_value=24),
  ny=st.integers(min_value=8, max_value=16),
)
@settings(max_examples=10, deadline=None)
def test_journal_load_increases_with_eccentricity(e1, e2, nx, ny):
  """Projected journal load grows with the eccentricity."""
  b = JournalBearing(nx=nx, ny=ny, nh=2, eccentricity_sweep=np.array([e1, e2]))
  w = solve_bearing_fem_2d(b).w
  assert w[1] > w[0]
  assert np.all(np.isfinite(w))


@given(
  e=st.floats(min_value=3e-6, max_value=12e-6),
  nx=st.integers(min_value=12, max_value=24),
  ny=st.integers(min_value=8, max_value=16),
)
@settings(max_examples=8, deadline=None)
def test_journal_centered_has_much_smaller_load(e, nx, ny):
  """At zero eccentricity the projected load nearly vanishes."""
  b0 = JournalBearing(nx=nx, ny=ny, nh=2, eccentricity_sweep=np.zeros(2))
  b1 = JournalBearing(nx=nx, ny=ny, nh=2, eccentricity_sweep=np.array([e, e]))
  w0 = solve_bearing_fem_2d(b0).w[0]
  w1 = solve_bearing_fem_2d(b1).w[0]
  assert abs(w0) < 0.05 * abs(w1) + 1e-9


# ── constructor/result fuzz ───────────────────────────────────────────────────


@given(
  cls=st.sampled_from(THRUST_CLASSES),
  xa=RADIUS,
  nh=st.integers(min_value=3, max_value=10),
)
@settings(max_examples=12, deadline=None)
def test_results_are_finite_for_valid_configs(cls, xa, nh):
  """No NaN/inf in any result channel for random valid configurations."""
  if cls is AnnularBearing:
    b = cls(xa=xa, xc=xa / 3, nh=nh)
  else:
    b = cls(xa=xa, nh=nh)
  r = solve_bearing_analytic(b)
  for channel in (r.w, r.k, r.qs, r.qa, r.qc, r.p):
    assert np.all(np.isfinite(channel))
  rf = solve_bearing_fem_1d(b)
  for channel in (rf.w, rf.k, rf.qs, rf.qa, rf.qc):
    assert np.all(np.isfinite(channel))


# ── further solver invariants ─────────────────────────────────────────────────


@given(
  nx=st.integers(min_value=8, max_value=16),
  ny=st.integers(min_value=6, max_value=12),
  nh=st.integers(min_value=2, max_value=4),
)
@settings(max_examples=8, deadline=None)
def test_fem_2d_rectangular_envelope_and_load_bound(nx, ny, nh):
  """Rectangular FEM: pa <= p <= ps and 0 <= w <= (ps - pa) * area."""
  b = RectangularBearing(nx=nx, ny=ny, nh=nh)
  r = solve_bearing_fem_2d(b)
  for snap in r.p_2d:
    assert np.all(snap >= b.pa - 1.0)
    assert np.all(snap <= b.ps + 1.0)
  bound = (b.ps - b.pa) * b.A * 1.02
  assert np.all(r.w >= 0.0)
  assert np.all(r.w <= bound)


@given(
  cls=st.sampled_from(THRUST_CLASSES),
  xa=RADIUS,
  amp=st.floats(min_value=0.0, max_value=5e-6),
)
@settings(max_examples=12, deadline=None)
def test_gaps_stay_positive_with_error_profile(cls, xa, amp):
  """Film gap is strictly positive everywhere, even with an error profile."""
  kind = np.random.choice(["none", "linear", "quadratic"])
  if cls is AnnularBearing:
    b = cls(xa=xa, xc=xa / 3, error_type=kind, error=amp)
  else:
    b = cls(xa=xa, error_type=kind, error=amp)
  geom = get_geom_1d(b, x=b.x)
  assert np.all(geom >= 0.0)
  assert np.all(b.ha[:, None] + geom[None, :] > 0.0) if geom.ndim == 1 else True


@given(cls=st.sampled_from(THRUST_CLASSES), xa=RADIUS)
@settings(max_examples=10, deadline=None)
def test_thrust_stiffness_positive_in_interior(cls, xa):
  """Interior stiffness samples are positive (load falls with gap)."""
  if cls is AnnularBearing:
    b = cls(xa=xa, xc=xa / 3)
  else:
    b = cls(xa=xa)
  k = solve_bearing_analytic(b).k
  assert np.all(k[1:-1] > 0.0)


def test_annular_chamber_edge_holds_pc():
  """Annular seal: inner edge pressure equals the chamber pressure."""
  pc = 0.3e6 + 101325.0
  b = AnnularBearing(pc=pc)
  r = solve_bearing_analytic(b)
  np.testing.assert_allclose(r.p[0, :], pc, rtol=1e-6)
