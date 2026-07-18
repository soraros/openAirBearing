"""Cross-validation of v2 against the legacy v1 implementation.

Tolerances: v1 rounds permeability to 3 significant digits, so 1-D results
agree to ~5e-4. The rectangular 2-D case agrees only loosely: v1 used grid
spacings inconsistent with its node coordinates (dx = lx/(nx+1) on a
linspace spanning lx), which v2 fixes — see the v1 utils/solvers source.
"""

from dataclasses import replace

import numpy as np
import pytest

from openairbearing.bearings import (
  AnnularBearing,
  CircularBearing,
  InfiniteLinearBearing,
  RectangularBearing,
)
from openairbearing.solvers import solve_bearing as solve_v1
from openairbearing.v2 import catalog
from openairbearing.v2.solvers.api import solve_bearing as solve_v2

from .conftest import RTOL_V1

PAIRS_1D = [
  ("circular", CircularBearing, catalog.circular),
  ("annular", AnnularBearing, catalog.annular),
  ("linear", InfiniteLinearBearing, catalog.linear),
]


@pytest.mark.parametrize("name, V1, v2_spec", PAIRS_1D)
@pytest.mark.parametrize("method", ["analytic", "numeric"])
def test_v1_pressure_load_stiffness_flow(name, V1, v2_spec, method):
  """1-D pads: v2 matches v1 within the v1 kappa rounding (5e-4)."""
  r1 = solve_v1(V1(), method)
  r2 = solve_v2(v2_spec(), method)
  np.testing.assert_allclose(r2.p, r1.p.T, rtol=RTOL_V1)
  np.testing.assert_allclose(r2.load, r1.w, rtol=RTOL_V1)
  np.testing.assert_allclose(r2.stiffness, r1.k, rtol=RTOL_V1)
  np.testing.assert_allclose(r2.q_supply, r1.qs, rtol=RTOL_V1)
  np.testing.assert_allclose(r2.q_ambient, r1.qa, rtol=RTOL_V1)
  # chamber flow is a near-zero residual for non-seal pads; floor the tolerance
  np.testing.assert_allclose(
    r2.q_chamber, r1.qc, rtol=RTOL_V1, atol=1e-4 * float(np.max(np.abs(r1.qc)))
  )


def test_v1_rectangular_numeric2d_loose(rectangular_spec):
  """Rectangular 2-D: same physics, looser tolerance for the v1 dx slop."""
  b1 = RectangularBearing(nx=15, ny=10, nh=3)
  r1 = solve_v1(b1, "numeric2d")
  spec = replace(rectangular_spec, grid=replace(rectangular_spec.grid, nx=15, ny=10, n_samples=3))
  r2 = solve_v2(spec, "numeric2d")
  # peak pressure is pinned by the supply pressure in both versions
  np.testing.assert_allclose(r2.p.max(), r1.p.max(), rtol=1e-3)
  # v2 load is closer to the physical bound (ps - pa)·A than v1's
  state = spec.state
  bound = (state.p_supply - state.p_ambient) * spec.pad.area
  assert r2.load[0] > r1.w[0]
  assert r2.load[0] < bound
  # same order of magnitude and same trend over the gap sweep; absolute
  # values diverge because v1's spacings/area weights were inconsistent
  np.testing.assert_allclose(r2.load, r1.w, rtol=1.0)
  assert r2.load[0] > r2.load[-1]


def test_v2_polar_2d_matches_1d(annular_spec):
  """v2 polar 2-D grid (v1's was broken) agrees with the 1-D solution."""
  spec = replace(annular_spec, grid=replace(annular_spec.grid, ny=24))
  r2d = solve_v2(spec, "numeric2d")
  r1d = solve_v2(annular_spec, "numeric")
  np.testing.assert_allclose(r2d.load, r1d.load, rtol=0.02)


def test_v1_journal_peak_pressure(journal_spec):
  """Journal: v1 and v2 peak pressures are both pinned near p_supply."""
  from openairbearing.bearings import JournalBearing

  b1 = JournalBearing(nx=40, ny=25, nh=5)
  r1 = solve_v1(b1, "numeric2d")
  spec = replace(journal_spec, grid=replace(journal_spec.grid, nx=40, ny=25, n_samples=5))
  r2 = solve_v2(spec, "numeric2d")
  np.testing.assert_allclose(r2.p.max(), r1.p.max(), rtol=1e-3)
