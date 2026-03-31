import numpy as np
import plotly.graph_objects as go
import pytest

from openairbearing.bearings import (
  AnnularBearing,
  CircularBearing,
  InfiniteLinearBearing,
  RectangularBearing,
)
from openairbearing.plots import (
  empty_figure,
  plot_ambient_flow_rate,
  plot_bearing_shape,
  plot_chamber_flow_rate,
  plot_key_results,
  plot_load_capacity,
  plot_pressure_distribution,
  plot_stiffness,
  plot_supply_flow_rate,
  plot_xy_shape,
  plot_xz_shape,
)
from openairbearing.solvers import solve_bearing


# ── Helpers ───────────────────────────────────────────────────────────


def _solve(bearing, soltype="analytic"):
  return solve_bearing(bearing, soltype)


# ── Individual plot functions ─────────────────────────────────────────


@pytest.mark.parametrize(
  "BearingCls", [CircularBearing, AnnularBearing, InfiniteLinearBearing]
)
def test_plot_load_capacity_returns_figure(BearingCls):
  """plot_load_capacity returns a Plotly Figure with traces."""
  b = BearingCls()
  r = _solve(b)
  fig = plot_load_capacity(b, [r])
  assert isinstance(fig, go.Figure)
  assert len(fig.data) > 0


@pytest.mark.parametrize(
  "BearingCls", [CircularBearing, AnnularBearing, InfiniteLinearBearing]
)
def test_plot_stiffness_returns_figure(BearingCls):
  b = BearingCls()
  r = _solve(b)
  fig = plot_stiffness(b, [r])
  assert isinstance(fig, go.Figure)
  assert len(fig.data) > 0


@pytest.mark.parametrize(
  "BearingCls", [CircularBearing, AnnularBearing, InfiniteLinearBearing]
)
def test_plot_pressure_distribution_1d(BearingCls):
  """1D pressure distribution returns figure with curves."""
  b = BearingCls()
  r = _solve(b)
  fig = plot_pressure_distribution(b, [r])
  assert isinstance(fig, go.Figure)
  assert len(fig.data) > 0


def test_plot_pressure_distribution_2d():
  """2D pressure produces contour plot."""
  b = RectangularBearing(nx=10, ny=8, nh=3)
  r = solve_bearing(b, "numeric2d")
  fig = plot_pressure_distribution(b, [r])
  assert isinstance(fig, go.Figure)
  assert len(fig.data) > 0


@pytest.mark.parametrize(
  "BearingCls", [CircularBearing, AnnularBearing, InfiniteLinearBearing]
)
def test_plot_flow_rates_return_figures(BearingCls):
  b = BearingCls()
  r = _solve(b)
  for plot_fn in [plot_supply_flow_rate, plot_ambient_flow_rate]:
    fig = plot_fn(b, [r])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_plot_chamber_flow_rate_seal():
  """Chamber flow plot works for seal-type (annular)."""
  b = AnnularBearing()
  r = _solve(b)
  fig = plot_chamber_flow_rate(b, [r])
  assert isinstance(fig, go.Figure)
  assert len(fig.data) > 0


# ── Composite plot functions ──────────────────────────────────────────


@pytest.mark.parametrize(
  "BearingCls, expected_count",
  [
    (CircularBearing, 5),  # bearing type: load, stiffness, pressure, qs, qa
    (AnnularBearing, 6),  # seal type: +qc
    (InfiniteLinearBearing, 6),
  ],
)
def test_plot_key_results_count(BearingCls, expected_count):
  """plot_key_results returns correct number of figures based on type."""
  b = BearingCls()
  r = _solve(b)
  figs = plot_key_results(b, [r])
  assert len(figs) == expected_count
  assert all(isinstance(f, go.Figure) for f in figs)


# ── Shape plots ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
  "BearingCls",
  [CircularBearing, AnnularBearing, InfiniteLinearBearing, RectangularBearing],
)
def test_plot_xy_shape_returns_figure(BearingCls):
  b = BearingCls()
  fig = plot_xy_shape(b)
  assert isinstance(fig, go.Figure)
  assert len(fig.data) > 0


@pytest.mark.parametrize(
  "BearingCls",
  [CircularBearing, AnnularBearing, InfiniteLinearBearing, RectangularBearing],
)
def test_plot_xz_shape_returns_figure(BearingCls):
  b = BearingCls()
  fig = plot_xz_shape(b)
  assert isinstance(fig, go.Figure)


@pytest.mark.parametrize(
  "BearingCls",
  [CircularBearing, AnnularBearing, InfiniteLinearBearing, RectangularBearing],
)
def test_plot_bearing_shape_returns_two(BearingCls):
  """plot_bearing_shape returns [xy, xz]."""
  b = BearingCls()
  figs = plot_bearing_shape(b)
  assert len(figs) == 2
  assert all(isinstance(f, go.Figure) for f in figs)


# ── Misc ──────────────────────────────────────────────────────────────


def test_empty_figure():
  fig = empty_figure()
  assert isinstance(fig, go.Figure)
