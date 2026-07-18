"""Tests for the v2 plotting layer."""

import plotly.graph_objects as go
import pytest

from openairbearing.v2 import catalog
from openairbearing.v2.postprocess import plots
from openairbearing.v2.solvers.api import solve_bearing

PADS_1D = ["circular", "annular", "linear"]


def _solve(name: str, method: str = "analytic"):
  return solve_bearing(getattr(catalog, name)(), method)


@pytest.mark.parametrize("name", PADS_1D)
def test_plot_load_capacity(name):
  fig = plots.plot_load_capacity(_solve(name))
  assert isinstance(fig, go.Figure)
  assert len(fig.data) > 0


@pytest.mark.parametrize("name", PADS_1D)
def test_plot_stiffness(name):
  fig = plots.plot_stiffness(_solve(name))
  assert isinstance(fig, go.Figure)
  assert len(fig.data) > 0


@pytest.mark.parametrize("name", PADS_1D)
def test_plot_pressure_distribution_1d(name):
  fig = plots.plot_pressure_distribution(_solve(name))
  assert isinstance(fig, go.Figure)
  assert len(fig.data) > 0


def test_plot_pressure_distribution_2d():
  fig = plots.plot_pressure_distribution(_solve("rectangular", "numeric2d"))
  assert isinstance(fig, go.Figure)
  assert len(fig.data) > 0


@pytest.mark.parametrize(
  "plot_fn",
  [plots.plot_supply_flow_rate, plots.plot_ambient_flow_rate, plots.plot_chamber_flow_rate],
)
def test_plot_flow_rates(plot_fn):
  fig = plot_fn(_solve("circular"))
  assert isinstance(fig, go.Figure)
  assert len(fig.data) > 0


def test_plot_multiple_solutions():
  sols = [_solve("circular", "analytic"), _solve("circular", "numeric")]
  fig = plots.plot_load_capacity(sols)
  assert len(fig.data) == 2


def test_key_results_bearing_vs_seal():
  """Seal pads add the chamber-flow figure."""
  assert len(plots.plot_key_results(_solve("circular"))) == 5
  assert len(plots.plot_key_results(_solve("annular"))) == 6


@pytest.mark.parametrize("name", ["circular", "annular", "linear", "rectangular", "journal"])
def test_plot_pad_shapes(name):
  spec = getattr(catalog, name)()
  assert isinstance(plots.plot_pad_xy(spec), go.Figure)
  assert isinstance(plots.plot_pad_xz(spec), go.Figure)


def test_solution_plot_methods():
  sol = _solve("circular")
  assert isinstance(sol.plot_pressure(), go.Figure)
  assert len(sol.plot_key_results()) == 5


def test_empty_input_raises():
  with pytest.raises(ValueError, match="at least one"):
    plots.plot_load_capacity([])
