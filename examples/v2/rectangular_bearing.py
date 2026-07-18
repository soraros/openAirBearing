"""Rectangular thrust bearing solved on the 2-D FDM grid.

Port of the v1 ``rectangular_bearing`` example. The v2 grid uses spacings
consistent with the node coordinates (v1's did not), so absolute values
differ slightly from v1 — v2's load integrates over the true pad area.
"""

from dataclasses import replace

import plotly.graph_objects as go

from openairbearing.v2 import catalog, solve_bearing
from openairbearing.v2.geometry.profiles import SurfaceError
from openairbearing.v2.postprocess import plots


def build_figures() -> tuple[go.Figure, ...]:
  """Solve numeric2d and assemble the figure set (no rendering)."""
  spec = replace(
    catalog.rectangular(),
    error=SurfaceError(kind="linear", amplitude=3e-6),
    grid=replace(catalog.rectangular().grid, nx=50, ny=30),
  )
  result = solve_bearing(spec, "numeric2d")
  return (
    plots.plot_pressure_distribution(result),
    plots.plot_load_capacity(result),
    plots.plot_stiffness(result),
    plots.plot_supply_flow_rate(result),
  )


def main() -> None:
  for fig in build_figures():
    fig.show()


if __name__ == "__main__":
  main()
