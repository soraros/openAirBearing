"""Journal bearing: load and stiffness over the eccentricity sweep.

Port of the v1 ``journal_bearing`` example. The sample axis sweeps the
eccentricity e; the film is the eccentric clearance field. v2 uses a
proper periodic θ grid (endpoint-free), removing v1's duplicated-node
artifacts (negative load at small e, diverging flow at high e).
"""

from dataclasses import replace

import plotly.graph_objects as go

from openairbearing.v2 import catalog, solve_bearing
from openairbearing.v2.postprocess import plots


def build_figures() -> tuple[go.Figure, ...]:
  """Solve the default journal bearing and assemble figures."""
  spec = replace(catalog.journal(), grid=replace(catalog.journal().grid, nx=60, ny=40))
  result = solve_bearing(spec, "numeric2d")
  return (
    plots.plot_pressure_distribution(result),
    plots.plot_load_capacity(result),
    plots.plot_stiffness(result),
    plots.plot_ambient_flow_rate(result),
  )


def main() -> None:
  for fig in build_figures():
    fig.show()


if __name__ == "__main__":
  main()
