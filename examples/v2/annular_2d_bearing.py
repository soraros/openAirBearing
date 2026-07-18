"""Annular bearing solved on a polar 2-D grid (r, θ).

Port of the v1 ``2d_bearing`` example — which could not actually run in
v1 (its 2-D polar path was broken: a no-op ``factors`` expression and an
incompatible epsilon tuple). v2 implements the polar metric properly, and
the 2-D result cross-checks against the 1-D solution to ~2%.
"""

from dataclasses import replace

import numpy as np
import plotly.graph_objects as go

from openairbearing.v2 import catalog, solve_bearing
from openairbearing.v2.postprocess import plots


def build_figures() -> tuple[go.Figure, ...]:
  """Solve polar 2-D and compare against the 1-D numeric solution."""
  base = catalog.annular()
  spec_2d = replace(base, grid=replace(base.grid, nx=60, ny=48))
  result_2d = solve_bearing(spec_2d, "numeric2d")
  result_1d = solve_bearing(base, "numeric")
  rel = np.max(np.abs(result_2d.load - result_1d.load) / result_1d.load)
  print(f"polar 2-D vs 1-D load, max relative difference: {rel:.2%}")
  return (
    plots.plot_pressure_distribution(result_2d),
    plots.plot_load_capacity([result_2d, result_1d]),
    plots.plot_stiffness([result_2d, result_1d]),
  )


def main() -> None:
  for fig in build_figures():
    fig.show()


if __name__ == "__main__":
  main()
