"""Circular thrust bearing with a quadratic error profile: analytic vs numeric.

Port of the v1 ``simple_circular_bearing`` example. Note the v2 style:
specs are immutable data customized with ``dataclasses.replace``, and one
``solve_bearing`` call produces the full sample sweep.
"""

from dataclasses import replace

import plotly.graph_objects as go

from openairbearing.v2 import catalog, solve_bearing
from openairbearing.v2.geometry.profiles import SurfaceError
from openairbearing.v2.operating.restrictor import PorousRestrictor
from openairbearing.v2.postprocess import plots
from openairbearing.v2.primitives.constants import P_ATM


def build_figures() -> tuple[go.Figure, ...]:
  """Solve and assemble the standard figure set (no rendering)."""
  base = catalog.circular()
  pad = replace(base.pad, r=40e-3 / 2)
  spec = replace(
    base,
    pad=pad,
    restrictor=PorousRestrictor.from_flow(5.0, 0.6e6 + P_ATM, pad.area),
    error=SurfaceError(kind="quadratic", amplitude=-2e-6),
    grid=replace(base.grid, nx=50, n_samples=60),
  )
  results = [solve_bearing(spec, "analytic"), solve_bearing(spec, "numeric")]
  return (
    plots.plot_load_capacity(results),
    plots.plot_stiffness(results),
    plots.plot_pressure_distribution(results),
    plots.plot_supply_flow_rate(results),
    plots.plot_ambient_flow_rate(results),
  )


def main() -> None:
  for fig in build_figures():
    fig.show()


if __name__ == "__main__":
  main()
