"""openAirBearing v2: data-oriented air bearing analysis.

Layered architecture in the absim_fvm style:

- ``primitives`` — constants (``Gas``, ``AIR``, ``P_ATM``), array type
  aliases, and scalar validation helpers.
- ``geometry`` — frozen pad specs, error profiles, and grid resolution.
- ``operating`` — ``PorousRestrictor`` and ``OperatingState``.
- ``problem`` — ``BearingSpec`` compiled into the immutable
  ``BearingProblem`` (grid axes, film stack, integration weights).
- ``solvers`` — analytic and numba-accelerated FDM film solvers,
  ``solve_bearing`` dispatch, and the frozen ``Solution`` record.
- ``postprocess`` — plotly figures built from explicit solution data.
- ``catalog`` — default specs reproducing the v1 bearing catalog.

Typical usage::

  from openairbearing.v2 import catalog, solve_bearing

  spec = catalog.circular()
  result = solve_bearing(spec, "analytic")
  result.plot_key_results()
"""

from openairbearing.v2 import catalog
from openairbearing.v2.geometry.grid import GridSpec
from openairbearing.v2.geometry.pads import (
  AnnularPad,
  CircularPad,
  JournalPad,
  LinearPad,
  Pad,
  RectangularPad,
)
from openairbearing.v2.geometry.profiles import SurfaceError
from openairbearing.v2.operating.restrictor import PorousRestrictor
from openairbearing.v2.operating.state import OperatingState
from openairbearing.v2.postprocess.plots import (
  plot_ambient_flow_rate,
  plot_chamber_flow_rate,
  plot_key_results,
  plot_load_capacity,
  plot_pad_xy,
  plot_pad_xz,
  plot_pressure_distribution,
  plot_stiffness,
  plot_supply_flow_rate,
)
from openairbearing.v2.primitives.constants import AIR, LPM_PER_M3S, P_ATM, PA_PER_BAR, Gas
from openairbearing.v2.problem import (
  BearingProblem,
  BearingSpec,
  BoundarySpec,
  EdgeBC,
  build_problem,
)
from openairbearing.v2.solvers.api import solve_bearing, solve_pressure
from openairbearing.v2.solvers.solution import Solution

__all__ = [
  "AIR",
  "LPM_PER_M3S",
  "PA_PER_BAR",
  "P_ATM",
  "AnnularPad",
  "BearingProblem",
  "BearingSpec",
  "BoundarySpec",
  "CircularPad",
  "EdgeBC",
  "Gas",
  "GridSpec",
  "JournalPad",
  "LinearPad",
  "OperatingState",
  "Pad",
  "PorousRestrictor",
  "RectangularPad",
  "Solution",
  "SurfaceError",
  "build_problem",
  "catalog",
  "plot_ambient_flow_rate",
  "plot_chamber_flow_rate",
  "plot_key_results",
  "plot_load_capacity",
  "plot_pad_xy",
  "plot_pad_xz",
  "plot_pressure_distribution",
  "plot_stiffness",
  "plot_supply_flow_rate",
  "solve_bearing",
  "solve_pressure",
]
