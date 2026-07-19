"""OpenAir - Air Bearing Analysis Tool.

This package provides tools for analyzing and visualizing air bearing performance.
"""

# Version information
__version__ = "0.1.8"

# Import constants from config (where they're actually defined)
# Import app
from .app.app import app

# Import specific bearing classes
from .bearings import (
    AnnularBearing,
    CircularBearing,
    InfiniteLinearBearing,
    JournalBearing,
    RectangularBearing,
)
from .config import DEMO_MODE

# Import visualization functions
from .plots import (
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

# Import solver function
from .solvers import get_pressure_numeric, solve_bearing

# Import utility functions from where they're defined
from .utils import (
    Result,
    get_area,
    get_beta,
    get_geom,
    get_kappa,
    get_Qsc,
)

# Define what should be available when using 'from openairbearing import *'
__all__ = [
    # App
    "app",
    # Bearing types
    "RectangularBearing",
    "CircularBearing",
    "AnnularBearing",
    "InfiniteLinearBearing",
    "JournalBearing",
    # Bearing parameters
    "get_kappa",
    "get_Qsc",
    "get_beta",
    "get_geom",
    "get_area",
    # Result type
    "Result",
    # Solver
    "solve_bearing",
    "get_pressure_numeric",
    # Configuration
    "DEMO_MODE",
    # Visualization
    "plot_bearing_shape",
    "plot_key_results",
    "plot_load_capacity",
    "plot_stiffness",
    "plot_pressure_distribution",
    "plot_supply_flow_rate",
    "plot_chamber_flow_rate",
    "plot_ambient_flow_rate",
    "plot_xy_shape",
    "plot_xz_shape",
]
