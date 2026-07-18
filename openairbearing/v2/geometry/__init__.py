"""Geometry layer: pad specs, boundary statements, error profiles, grid."""

from openairbearing.v2.geometry.boundaries import (
  BC_DIRICHLET,
  BC_NEUMANN,
  BC_PERIODIC,
  BCKind,
  BoundarySpec,
  EdgeBC,
  PressureSlot,
)
from openairbearing.v2.geometry.grid import GridSpec
from openairbearing.v2.geometry.pads import (
  AnnularPad,
  CircularPad,
  CoordinateSystem,
  JournalPad,
  LinearPad,
  Pad,
  RectangularPad,
)
from openairbearing.v2.geometry.profiles import (
  ErrorKind,
  ProfileLayout,
  SurfaceError,
  eval_surface_error,
)

__all__ = [
  "BC_DIRICHLET",
  "BC_NEUMANN",
  "BC_PERIODIC",
  "AnnularPad",
  "BCKind",
  "BoundarySpec",
  "CircularPad",
  "CoordinateSystem",
  "EdgeBC",
  "ErrorKind",
  "GridSpec",
  "JournalPad",
  "LinearPad",
  "Pad",
  "PressureSlot",
  "ProfileLayout",
  "RectangularPad",
  "SurfaceError",
  "eval_surface_error",
]
