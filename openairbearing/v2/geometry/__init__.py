"""Geometry layer: pad specs, error profiles, and grid resolution."""

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
  "AnnularPad",
  "CircularPad",
  "CoordinateSystem",
  "ErrorKind",
  "GridSpec",
  "JournalPad",
  "LinearPad",
  "Pad",
  "ProfileLayout",
  "RectangularPad",
  "SurfaceError",
  "eval_surface_error",
]
