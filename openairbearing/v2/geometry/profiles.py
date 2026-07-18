"""Manufacturing-error surface profiles added to the nominal film gap.

Profiles are pure data; ``eval_surface_error`` evaluates one profile on a
compiled grid. The evaluated field is always shifted so its minimum is
zero (the error opens the gap, it never closes it below nominal).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from openairbearing.v2.primitives.types import AxisScalar, FieldScalar
from openairbearing.v2.primitives.validation import real

__all__ = ["ErrorKind", "ProfileLayout", "SurfaceError", "eval_surface_error"]

type ErrorKind = Literal["none", "linear", "quadratic", "tiltx", "tilty"]
type ProfileLayout = Literal["line", "cartesian", "polar"]

_ERROR_KINDS: tuple[str, ...] = ("none", "linear", "quadratic", "tiltx", "tilty")


@dataclass(frozen=True, slots=True)
class SurfaceError:
  """Profiled manufacturing error; ``amplitude`` is the peak deviation [m]."""

  kind: ErrorKind = "none"
  amplitude: float = 0.0

  def __post_init__(self) -> None:
    if self.kind not in _ERROR_KINDS:
      raise ValueError(f"SurfaceError.kind must be one of {_ERROR_KINDS}, got {self.kind!r}")
    object.__setattr__(self, "amplitude", real(self.amplitude, "SurfaceError.amplitude"))


def eval_surface_error(
  error: SurfaceError,
  *,
  x: AxisScalar,
  y: AxisScalar,
  x_extent: float,
  y_extent: float,
  layout: ProfileLayout,
) -> FieldScalar:
  """Evaluate one error profile on a grid, shifted to a zero minimum.

  ``x_extent``/``y_extent`` are the full axis spans used to normalize the
  profile (pad length for cartesian grids, outer radius for polar grids,
  2π for the unwrapped journal coordinate).
  """
  a = error.amplitude
  kind = error.kind
  if y.size <= 1:
    if kind == "none":
      geom = np.zeros_like(x)
    elif kind == "linear":
      geom = a * (1.0 - x / x_extent)
    elif kind == "quadratic":
      geom = a * (1.0 - (x / x_extent) ** 2)
    else:
      raise ValueError(f"error kind {kind!r} is not supported on 1-D grids")
    return geom - float(np.min(geom))

  if layout == "cartesian":
    u = x[:, None] / x_extent + np.zeros((x.size, y.size))
    v = y[None, :] / y_extent + np.zeros((x.size, y.size))
    if kind == "none":
      geom = np.zeros((x.size, y.size))
    elif kind == "linear":
      geom = 2.0 * a * np.maximum(np.abs(u), np.abs(v))
    elif kind == "quadratic":
      geom = 4.0 * a * np.maximum(u**2, v**2)
    elif kind == "tiltx":
      geom = a * u
    elif kind == "tilty":
      geom = a * v
    else:
      raise ValueError(f"unknown error kind {kind!r}")
  elif layout == "polar":
    r = x[:, None]
    if kind == "none":
      geom = np.zeros((x.size, y.size))
    elif kind == "linear":
      geom = a * (1.0 - r / x_extent) + np.zeros((x.size, y.size))
    elif kind == "quadratic":
      geom = a * (1.0 - (r / x_extent) ** 2) + np.zeros((x.size, y.size))
    else:
      raise ValueError(f"error kind {kind!r} is not supported on polar grids")
  else:
    raise ValueError(f"unknown profile layout {layout!r}")
  return geom - float(np.min(geom))
