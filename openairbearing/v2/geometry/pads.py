"""Pad geometry specs: pure declarative data, no derived arrays.

Grid coordinates, film stacks, and integration weights are compiled from
these specs by ``openairbearing.v2.problem.build_problem`` — the pads
themselves only own dimensions, areas, and class-level layout metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np

from openairbearing.v2.primitives.validation import pos_real

__all__ = [
  "AnnularPad",
  "CircularPad",
  "CoordinateSystem",
  "JournalPad",
  "LinearPad",
  "Pad",
  "RectangularPad",
]

type CoordinateSystem = Literal["polar", "cartesian"]


@dataclass(frozen=True, slots=True)
class CircularPad:
  """Circular thrust pad; polar 1-D grid over r in [r_center, r]."""

  r: float = 37e-3 / 2
  r_center: float = 1e-6

  CSYS: ClassVar[CoordinateSystem] = "polar"
  DIM: ClassVar[int] = 1
  SEAL: ClassVar[bool] = False
  ANALYTIC: ClassVar[bool] = True

  def __post_init__(self) -> None:
    object.__setattr__(self, "r", pos_real(self.r, "CircularPad.r"))
    r_center = pos_real(self.r_center, "CircularPad.r_center")
    if r_center >= self.r:
      raise ValueError(f"CircularPad.r_center must be below r, got {self.r_center!r}")
    object.__setattr__(self, "r_center", r_center)

  @property
  def area(self) -> float:
    """Pad face area [m²]."""
    return float(np.pi * self.r**2)

  @property
  def extent(self) -> float:
    """Characteristic length for the porous feeding parameter β."""
    return self.r


@dataclass(frozen=True, slots=True)
class AnnularPad:
  """Annular thrust pad (ring seal); polar 1-D grid over r in [r_inner, r]."""

  r: float = 58e-3 / 2
  r_inner: float = 25e-3 / 2

  CSYS: ClassVar[CoordinateSystem] = "polar"
  DIM: ClassVar[int] = 1
  SEAL: ClassVar[bool] = True
  ANALYTIC: ClassVar[bool] = True

  def __post_init__(self) -> None:
    object.__setattr__(self, "r", pos_real(self.r, "AnnularPad.r"))
    r_inner = pos_real(self.r_inner, "AnnularPad.r_inner")
    if r_inner >= self.r:
      raise ValueError(f"AnnularPad.r_inner must be below r, got {self.r_inner!r}")
    object.__setattr__(self, "r_inner", r_inner)

  @property
  def area(self) -> float:
    """Pad face area [m²]."""
    return float(np.pi * (self.r**2 - self.r_inner**2))

  @property
  def extent(self) -> float:
    """Characteristic length for the porous feeding parameter β."""
    return self.r


@dataclass(frozen=True, slots=True)
class LinearPad:
  """Infinitely wide linear pad; cartesian 1-D grid over x in [0, length].

  Loads and flows are per unit width [N/m, L/min per m].
  """

  length: float = 40e-3

  CSYS: ClassVar[CoordinateSystem] = "cartesian"
  DIM: ClassVar[int] = 1
  SEAL: ClassVar[bool] = True
  ANALYTIC: ClassVar[bool] = True

  def __post_init__(self) -> None:
    object.__setattr__(self, "length", pos_real(self.length, "LinearPad.length"))

  @property
  def area(self) -> float:
    """Pad face area per unit width [m]."""
    return self.length

  @property
  def extent(self) -> float:
    """Characteristic length for the porous feeding parameter β."""
    return self.length


@dataclass(frozen=True, slots=True)
class RectangularPad:
  """Rectangular thrust pad; cartesian 2-D grid centered on the origin."""

  lx: float = 80e-3
  ly: float = 40e-3

  CSYS: ClassVar[CoordinateSystem] = "cartesian"
  DIM: ClassVar[int] = 2
  SEAL: ClassVar[bool] = False
  ANALYTIC: ClassVar[bool] = False

  def __post_init__(self) -> None:
    object.__setattr__(self, "lx", pos_real(self.lx, "RectangularPad.lx"))
    object.__setattr__(self, "ly", pos_real(self.ly, "RectangularPad.ly"))

  @property
  def area(self) -> float:
    """Pad face area [m²]."""
    return self.lx * self.ly

  @property
  def extent(self) -> float:
    """Characteristic length for the porous feeding parameter β."""
    return self.lx


@dataclass(frozen=True, slots=True)
class JournalPad:
  """Journal bearing pad; unwrapped 2-D grid over (θ, y).

  ``clearance`` is the diameter clearance (journal radius minus shaft
  radius is ``clearance / 2`` in the film model, matching the v1
  convention). The sample axis sweeps eccentricity instead of gap.
  """

  r: float = 50.02e-3 / 2
  length: float = 89e-3
  clearance: float = 40e-6

  CSYS: ClassVar[CoordinateSystem] = "cartesian"
  DIM: ClassVar[int] = 2
  SEAL: ClassVar[bool] = True
  ANALYTIC: ClassVar[bool] = False

  def __post_init__(self) -> None:
    object.__setattr__(self, "r", pos_real(self.r, "JournalPad.r"))
    object.__setattr__(self, "length", pos_real(self.length, "JournalPad.length"))
    object.__setattr__(self, "clearance", pos_real(self.clearance, "JournalPad.clearance"))

  @property
  def area(self) -> float:
    """Unwrapped pad face area [m²]."""
    return float(2.0 * np.pi * self.r * self.length)

  @property
  def extent(self) -> float:
    """Characteristic length for the porous feeding parameter β."""
    return self.r


type Pad = CircularPad | AnnularPad | LinearPad | RectangularPad | JournalPad
