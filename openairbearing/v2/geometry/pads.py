"""Pad geometry specs: declarative data plus pad-owned grid semantics.

Pads are frozen dimension records that also own their geometry semantics:
grid axes, boundary statements, error-profile extents, film stack, and
integration weights. ``build_problem`` (openairbearing.v2.problem)
orchestrates these methods into the compiled ``BearingProblem``.

``PadBase`` carries the shared defaults (thrust film, direct load
projection, extent-driven profile, boundary statements from ``EDGES``);
each pad declares only what is special about it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np

from openairbearing.v2.geometry.boundaries import BoundarySpec, EdgeBC
from openairbearing.v2.geometry.profiles import ProfileLayout
from openairbearing.v2.primitives.types import F64, AxisScalar, FieldScalar, SampleScalar
from openairbearing.v2.primitives.validation import pos_real

__all__ = [
  "AnnularPad",
  "CircularPad",
  "CoordinateSystem",
  "JournalPad",
  "LinearPad",
  "Pad",
  "PadBase",
  "RectangularPad",
]

type CoordinateSystem = Literal["polar", "cartesian"]

_AMBIENT = EdgeBC("dirichlet", "ambient")
_CHAMBER = EdgeBC("dirichlet", "chamber")
_NEUMANN = EdgeBC("neumann")
_PERIODIC = EdgeBC("periodic")


# ── Shared helpers ────────────────────────────────────────────────────────────


def _validate_pos(instance: object, *names: str) -> None:
  """pos_real-validate and write back the named fields of a frozen pad."""
  cls = type(instance)
  for name in names:
    object.__setattr__(instance, name, pos_real(getattr(instance, name), f"{cls.__name__}.{name}"))


def _theta_or_placeholder(ny: int) -> AxisScalar:
  """Periodic θ axis for polar pads solved on a 2-D grid."""
  if ny <= 1:
    return np.zeros(1)
  return np.linspace(0.0, 2.0 * np.pi, ny, endpoint=False)


def _gap_film(samples: SampleScalar, geom: FieldScalar) -> F64:
  """Nominal gap plus the error profile (thrust pads)."""
  if geom.ndim == 1:
    return samples[:, None] + geom[None, :]
  return samples[:, None, None] + geom[None, :, :]


def _dA_line(x: AxisScalar) -> AxisScalar:
  """1-D trapezoidal weights."""
  dA = np.gradient(x)
  dA[[0, -1]] /= 2.0
  return dA


def _dA_polar(x: AxisScalar, ny: int) -> FieldScalar:
  """Polar area weights: 1-D trapezoidal ring or exact 2-D annular cells."""
  if ny <= 1:
    dA = np.pi * np.gradient(x**2)
    dA[[0, -1]] /= 2.0
    return dA
  dx2 = x**2 - np.insert(x[:-1], 0, x[0]) ** 2
  dtheta = 2.0 * np.pi / ny
  return 0.5 * dx2[:, None] * dtheta * np.ones((x.size, ny))


# ── Pads ──────────────────────────────────────────────────────────────────────


class PadBase:
  """Shared pad defaults; subclasses declare fields and what is special."""

  CSYS: ClassVar[CoordinateSystem]
  DIM: ClassVar[int]
  SEAL: ClassVar[bool]
  ANALYTIC: ClassVar[bool]
  EDGES: ClassVar[tuple[EdgeBC, EdgeBC, EdgeBC, EdgeBC]]
  STIFFNESS_SIGN: ClassVar[float] = -1.0

  @property
  def area(self) -> float:
    raise NotImplementedError

  @property
  def extent(self) -> float:
    """Characteristic length for the porous feeding parameter β."""
    raise NotImplementedError

  def axes(self, nx: int, ny: int) -> tuple[AxisScalar, AxisScalar]:
    raise NotImplementedError

  def area_weights(self, x: AxisScalar, y: AxisScalar) -> FieldScalar:
    raise NotImplementedError

  def boundaries(self) -> BoundarySpec:
    return BoundarySpec(*self.EDGES)

  def profile_args(self) -> tuple[float, float, ProfileLayout]:
    return self.extent, self.extent, self.CSYS

  def film(self, samples: SampleScalar, geom: FieldScalar, x: AxisScalar, y: AxisScalar) -> F64:
    """Nominal gap plus the error profile."""
    return _gap_film(samples, geom)

  def load_weights(self, x: AxisScalar, y: AxisScalar, dA: FieldScalar) -> FieldScalar:
    """Thrust pads integrate gauge pressure directly."""
    return dA


@dataclass(frozen=True, slots=True)
class CircularPad(PadBase):
  """Circular thrust pad; polar 1-D grid over r in [r_center, r]."""

  r: float = 37e-3 / 2
  r_center: float = 1e-6

  CSYS: ClassVar[CoordinateSystem] = "polar"
  DIM: ClassVar[int] = 1
  SEAL: ClassVar[bool] = False
  ANALYTIC: ClassVar[bool] = True
  EDGES: ClassVar[tuple[EdgeBC, EdgeBC, EdgeBC, EdgeBC]] = (
    _NEUMANN,
    _AMBIENT,
    _PERIODIC,
    _PERIODIC,
  )

  def __post_init__(self) -> None:
    _validate_pos(self, "r", "r_center")
    if self.r_center >= self.r:
      raise ValueError(f"CircularPad.r_center must be below r, got {self.r_center!r}")

  @property
  def area(self) -> float:
    """Pad face area [m²]."""
    return float(np.pi * self.r**2)

  @property
  def extent(self) -> float:
    return self.r

  def axes(self, nx: int, ny: int) -> tuple[AxisScalar, AxisScalar]:
    return np.linspace(self.r_center, self.r, nx), _theta_or_placeholder(ny)

  def area_weights(self, x: AxisScalar, y: AxisScalar) -> FieldScalar:
    return _dA_polar(x, y.size)


@dataclass(frozen=True, slots=True)
class AnnularPad(PadBase):
  """Annular thrust pad (ring seal); polar 1-D grid over r in [r_inner, r]."""

  r: float = 58e-3 / 2
  r_inner: float = 25e-3 / 2

  CSYS: ClassVar[CoordinateSystem] = "polar"
  DIM: ClassVar[int] = 1
  SEAL: ClassVar[bool] = True
  ANALYTIC: ClassVar[bool] = True
  EDGES: ClassVar[tuple[EdgeBC, EdgeBC, EdgeBC, EdgeBC]] = (
    _CHAMBER,
    _AMBIENT,
    _PERIODIC,
    _PERIODIC,
  )

  def __post_init__(self) -> None:
    _validate_pos(self, "r", "r_inner")
    if self.r_inner >= self.r:
      raise ValueError(f"AnnularPad.r_inner must be below r, got {self.r_inner!r}")

  @property
  def area(self) -> float:
    """Pad face area [m²]."""
    return float(np.pi * (self.r**2 - self.r_inner**2))

  @property
  def extent(self) -> float:
    return self.r

  def axes(self, nx: int, ny: int) -> tuple[AxisScalar, AxisScalar]:
    return np.linspace(self.r_inner, self.r, nx), _theta_or_placeholder(ny)

  def area_weights(self, x: AxisScalar, y: AxisScalar) -> FieldScalar:
    return _dA_polar(x, y.size)


@dataclass(frozen=True, slots=True)
class LinearPad(PadBase):
  """Infinitely wide linear pad; cartesian 1-D grid over x in [0, length].

  Loads and flows are per unit width [N/m, L/min per m].
  """

  length: float = 40e-3

  CSYS: ClassVar[CoordinateSystem] = "cartesian"
  DIM: ClassVar[int] = 1
  SEAL: ClassVar[bool] = True
  ANALYTIC: ClassVar[bool] = True
  EDGES: ClassVar[tuple[EdgeBC, EdgeBC, EdgeBC, EdgeBC]] = (
    _CHAMBER,
    _AMBIENT,
    _PERIODIC,
    _PERIODIC,
  )

  def __post_init__(self) -> None:
    _validate_pos(self, "length")

  @property
  def area(self) -> float:
    """Pad face area per unit width [m]."""
    return self.length

  @property
  def extent(self) -> float:
    return self.length

  def axes(self, nx: int, ny: int) -> tuple[AxisScalar, AxisScalar]:
    return np.linspace(0.0, self.length, nx), np.zeros(1)

  def area_weights(self, x: AxisScalar, y: AxisScalar) -> FieldScalar:
    return _dA_line(x)


@dataclass(frozen=True, slots=True)
class RectangularPad(PadBase):
  """Rectangular thrust pad; cartesian 2-D grid centered on the origin."""

  lx: float = 80e-3
  ly: float = 40e-3

  CSYS: ClassVar[CoordinateSystem] = "cartesian"
  DIM: ClassVar[int] = 2
  SEAL: ClassVar[bool] = False
  ANALYTIC: ClassVar[bool] = False
  EDGES: ClassVar[tuple[EdgeBC, EdgeBC, EdgeBC, EdgeBC]] = (
    _AMBIENT,
    _AMBIENT,
    _AMBIENT,
    _AMBIENT,
  )

  def __post_init__(self) -> None:
    _validate_pos(self, "lx", "ly")

  @property
  def area(self) -> float:
    """Pad face area [m²]."""
    return self.lx * self.ly

  @property
  def extent(self) -> float:
    return self.lx

  def axes(self, nx: int, ny: int) -> tuple[AxisScalar, AxisScalar]:
    return (
      np.linspace(-self.lx / 2, self.lx / 2, nx),
      np.linspace(-self.ly / 2, self.ly / 2, ny),
    )

  def profile_args(self) -> tuple[float, float, ProfileLayout]:
    return self.lx, self.ly, "cartesian"

  def area_weights(self, x: AxisScalar, y: AxisScalar) -> FieldScalar:
    return _dA_line(x)[:, None] * _dA_line(y)[None, :]


@dataclass(frozen=True, slots=True)
class JournalPad(PadBase):
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
  STIFFNESS_SIGN: ClassVar[float] = 1.0
  EDGES: ClassVar[tuple[EdgeBC, EdgeBC, EdgeBC, EdgeBC]] = (
    _PERIODIC,
    _PERIODIC,
    _AMBIENT,
    _AMBIENT,
  )

  def __post_init__(self) -> None:
    _validate_pos(self, "r", "length", "clearance")

  @property
  def area(self) -> float:
    """Unwrapped pad face area [m²]."""
    return float(2.0 * np.pi * self.r * self.length)

  @property
  def extent(self) -> float:
    return self.r

  def axes(self, nx: int, ny: int) -> tuple[AxisScalar, AxisScalar]:
    theta = np.linspace(-np.pi, np.pi, nx, endpoint=False)
    return theta, np.linspace(-self.length / 2, self.length / 2, ny)

  def profile_args(self) -> tuple[float, float, ProfileLayout]:
    return 2.0 * np.pi, self.length, "cartesian"

  def film(self, samples: SampleScalar, geom: FieldScalar, x: AxisScalar, y: AxisScalar) -> F64:
    """Eccentric clearance field h(e, θ) plus the error profile (v1 formula)."""
    e = samples[:, None]
    theta = x[None, :]
    c = self.clearance
    clearance = self.r - np.sqrt(
      (self.r - c / 2) ** 2 + e**2 + 2.0 * e * (self.r - c / 2) * np.cos(theta)
    )
    return clearance[:, :, None] + geom[None, :, :]

  def area_weights(self, x: AxisScalar, y: AxisScalar) -> FieldScalar:
    dtheta = 2.0 * np.pi / x.size
    return self.r * dtheta * _dA_line(y)[None, :] * np.ones((x.size, y.size))

  def load_weights(self, x: AxisScalar, y: AxisScalar, dA: FieldScalar) -> FieldScalar:
    """Project pressure onto cos θ along the eccentricity direction."""
    return np.cos(x)[:, None] * dA


type Pad = CircularPad | AnnularPad | LinearPad | RectangularPad | JournalPad
