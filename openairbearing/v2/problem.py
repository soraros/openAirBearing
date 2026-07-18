"""Declarative bearing specs and the compiled film problem.

The data spine of v2: a ``BearingSpec`` is pure declarative data (pad,
restrictor, reservoir state, error profile, grid). ``build_problem``
compiles it once into a ``BearingProblem`` by orchestrating the pad-owned
grid semantics (see ``openairbearing.v2.geometry.pads``) into immutable
arrays with clean conventions:

- 1-D pads: fields are ``(nx,)``, sample stacks are ``(nh, nx)``.
- 2-D pads: fields are ``(nx, ny)``, sample stacks are ``(nh, nx, ny)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

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
  JournalPad,
  LinearPad,
  Pad,
  RectangularPad,
)
from openairbearing.v2.geometry.profiles import SurfaceError, eval_surface_error
from openairbearing.v2.operating.restrictor import PorousRestrictor
from openairbearing.v2.operating.state import OperatingState
from openairbearing.v2.primitives.constants import AIR, Gas
from openairbearing.v2.primitives.types import F64, AxisScalar, FieldScalar, SampleScalar

__all__ = [
  "BC_DIRICHLET",
  "BC_NEUMANN",
  "BC_PERIODIC",
  "BCKind",
  "BearingProblem",
  "BearingSpec",
  "BoundarySpec",
  "EdgeBC",
  "PressureSlot",
  "build_problem",
]

_PAD_TYPES: tuple[type, ...] = (CircularPad, AnnularPad, LinearPad, RectangularPad, JournalPad)


@dataclass(frozen=True, slots=True)
class BearingSpec:
  """Declarative bearing definition: pad, restrictor, state, error, grid."""

  pad: Pad
  restrictor: PorousRestrictor
  state: OperatingState = field(default_factory=OperatingState)
  error: SurfaceError = field(default_factory=SurfaceError)
  grid: GridSpec = field(default_factory=GridSpec)
  gas: Gas = AIR

  def __post_init__(self) -> None:
    if not isinstance(self.pad, _PAD_TYPES):
      raise TypeError(f"pad must be a v2 pad spec, got {self.pad!r}")
    if self.pad.DIM == 2 and self.grid.ny < 2:
      raise ValueError(f"{type(self.pad).__name__} requires grid.ny >= 2")


@dataclass(frozen=True, slots=True, init=False)
class BearingProblem:
  """Compiler-produced grid arrays, film stack, weights, and boundaries."""

  spec: BearingSpec
  x: AxisScalar
  y: AxisScalar
  dx: AxisScalar
  dy: AxisScalar
  samples: SampleScalar
  geom: FieldScalar
  gaps: F64
  dA: FieldScalar
  load_weights: FieldScalar
  stiffness_sign: float
  beta: SampleScalar
  source: float
  boundaries: BoundarySpec

  def __init__(self) -> None:
    raise TypeError("BearingProblem is compiler-produced; use build_problem")

  @classmethod
  def _from_compiled(
    cls,
    *,
    spec: BearingSpec,
    x: AxisScalar,
    y: AxisScalar,
    samples: SampleScalar,
    geom: FieldScalar,
    gaps: F64,
    dA: FieldScalar,
    load_weights: FieldScalar,
    boundaries: BoundarySpec,
  ) -> BearingProblem:
    """Join arrays emitted by the one canonical problem compiler."""
    out = object.__new__(cls)
    object.__setattr__(out, "spec", spec)
    object.__setattr__(out, "x", _readonly(x))
    object.__setattr__(out, "y", _readonly(y))
    object.__setattr__(out, "dx", _readonly(np.gradient(x)))
    object.__setattr__(out, "dy", _readonly(np.gradient(y) if y.size > 1 else np.ones(1)))
    object.__setattr__(out, "samples", _readonly(samples))
    object.__setattr__(out, "geom", _readonly(geom))
    object.__setattr__(out, "gaps", _readonly(gaps))
    object.__setattr__(out, "dA", _readonly(dA))
    object.__setattr__(out, "load_weights", _readonly(load_weights))
    object.__setattr__(out, "stiffness_sign", float(spec.pad.STIFFNESS_SIGN))
    object.__setattr__(
      out, "beta", _readonly(spec.restrictor.feeding_parameter(spec.pad.extent, samples))
    )
    object.__setattr__(out, "source", spec.restrictor.source_coeff(spec.gas))
    object.__setattr__(out, "boundaries", boundaries)
    return out

  @property
  def pad(self) -> Pad:
    return self.spec.pad

  @property
  def restrictor(self) -> PorousRestrictor:
    return self.spec.restrictor

  @property
  def state(self) -> OperatingState:
    return self.spec.state

  @property
  def gas(self) -> Gas:
    return self.spec.gas

  @property
  def area(self) -> float:
    return self.spec.pad.area

  @property
  def dim(self) -> int:
    """Spatial dimension of the compiled grid (1 or 2)."""
    return 1 if self.y.size <= 1 else 2

  @property
  def polar(self) -> bool:
    return self.pad.CSYS == "polar"

  @property
  def journal(self) -> bool:
    return isinstance(self.pad, JournalPad)

  @property
  def n_samples(self) -> int:
    return int(self.samples.size)


def build_problem(spec: BearingSpec) -> BearingProblem:
  """Compile one declarative spec into the immutable film problem."""
  if not isinstance(spec, BearingSpec):
    raise TypeError(f"spec must be a BearingSpec, got {spec!r}")
  pad = spec.pad
  x, y = pad.axes(spec.grid.nx, spec.grid.ny)
  samples = np.linspace(spec.grid.sample_min, spec.grid.sample_max, spec.grid.n_samples)
  x_extent, y_extent, layout = pad.profile_args()
  geom = eval_surface_error(
    spec.error, x=x, y=y, x_extent=x_extent, y_extent=y_extent, layout=layout
  )
  dA = pad.area_weights(x, y)
  return BearingProblem._from_compiled(
    spec=spec,
    x=x,
    y=y,
    samples=samples,
    geom=geom,
    gaps=pad.film(samples, geom, x, y),
    dA=dA,
    load_weights=pad.load_weights(x, y, dA),
    boundaries=pad.boundaries(),
  )


def _readonly(arr: F64) -> F64:
  out = np.ascontiguousarray(arr, dtype=np.float64)
  out.setflags(write=False)
  return out
