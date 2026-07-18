"""Default bearing specs reproducing the v1 catalog.

Each factory returns a fully wired ``BearingSpec`` with the v1 default
geometry, porous-restrictor calibration, reservoir pressures, and grid.
Customize with ``dataclasses.replace`` or by constructing ``BearingSpec``
directly.

Note: v1 calibrates every default restrictor at ``0.6e6 + p_ambient`` —
for linear/rectangular/journal that is the *base-class* default ``psc``
in effect when ``get_kappa`` runs during ``__post_init__``, before the
subclass override lands — while solving at ``p_supply = 0.41e6``. The
catalog reproduces the actual v1 outputs with that convention made
explicit data.
"""

from __future__ import annotations

from openairbearing.v2.geometry.grid import GridSpec
from openairbearing.v2.geometry.pads import (
  AnnularPad,
  CircularPad,
  JournalPad,
  LinearPad,
  RectangularPad,
)
from openairbearing.v2.operating.restrictor import PorousRestrictor
from openairbearing.v2.operating.state import OperatingState
from openairbearing.v2.primitives.constants import P_ATM
from openairbearing.v2.problem import BearingSpec

__all__ = ["annular", "circular", "journal", "linear", "rectangular"]

_P_HIGH = 0.6e6 + P_ATM
_P_LOW = 0.41e6


def circular() -> BearingSpec:
  """Default circular thrust bearing (v1 CircularBearing)."""
  pad = CircularPad()
  return BearingSpec(
    pad=pad,
    restrictor=PorousRestrictor.from_flow(2.8, _P_HIGH, pad.area),
    state=OperatingState(p_supply=_P_HIGH),
    grid=GridSpec(nx=30, n_samples=20),
  )


def annular() -> BearingSpec:
  """Default annular thrust bearing (v1 AnnularBearing)."""
  pad = AnnularPad()
  return BearingSpec(
    pad=pad,
    restrictor=PorousRestrictor.from_flow(3.0, _P_HIGH, pad.area),
    state=OperatingState(p_supply=_P_HIGH),
    grid=GridSpec(nx=30, n_samples=20),
  )


def linear() -> BearingSpec:
  """Default infinitely wide linear bearing (v1 InfiniteLinearBearing)."""
  pad = LinearPad()
  return BearingSpec(
    pad=pad,
    restrictor=PorousRestrictor.from_flow(37.0, _P_HIGH, pad.area),
    state=OperatingState(p_supply=_P_LOW),
    grid=GridSpec(nx=30, n_samples=20),
  )


def rectangular() -> BearingSpec:
  """Default rectangular thrust bearing (v1 RectangularBearing)."""
  pad = RectangularPad()
  return BearingSpec(
    pad=pad,
    restrictor=PorousRestrictor.from_flow(2.94, _P_HIGH, pad.area),
    state=OperatingState(p_supply=_P_LOW),
    grid=GridSpec(nx=40, ny=20, n_samples=20),
  )


def journal() -> BearingSpec:
  """Default journal bearing (v1 JournalBearing).

  The sample axis sweeps eccentricity over [0.01e-6, c/2 - 1e-6].
  """
  pad = JournalPad()
  return BearingSpec(
    pad=pad,
    restrictor=PorousRestrictor.from_flow(15.0, _P_HIGH, pad.area, thickness=3e-3),
    state=OperatingState(p_supply=_P_LOW),
    grid=GridSpec(nx=80, ny=50, n_samples=20, sample_min=0.01e-6, sample_max=19e-6),
  )
