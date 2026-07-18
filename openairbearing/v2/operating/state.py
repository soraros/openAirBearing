"""Operating state: reservoir pressures and slip correction."""

from __future__ import annotations

from dataclasses import dataclass

from openairbearing.v2.primitives.constants import P_ATM
from openairbearing.v2.primitives.validation import nonneg_real, pos_real

__all__ = ["OperatingState"]


@dataclass(frozen=True, slots=True)
class OperatingState:
  """Reservoir pressures [Pa, absolute] and the slip-flow factor Ψ.

  ``p_supply`` feeds the porous restrictor, ``p_chamber`` is the inner
  (recess/chamber-side) boundary of seal pads, and ``p_ambient`` is the
  outer boundary and the structural load reference.
  """

  p_supply: float = 0.6e6 + P_ATM
  p_chamber: float = P_ATM
  p_ambient: float = P_ATM
  slip: float = 0.0

  def __post_init__(self) -> None:
    object.__setattr__(self, "p_supply", pos_real(self.p_supply, "OperatingState.p_supply"))
    object.__setattr__(self, "p_chamber", pos_real(self.p_chamber, "OperatingState.p_chamber"))
    object.__setattr__(self, "p_ambient", pos_real(self.p_ambient, "OperatingState.p_ambient"))
    object.__setattr__(self, "slip", nonneg_real(self.slip, "OperatingState.slip"))
