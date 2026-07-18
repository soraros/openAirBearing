"""Constants and gas properties. Units: SI."""

from __future__ import annotations

from dataclasses import dataclass

from openairbearing.v2.primitives.validation import pos_real

__all__ = ["AIR", "LPM_PER_M3S", "PA_PER_BAR", "P_ATM", "Gas"]

PA_PER_BAR: float = 1e5
P_ATM: float = 101325.0  # Pa (1 atm)
LPM_PER_M3S: float = 6e4  # L/min per m³/s


@dataclass(frozen=True, slots=True)
class Gas:
  """Working-gas properties.

  ``rho`` is the density measured at ``p_rho``. For air at 20 °C:
  rho = 1.293 kg/m³ at p_rho = 1 atm.
  """

  rho: float = 1.293  # kg/m³ at p_rho
  mu: float = 1.85e-5  # Pa·s
  p_rho: float = P_ATM  # Pa

  def __post_init__(self) -> None:
    object.__setattr__(self, "rho", pos_real(self.rho, "Gas.rho"))
    object.__setattr__(self, "mu", pos_real(self.mu, "Gas.mu"))
    object.__setattr__(self, "p_rho", pos_real(self.p_rho, "Gas.p_rho"))


AIR = Gas()
