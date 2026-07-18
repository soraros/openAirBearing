"""Porous-media restrictor model: permeability plus calibration provenance.

The restrictor is characterized by its Darcy permeability κ [m²] and wall
thickness [m]. Porous media are commonly specified by a measured free-flow
rate ``q_lpm`` [L/min] at a calibration pressure; ``from_flow`` converts
that spec into permeability, and ``flow_lpm`` converts back:

    Q = κ · A · (p² − p_a²) / (2 · μ · h_p · p_a)
"""

from __future__ import annotations

from dataclasses import dataclass

from openairbearing.v2.primitives.constants import AIR, LPM_PER_M3S, P_ATM, Gas
from openairbearing.v2.primitives.types import F64
from openairbearing.v2.primitives.validation import pos_real

__all__ = ["PorousRestrictor"]


@dataclass(frozen=True, slots=True)
class PorousRestrictor:
  """Porous-media feed characterized by permeability κ and wall thickness."""

  permeability: float  # m²
  thickness: float = 4.5e-3  # m

  def __post_init__(self) -> None:
    object.__setattr__(
      self, "permeability", pos_real(self.permeability, "PorousRestrictor.permeability")
    )
    object.__setattr__(self, "thickness", pos_real(self.thickness, "PorousRestrictor.thickness"))

  @classmethod
  def from_flow(
    cls,
    q_lpm: float,
    p_calibration: float,
    area: float,
    *,
    thickness: float = 4.5e-3,
    gas: Gas = AIR,
    p_ambient: float = P_ATM,
  ) -> PorousRestrictor:
    """Build κ from the standard free-flow spec [L/min] at ``p_calibration``."""
    q = pos_real(q_lpm, "q_lpm") / LPM_PER_M3S
    p_c = pos_real(p_calibration, "p_calibration")
    area = pos_real(area, "area")
    thickness = pos_real(thickness, "thickness")
    p_a = pos_real(p_ambient, "p_ambient")
    if p_c <= p_a:
      raise ValueError(f"p_calibration must exceed p_ambient, got {p_calibration!r}")
    kappa = 2.0 * q * gas.mu * thickness * p_a / (area * (p_c**2 - p_a**2))
    return cls(permeability=kappa, thickness=thickness)

  def flow_lpm(
    self,
    area: float,
    p_supply: float,
    *,
    gas: Gas = AIR,
    p_ambient: float = P_ATM,
  ) -> float:
    """Free-flow rate [L/min] through ``area`` at ``p_supply`` (inverse of from_flow)."""
    area = pos_real(area, "area")
    p_s = pos_real(p_supply, "p_supply")
    p_a = pos_real(p_ambient, "p_ambient")
    if p_s <= p_a:
      raise ValueError(f"p_supply must exceed p_ambient, got {p_supply!r}")
    q = self.permeability * area * (p_s**2 - p_a**2) / (2.0 * gas.mu * self.thickness * p_a)
    return float(q * LPM_PER_M3S)

  def source_coeff(self, gas: Gas) -> float:
    """Sink coefficient in the p² film equation: −κ / (2 · h_p · μ)."""
    return -self.permeability / (2.0 * self.thickness * gas.mu)

  def feeding_parameter(self, extent: float, gaps: F64) -> F64:
    """Porous feeding parameter β = 6κL²/(h_p·h³) evaluated per sample gap."""
    extent = pos_real(extent, "extent")
    return 6.0 * self.permeability * extent**2 / (self.thickness * gaps**3)
