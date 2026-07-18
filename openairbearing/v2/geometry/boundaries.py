"""Boundary statements on grid edges: pure declarative data.

Dirichlet edges name a pressure slot resolved against the operating state
at solve time; Neumann and periodic edges carry no value. Integer codes
are what the numba kernels receive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from openairbearing.v2.operating.state import OperatingState

__all__ = [
  "BC_DIRICHLET",
  "BC_NEUMANN",
  "BC_PERIODIC",
  "BCKind",
  "BoundarySpec",
  "EdgeBC",
  "PressureSlot",
]

BC_DIRICHLET: int = 0
BC_NEUMANN: int = 1
BC_PERIODIC: int = 2

type BCKind = Literal["dirichlet", "neumann", "periodic"]
type PressureSlot = Literal["supply", "chamber", "ambient", ""]

_BC_CODES: dict[str, int] = {
  "dirichlet": BC_DIRICHLET,
  "neumann": BC_NEUMANN,
  "periodic": BC_PERIODIC,
}

_PRESSURE_SLOTS: tuple[str, ...] = ("supply", "chamber", "ambient", "")


@dataclass(frozen=True, slots=True)
class EdgeBC:
  """One grid-edge boundary statement; ``slot`` names the reservoir pressure."""

  kind: BCKind
  slot: PressureSlot = ""

  def __post_init__(self) -> None:
    if self.kind not in _BC_CODES:
      raise ValueError(f"EdgeBC.kind must be one of {tuple(_BC_CODES)}, got {self.kind!r}")
    if self.slot not in _PRESSURE_SLOTS:
      raise ValueError(f"EdgeBC.slot must be a pressure slot, got {self.slot!r}")
    if self.kind == "dirichlet" and not self.slot:
      raise ValueError("dirichlet EdgeBC requires a pressure slot")
    if self.kind != "dirichlet" and self.slot:
      raise ValueError(f"{self.kind} EdgeBC takes no pressure slot")

  @property
  def code(self) -> int:
    """Integer kind code passed to numba kernels."""
    return _BC_CODES[self.kind]

  def pressure(self, state: OperatingState) -> float:
    """Resolve the slot to an absolute pressure; 0 for value-free kinds."""
    if self.slot == "supply":
      return state.p_supply
    if self.slot == "chamber":
      return state.p_chamber
    if self.slot == "ambient":
      return state.p_ambient
    return 0.0


@dataclass(frozen=True, slots=True)
class BoundarySpec:
  """Boundary statements on the four grid edges (y edges unused in 1-D)."""

  x_lo: EdgeBC
  x_hi: EdgeBC
  y_lo: EdgeBC = field(default_factory=lambda: EdgeBC("periodic"))
  y_hi: EdgeBC = field(default_factory=lambda: EdgeBC("periodic"))
