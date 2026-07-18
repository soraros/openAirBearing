"""Grid resolution spec: node counts plus the film-sample sweep."""

from __future__ import annotations

from dataclasses import dataclass

from openairbearing.v2.primitives.validation import int_ge, pos_real

__all__ = ["GridSpec"]


@dataclass(frozen=True, slots=True)
class GridSpec:
  """Discretization and film-sample sweep.

  The sample axis sweeps the nominal film gap [m] for thrust pads and the
  eccentricity [m] for journal pads.
  """

  nx: int = 30
  ny: int = 1
  n_samples: int = 20
  sample_min: float = 1e-6
  sample_max: float = 20e-6

  def __post_init__(self) -> None:
    object.__setattr__(self, "nx", int_ge(self.nx, "GridSpec.nx", 3))
    object.__setattr__(self, "ny", int_ge(self.ny, "GridSpec.ny", 1))
    object.__setattr__(self, "n_samples", int_ge(self.n_samples, "GridSpec.n_samples", 3))
    sample_min = pos_real(self.sample_min, "GridSpec.sample_min")
    sample_max = pos_real(self.sample_max, "GridSpec.sample_max")
    if sample_max <= sample_min:
      raise ValueError(f"GridSpec.sample_max must exceed sample_min, got {self.sample_max!r}")
    object.__setattr__(self, "sample_min", sample_min)
    object.__setattr__(self, "sample_max", sample_max)
