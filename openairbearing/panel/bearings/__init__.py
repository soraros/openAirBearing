from ._base import BearingTypeConfig
from .annular import CONFIG as _annular
from .circular import CONFIG as _circular
from .infinite import CONFIG as _infinite
from .journal import CONFIG as _journal
from .rectangular import CONFIG as _rectangular

__all__ = [
  "BearingTypeConfig",
  "REGISTRY",
  "BEARING_CLASSES",
  "SOLVER_AVAILABILITY",
  "SOLVER_DEFAULTS",
  "VISIBILITY",
]

REGISTRY: dict[str, BearingTypeConfig] = {
  c.key: c for c in [_circular, _annular, _infinite, _rectangular, _journal]
}

# Convenience dicts derived from registry.
BEARING_CLASSES: dict[str, type] = {k: c.cls for k, c in REGISTRY.items()}
SOLVER_AVAILABILITY: dict[str, list[str]] = {k: c.solvers for k, c in REGISTRY.items()}
SOLVER_DEFAULTS: dict[str, list[str]] = {
  k: c.default_solvers for k, c in REGISTRY.items()
}
VISIBILITY: dict[str, dict[str, bool]] = {k: c.visibility for k, c in REGISTRY.items()}
