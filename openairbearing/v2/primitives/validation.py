"""Small validation vocabulary for public scalar and name boundaries."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np

__all__ = [
  "int_ge",
  "nonempty_str",
  "nonneg_int",
  "nonneg_real",
  "pos_int",
  "pos_real",
  "real",
]


def real(value: float, name: str) -> float:
  """Normalize one finite real scalar."""
  out = _real_number(value, name)
  if not np.isfinite(out):
    raise ValueError(f"{name} must be finite, got {value!r}")
  return out


def pos_real(value: float, name: str) -> float:
  """Normalize one finite real scalar strictly above zero."""
  out = _real_number(value, name)
  if not np.isfinite(out) or out <= 0.0:
    raise ValueError(f"{name} must be positive and finite, got {value!r}")
  return out


def nonneg_real(value: float, name: str) -> float:
  """Normalize one finite real scalar greater than or equal to zero."""
  out = _real_number(value, name)
  if not np.isfinite(out) or out < 0.0:
    raise ValueError(f"{name} must be non-negative and finite, got {value!r}")
  return out


def int_ge(value: int, name: str, minimum: int) -> int:
  """Normalize one integral scalar with an inclusive lower bound."""
  if isinstance(value, bool) or not isinstance(value, Integral):
    raise TypeError(f"{name} must be an int, got {value!r}")
  out = int(value)
  if out < minimum:
    raise ValueError(f"{name} must be at least {minimum}, got {value!r}")
  return out


def pos_int(value: int, name: str) -> int:
  """Normalize one integral scalar strictly above zero."""
  return int_ge(value, name, 1)


def nonneg_int(value: int, name: str) -> int:
  """Normalize one integral scalar greater than or equal to zero."""
  return int_ge(value, name, 0)


def nonempty_str(value: str, name: str) -> str:
  """Normalize one non-empty string."""
  if not isinstance(value, str):
    raise TypeError(f"{name} must be a string, got {value!r}")
  if not value:
    raise ValueError(f"{name} must be non-empty")
  return value


def _real_number(value: float, name: str) -> float:
  if isinstance(value, bool) or not isinstance(value, Real):
    raise TypeError(f"{name} must be a real number, got {value!r}")
  return float(value)
