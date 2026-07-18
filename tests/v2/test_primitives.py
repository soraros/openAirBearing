"""Tests for primitives: validation, gas, and the porous restrictor."""

import numpy as np
import pytest

from openairbearing.v2.primitives.constants import AIR, P_ATM, Gas
from openairbearing.v2.primitives.validation import (
  nonempty_str,
  nonneg_real,
  pos_int,
  pos_real,
  real,
)
from openairbearing.v2.operating.restrictor import PorousRestrictor

# ── validation ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("value", [1.0, 0.5, 1e-15])
def test_pos_real_accepts(value):
  """pos_real accepts positive finite reals."""
  assert pos_real(value, "x") == float(value)


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, "1", None, True])
def test_pos_real_rejects(value):
  with pytest.raises((TypeError, ValueError)):
    pos_real(value, "x")


@pytest.mark.parametrize("value", [0.0, 1.0, -3.5])
def test_real_and_nonneg(value):
  assert real(value, "x") == float(value)
  if value < 0:
    with pytest.raises(ValueError):
      nonneg_real(value, "x")
  else:
    assert nonneg_real(value, "x") == float(value)


@pytest.mark.parametrize("value", [1, 3, 10])
def test_pos_int_accepts(value):
  assert pos_int(value, "n") == value


@pytest.mark.parametrize("value", [0, -1, 2.5, "3", True])
def test_pos_int_rejects(value):
  with pytest.raises((TypeError, ValueError)):
    pos_int(value, "n")


def test_nonempty_str():
  assert nonempty_str("a", "n") == "a"
  with pytest.raises(ValueError):
    nonempty_str("", "n")
  with pytest.raises(TypeError):
    nonempty_str(3, "n")


# ── Gas ───────────────────────────────────────────────────────────────────────


def test_air_defaults():
  assert AIR.rho == pytest.approx(1.293)
  assert AIR.mu == pytest.approx(1.85e-5)
  assert AIR.p_rho == pytest.approx(P_ATM)


def test_gas_invalid():
  with pytest.raises(ValueError):
    Gas(rho=0.0)
  with pytest.raises(ValueError):
    Gas(mu=-1.0)


# ── PorousRestrictor ──────────────────────────────────────────────────────────


def test_from_flow_pinned_circular():
  """kappa from the default circular flow spec (no sig-dig rounding in v2)."""
  r = PorousRestrictor.from_flow(2.8, 0.6e6 + P_ATM, np.pi * (37e-3 / 2) ** 2)
  np.testing.assert_allclose(r.permeability, 1.5204314392561677e-15, rtol=1e-9)


def test_flow_roundtrip():
  """from_flow and flow_lpm are mutual inverses."""
  area = 2.0e-3
  r = PorousRestrictor.from_flow(5.0, 0.5e6 + P_ATM, area)
  q = r.flow_lpm(area, 0.5e6 + P_ATM)
  assert q == pytest.approx(5.0, rel=1e-12)


def test_flow_requires_pressure_above_ambient():
  """Supply at ambient gives no driving pressure difference."""
  r = PorousRestrictor(1e-15)
  with pytest.raises(ValueError):
    r.flow_lpm(1e-3, P_ATM)
  with pytest.raises(ValueError):
    PorousRestrictor.from_flow(1.0, P_ATM, 1e-3)


def test_source_coeff_negative():
  """The source coefficient is a sink in the p² equation: -κ/(2·hp·μ)."""
  r = PorousRestrictor(1e-15, thickness=4.5e-3)
  expected = -1e-15 / (2.0 * 4.5e-3 * AIR.mu)
  assert r.source_coeff(AIR) == pytest.approx(expected)
  assert r.source_coeff(AIR) < 0.0


def test_feeding_parameter(circular_spec):
  """β = 6κL²/(hp·h³) evaluated per sample."""
  spec = circular_spec
  gaps = np.array([1e-6, 20e-6])
  beta = spec.restrictor.feeding_parameter(spec.pad.extent, gaps)
  expected = (
    6 * spec.restrictor.permeability * spec.pad.extent**2 / (spec.restrictor.thickness * gaps**3)
  )
  np.testing.assert_allclose(beta, expected, rtol=1e-12)
  # pinned endpoints for the default circular restrictor
  np.testing.assert_allclose(beta, [693.82354678, 0.0867279433], rtol=1e-8)
