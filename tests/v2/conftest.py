"""Shared fixtures and tolerances for the v2 test suite."""

from dataclasses import replace

import pytest

from openairbearing.v2 import catalog
from openairbearing.v2.problem import BearingSpec

# v1 rounds permeability to 3 significant digits; v2 keeps full precision.
RTOL_V1 = 5e-4
# FDM vs analytic on the coarse default grid (nx=30).
RTOL_NUMERIC = 0.05
# Fine-grid comparison (nx>=100, n_samples>=200).
RTOL_NUMERIC_FINE = 0.02
# Machine-precision checks (boundary values, identities).
RTOL_EXACT = 1e-12

FINE_GRID = dict(nx=100, n_samples=200, sample_min=1e-6, sample_max=100e-6)


def fine(spec: BearingSpec) -> BearingSpec:
  """Spec with a fine sample/grid resolution for cross-validation."""
  return replace(spec, grid=replace(spec.grid, **FINE_GRID))


@pytest.fixture
def circular_spec() -> BearingSpec:
  return catalog.circular()


@pytest.fixture
def annular_spec() -> BearingSpec:
  return catalog.annular()


@pytest.fixture
def linear_spec() -> BearingSpec:
  return catalog.linear()


@pytest.fixture
def rectangular_spec() -> BearingSpec:
  return catalog.rectangular()


@pytest.fixture
def journal_spec() -> BearingSpec:
  return catalog.journal()


@pytest.fixture
def circular_fine() -> BearingSpec:
  return fine(catalog.circular())


@pytest.fixture
def annular_fine() -> BearingSpec:
  return fine(catalog.annular())


@pytest.fixture
def linear_fine() -> BearingSpec:
  return fine(catalog.linear())
