import pytest

from openairbearing.bearings import (
  AnnularBearing,
  CircularBearing,
  InfiniteLinearBearing,
  RectangularBearing,
)

# --- Tolerance constants ---
# FDM vs analytic on a coarse grid (default nx=30)
RTOL_NUMERIC = 0.05
# Fine-grid comparison (nx>=100, nh>=200)
RTOL_NUMERIC_FINE = 0.02
# Machine-precision checks (boundary values, identities)
RTOL_EXACT = 1e-12


# --- Shared solver parameters for cross-validation ---
FINE_GRID = dict(
  nh=200,
  ha_min=1e-6,
  ha_max=100e-6,
  nx=100,
  error_type="none",
  error=0,
  Psi=0,
)


# --- Default bearing fixtures ---
@pytest.fixture
def circular():
  return CircularBearing()


@pytest.fixture
def annular():
  return AnnularBearing()


@pytest.fixture
def infinite():
  return InfiniteLinearBearing()


@pytest.fixture
def rectangular():
  return RectangularBearing()


@pytest.fixture
def circular_fine():
  """High-resolution circular bearing for analytic-numeric cross-validation."""
  return CircularBearing(**FINE_GRID)


@pytest.fixture
def annular_fine():
  """High-resolution annular bearing for analytic-numeric cross-validation."""
  return AnnularBearing(**FINE_GRID)


@pytest.fixture
def infinite_fine():
  """High-resolution infinite bearing for analytic-numeric cross-validation."""
  return InfiniteLinearBearing(**FINE_GRID)
