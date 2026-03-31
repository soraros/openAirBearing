import numpy as np
import pytest

from openairbearing.bearings import *
from openairbearing.solvers import (
  get_pressure_analytic_annular,
  get_pressure_analytic_circular,
  get_pressure_analytic_infinite,
  get_pressure_numeric,
  solve_bearing,
)


def test_get_pressure_analytic_circular():
  """Test the analytic pressure distribution for a circular bearing."""
  bearing = CircularBearing()
  p = get_pressure_analytic_circular(bearing)
  assert p.shape == (bearing.nx, bearing.nh)
  e = 1e-6
  print(np.min(p - bearing.ps), np.max(p - bearing.ps))
  assert np.all(p - bearing.pa > -e)
  assert np.all(p - bearing.ps < e)


def test_get_pressure_analytic_annular():
  """Test the analytic pressure distribution for a annular bearing."""
  bearing = AnnularBearing()
  p = get_pressure_analytic_annular(bearing)
  assert p.shape == (bearing.nx, bearing.nh)
  assert p[0, :] == pytest.approx(bearing.pc)
  assert p[-1, :] == pytest.approx(bearing.pa)
  e = 1e-6
  assert np.all(p - bearing.pa > -e)
  assert np.all(p - bearing.ps < e)
  # assert np.all(p <= bearing.ps)


def test_get_pressure_analytic_infinite():
  """Test the analytic pressure distribution for a infinite linear bearing."""
  bearing = InfiniteLinearBearing()
  p = get_pressure_analytic_infinite(bearing)
  assert p.shape == (bearing.nx, bearing.nh)
  e = 1e-6
  assert np.all(p - bearing.pa > -e)
  assert np.all(p - bearing.ps < e)


def test_get_pressure_numeric():
  """Test the numeric pressure distribution for a circular bearing."""
  bearing = CircularBearing()
  p = get_pressure_numeric(bearing)
  assert p.shape == (bearing.nx, bearing.nh)
  e = 1e-6
  assert np.all(p - bearing.pa > -e)
  assert np.all(p - bearing.ps < e)


# def test_get_pressure_2d_numeric():
#     """Test the numeric pressure distribution for a rectangular bearing."""
#     bearing = RectangularBearing()
#     p = get_pressure_2d_numeric(bearing)
#     assert p.shape == (bearing.ny, bearing.nx, bearing.nh)
#     e = 1e-6
#     assert np.all(p - bearing.pa > -e)
#     assert np.all(p - bearing.ps < e)


def test_solve_bearing():
  """Test the solve_bearing function."""
  par = {
    "nh": 200,
    "ha_min": 1e-6,
    "ha_max": 100e-6,
    "nx": 100,
    "error_type": "none",
    "error": 0,
    "Psi": 0,
  }
  # test that analytical solutions match the numerical solutions with tolerance e
  for bearing in [
    CircularBearing(**par),
    AnnularBearing(**par),
    InfiniteLinearBearing(**par),
  ]:
    results = [
      solve_bearing(bearing, "analytic"),
      solve_bearing(bearing, "numeric"),
    ]
    e = 0.05
    assert np.allclose(results[0].p, results[1].p, rtol=e)
    assert np.allclose(results[0].w, results[1].w, rtol=e)
    assert np.allclose(results[0].k, results[1].k, rtol=e)
    assert np.allclose(results[0].qs, results[1].qs, rtol=e)
