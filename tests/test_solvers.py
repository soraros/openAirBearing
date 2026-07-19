import pytest
import numpy as np

from openairbearing.bearings import *
from openairbearing.solvers import (
    solve_bearing,
    get_pressure_analytic_circular,
    get_pressure_analytic_annular,
    get_pressure_analytic_infinite,
    get_pressure_numeric,
    get_pressure_2d_numeric,
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


def test_get_pressure_2d_numeric_polar_raises():
    """Polar 2D numeric raises a clear error instead of crashing on `factors`."""
    bearing = AnnularBearing(ny=8)
    with pytest.raises(NotImplementedError, match="polar"):
        get_pressure_2d_numeric(bearing)


def test_rectangular_grid_spacing_consistent():
    """dx/dy match the actual node spacing of the linspace grids."""
    bearing = RectangularBearing()
    assert bearing.dx == pytest.approx(bearing.x[1] - bearing.x[0])
    assert bearing.dy == pytest.approx(bearing.y[1] - bearing.y[0])


def test_journal_theta_grid_periodic():
    """Journal theta grid is endpoint-free with uniform 2*pi/nx spacing."""
    bearing = JournalBearing()
    np.testing.assert_allclose(np.diff(bearing.theta), 2 * np.pi / bearing.nx)
    assert bearing.dx == pytest.approx(2 * np.pi / bearing.nx)
    assert bearing.dy == pytest.approx(bearing.y[1] - bearing.y[0])


def test_blocked_restrictor_circular():
    """blocked=True zeroes permeability in the blocked band and solves."""
    b = CircularBearing(blocked=True)
    assert b.block_in is not None and b.block_in.any()
    assert 0 < b.block_A < b.A
    p = get_pressure_numeric(b)
    p_open = get_pressure_numeric(CircularBearing())
    assert p.shape == p_open.shape
    assert not np.allclose(p, p_open)


def test_blocked_restrictor_rectangular():
    """blocked=True works on the centered 2D grid as well."""
    b = RectangularBearing(nx=15, ny=10, nh=3, blocked=True, block_w=8e-3)
    assert b.block_in is not None and b.block_in.any()
    assert 0 < b.block_A < b.A
    p = get_pressure_2d_numeric(b)
    assert p.shape == (b.ny, b.nx, b.nh)


def test_blocked_restrictor_journal_raises():
    with pytest.raises(NotImplementedError, match="journal"):
        JournalBearing(blocked=True)
