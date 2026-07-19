import pytest
import numpy as np
from openairbearing.utils import (
    get_area,
    get_geom,
    get_beta,
    get_kappa,
    get_Qsc,
    get_dA,
    get_load_capacity,
    get_stiffness,
)
from openairbearing.bearings import BaseBearing, InfiniteLinearBearing


def test_get_area():
    """Test the get_area function for different bearing cases."""
    bearing = BaseBearing(case="circular", xa=10)
    assert get_area(bearing) == pytest.approx(np.pi * 10**2)

    bearing = BaseBearing(case="annular", xa=10, xc=5)
    assert get_area(bearing) == pytest.approx(np.pi * (10**2 - 5**2))

    bearing = BaseBearing(case="infinite", xa=10)
    assert get_area(bearing) == pytest.approx(10)

    bearing = BaseBearing(case="rectangular", xa=10, ya=5)
    assert get_area(bearing) == pytest.approx(10 * 5)

    with pytest.raises(ValueError, match="Unknown case: unknown"):
        bearing = BaseBearing(case="unknown")
        get_area(bearing)


def test_get_geom():
    """Test the get_geom function for different error types."""
    nx = 10
    ny = 5

    bearing = BaseBearing(
        case="circular", csys="polar", nx=nx, xa=10, ya=5, error_type="none"
    )
    geom = get_geom(bearing)
    assert np.all(geom == 0)

    bearing.error_type = "linear"
    geom = get_geom(bearing)
    assert geom.shape == (nx,)

    bearing.error_type = "quadratic"
    geom = get_geom(bearing)
    assert geom.shape == (nx,)

    with pytest.raises(ValueError, match="Unknown error type: invalid"):
        bearing.error_type = "invalid"
        get_geom(bearing)

    bearing = BaseBearing(
        case="rectangular", nx=nx, ny=ny, xa=10, ya=5, error_type="none"
    )
    geom = get_geom(bearing)
    assert np.all(geom == 0)

    bearing.error_type = "linear"
    geom = get_geom(bearing)
    assert geom.shape == (nx, ny)

    bearing.error_type = "quadratic"
    geom = get_geom(bearing)
    assert geom.shape == (nx, ny)

    with pytest.raises(ValueError, match="Unknown error type: invalid"):
        bearing.error_type = "invalid"
        get_geom(bearing)


def test_get_beta():
    """Test the get_beta function."""
    ha = np.linspace(1e-6, 20e-6, 10)
    hp = 4e-3
    kappa = 1e-15
    xa = 15
    bearing = BaseBearing(case="circular", xa=xa, hp=hp)
    bearing.kappa = kappa
    bearing.ha = ha
    beta = get_beta(bearing)
    assert beta.shape == ha.shape
    assert beta == pytest.approx(6 * kappa * xa**2 / (hp * ha**3))


def test_get_kappa():
    """Test the get_kappa function."""
    bearing = BaseBearing(
        case="circular", Qsc=3, mu=1.85e-5, hp=1e-3, pa=101325, psc=0.6e6 + 101325
    )
    bearing.A = get_area(bearing)
    kappa = get_kappa(bearing)
    assert kappa > 0


def test_get_Qsc():
    """Test the get_Qsc function."""
    bearing = BaseBearing(case="circular", psc=0.6e6)
    bearing.kappa = 1e-15
    Qsc = get_Qsc(bearing)
    assert Qsc > 0


def test_get_dA():
    """Test the get_dA function."""
    bearing = BaseBearing(case="circular", csys="polar", nx=10, ny=1, xa=10)
    dA = get_dA(bearing)
    bearing = BaseBearing(case="infinite", csys="cartesian", nx=10, ny=1, xa=10)
    dA = get_dA(bearing)
    assert dA.shape == (10,)


# def test_get_load_capacity():
#     """Test the get_load_capacity function."""
#     bearing = BaseBearing(case="circular", type="bearing", csys="polar", nx=10, ny=1, xa=10, pa=101325)
#     p = np.ones(10) * 101325
#     w = get_load_capacity(bearing, p)
#     assert w.shape == p.shape ,  f"w {w.shape}, p {p.shape}"
#     # assert np.allclose(w, np.sum (p * bearing.dA, axis=0))


def test_get_stiffness():
    """Test the get_stiffness function."""
    nh = 40
    bearing = BaseBearing(case="circular", ha_min=1e-6, ha_max=20e-6, nh=40)
    w = 2 ** np.linspace(0, 100, nh)
    k = get_stiffness(bearing, w)
    assert np.allclose(k, -np.gradient(w, bearing.ha.flatten()))


def test_kappa_uses_subclass_psc():
    """get_kappa calibrates with the psc set by the subclass, so the
    configured flow rate round-trips through get_Qsc."""
    bearing = InfiniteLinearBearing()
    assert get_Qsc(bearing) == pytest.approx(bearing.Qsc, rel=0.01)
