"""Tests for geometry: pad specs, error profiles, and the grid spec."""

import numpy as np
import pytest

from openairbearing.v2.geometry.grid import GridSpec
from openairbearing.v2.geometry.pads import (
  AnnularPad,
  CircularPad,
  JournalPad,
  LinearPad,
  RectangularPad,
)
from openairbearing.v2.geometry.profiles import SurfaceError, eval_surface_error

# ── Pad specs ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
  "pad, expected",
  [
    (CircularPad(), np.pi * (37e-3 / 2) ** 2),
    (AnnularPad(), np.pi * ((58e-3 / 2) ** 2 - (25e-3 / 2) ** 2)),
    (LinearPad(), 40e-3),
    (RectangularPad(), 80e-3 * 40e-3),
    (JournalPad(), 2 * np.pi * (50.02e-3 / 2) * 89e-3),
  ],
)
def test_pad_area(pad, expected):
  """Area matches the closed form for each pad."""
  assert pad.area == pytest.approx(expected)


def test_pad_metadata():
  assert CircularPad.CSYS == "polar" and CircularPad.DIM == 1 and not CircularPad.SEAL
  assert AnnularPad.SEAL and AnnularPad.ANALYTIC
  assert LinearPad.CSYS == "cartesian" and LinearPad.ANALYTIC
  assert RectangularPad.DIM == 2 and not RectangularPad.ANALYTIC
  assert JournalPad.SEAL and JournalPad.DIM == 2


def test_pad_validation():
  with pytest.raises(ValueError):
    CircularPad(r=-1.0)
  with pytest.raises(ValueError):
    AnnularPad(r=10e-3, r_inner=20e-3)
  with pytest.raises(ValueError):
    LinearPad(length=0.0)
  with pytest.raises(ValueError):
    JournalPad(clearance=-1e-6)


def test_pads_are_immutable(circular_spec):
  with pytest.raises(AttributeError):
    circular_spec.pad.r = 1.0


# ── SurfaceError ──────────────────────────────────────────────────────────────


def test_surface_error_validation():
  with pytest.raises(ValueError, match="kind"):
    SurfaceError(kind="wavy")
  with pytest.raises(TypeError):
    SurfaceError(amplitude="big")


def test_eval_none_is_zero():
  x = np.linspace(1e-6, 18.5e-3, 30)
  geom = eval_surface_error(
    SurfaceError(), x=x, y=np.zeros(1), x_extent=18.5e-3, y_extent=0.0, layout="polar"
  )
  np.testing.assert_array_equal(geom, np.zeros_like(x))


@pytest.mark.parametrize("kind", ["linear", "quadratic"])
def test_eval_1d_profiles(kind):
  """1-D profiles have the right shape and a zero minimum."""
  x = np.linspace(1e-6, 18.5e-3, 20)
  geom = eval_surface_error(
    SurfaceError(kind=kind, amplitude=5e-6),
    x=x,
    y=np.zeros(1),
    x_extent=18.5e-3,
    y_extent=0.0,
    layout="polar",
  )
  assert geom.shape == (20,)
  assert geom.min() == pytest.approx(0.0)
  assert geom.max() > 0.0


@pytest.mark.parametrize("kind", ["none", "linear", "quadratic", "tiltx", "tilty"])
def test_eval_2d_cartesian_shapes(kind):
  """2-D cartesian profiles have (nx, ny) shape and zero minimum."""
  x = np.linspace(-40e-3, 40e-3, 15)
  y = np.linspace(-20e-3, 20e-3, 10)
  geom = eval_surface_error(
    SurfaceError(kind=kind, amplitude=5e-6),
    x=x,
    y=y,
    x_extent=80e-3,
    y_extent=40e-3,
    layout="cartesian",
  )
  assert geom.shape == (15, 10)
  assert geom.min() == pytest.approx(0.0)


def test_eval_tilt_unsupported_on_1d():
  x = np.linspace(0.0, 40e-3, 10)
  with pytest.raises(ValueError, match="1-D"):
    eval_surface_error(
      SurfaceError(kind="tiltx", amplitude=5e-6),
      x=x,
      y=np.zeros(1),
      x_extent=40e-3,
      y_extent=0.0,
      layout="cartesian",
    )


def test_eval_tilt_unsupported_on_polar():
  x = np.linspace(12.5e-3, 29e-3, 10)
  y = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
  with pytest.raises(ValueError, match="polar"):
    eval_surface_error(
      SurfaceError(kind="tilty", amplitude=5e-6),
      x=x,
      y=y,
      x_extent=29e-3,
      y_extent=2.0 * np.pi,
      layout="polar",
    )


def test_eval_tiltx_is_signed_slope():
  """tiltx ramps from -a/2 to +a/2 across the pad (shifted to zero min)."""
  x = np.linspace(-40e-3, 40e-3, 15)
  y = np.linspace(-20e-3, 20e-3, 10)
  geom = eval_surface_error(
    SurfaceError(kind="tiltx", amplitude=4e-6),
    x=x,
    y=y,
    x_extent=80e-3,
    y_extent=40e-3,
    layout="cartesian",
  )
  np.testing.assert_allclose(geom[-1, :] - geom[0, :], 4e-6, rtol=1e-12)


# ── GridSpec ──────────────────────────────────────────────────────────────────


def test_grid_validation():
  with pytest.raises(ValueError):
    GridSpec(nx=2)
  with pytest.raises(ValueError):
    GridSpec(sample_min=20e-6, sample_max=1e-6)
  with pytest.raises(ValueError):
    GridSpec(n_samples=2)
