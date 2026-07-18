"""Finite-difference film solvers (numba kernels plus thin Python wrappers).

Solves the isothermal Reynolds equation with porous feed in the squared
pressure ψ = p²:

- 1-D pads: tridiagonal system per sample, assembled and solved
  (Thomas algorithm) inside ``@njit`` kernels, parallel over samples.
- 2-D pads: COO assembly per sample inside an ``@njit`` kernel, solved
  with scipy sparse. Metric factors (polar 1/r, 1/r², journal 1/r²) enter
  as precomputed face-coefficient arrays, so the kernel stays generic.

Neumann edges fold the missing face onto the opposite neighbor (ghost
value ψ₋₁ = ψ₊₁), a second-order zero-gradient condition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numba import njit, prange

from openairbearing.v2.geometry.pads import JournalPad
from openairbearing.v2.primitives.types import F64, I32, StackScalar
from openairbearing.v2.problem import BC_DIRICHLET, BC_NEUMANN, BC_PERIODIC

if TYPE_CHECKING:
  from openairbearing.v2.problem import BearingProblem

__all__ = ["assemble_1d", "solve_fdm_1d", "solve_fdm_2d", "thomas_1d"]


# ── 1-D kernels ───────────────────────────────────────────────────────────────


@njit(cache=True, parallel=True)
def assemble_1d(
  gaps: F64,
  x: F64,
  dx: F64,
  source: float,
  slip: float,
  mu: float,
  polar: bool,
  west_code: int,
  east_code: int,
  psi_west: float,
  psi_east: float,
  ps2: float,
) -> tuple[F64, F64, F64, F64]:
  """Assemble the tridiagonal system for every sample.

  Returns ``(lower, main, upper, rhs)``, each ``(nh, nx)``; ``lower[j]`` is
  the west coefficient of row ``j`` (``lower[:, 0]`` unused), ``upper[j]``
  the east coefficient of row ``j`` (``upper[:, -1]`` unused).
  """
  nh, nx = gaps.shape
  lower = np.zeros((nh, nx))
  main = np.zeros((nh, nx))
  upper = np.zeros((nh, nx))
  rhs = np.empty((nh, nx))
  for k in prange(nh):
    eps = np.empty(nx)
    for j in range(nx):
      eps[j] = (1.0 + slip) * gaps[k, j] ** 3 / (24.0 * mu)
      if polar:
        eps[j] *= x[j]
    eh = np.empty(nx - 1)
    for j in range(nx - 1):
      eh[j] = 0.5 * (eps[j] + eps[j + 1])
    for j in range(1, nx - 1):
      c = 1.0 / x[j] if polar else 1.0
      main[k, j] = -(eh[j] + eh[j - 1]) / dx[j] ** 2 * c + source
      upper[k, j] = eh[j] / dx[j] ** 2 * c
      lower[k, j] = eh[j - 1] / dx[j] ** 2 * c
    for j in range(nx):
      rhs[k, j] = ps2 * source
    # west edge (row 0)
    if west_code == BC_NEUMANN:
      main[k, 0] = source
      upper[k, 0] = -source
      rhs[k, 0] = 0.0
    elif west_code == BC_DIRICHLET:
      main[k, 0] = 1.0
      upper[k, 0] = 0.0
      rhs[k, 0] = psi_west
    # east edge (row nx-1)
    if east_code == BC_NEUMANN:
      main[k, nx - 1] = source
      lower[k, nx - 1] = -source
      rhs[k, nx - 1] = 0.0
    elif east_code == BC_DIRICHLET:
      main[k, nx - 1] = 1.0
      lower[k, nx - 1] = 0.0
      rhs[k, nx - 1] = psi_east
  return lower, main, upper, rhs


@njit(cache=True, parallel=True)
def thomas_1d(lower: F64, main: F64, upper: F64, rhs: F64) -> F64:
  """Thomas-solve one tridiagonal system per sample. Returns ψ = p²."""
  nh, nx = main.shape
  out = np.empty((nh, nx))
  for k in prange(nh):
    c = main[k].copy()
    d = rhs[k].copy()
    for j in range(1, nx):
      w = lower[k, j] / c[j - 1]
      c[j] -= w * upper[k, j - 1]
      d[j] -= w * d[j - 1]
    out[k, nx - 1] = d[nx - 1] / c[nx - 1]
    for j in range(nx - 2, -1, -1):
      out[k, j] = (d[j] - upper[k, j] * out[k, j + 1]) / c[j]
  return out


def solve_fdm_1d(problem: BearingProblem) -> StackScalar:
  """1-D FDM pressure stack ``(nh, nx)`` for one compiled problem."""
  state = problem.state
  bc = problem.boundaries
  lower, main, upper, rhs = assemble_1d(
    problem.gaps,
    problem.x,
    problem.dx,
    problem.source,
    state.slip,
    problem.gas.mu,
    problem.polar,
    bc.x_lo.code,
    bc.x_hi.code,
    bc.x_lo.pressure(state) ** 2,
    bc.x_hi.pressure(state) ** 2,
    state.p_supply**2,
  )
  psi = thomas_1d(lower, main, upper, rhs)
  return np.sqrt(psi)


# ── 2-D kernel ────────────────────────────────────────────────────────────────


@njit(cache=True)
def assemble_2d(
  ax_w: F64,
  ax_e: F64,
  ay_s: F64,
  ay_n: F64,
  src: F64,
  is_dir: F64,
  dir_val: F64,
  bc_xlo: int,
  bc_xhi: int,
  bc_ylo: int,
  bc_yhi: int,
  row: I32,
  col: I32,
  val: F64,
  rhs: F64,
) -> int:
  """Assemble the 5-point stencil into COO buffers. Returns the nnz count.

  Face coefficients already include metric factors and Neumann folding;
  ``src`` is zeroed on non-periodic edges by the caller; ``rhs`` is
  pre-filled with ``src * ps²`` and only Dirichlet rows are overwritten.
  """
  nx, ny = src.shape
  m = 0
  for i in range(nx):
    for j in range(ny):
      r = i * ny + j
      if is_dir[i, j]:
        row[m] = r
        col[m] = r
        val[m] = 1.0
        rhs[r] = dir_val[i, j]
        m += 1
        continue
      center = src[i, j]
      # x- face
      if i > 0:
        row[m], col[m], val[m] = r, r - ny, ax_w[i, j]
        m += 1
      elif bc_xlo == BC_PERIODIC:
        row[m], col[m], val[m] = r, (nx - 1) * ny + j, ax_w[i, j]
        m += 1
      center -= ax_w[i, j]
      # x+ face
      if i < nx - 1:
        row[m], col[m], val[m] = r, r + ny, ax_e[i, j]
        m += 1
      elif bc_xhi == BC_PERIODIC:
        row[m], col[m], val[m] = r, j, ax_e[i, j]
        m += 1
      center -= ax_e[i, j]
      # y- face
      if j > 0:
        row[m], col[m], val[m] = r, r - 1, ay_s[i, j]
        m += 1
      elif bc_ylo == BC_PERIODIC:
        row[m], col[m], val[m] = r, i * ny + ny - 1, ay_s[i, j]
        m += 1
      center -= ay_s[i, j]
      # y+ face
      if j < ny - 1:
        row[m], col[m], val[m] = r, r + 1, ay_n[i, j]
        m += 1
      elif bc_yhi == BC_PERIODIC:
        row[m], col[m], val[m] = r, i * ny, ay_n[i, j]
        m += 1
      center -= ay_n[i, j]
      row[m], col[m], val[m] = r, r, center
      m += 1
  return m


def solve_fdm_2d(problem: BearingProblem) -> StackScalar:
  """2-D FDM pressure stack ``(nh, nx, ny)`` for one compiled problem."""
  state = problem.state
  bc = problem.boundaries
  gas = problem.gas
  x, y = problem.x, problem.y
  nx, ny = x.size, y.size
  n = nx * ny
  dx = float(problem.dx[0])
  dy = float(problem.dy[0])
  ps2 = state.p_supply**2

  bc_codes = (bc.x_lo.code, bc.x_hi.code, bc.y_lo.code, bc.y_hi.code)
  psi_bc = (
    bc.x_lo.pressure(state) ** 2,
    bc.x_hi.pressure(state) ** 2,
    bc.y_lo.pressure(state) ** 2,
    bc.y_hi.pressure(state) ** 2,
  )
  is_dir = np.zeros((nx, ny))
  dir_val = np.zeros((nx, ny))
  for code, value, slc in (
    (bc_codes[0], psi_bc[0], (0, slice(None))),
    (bc_codes[1], psi_bc[1], (-1, slice(None))),
    (bc_codes[2], psi_bc[2], (slice(None), 0)),
    (bc_codes[3], psi_bc[3], (slice(None), -1)),
  ):
    if code == BC_DIRICHLET:
      is_dir[slc] = 1.0
      dir_val[slc] = value

  row = np.empty(5 * n, dtype=np.int32)
  col = np.empty(5 * n, dtype=np.int32)
  val = np.empty(5 * n)

  p = np.empty((problem.n_samples, nx, ny))
  for k in range(problem.n_samples):
    h = problem.gaps[k]
    eps = (1.0 + state.slip) * h**3 / (24.0 * gas.mu)
    cx, cy, ex, ey = _metrics(problem, eps)
    ax_w, ax_e = _face_coeffs(ex, cx, dx, bc_codes[0], bc_codes[1], axis=0)
    ay_s, ay_n = _face_coeffs(ey, cy, dy, bc_codes[2], bc_codes[3], axis=1)
    src = np.full((nx, ny), problem.source)
    if bc_codes[0] != BC_PERIODIC:
      src[0, :] = 0.0
    if bc_codes[1] != BC_PERIODIC:
      src[-1, :] = 0.0
    if bc_codes[2] != BC_PERIODIC:
      src[:, 0] = 0.0
    if bc_codes[3] != BC_PERIODIC:
      src[:, -1] = 0.0
    rhs = (src * ps2).ravel()
    nnz = assemble_2d(
      ax_w,
      ax_e,
      ay_s,
      ay_n,
      src,
      is_dir,
      dir_val,
      bc_codes[0],
      bc_codes[1],
      bc_codes[2],
      bc_codes[3],
      row,
      col,
      val,
      rhs,
    )
    mat = sp.coo_array((val[:nnz], (row[:nnz], col[:nnz])), shape=(n, n)).tocsr()
    p[k] = np.sqrt(spla.spsolve(mat, rhs)).reshape(nx, ny)
  return p


# ── 2-D assembly helpers ──────────────────────────────────────────────────────


def _metrics(problem: BearingProblem, eps: F64) -> tuple[F64 | float, F64 | float, F64, F64]:
  """Metric factors: Lψ = cx·D_x(ex·D_xψ) + cy·D_y(ey·D_yψ)."""
  pad = problem.pad
  if problem.polar:
    r = problem.x[:, None]
    return 1.0 / r, 1.0 / r**2, r * eps, eps
  if isinstance(pad, JournalPad):
    return 1.0 / pad.r**2, 1.0, eps, eps
  return 1.0, 1.0, eps, eps


def _face_coeffs(
  e: F64, c: F64 | float, spacing: float, bc_lo: int, bc_hi: int, axis: int
) -> tuple[F64, F64]:
  """Face-averaged coefficients with periodic wrap and Neumann folding.

  Returns ``(a_lo, a_hi)`` where ``a_lo[i]`` multiplies the neighbor below
  node ``i``. At Neumann edges the missing face folds onto the opposite
  neighbor (ghost value ψ₋₁ = ψ₊₁); at Dirichlet edges the row is replaced
  by the caller, so the face value is irrelevant (mirror convention).
  """
  a_lo = 0.5 * (np.roll(e, 1, axis=axis) + e)
  a_hi = 0.5 * (np.roll(e, -1, axis=axis) + e)
  if bc_lo != BC_PERIODIC:
    lo = (0, slice(None)) if axis == 0 else (slice(None), 0)
    a_lo[lo] = a_hi[lo]  # mirror convention (fold target / unused)
  if bc_hi != BC_PERIODIC:
    hi = (-1, slice(None)) if axis == 0 else (slice(None), -1)
    a_hi[hi] = a_lo[hi]
  a_lo = c * a_lo / spacing**2
  a_hi = c * a_hi / spacing**2
  if bc_lo == BC_NEUMANN:
    lo = (0, slice(None)) if axis == 0 else (slice(None), 0)
    a_hi[lo] = a_hi[lo] + a_lo[lo]
    a_lo[lo] = 0.0
  if bc_hi == BC_NEUMANN:
    hi = (-1, slice(None)) if axis == 0 else (slice(None), -1)
    a_lo[hi] = a_lo[hi] + a_hi[hi]
    a_hi[hi] = 0.0
  return (
    np.ascontiguousarray(a_lo, dtype=np.float64),
    np.ascontiguousarray(a_hi, dtype=np.float64),
  )
