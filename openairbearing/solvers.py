import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import i0, k0

from openairbearing.utils import (
  Result,
  get_load_capacity,
  get_stiffness,
  get_volumetric_flow,
)


def solve_bearing(bearing, soltype: str) -> Result:
  match soltype:
    case "analytic":
      name = "analytic"
      match bearing.case:
        case "circular":
          p = get_pressure_analytic_circular(bearing)
        case "annular":
          p = get_pressure_analytic_annular(bearing)
        case "infinite":
          p = get_pressure_analytic_infinite(bearing)
        case _:
          return Result(
            name="none",
            p=np.array([]),
            w=np.array([]),
            k=np.array([]),
            qs=np.array([]),
            qa=np.array([]),
            qc=np.array([]),
          )
    case "numeric":
      name = "numeric"
      p = get_pressure_numeric(bearing)
    case "numeric2d":
      name = "numeric2d"
      p = get_pressure_2d_numeric(bearing)
    case _:
      raise ValueError("Invalid solution type")

  w = get_load_capacity(bearing=bearing, p=p)
  k = get_stiffness(bearing=bearing, w=w)
  qs, qa, qc = get_volumetric_flow(bearing=bearing, p=p, soltype=soltype)

  return Result(name=name, p=p, w=w, k=k, qs=qs, qa=qa, qc=qc)


def get_pressure_analytic_infinite(bearing):
  """
  Calculates the solution for the pressure distribution in infinitely long bearings and seals.
  """

  b = bearing

  f = (2 * b.beta) ** 0.5
  slip = (1 + b.Psi) ** 0.5

  # nondimensionals
  Pa = 1
  Ra = 1
  R = b.x / b.xa
  Ps = b.ps / b.pa
  Pc = b.pc / b.pa

  exp_f = np.exp((f * Ra) / slip)

  numer1 = -(Pc**2) + Ps**2 + exp_f * (Pa**2 - Ps**2)
  numer2 = exp_f * (-(Pa**2) + Ps**2 + exp_f * (Pc**2 - Ps**2))

  denom = -1 + np.exp((2 * f * Ra) / slip)

  C1 = numer1 / denom
  C2 = numer2 / denom

  p = (
    b.pa
    * (Ps**2 + C1 * np.exp(np.outer(R, f) / slip) + C2 * np.exp(-np.outer(R, f) / slip))
    ** 0.5
  )
  return p


def get_pressure_analytic_annular(bearing):
  """
  Calculates the Bessel function solution for the pressure distribution in annular bearings and seals.
  """

  b = bearing

  f = (2 * b.beta) ** 0.5

  # nondimensionals
  Pa = 1
  Ra = 1
  R = b.x / b.xa
  Ps = b.ps / b.pa
  Pc = b.pc / b.pa
  Rc = b.xc / b.xa

  numer1 = (Pa**2 - Ps**2) * k0(f * Rc) + (Ps**2 - Pc**2) * k0(f * Ra)
  numer2 = (Pa**2 - Ps**2) * i0(f * Rc) + (Ps**2 - Pc**2) * i0(f * Ra)

  denom = i0(f * Rc) * k0(f * Ra) - i0(f * Ra) * k0(f * Rc)

  C1 = numer1 / denom
  C2 = numer2 / denom

  p = b.pa * (Ps**2 - C1 * i0(np.outer(R, f)) + C2 * k0(np.outer(R, f))) ** 0.5
  return p


def get_pressure_analytic_circular(bearing):
  """
  Calculates the Bessel function solution for the pressure distribution in circluar trust bearings.
  """
  b = bearing
  p = (
    b.ps
    * (
      1
      - (1 - b.pa**2 / b.ps**2)
      * i0(np.outer(b.x / b.xa, (2 * b.beta) ** 0.5))
      / i0((2 * b.beta) ** 0.5)
    )
    ** 0.5
  )
  return p


def get_pressure_numeric(bearing, ha=None, ps=None):
  b = bearing

  # uniform kappa
  kappa = b.kappa * np.ones_like(b.x)

  # Partially blocked restrictors, set blocked region to 0 permeability
  if b.blocked:
    kappa[b.block_in] = 0

  # porous feeding terms
  porous_source = -kappa / (2 * b.hp * b.mu)

  if ha is None:
    ha = b.ha
  else:
    ha = np.atleast_1d(ha)
  if ps is None:
    ps = b.ps

  p = np.zeros((len(b.x), len(ha)))

  for i in range(len(ha)):
    h = ha[i] + b.geom

    if b.csys == "polar":
      epsilon = (1 + b.Psi) * b.x * h**3 / (24 * b.mu)
      coefficient = sp.diags(1 / b.x, 0)
    elif b.csys == "cartesian":
      epsilon = (1 + b.Psi) * h**3 / (24 * b.mu)
      coefficient = 1

    diff_mat = build_diff_matrix(coefficient, epsilon, b.dx)
    A = sp.lil_matrix(diff_mat + sp.diags(porous_source, 0))

    f = ps**2 * porous_source

    # Boundary conditions
    if b.type == "bearing":
      # Neumann at r=0
      A[0, 1] = -A[0, 0]
      f[0] = 0
    elif b.type == "seal":
      # Dirilecht at r=rc
      A[0, 0] = 1
      A[0, 1] = 0
      f[0] = b.pc**2

    # Dirichlet at r=ra
    A[-1, -2] = 0
    A[-1, -1] = 1
    f[-1] = b.pa**2

    A = A.tocsr()
    p[:, i] = spla.spsolve(A, f) ** 0.5

  return p


def build_diff_matrix(
  coef: np.ndarray, eps: np.ndarray, dr: np.ndarray
) -> sp.csr_matrix:
  """Construct finite-difference matrix for coefficient @ D_r(epsilon * D_r(f(r)))

  Builds a sparse matrix representing the discretized differential operator
  using second-order central differences with variable coefficients.
  """

  N = len(dr)

  # Compute epsilon at half-points
  eps_half = (eps[:-1] + eps[1:]) / 2
  # Finite difference second derivative matrix with variable coefficient
  diag_main = np.zeros(N)
  diag_upper = np.zeros(N - 1)
  diag_lower = np.zeros(N - 1)

  # interior points with 3 point stencil
  diag_main[1:-1] = -(eps_half[1:] + eps_half[:-1]) / dr[1:-1] ** 2
  diag_upper[1:] = eps_half[1:] / dr[1:-1] ** 2
  diag_lower[:-1] = eps_half[:-1] / dr[1:-1] ** 2

  # Assemble sparse matrix
  L_mat = sp.diags([diag_lower, diag_main, diag_upper], [-1, 0, 1], format="csr")
  # Handle also scalar coefficients
  if isinstance(coef, (float, int)):
    return coef * L_mat
  else:
    return coef @ L_mat


def get_pressure_2d_numeric(bearing):
  """
  Solve the 2D pressure distribution for the bearing using finite differences.

  Args:
      bearing: Bearing object containing geometry and properties.

  Returns:
      np.ndarray: 2D pressure distribution array.
  """
  b = bearing
  N = b.nx
  M = b.ny

  p = np.zeros((M, N, len(b.ha)))

  # Uniform kappa
  kappa = b.kappa * np.ones((N, M))

  # Partially blocked restrictors, set blocked region to 0 permeability
  if b.blocked:
    kappa[b.block_in] = 0

  # Porous feeding terms
  porous_source = -kappa / (2 * b.hp * b.mu)

  for i in range(b.nh):
    if b.case == "journal":
      h = (b.clearance + b.geom[:, :, None])[:, :, i]
    else:
      h = b.ha[i] + b.geom

    if b.csys == "polar":
      epsilon_r = b.x[:, None] * (1 + b.Psi) * h**3 / (24 * b.mu)
      epsilon_theta = (1 + b.Psi) * h**3 / (24 * b.mu)
      epsilon = (epsilon_r, epsilon_theta)
    elif b.csys == "cartesian":
      epsilon = (1 + b.Psi) * h**3 / (24 * b.mu)

    # boundary conditions
    if b.case == "rectangular":
      bc = {
        "west": "Dirichlet",
        "east": "Dirichlet",
        "north": "Dirichlet",
        "south": "Dirichlet",
      }
      bc_vals = {
        "west": b.pa,
        "east": b.pa,
        "north": b.pa,
        "south": b.pa,
      }
      factors = [1, 1]
    elif b.case == "circular":
      bc = {
        "west": "Neumann",
        "east": "Dirichlet",
        "north": "Periodic",
        "south": "Periodic",
      }
      bc_vals = {
        "west": b.pa,
        "east": b.pa,
        "north": b.pa,
        "south": b.pa,
      }
      factors[b.x, b.x**2]
    elif b.case == "annular":
      bc = {
        "west": "Dirichlet",
        "east": "Dirichlet",
        "north": "Periodic",
        "south": "Periodic",
      }
      bc_vals = {
        "west": b.pa,
        "east": b.pc,
      }
      factors[b.x, b.x**2]
    elif b.case == "journal":
      bc = {
        "west": "Dirichlet",
        "east": "Dirichlet",
        "north": "Periodic",
        "south": "Periodic",
      }
      bc_vals = {
        "west": b.pa,
        "east": b.pc,
      }
      factors = [b.xa**2, 1]

    p[:, :, i] = fdm_2d(
      epsilon=epsilon,
      porous_source=porous_source,
      ps=b.ps,
      dx=b.dx,
      dy=b.dy,
      bc=bc,
      bc_vals=bc_vals,
      N=N,
      M=M,
      factors=factors,
    )
  return p


def fdm_2d(
  epsilon: np.ndarray,
  porous_source: np.ndarray,
  ps: float,
  dx: np.ndarray,
  dy: np.ndarray,
  bc: dict,
  bc_vals: dict,
  N: int,
  M: int,
  factors: list = [1, 1],
) -> np.ndarray:
  """
  Solve air gap pressure in 2D using a finite difference scheme.

  Args:
      epsilon (np.ndarray): Coefficient matrix (N, M).
      porous_source (float): Source term through porous restrictor.
      ps (float): Supply pressure
      dx (float): Grid spacing in the x-direction.
      dy (float): Grid spacing in the y-direction.
      bc (dict): Boundary conditions for "west", "east", "north", "south".
                 Each can be "Dirichlet", "Neumann", or "Periodic".
      dirichlet_values (dict): Dirichlet values for "west", "east", "north", "south".
      N (int): Number of grid points in the x-direction.
      M (int): Number of grid points in the y-direction.

  Returns:
      np.ndarray: Solution pressure matrix p (N, M).
  """
  # Number of unknowns
  num_points = N * M
  fx = factors[0]
  fy = factors[1]

  # Create sparse matrix and right-hand side
  A = sp.lil_matrix((num_points, num_points))

  # edge source = 0 except for periodic boundaries
  if bc["west"] != "Periodic":
    porous_source[:, 0] = 0
  if bc["east"] != "Periodic":
    porous_source[:, -1] = 0
  if bc["north"] != "Periodic":
    porous_source[-1, :] = 0
  if bc["south"] != "Periodic":
    porous_source[0, :] = 0

  b = porous_source.flatten() * ps**2

  # Helper function to convert 2D indices to 1D
  def idx(i, j):
    return i * M + j

  # Compute epsilon at half-points
  eps_w = np.zeros_like(epsilon)
  eps_e = np.zeros_like(epsilon)
  eps_n = np.zeros_like(epsilon)
  eps_s = np.zeros_like(epsilon)

  # West/east half-points
  eps_w[:, :-1] = (epsilon[:, :-1] + epsilon[:, 1:]) / 2
  eps_e[:, 1:] = (epsilon[:, :-1] + epsilon[:, 1:]) / 2
  # North/south half-points
  eps_n[:-1, :] = (epsilon[:-1, :] + epsilon[1:, :]) / 2
  eps_s[1:, :] = (epsilon[:-1, :] + epsilon[1:, :]) / 2

  # Build the finite difference matrix
  for i in range(N):
    for j in range(M):
      row = idx(i, j)

      # Coefficients for the 5-point stencil
      center = (
        -(eps_w[i, j] + eps_e[i, j]) / (dy**2 * fy)
        - (eps_n[i, j] + eps_s[i, j]) / (dx**2 * fx)
        + porous_source[i, j]
      )
      west = eps_w[i, j] / (dy**2 * fy)
      east = eps_e[i, j] / (dy**2 * fy)
      north = eps_n[i, j] / (dx**2 * fx)
      south = eps_s[i, j] / (dx**2 * fx)

      # center point
      A[row, row] = center

      # West neighbor
      if j > 0:
        A[row, idx(i, j - 1)] = west
      else:
        match bc["west"]:
          case "Periodic":
            A[row, idx(i, M - 1)] = west
          case "Neumann":
            A[row, row] += west
          case "Dirichlet":
            A[row, :] = 0
            A[row, row] = 1
            b[row] = bc_vals["west"] ** 2
          case _:
            raise ValueError("invalid BC")

      # East neighbor
      if j < M - 1:
        A[row, idx(i, j + 1)] = east
      else:
        match bc["east"]:
          case "Periodic":
            A[row, idx(i, 0)] = east
          case "Neumann":
            A[row, row] += east
          case "Dirichlet":
            A[row, :] = 0
            A[row, row] = 1
            b[row] = bc_vals["east"] ** 2
          case _:
            raise ValueError("invalid BC")

      # North neighbor
      if i > 0:
        A[row, idx(i - 1, j)] = north
      else:
        match bc["north"]:
          case "Periodic":
            A[row, idx(N - 1, j)] = north
          case "Neumann":
            A[row, row] += north
          case "Dirichlet":
            A[row, :] = 0
            A[row, row] = 1
            b[row] = bc_vals["north"] ** 2
          case _:
            raise ValueError("invalid BC")

      # South neighbor
      if i < N - 1:
        A[row, idx(i + 1, j)] = south
      else:
        match bc["south"]:
          case "Periodic":
            A[row, idx(0, j)] = south
          case "Neumann":
            A[row, row] += south
          case "Dirichlet":
            A[row, :] = 0
            A[row, row] = 1
            b[row] = bc_vals["south"] ** 2
          case _:
            raise ValueError("invalid BC")
      # Source term
      # b[row] += porous_source[i, j] * ps**2

  # Solve the linear system
  A = A.tocsr()

  p_flat = spla.spsolve(A, b) ** 0.5
  p = p_flat.reshape((N, M)).T
  return p
