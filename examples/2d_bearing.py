import matplotlib.pyplot as plt
import numpy as np

import openairbearing as ab

if __name__ == "__main__":
    # initialize circular bearing with parameters
    bearing = ab.AnnularBearing(
        xa=40e-3,
        ya=2 * np.pi,
        pc=0.3e6,
        nx=100,
        ny=120,
    )
    # plot bearing shape
    # figure = ab.plot_bearing_shape(bearing)
    # figure.show()
    # solve with analytic and numeric methods

    result = [
        ab.solve_bearing(bearing, "analytic"),
        ab.solve_bearing(bearing, "numeric2d"),
    ]
    plt.contourf(bearing.x * 1e3, bearing.y * 1e3, result[1].p[:, :, 3] * 1e-6)
    plt.colorbar()
    plt.show()

    # plot results
    figure = ab.plot_key_results(bearing, result)
    figure.show()


# def get_pressure_2d_numeric(bearing):
#     """
#     Solve the 2D pressure distribution for the bearing using finite differences.

#     Args:
#         bearing: Bearing object containing geometry and properties.

#     Returns:
#         np.ndarray: 2D pressure distribution array.
#     """
#     b = bearing
#     N = b.nx
#     M = b.ny

#     p = np.zeros((M, N, len(b.ha)))

#     # Uniform kappa
#     kappa = b.kappa * np.ones((N, M))

#     # Partially blocked restrictors, set blocked region to 0 permeability
#     if b.blocked:
#         kappa[b.block_in] = 0

#     # Porous feeding terms
#     porous_source = -kappa / (2 * b.hp * b.mu)

#     for i, ha in enumerate(b.ha):
#         h = ha + b.geom
#         if b.csys == "polar":
#             epsilon_r = b.x[:, None] * (1 + b.Psi) * h**3 / (24 * b.mu)
#             epsilon_theta = (1 + b.Psi) * h**3 / (24 * b.mu)
#             epsilon = (epsilon_r, epsilon_theta)
#         elif b.csys == "cartesian":
#             epsilon = (1 + b.Psi) * h**3 / (24 * b.mu)

#         # boundary conditions
#         if b.case == "rectangular":
#             bc = {
#                 "west": "Dirichlet",
#                 "east": "Dirichlet",
#                 "north": "Dirichlet",
#                 "south": "Dirichlet",
#             }
#         elif b.case == "circular":
#             bc = {
#                 "west": "Neumann",
#                 "east": "Dirichlet",
#                 "north": "Periodic",
#                 "south": "Periodic",
#             }
#         elif b.case == "annular":
#             bc = {
#                 "west": "Dirichlet",
#                 "east": "Dirichlet",
#                 "north": "Periodic",
#                 "south": "Periodic",
#             }
#         elif b.case == "journal":
#             bc = {
#                 "west": "Dirichlet",
#                 "east": "Dirichlet",
#                 "north": "Neumann",
#                 "south": "Neumann",
#             }

#         # Build the 2D differential matrix
#         A = build_2d_diff_matrix(
#             epsilon=epsilon,
#             porous_source=porous_source,
#             dx=b.dx,
#             dy=b.dy,
#             bc=bc,
#             N=N,
#             M=M,
#         )

#         # Right-hand side (forcing term)
#         f = b.ps**2 * porous_source

#         match bc["west"]:
#             case "Dirichlet":
#                 f[0, :] = b.pa**2 if b.type == "bearing" else b.pc**2
#             case "Neumann":
#                 f[0, :] = 0
#             case "Periodic":
#                 pass
#             case _:
#                 raise ValueError("invalid BC")

#         match bc["east"]:
#             case "Dirichlet":
#                 f[-1, :] = b.pa**2
#             case "Neumann":
#                 f[-1, :] = 0
#             case "Periodic":
#                 pass
#             case _:
#                 raise ValueError("invalid BC")

#         match bc["north"]:
#             case "Dirichlet":
#                 f[1:-1, -1] = b.pa**2
#             case "Neumann":
#                 f[1:-1, -1] = 0
#             case "Periodic":
#                 pass
#             case _:
#                 raise ValueError("invalid BC")

#         match bc["south"]:
#             case "Dirichlet":
#                 f[1:-1, 0] = b.pa**2
#             case "Neumann":
#                 f[1:-1, 0] = 0
#             case "Periodic":
#                 pass
#             case _:
#                 raise ValueError("invalid BC")

#         # Solve the linear system
#         A = A.tocsr()
#         f = f.flatten("F")
#         p_flat = spla.spsolve(A, f)
#         p[:, :, i] = p_flat.reshape((M, N)) ** 0.5

#         if i == 5:
#             try:
#                 import matplotlib.pyplot as plt

#                 arr = np.hstack([A.toarray(), f[:, None]])
#                 arr[arr == 0] = np.nan
#                 plt.imshow(np.log(np.abs(arr)))
#                 plt.show()
#             except:
#                 pass
#     return p

# def build_2d_diff_matrix(
#     epsilon: np.ndarray,
#     porous_source: np.ndarray,
#     dx: float,
#     dy: float,
#     bc: dict,
#     N: int,
#     M: int,
# ) -> sp.csr_matrix:
#     """
#     Construct a finite-difference matrix for 2D differential operator:
#     coef @ D_r(epsilon * D_r(f(r))) + coef @ D_x(epsilon * D_x(f(x)))

#     Args:
#         coef (np.ndarray): Coefficient matrix.
#         eps (np.ndarray): Epsilon matrix (variable coefficients) (N,M).
#         dr (np.ndarray): Radial grid spacing.
#         dx (np.ndarray): angular grid spacing.
#         bc (dict): Dictionary specifying boundary conditions for "left", "right", "top", and "bottom".

#     Returns:
#         sp.csr_matrix: Sparse matrix representing the 2D differential operator.
#     """

#     eps_x = (epsilon[:-1, :] + epsilon[1:, :]) / 2
#     eps_y = (epsilon[:, :-1] + epsilon[:, 1:]) / 2


#     eps_w = np.vstack([eps_x, np.zeros((1, M))])
#     eps_e = np.vstack([np.zeros((1, M)), eps_x])
#     eps_s = np.hstack([eps_y, np.zeros((N, 1))])
#     eps_n = np.hstack([np.zeros((N, 1)), eps_y])

#     eps_w[:, [0,-1]] = 0
#     eps_e[:, [0,-1]] = 0
#     eps_s[[0, -1], :] = 0
#     eps_n[[0, -1], :] = 0

#     # stencil coefficeints
#     center = (
#         -(eps_w + eps_e) / dx[:, None] ** 2
#         - (eps_n + eps_s) / dy[None, :] ** 2
#         + porous_source
#     )
#     west = eps_w / dx[:, None] ** 2
#     east = eps_e / dx[:, None] ** 2
#     north = eps_n / dy[None, :] ** 2
#     south = eps_s / dy[None, :] ** 2

#     diag_north_periodic = np.zeros(N)
#     diag_south_periodic = np.zeros(N)

#     west[:, [0, -1]] = 0
#     east[:, [0, -1]] = 0

#     north[[0, -1], :] = 0
#     south[[0, -1], :] = 0

#     # Boundary conditions
#     match bc["west"]:
#         case "Dirichlet":
#             center[0, :] = 1
#             east[0, :] = 0
#         case "Neumann":
#             east[0, :] = -center[0, :]
#         case "Periodic":
#             pass
#         case _:
#             raise ValueError("invalid BC")

#     match bc["east"]:
#         case "Dirichlet":
#             center[-1, :] = 1
#             west[-1, :] = 0
#         case "Neumann":
#             west[-1, :] = -center[-1, :]
#         case "Periodic":
#             pass
#         case _:
#             raise ValueError("invalid BC")

#     match bc["north"]:
#         case "Dirichlet":
#             center[1:-1, -1] = 1
#             south[1:-1, -1] = 0
#         case "Neumann":
#             south[:, -1] = -center[:, -1]
#         case "Periodic":
#             diag_north_periodic = north[:, -1]
#             diag_north_periodic[[0, -1]] = 0
#         case _:
#             raise ValueError("invalid BC")

#     match bc["south"]:
#         case "Dirichlet":
#             center[1:-1, 0] = 1
#             north[1:-1, 0] = 0
#         case "Neumann":
#             north[:, 0] = -center[:, 0]
#         case "Periodic":
#             diag_south_periodic = south[:, 0]
#             diag_south_periodic[[0, -1]] = 0
#         case _:
#             raise ValueError("invalid BC")

#     # sparse matrix diagonals
#     diag_center = center.flatten("F")
#     diag_west = west.flatten("F")[1:]
#     diag_east = east.flatten("F")[:-1]
#     diag_north = north.flatten("F")[:-N]
#     diag_south = south.flatten("F")[N:]

#     # diag_east[N - 1 :: N] = 0
#     # diag_west[N - 2 :: N] = 0

#     # diag_south[0::N] = 0
#     # diag_south[N - 1 :: N] = 0

#     # diag_north[0::N] = 0
#     # diag_north[N - 1 :: N] = 0

#     # diag_east[:N] = 0
#     # diag_east[-N:] = 0

#     # diag_west[:N] = 0
#     # diag_west[-N:] = 0


#     ind_periodic = N * (M - 1)

#     L_mat = sp.diags(
#         [
#             diag_center,
#             diag_east,
#             diag_west,
#             diag_north,
#             diag_south,
#             diag_north_periodic,
#             diag_south_periodic,
#         ],
#         [0, 1, -1, N, -N, ind_periodic, -ind_periodic],
#         format="csr",
#     )
#     return L_mat
