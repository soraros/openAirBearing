import matplotlib.pyplot as plt

import openairbearing as ab

if __name__ == "__main__":
  # initialize circular bearing with parameters
  bearing = ab.RectangularBearing(
    nx=50,
    ny=60,
    error_type="none",
    error=-3e-6,
  )
  # print(np.round(bearing.geom*1e6, 2))
  # # plot bearing shape
  figure = ab.plot_bearing_shape(bearing)
  figure.show()

  # solve with analytic and numeric methods
  result = ab.solve_bearing(bearing, "numeric2d")

  plt.contourf(bearing.x * 1e3, bearing.y, result.p[:, :, 2] * 1e-6)
  plt.colorbar()
  plt.show()
  # plot results
  figure = ab.plot_key_results(bearing, result)
  figure.show()
