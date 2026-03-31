import openairbearing as ab


def circular_bearing_example():
  """
  Example of a circular bearing with a quadratic error profile.
  """

  # initialize circular bearing with parameters
  # 2 micrometer error with concave quadratic profile
  bearing = ab.CircularBearing(
    xa=40, Qsc=5, nx=50, nh=60, error_type="quadratic", error=-2e-6
  )

  # solve with analytic and numeric methods
  result = [
    ab.solve_bearing(bearing, "analytic"),
    ab.solve_bearing(bearing, "numeric"),
  ]

  ab.plot_load_capacity(bearing, result).show()
  ab.plot_stiffness(bearing, result).show()
  ab.plot_pressure_distribution(bearing, result).show()
  ab.plot_supply_flow_rate(bearing, result).show()
  ab.plot_chamber_flow_rate(bearing, result).show()
  ab.plot_ambient_flow_rate(bearing, result).show()


if __name__ == "__main__":
  circular_bearing_example()
