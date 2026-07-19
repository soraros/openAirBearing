import openairbearing as ab

if __name__ == "__main__":
    # initialize circular bearing with parameters
    bearing = ab.JournalBearing()

    # solve with analytic and numeric methods
    result = ab.solve_bearing(bearing, "numeric2d")

    # plot results
    # figure = ab.plot_key_results(bearing, result)
    # ab.plot_load_capacity(bearing, result).show()
    # ab.plot_stiffness(bearing, result).show()
    # ab.plot_pressure_distribution(bearing, result).show()
    # ab.plot_supply_flow_rate(bearing, result).show()
    # ab.plot_chamber_flow_rate(bearing, result).show()
    # ab.plot_ambient_flow_rate(bearing, result).show()
    # ab.plot_key_results(bearing, result).show()
