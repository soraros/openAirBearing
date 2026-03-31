# def test_plot_key_results():
#     """Test the plot_key_results function."""
#     # Mock a bearing object
#     bearings = [
#         CircularBearing(),
#         AnnularBearing(),
#         InfiniteLinearBearing(),
#         RectangularBearing(),
#     ]
#     for bearing in bearings:

#         # Mock results
#         class MockResult:
#             def __init__(self, name):
#                 self.name = name
#                 self.w = np.random.rand(bearing.nh)
#                 self.k = np.random.rand(bearing.nh)
#                 self.qs = np.random.rand(bearing.nh)
#                 self.qc = np.random.rand(bearing.nh)
#                 self.qa = np.random.rand(bearing.nh)
#                 if bearing.case == "rectangular":
#                     self.p = np.random.rand(bearing.ny, bearing.nx, bearing.nh)
#                 else:
#                     self.p = np.random.rand(bearing.nx, bearing.nh)

#         results = [MockResult("Analytic"), MockResult("Numeric")]

#         # Call the function
#         fig = plot_key_results(bearing, results)

#         # Assertions
#         assert isinstance(fig, Figure), "The output should be a Plotly Figure."
#         assert len(fig.data) > 0, "The figure should contain traces."
#         assert len(fig.layout.annotations) > 0, "The figure should have subplot titles."


# def test_plot_key_results_no_results():
#     """Test plot_key_results with no results."""
#     # Mock a bearing object
#     bearing = CircularBearing()

#     # Call the function with no results
#     fig = plot_key_results(bearing, [])

#     # Assertions
#     assert isinstance(fig, Figure), "The output should be a Plotly Figure."
#     assert len(fig.data) == 0, "The figure should contain no traces."
#     assert (
#         fig.layout.title.text == "No suitable solver selected"
#     ), "The figure should display a message when no results are provided."


# def test_plot_bearing_shape():
#     """Test the plot_bearing_shape function."""
#     # Mock a bearing object
#     bearings = [
#         CircularBearing(),
#         AnnularBearing(),
#         InfiniteLinearBearing(),
#         RectangularBearing(),
#     ]
#     for bearing in bearings:
#         bearing.ha = np.linspace(bearing.ha_min, bearing.ha_max, bearing.nx)
#         bearing.x = np.linspace(0, bearing.xa, bearing.nx)

#         # Call the function
#         fig = plot_bearing_shape(bearing)

#         # Assertions
#         assert isinstance(fig, Figure), "The output should be a Plotly Figure."
#         assert len(fig.data) > 0, "The figure should contain traces."
#         assert len(fig.layout.annotations) > 0, "The figure should have subplot titles."
