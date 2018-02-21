import numpy as np

from SSplines import sample_triangle
from SSplines.helper_functions import barycentric_coordinates, points_from_barycentric_coordinates


def test_points_from_barycentric_coordinates():
    vertices = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])
    points = sample_triangle(vertices, 450)

    bary_coords = barycentric_coordinates(vertices, points)
    points_from_bary_coords = points_from_barycentric_coordinates(vertices, bary_coords)
    np.testing.assert_almost_equal(points, points_from_bary_coords)
