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


def test_points_from_barycentric_coordinates_cubic_quadrature_points():
    vertices = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])

    b = np.array([
        [1 / 3, 1 / 3, 1 / 3],
        [0.13333333, 0.13333333, 0.73333333],
        [0.73333333, 0.13333333, 0.13333333],
        [0.13333333, 0.73333333, 0.13333333]
    ])

    expected_points = np.array([
        [1 / 3, 1 / 3],
        [2 / 15, 11 / 15],
        [2 / 15, 2 / 15],
        [11 / 15, 2 / 15]
    ])

    computed_points = points_from_barycentric_coordinates(vertices, b)

    np.testing.assert_almost_equal(computed_points, expected_points)


def test_points_from_barycentric_coordinates_cubic_quadrature_points_arbitrary_triangle():
    vertices = np.array([
        [0, 0],
        [2, 0],
        [2, 1]
    ])

    b = np.array([
        [1 / 3, 1 / 3, 1 / 3],
        [0.13333333, 0.13333333, 0.73333333],
        [0.73333333, 0.13333333, 0.13333333],
        [0.13333333, 0.73333333, 0.13333333]
    ])

    expected_points = np.array([
        [4 / 3, 1 / 3],
        [26 / 15, 11 / 15],
        [8 / 15, 2 / 15],
        [26 / 15, 2 / 15]
    ])

    computed_points = points_from_barycentric_coordinates(vertices, b)

    np.testing.assert_almost_equal(computed_points, expected_points)
