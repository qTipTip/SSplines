import numpy as np
import pytest
from SSplines.helper_functions import barycentric_coordinates, barycentric_coordinates_multiple_triangles


def test_barycentric_coordinates_multiple():
    """
    Verifies that ~`barycentric_coordinates` returns the correct values when supplied with an array of input values.
    """

    triangle = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])

    points = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [0.5, 0],
        [-1, 0],
        [1.1, 0],
        [1 / 3, 1 / 3]
    ])

    expected = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0.5, 0.5, 0],
        [2, -1, 0],
        [-0.1, 1.1, 0],
        [1 / 3, 1 / 3, 1 / 3]
    ])

    computed = barycentric_coordinates(triangle, points)
    np.testing.assert_almost_equal(computed, expected)


def test_barycentric_coordinates_single():
    """
    Verifies that ~`barycentric_coordinates` returns the correct values when supplied with a single input value.
    """
    triangle = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])

    points = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [0.5, 0],
        [-1, 0],
        [1.1, 0],
        [1 / 3, 1 / 3]
    ])

    expected = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0.5, 0.5, 0],
        [2, -1, 0],
        [-0.1, 1.1, 0],
        [1 / 3, 1 / 3, 1 / 3]
    ])

    for b, p in zip(expected, points):
        computed = barycentric_coordinates(triangle, p)
        np.testing.assert_almost_equal(computed, np.atleast_2d(b))


@pytest.mark.skip('Not implemented yet')
def test_barycentric_coordinates_two_triangles_single_point():
    t1 = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])
    t2 = np.array([
        [1, 0],
        [1, 1],
        [0, 1]
    ])

    triangle_vertices = np.array([
        t1,
        t2
    ])

    x = np.array([0.5, 0.5])

    expected = np.array([
        [0, 0.5, 0.5],
        [0.5, 0, 0.5]
    ])
    computed = barycentric_coordinates_multiple_triangles(triangle_vertices, x)

    np.testing.assert_almost_equal(computed, expected)
