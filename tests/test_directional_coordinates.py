import numpy as np

from SSplines.helper_functions import directional_coordinates


def test_directional_coordinates_multiple():
    """
    Verifies that ~`directional_coordinates` returns the correct values when supplied with an array of input values.
    """

    triangle = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])

    directions = np.array([
        [-1, 0],
        [-12, 3]
    ])

    expected = np.array([
        [1, -1, 0],
        [9, -12, 3]
    ])

    computed = directional_coordinates(triangle, directions)
    np.testing.assert_almost_equal(computed, expected)


def test_directional_coordinates_single():
    """
    Verifies that ~`directional_coordinates` returns the correct values when supplied with a single input.
    """

    triangle = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])

    directions = np.array([
        [-1, 0],
        [-12, 3]
    ])

    expected = np.array([
        [1, -1, 0],
        [9, -12, 3]
    ])

    for u, a in zip(directions, expected):
        computed = directional_coordinates(triangle, u)
        np.testing.assert_almost_equal(computed, np.atleast_2d(a))
