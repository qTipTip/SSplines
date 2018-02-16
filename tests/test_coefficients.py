import numpy as np

from SSplines.helper_functions import determine_sub_triangle, coefficients_linear, barycentric_coordinates


def test_coefficients_linear_multiple():
    vertices = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])

    points = np.array([
        [0, 0.25],
        [0, 0.5],
        [0.25, 0.25],
        [0.5, 0],
    ])

    b = barycentric_coordinates(vertices, points)
    k = determine_sub_triangle(vertices, b)

    expected = np.array([
        [0, 5, 6],
        [2, 5, 8],
        [3, 6, 9],
        [1, 3, 7]
    ])

    computed = coefficients_linear(k)

    np.testing.assert_almost_equal(computed, expected)


def test_coefficients_linear_single():
    vertices = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])

    points = np.array([
        [0, 0.25],
        [0, 0.5],
        [0.25, 0.25],
        [0.5, 0],
    ])

    b = barycentric_coordinates(vertices, points)
    k = determine_sub_triangle(vertices, b)

    expected = np.array([
        [0, 5, 6],
        [2, 5, 8],
        [3, 6, 9],
        [1, 3, 7]
    ])

    for c, e in zip(k, expected):
        computed = coefficients_linear(c)
        np.testing.assert_almost_equal(computed, e)
