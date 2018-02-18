import numpy as np

from SSplines.helper_functions import barycentric_coordinates, determine_sub_triangle, directional_coordinates, \
    evaluate_non_zero_basis_derivatives


def edge_1_derivative(t):
    return [8 * t - 4, 4 - 12 * t, 0, 4 * t, 0, 0]


def edge_2_derivative(t):
    return [-4 * (1 - t), 0, 8 - 12 * t, 8 * t - 4, 0, 0]


def test_derivative_evaluation_along_edge():
    """
    Test to see if the derivative matches that of the univariate quadratic
    spline along the bottom edge of unit simplex
    """

    triangle = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])
    d = 2
    r = 1
    a = directional_coordinates(triangle, np.array([1, 0]))

    n = 1000
    t_values_1 = np.linspace(0, 0.5, n, endpoint=False)
    t_values_2 = np.linspace(0.5, 1, n, endpoint=False)

    p1 = np.array([
        t_values_1,
        np.zeros(n)
    ]).T

    b1 = barycentric_coordinates(triangle, p1)
    k1 = determine_sub_triangle(b1)

    expected_derivative_1 = np.array([edge_1_derivative(t) for t in t_values_1])
    computed_derivative_1 = evaluate_non_zero_basis_derivatives(d=d, r=r, k=k1, b=b1, a=a)

    np.testing.assert_almost_equal(expected_derivative_1, computed_derivative_1)

    p2 = np.array([
        t_values_2,
        np.zeros(n)
    ]).T

    b2 = barycentric_coordinates(triangle, p2)
    k2 = determine_sub_triangle(b2)

    expected_derivative_2 = np.array([edge_2_derivative(t) for t in t_values_2])
    computed_derivative_2 = evaluate_non_zero_basis_derivatives(d=d, r=r, k=k2, b=b2, a=a)

    np.testing.assert_almost_equal(expected_derivative_2, computed_derivative_2)
