import numpy as np

from SSplines.helper_functions import sample_triangle
from SSplines.spline_function import SplineFunction


def test_spline_function_evaluation_multiple_linear():
    """
    Evaluates the linear spline function with all coefficients equal to one, and assert that the value
    is indeed 1 for all points.
    """

    triangle = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])

    d = 1
    c = np.ones(10)
    f = SplineFunction(coefficients=c, degree=d, triangle=triangle)

    points = sample_triangle(triangle, 30)

    expected_values = np.ones(len(points))
    computed_values = f(points)

    np.testing.assert_almost_equal(expected_values, computed_values)


def test_spline_function_evaluation_multiple_quadratic():
    """
    Computes all the non-zero quadratic basis splines at a set of points, and asserts
    that for each point, the basis functions sum to one.
    """

    triangle = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])

    d = 2
    c = np.ones(12)
    f = SplineFunction(coefficients=c, degree=d, triangle=triangle)

    points = sample_triangle(triangle, 30)

    expected_values = np.ones(len(points))
    computed_values = f(points)

    np.testing.assert_almost_equal(expected_values, computed_values)
