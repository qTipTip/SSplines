import numpy as np
import pytest

from SSplines.helper_functions import gaussian_quadrature_data, gaussian_quadrature


def test_linear_reproduction():
    def f(p):
        return p[:, 0] + p[:, 1]

    triangle = np.array([
        [0, 0],
        [2, 0],
        [2, 1]
    ])
    b, w = gaussian_quadrature_data(order=1)

    expected_integral = 5 / 3
    computed_integral = gaussian_quadrature(triangle, f, b, w)

    np.testing.assert_almost_equal(expected_integral, computed_integral)


def test_quadratic_reproduction():
    def f(p):
        return 6 * p[:, 0] ** 2 - 40 * p[:, 1]

    triangle = np.array([
        [0, 3],
        [1, 1],
        [5, 3]
    ])
    b, w = gaussian_quadrature_data(order=2)

    expected_integral = -935 / 3
    computed_integral = gaussian_quadrature(triangle, f, b, w)

    np.testing.assert_almost_equal(expected_integral, computed_integral)


def test_quadratic_reproduction_arbitrary_triangle():
    def f(p):
        return p[:, 0] ** 2 + p[:, 1] ** 2 + p[:, 0] * p[:, 1]

    triangle = np.array([
        [3, 1.5],
        [5, 1.5],
        [5, 2.5]
    ])
    b, w = gaussian_quadrature_data(order=2)

    expected_integral = 30.4167
    computed_integral = gaussian_quadrature(triangle, f, b, w)

    np.testing.assert_almost_equal(expected_integral, computed_integral, decimal=3)


@pytest.mark.skip(reason='Error somewhere in cubic integration')
def test_cubic_reproduction():
    def f(p):
        return p[:, 0] ** 3

    triangle = np.array([
        [0, 0],
        [2, 0],
        [2, 1]
    ])

    b, w = gaussian_quadrature_data(order=3)

    expected_integral = 16 / 5
    computed_integral = gaussian_quadrature(triangle, f, b, w)

    np.testing.assert_almost_equal(computed_integral, expected_integral)


@pytest.mark.skip(reason='Error somewhere in cubic integration')
def test_cubic_reproduction_unit_simplex():
    def f(p):
        return p[:, 0] ** 3

    triangle = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])
    b, w = gaussian_quadrature_data(order=3)

    expected_integral = 1 / 20
    computed_integral = gaussian_quadrature(triangle, f, b, w)

    np.testing.assert_almost_equal(computed_integral, expected_integral)


@pytest.mark.skip(reason='Error somewhere in cubic integration')
def test_cubic_reproduction_of_linear_polynomial():
    def f(p):
        return p[:, 0] + p[:, 1]

    triangle = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])
    b, w = gaussian_quadrature_data(order=3)

    expected_integral = 1 / 3
    computed_integral = gaussian_quadrature(triangle, f, b, w)
    np.testing.assert_almost_equal(computed_integral, expected_integral)
