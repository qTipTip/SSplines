import numpy as np

from SSplines.helper_functions import gaussian_quadrature_data, gaussian_quadrature


def test_linear_reproduction():
    def f(p):
        return


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
