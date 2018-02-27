import numpy as np
from SSplines import gaussian_quadrature_data, gaussian_quadrature_ps12


def test_quadrature_over_multiple_triangles():
    triangle = np.array([
        [0, 3],
        [1, 1],
        [5, 3]
    ])

    def f(p):
        return 6 * p[:, 0] ** 2 - 40 * p[:, 1]

    b, w = gaussian_quadrature_data(order=2)

    expected_integral = -935 / 3
    computed_integral = gaussian_quadrature_ps12(triangle, f, b, w)

    np.testing.assert_almost_equal(expected_integral, computed_integral)
