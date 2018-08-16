import numpy as np

from SSplines import barycentric_coordinates


def test_barycentric_coordinates():
    triangle = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])
    x = np.array([0.5, 0.5])

    b_numerical = barycentric_coordinates(triangle, x, exact=False)
    b_exact = barycentric_coordinates(triangle, x, exact=True)

    for num, exac in zip(b_numerical, b_exact):
        np.testing.assert_almost_equal(num, exac)
