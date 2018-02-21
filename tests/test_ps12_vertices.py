import numpy as np

from SSplines.helper_functions import ps12_vertices


def test_ps12_vertices():
    vertices = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])

    expected = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [0.5, 0],
        [0.5, 0.5],
        [0, 0.5],
        [0.25, 0.25],
        [0.5, 0.25],
        [0.25, 0.5],
        [1 / 3, 1 / 3]
    ])

    computed = ps12_vertices(vertices)
    np.testing.assert_almost_equal(computed, expected)
