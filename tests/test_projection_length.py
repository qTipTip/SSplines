import numpy as np

from SSplines.helper_functions import projection_length


def test_projection_length_unit():
    u = np.array([1, 0])
    v = np.array([1, np.sqrt(3)])

    computed_projection_length = projection_length(u, v)
    expected_projection_length = 1

    np.testing.assert_almost_equal(computed_projection_length, expected_projection_length)


def test_projection_length_arbitrary():
    u = np.array([3, 1])
    v = np.array([2, 2])

    computed_projection_length = projection_length(u, v)
    expected_projection_length = 4 / 5

    np.testing.assert_almost_equal(computed_projection_length, expected_projection_length)


def test_projection_length_orthogonal():
    u = np.array([0, 1, 0])
    v = np.array([3, 0, 0])

    computed_projection_length = projection_length(u, v)
    expected_projection_length = 0

    np.testing.assert_almost_equal(computed_projection_length, expected_projection_length)
