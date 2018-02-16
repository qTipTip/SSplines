import numpy as np

from SSplines.helper_functions import coefficients_linear, sub_matrix, coefficients_quadratic


def test_sub_matrix_multiple_linear():
    k = np.array([0, 1])
    matrix = np.array([
        np.zeros((12, 10)),
        np.zeros((12, 10))
    ])
    c = coefficients_linear(k)
    d = 1
    matrix[0, k[0], c[0]] = np.ones(3)
    matrix[1, k[1], c[1]] = np.ones(3)

    sub = sub_matrix(matrix, d, k)

    # sum over the sub-matrix to see if it is the expected value, in which case
    # the correct sub-matrix was extracted.
    computed_sum = np.squeeze(np.apply_along_axis(np.sum, 2, sub))
    expected_sum = [3, 3]

    np.testing.assert_almost_equal(computed_sum, expected_sum)
    assert sub.shape == (2, 1, 3)


def test_sub_matrix_multiple_quadratic():
    k = np.array([0, 1])
    matrix = np.array([
        np.zeros((10, 12)),
        np.zeros((10, 12))
    ])
    cl = coefficients_linear(k)
    cq = coefficients_quadratic(k)
    d = 2

    matrix[np.ix_([0], cl[0], cq[0])] = np.ones((3, 6))
    matrix[np.ix_([1], cl[1], cq[1])] = np.ones((3, 6))

    sub = sub_matrix(matrix, d, k)

    # sum over the sub-matrix to see if it is the expected value, in which case
    # the correct sub-matrix was extracted.
    computed_sum = np.sum(sub)
    expected_sum = [36]

    np.testing.assert_almost_equal(computed_sum, expected_sum)
    assert sub.shape == (2, 3, 6)
