import numpy as np

from SSplines.helper_functions import sample_triangle, barycentric_coordinates, evaluate_non_zero_basis_splines


def test_non_zero_splines_evaluation_multiple():
    """
    Computes all the non-zero basis splines at a set of points, and asserts
    that for each point, the basis functions sum to one.
    """

    triangle = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])

    d = 2

    points = sample_triangle(triangle, 30)
    bary_coords = barycentric_coordinates(triangle, points)

    s = evaluate_non_zero_basis_splines(triangle, d, bary_coords)

    expected_sum = np.ones((len(points)))
    computed_sum = s.sum(axis=1)

    np.testing.assert_almost_equal(expected_sum, computed_sum)
