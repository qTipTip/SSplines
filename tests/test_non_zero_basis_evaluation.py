import numpy as np

from SSplines.helper_functions import sample_triangle, barycentric_coordinates, evaluate_non_zero_basis_splines, \
    determine_sub_triangle


def test_non_zero_splines_evaluation_multiple_quadratic():
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

    points = sample_triangle(triangle, 30)
    bary_coords = barycentric_coordinates(triangle, points)
    k = determine_sub_triangle(bary_coords)
    s = evaluate_non_zero_basis_splines(d, bary_coords, k)

    expected_sum = np.ones((len(points)))
    computed_sum = s.sum(axis=1)

    np.testing.assert_almost_equal(expected_sum, computed_sum)


def test_non_zero_splines_evaluation_multiple_linear():
    """
    Computes all the non-zero linear basis splines at a set of points, and asserts
    that for each point, the basis functions sum to one.
    """

    triangle = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])

    d = 1

    points = sample_triangle(triangle, 30)
    bary_coords = barycentric_coordinates(triangle, points)
    k = determine_sub_triangle(bary_coords)

    s = evaluate_non_zero_basis_splines(d, bary_coords, k)
    expected_sum = np.ones((len(points)))
    computed_sum = s.sum(axis=1)

    np.testing.assert_almost_equal(expected_sum, computed_sum)
