import numpy as np

from SSplines.helper_functions import sample_triangle, barycentric_coordinates, determine_sub_triangle, \
    directional_coordinates, evaluate_non_zero_basis_derivatives


def test_non_zero_splines_derivatives_multiple_zeroth():
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
    r = 0
    u = np.array([0.5, 0.5])
    points = sample_triangle(triangle, 1)
    bary_coords = barycentric_coordinates(triangle, points)
    directional_coords = directional_coordinates(triangle, u)

    k = determine_sub_triangle(bary_coords)
    s = evaluate_non_zero_basis_derivatives(d, r, bary_coords, directional_coords, k)
    expected_sum = np.ones((len(points)))
    computed_sum = s.sum(axis=1)

    np.testing.assert_almost_equal(expected_sum, computed_sum)
