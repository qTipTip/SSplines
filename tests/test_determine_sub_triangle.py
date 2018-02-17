import numpy as np

from SSplines.helper_functions import determine_sub_triangle, barycentric_coordinates


def test_determine_sub_triangle_multiple():
    triangle = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])

    points = np.array([
        [0, 0.25],
        [0, 0.5],
        [0.25, 0.25],
        [0.5, 0],
    ])
    bary_coords = barycentric_coordinates(triangle, points)
    expected = np.array([
        0, 5, 7, 2
    ])

    computed = determine_sub_triangle(bary_coords)
    np.testing.assert_almost_equal(computed, expected)

    assert computed.dtype == np.int


def test_determine_sub_triangle_single():
    triangle = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])

    points = np.array([
        [0, 0.25],
        [0, 0.5],
        [0.25, 0.25],
        [0.5, 0],
    ])
    bary_coords = barycentric_coordinates(triangle, points)
    expected = np.array([
        0, 5, 7, 2
    ])

    for b, e in zip(bary_coords, expected):
        computed = determine_sub_triangle(b)
        np.testing.assert_almost_equal(computed, e)

        assert computed.dtype == np.int
