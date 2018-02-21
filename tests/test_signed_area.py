import numpy as np

from SSplines.helper_functions import signed_area


# noinspection PyTypeChecker
def test_signed_area_unit_simplex():
    vertices = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])

    expected_area = 0.5
    computed_area = signed_area(vertices)

    np.testing.assert_almost_equal(expected_area, computed_area)


# noinspection PyTypeChecker
def test_signed_area_unit_simplex_reverse_orientation():
    vertices = np.array([
        [0, 0],
        [0, 1],
        [1, 0]
    ])

    expected_area = -0.5
    computed_area = signed_area(vertices)

    np.testing.assert_almost_equal(expected_area, computed_area)


def test_signed_area_arbitrary_triangle():
    vertices = np.array([
        [5, 1],
        [6, 1],
        [5.5, 1 + np.sqrt(3) / 2]
    ])

    expected_area = np.sqrt(3) / 4
    computed_area = signed_area(vertices)

    np.testing.assert_almost_equal(expected_area, computed_area)


def test_signed_area_multiple_triangles():
    vertices = np.array([
        [
            [0, 0],
            [0, 1],
            [1, 0]
        ],
        [
            [5, 1],
            [6, 1],
            [5.5, 1 + np.sqrt(3) / 2]
        ]
    ])

    expected_area = np.array([-1 / 2, np.sqrt(3) / 4])
    computed_area = signed_area(vertices)

    np.testing.assert_almost_equal(expected_area, computed_area)
