import numpy as np

from SSplines import SimplexSpline


def test_evaluation_coordinates_single():
    triangle = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])
    knot_multiplicities = [3, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    s = SimplexSpline(triangle, knot_multiplicities)

    x = np.array([0.5, 0.5])
    b = np.array([0, 0.5, 0.5])

    computed_barycentric = s(b, barycentric=True)
    computed_cartesian = s(x, barycentric=False)

    np.testing.assert_array_almost_equal_nulp(computed_barycentric, computed_cartesian)


def test_evaluation_coordinates_multiple():
    triangle = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])
    knot_multiplicities = [3, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    s = SimplexSpline(triangle, knot_multiplicities)

    x = np.array([[0.5, 0.5],
                  [0.5, 0.5]])
    b = np.array([[0, 0.5, 0.5],
                  [0, 0.5, 0.5]])

    computed_barycentric = s(b, barycentric=True)
    computed_cartesian = s(x, barycentric=False)

    np.testing.assert_array_almost_equal_nulp(computed_barycentric, computed_cartesian)
