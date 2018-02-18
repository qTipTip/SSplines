import numpy as np
import pytest

from SSplines.spline_space import SplineSpace


def test_hermite_basis_nodal_properties():
    triangle = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])

    d = 2
    S = SplineSpace(triangle, d)
    B = S.hermite_basis()

    n1 = np.array([0, 1])
    n2 = np.array([-np.sqrt(2) / 2, -np.sqrt(2) / 2])
    n3 = np.array([1, 0])

    m1 = [0.5, 0]
    m2 = [0.5, 0.5]
    m3 = [0, 0.5]

    p1, p2, p3 = triangle

    np.testing.assert_almost_equal(B[0](p1), 1)
    np.testing.assert_almost_equal(B[4](p2), 1)
    np.testing.assert_almost_equal(B[8](p3), 1)

    np.testing.assert_almost_equal(B[1].dx(p1), 1)
    np.testing.assert_almost_equal(B[5].dx(p2), 1)
    np.testing.assert_almost_equal(B[9].dx(p3), 1)

    np.testing.assert_almost_equal(B[2].dy(p1), 1)
    np.testing.assert_almost_equal(B[6].dy(p2), 1)
    np.testing.assert_almost_equal(B[10].dy(p3), 1)

    np.testing.assert_almost_equal(B[3].D(m1, n1, 1), 1)
    np.testing.assert_almost_equal(B[7].D(m2, n2, 1), 1)
    np.testing.assert_almost_equal(B[11].D(m3, n3, 1), 1)


def test_hermite_basis_wrong_degree():
    triangle = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])

    d = 1
    S = SplineSpace(triangle, d)

    with pytest.raises(AssertionError):
        S.hermite_basis()
