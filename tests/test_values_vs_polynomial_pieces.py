import numpy as np
import sympy as sy

from SSplines import SplineSpace, ps12_sub_triangles, sample_triangle
from SSplines.constants import KNOT_MULTIPLICITIES_QUADRATIC
from SSplines.symbolic import polynomial_pieces


def test_values_quadratic_basis():
    """
    tests that the polynomial pieces and the numerical splines evaluate to the same value over all the
    subtriangles.
    """
    X, Y = sy.symbols('X Y')

    triangle = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])

    S = SplineSpace(triangle, 2)

    for basis_num in range(12):
        b_num = S.basis()[basis_num]
        b_sym = polynomial_pieces(triangle, KNOT_MULTIPLICITIES_QUADRATIC[basis_num])

        subtriangles = ps12_sub_triangles(triangle)

        for k in range(12):
            t = subtriangles[k]
            points = sample_triangle(t, 1)
            num_b_sym = sy.lambdify([X, Y], b_sym[k])

            for p in points:
                np.testing.assert_almost_equal(b_num(p), num_b_sym(p[0], p[1]))
