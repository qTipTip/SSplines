import numpy as np
import sympy as sy
from SSplines import SplineSpace, sample_triangle
from SSplines.constants import KNOT_MULTIPLICITIES_QUADRATIC
from SSplines.helper_functions import determine_sub_triangle, barycentric_coordinates
from SSplines.symbolic import polynomial_pieces


def test_gradient_quadratic_corner_basis_functions():
    X, Y = sy.symbols('X Y')

    triangle = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])

    S = SplineSpace(triangle, 2)
    points = sample_triangle(triangle, 15)
    for basis_num in [0, 4, 8]:
        b_num = S.basis()[basis_num]
        b_sym = polynomial_pieces(triangle, KNOT_MULTIPLICITIES_QUADRATIC[basis_num])

        for p in points:
            b = barycentric_coordinates(triangle, p)
            k = determine_sub_triangle(b)[0]
            b_lapl = sy.diff(b_sym[k], X, X) + sy.diff(b_sym[k], Y, Y)
            num_b_sym = sy.lambdify([X, Y], b_lapl)
            np.testing.assert_almost_equal(b_num.lapl(p), num_b_sym(p[0], p[1]))
