import numpy as np
from SSplines.constants import KNOT_MULTIPLICITIES_QUADRATIC
from SSplines.symbolic import polynomial_pieces
from sympy import symbols, simplify


def test_polynomial_pieces_quadratic_basis():
    X, Y = symbols('X Y')
    triangle = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])

    # knot multiplicities corresponding to the first quadratic basis function
    multiplicities = KNOT_MULTIPLICITIES_QUADRATIC[0]

    expected_polynomials = [1.0 * (-2.0 * X - 2.0 * Y + 1) ** 2, 1.0 * (-2.0 * X - 2.0 * Y + 1) ** 2, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0]
    computed_polynomials = polynomial_pieces(triangle, multiplicities)
    for e, c in zip(expected_polynomials, computed_polynomials):
        np.testing.assert_almost_equal(simplify(e - c), 0)

