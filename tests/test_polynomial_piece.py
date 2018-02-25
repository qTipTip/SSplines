import numpy as np
from sympy import symbols, simplify

from SSplines.constants import KNOT_MULTIPLICITIES_QUADRATIC
from SSplines.symbolic import polynomial_pieces


def test_polynomial_pieces_quadratic_basis():
    X, Y = symbols('X Y')
    triangle = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])

    # knot multiplicities corresponding to the first quadratic basis function
    multiplicities = KNOT_MULTIPLICITIES_QUADRATIC[0]

    expected_polynomials = (16 * X ** 2 + 32 * X * Y + 16 * Y ** 2 - 16 * X - 16 * Y + 4,
                            16 * X ** 2 + 32 * X * Y + 16 * Y ** 2 - 16 * X - 16 * Y + 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    computed_polynomials = polynomial_pieces(triangle, multiplicities)

    for e, c in zip(expected_polynomials, computed_polynomials):
        np.testing.assert_almost_equal(simplify(e - c), 0)


def test_polynomial_pieces_recurrence():
    X, Y = symbols('X Y')
    triangle = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])

    # knot multiplicities corresponding to the fourth quadratic basis function
    multiplicities = KNOT_MULTIPLICITIES_QUADRATIC[3]

    expected_polynomials = (
        0, 4 * X ** 2 - 8 * X * Y + 4 * Y ** 2, -12 * X ** 2 - 8 * X * Y + 4 * Y ** 2 + 16 * X - 4,
        -12 * X ** 2 - 8 * X * Y + 4 * Y ** 2 + 16 * X - 4, 0, 0, 0, 4 * X ** 2 - 8 * X * Y + 4 * Y ** 2,
        4 * X ** 2 - 8 * X * Y + 4 * Y ** 2, 4 * X ** 2 - 8 * X * Y + 4 * Y ** 2, 0, 0)
    computed_polynomials = polynomial_pieces(triangle, multiplicities)

    for e, c in zip(expected_polynomials, computed_polynomials):
        np.testing.assert_almost_equal(simplify(e - c), 0)
