from math import factorial

import sympy as sp
from SSplines import ps12_vertices
from SSplines.constants import DELETED_KNOT_TO_TRIANGLE, KNOT_CONFIGURATION_TO_FACE_INDICES


def degenerate_support(s):
    """
    Checks whether the given list of vertex indices describe
    a degenerate triangle in the PS12-split.
    :param s: list of vertex indices.
    :return: true/false
    """

    # not enough vertices to describe a triangle.
    if len(s) < 3:
        return True
    # the following configurations only correspond to splines of degree at most two.
    if s == (0, 1, 3) or s == (1, 2, 4) or s == (0, 2, 5):
        return True

    # the following configurations corresponds to more general splines as the vertices are not used for higher degree
    # splines,
    # and these might not be needed.
    # implemented for completeness.
    if s == (3, 5, 6) or s == (3, 4, 7) or s == (4, 5, 8):
        return True
    if s == (0, 6, 9) or s == (0, 4, 6) or s == (0, 4, 9) or s == (4, 6, 9) \
            or s == (3, 8, 9) or s == (2, 3, 9) or s == (2, 3, 8) or s == (2, 8, 9) \
            or s == (1, 7, 9) or s == (1, 5, 7) or s == (1, 5, 9) or s == (5, 7, 9):
        return True
    return False


def barycentric_coordinates_symbolic(p0, p1, p2):
    X, Y = sp.symbols('X Y')

    A = sp.Matrix([[p0[0], p1[0], p2[0]],
                   [p0[1], p1[1], p2[1]],
                   [1, 1, 1]])
    b = sp.Matrix([[X], [Y], [1]])

    barycentric_coords = A.LUsolve(b)

    return list(barycentric_coords)


def polynomial_pieces(triangle, multiplicities):
    """
    Given a set of vertices, and a list of knot multiplicities, returns the twelve polynomial pieces corresponding to
    the S-spline basis function.
    :param triangle: vertices of triangle
    :param multiplicities: list of knot multiplicites
    :return: list of twelve polynomial pieces
    """

    ps12 = ps12_vertices(triangle)

    non_zero_multiplicities = [(k, m) for k, m in enumerate(multiplicities) if m != 0]
    knot_configuration = tuple([k for k, _ in non_zero_multiplicities])

    polynomials = [0] * 12

    if degenerate_support(knot_configuration):
        return polynomials

    elif len(knot_configuration) == 3:

        v1, v2, v3 = [ps12[i] for i in knot_configuration]
        b1, b2, b3 = barycentric_coordinates_symbolic(v1, v2, v3)

        mu1, mu2, mu3 = [m - 1 for _, m in non_zero_multiplicities]

        multinomial = factorial(mu1 + mu2 + mu3) / (factorial(mu1) * factorial(mu2) * factorial(mu3))
        Q = (b1 ** mu1 * b2 ** mu2 * b3 ** mu3) * multinomial

        faces = KNOT_CONFIGURATION_TO_FACE_INDICES[knot_configuration]
        for face in faces:
            polynomials[face] = Q

        return polynomials

    elif len(knot_configuration) > 3:
        # find a non-degenerate triangle in the knot_configuration and compute
        # barycentric coordinates with respect to this new triangle.
        reduced_knot_configuration = DELETED_KNOT_TO_TRIANGLE[knot_configuration]
        v1, v2, v3 = [ps12[i] for i in reduced_knot_configuration]
        bary_coordinates = barycentric_coordinates_symbolic(v1, v2, v3)

        Q = []

        # compute contributions recursively
        for i in reduced_knot_configuration:
            reduced_multiplicities = [m - (i == j) for j, m in enumerate(multiplicities)]
            reduced_polynomial_pieces = polynomial_pieces(triangle, reduced_multiplicities)
            Q.append(reduced_polynomial_pieces)

        # add up total
        for i in range(len(polynomials)):
            # for each of the 12 faces
            for j in range(len(Q)):
                # add the corresponding three contributions
                polynomials[i] += Q[j][i] * bary_coordinates[j]

        # Normalize by area
        # area_triangle = abs(signed_area(triangle))
        # area_knot_configuration = abs(signed_area(np.array([v1, v2, v3])))
        # factor = area_triangle / area_knot_configuration
        # for i in range(len(polynomials)):
        #    polynomials[i] *= factor

        return polynomials
