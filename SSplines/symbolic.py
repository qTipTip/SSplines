import math

import numpy as np
import sympy as sp

import SSplines
from SSplines.dicts import KNOT_CONFIGURATION_TO_SUBTRIANGLES, DELETED_KNOT_TO_TRIANGLE, \
    KNOT_CONFIGURATION_TO_FACE_INDICES

X, Y = sp.symbols('X Y')


def degenerate(knot_configuration):
    """
    Given a knot configuration, return True if the corresponding triangle
    is degenerate, of area zero, or False otherwise.
    """

    if knot_configuration == (0, 1, 3) or knot_configuration == (1, 2, 4) or knot_configuration == (0, 2, 5):
        return True
    return False


def non_zero_multiplicities(knot_multiplicities):
    """
    Returns a list of the non-zero multiplicities in a list of knot
    multiplicities, as well as the corresponding knot configuration.
    """

    non_zero_m = []
    knot_confg = []
    for i, m in enumerate(knot_multiplicities):
        if m != 0:
            non_zero_m.append(m)
            knot_confg.append(i)
    return non_zero_m, knot_confg


def knot_configuration_to_non_degenerate_interior_triangle(knot_configuration):
    """
    Given a knot configuration, returns the indices of a triangle contained in
    the convex hull of the knots.
    """
    return DELETED_KNOT_TO_TRIANGLE[tuple(knot_configuration)]


def knot_configuration_to_face(knot_configuration):
    """
    Given a knot configuration, returns a list of face indices corresponding to
    the faces that lie within the convex hull of the knot configuration.
    """
    return KNOT_CONFIGURATION_TO_FACE_INDICES[tuple(knot_configuration)]


def barycentric_coordinates(triangle):
    """
    Returns barycentric coordinates with respect to the given triangle as
    functions of X, Y.
    """
    p0, p1, p2 = triangle
    A = sp.Matrix([[p0[0], p1[0], p2[0]],
                   [p0[1], p1[1], p2[1]],
                   [1, 1, 1]])
    b = sp.Matrix([[X], [Y], [1]])
    barycentric_coords = A.LUsolve(b)

    return list(barycentric_coords)


def multinomial(i, j, k):
    """
    Computes the trinomial for use in evaluation of Bernstein polynomials.
    """
    return math.factorial(i + j + k) / (math.factorial(i) * math.factorial(j) * math.factorial(k))


def bernstein_polynomial(full_triangle, sub_triangle, non_zero_multiplicities):
    """
    Computes the bernstein polynomial over given triangle with knot
    multiplicities. Non_zero_multiplicities has to be of length 3!
    """
    b1, b2, b3 = barycentric_coordinates(sub_triangle)
    m1, m2, m3 = [m - 1 for m in non_zero_multiplicities]

    area_full = SSplines.area(full_triangle)
    area_sub = SSplines.area(sub_triangle)

    Q = b1 ** m1 * b2 ** m2 * b3 ** m3 * multinomial(m1, m2, m3) * area_full / area_sub
    return Q


def remove_knot(knot_multiplicities, knot):
    """
    Given a list of knot multiplicities, and a knot, reduce the corresponding
    knot multiplicity by one.
    """

    reduced_knot_multiplicities = [m - (knot == j) for j, m in enumerate(knot_multiplicities)]

    return reduced_knot_multiplicities


def knot_configuration_area(knot_configuration, ps12):
    """
    Given a knot configuration corresponding to a quadratic S-spline basis,
    return the area of the convex hull.
    """

    sub_triangle_configurations = KNOT_CONFIGURATION_TO_SUBTRIANGLES[tuple(knot_configuration)]
    a = 0
    for triangle in sub_triangle_configurations:
        points = ps12[triangle]
        a += SSplines.area(points)
    return a


def polynomial_pieces(triangle, knot_multiplicities, first_call=True):
    """
    Computes the twelve polynomial pieces of the spline with given knot
    multiplicities over the given triangle.
    """

    ps12 = SSplines.ps12_vertices(triangle)
    non_zero_knots, knot_configuration = non_zero_multiplicities(knot_multiplicities)
    polynomials = [0] * 12
    # if the triangle has zero area, return zeros.
    if degenerate(tuple(knot_configuration)):
        return polynomials

    # or, if there are three distinct knots, compute bernstein)
    # polynomial contributions
    elif len(knot_configuration) == 3:
        sub_triangle = ps12[knot_configuration]
        full_triangle = ps12[[0, 1, 2]]
        Q = bernstein_polynomial(full_triangle, sub_triangle, non_zero_knots)
        for face in knot_configuration_to_face(knot_configuration):
            polynomials[face] += Q

    # otherwise, find a non-degenerate triangle in the interior of the knot
    # configuration, and compute contributions recursively by removing knots
    # one by one.
    else:
        interior_triangle_knots = knot_configuration_to_non_degenerate_interior_triangle(knot_configuration)
        interior_triangle_verts = ps12[list(interior_triangle_knots)]
        barycentric_coords = barycentric_coordinates(interior_triangle_verts)

        Q = []
        for knot in interior_triangle_knots:
            reduced_knot_multiplicities = remove_knot(knot_multiplicities, knot)
            reduced_polynomial_pieces = polynomial_pieces(triangle, reduced_knot_multiplicities, first_call=False)
            Q.append(reduced_polynomial_pieces)

        for i in range(len(polynomials)):
            for j in range(len(Q)):
                polynomials[i] += Q[j][i] * barycentric_coords[j]

    # If this is the final return statement in the recurrence, then normalize
    # by area to obtain the S-spline basis.
    if first_call:
        area_knot_config = knot_configuration_area(knot_configuration, ps12)
        area_full_triang = SSplines.area(ps12[[0, 1, 2]])

        for i in range(len(polynomials)):
            polynomials[i] *= (area_knot_config / area_full_triang)
    return polynomials


if __name__ == "__main__":
    triangle = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])

    p = SSplines.sample_triangle(triangle, 30)
    b = SSplines.barycentric_coordinates(triangle, p)
    k = SSplines.determine_sub_triangle(b)
