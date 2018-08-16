import numpy as np

from SSplines.dicts import KNOT_CONFIGURATION_TO_FACE_INDICES
from SSplines.helper_functions import barycentric_coordinates, determine_sub_triangle, \
    points_from_barycentric_coordinates, simplex_spline_graphic_small
from SSplines.symbolic import polynomial_pieces


class SimplexSpline(object):
    """
    Represents a simplex spline in terms of knot multiplicities at the vertices of the PS12-split of a given triangle.
    """

    def __init__(self, triangle, knot_multiplicities):
        self.triangle = triangle
        self.knot_multiplicities = knot_multiplicities + (10 - len(knot_multiplicities)) * [0]
        self.knot_indices = [i for i in range(len(knot_multiplicities)) if knot_multiplicities[i] != 0]
        self.face_indices = KNOT_CONFIGURATION_TO_FACE_INDICES[tuple(self.knot_indices)]
        self.degree = sum(knot_multiplicities) - 3
        self.polynomial_pieces = polynomial_pieces(triangle, knot_multiplicities)

    def __call__(self, y, exact=False, barycentric=False):
        """
        Evaluates the spline function at point(s) y, either with Cartesian or with barycentric coordinates.
        :param y: set of points
        :return: f(y)
        """
        if barycentric:
            b = y
            x = points_from_barycentric_coordinates(self.triangle, b)
        else:
            x = np.atleast_2d(y)
            b = barycentric_coordinates(self.triangle, x, exact=exact)

        k = determine_sub_triangle(b)

        if exact:
            return np.array([self.polynomial_pieces[k[i]].subs({'X': x[i][0], 'Y': x[i][1]}) for i in range(len(x))],
                            dtype=object)
        else:
            return np.array([self.polynomial_pieces[k[i]].subs({'X': x[i][0], 'Y': x[i][1]}) for i in range(len(x))],
                            dtype=np.float)

    def display(self):
        """
        Displays the SimplexSpline using graphical notation.
        """
        simplex_spline_graphic_small(self.knot_multiplicities)
