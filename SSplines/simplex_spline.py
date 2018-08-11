import numpy as np
import sympy as sp

from SSplines.constants import UY, UX
from SSplines.dicts import KNOT_CONFIGURATION_TO_FACE_INDICES
from SSplines.helper_functions import coefficients_linear, coefficients_quadratic, coefficients_cubic, \
    barycentric_coordinates, determine_sub_triangle, evaluate_non_zero_basis_splines, evaluate_non_zero_basis_derivatives, \
    directional_coordinates
from SSplines.symbolic import polynomial_pieces
    
class SimplexSpline(object):
    """
    Represents a simplex spline in terms of knot multiplicities at the vertices of the PS12-split of a given triangle.
    """
    def __init__(self, triangle, knot_multiplicities):
        self.triangle = triangle
        self.knot_multiplicities = knot_multiplicities + (10 - len(knot_multiplicities))* [0]
        self.knot_indices = [i for i in range(len(knot_multiplicities)) if knot_multiplicities[i] != 0]
        self.face_indices = KNOT_CONFIGURATION_TO_FACE_INDICES[tuple(self.knot_indices)]
        self.degree = sum(knot_multiplicities) - 3
        self.polynomial_pieces = polynomial_pieces(triangle, knot_multiplicities)

    def __call__(self, x, exact = False, barycentric = False):
        """
        Evaluates the spline function at point(s) x.
        :param x: set of points
        :return: f(x)
        """
        if barycentric:
            b = x
        else:
            b = barycentric_coordinates(self.triangle, x, exact = exact)

        k = determine_sub_triangle(b)
        
        if exact:
            return np.array([ self.polynomial_pieces[k[i]].subs({'X': x[i][0], 'Y': x[i][1]}) for i in range(len(x))], dtype = object)
        else:
            return np.array([ self.polynomial_pieces[k[i]].subs({'X': x[i][0], 'Y': x[i][1]}) for i in range(len(x))], dtype = np.float)
