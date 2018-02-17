import numpy as np

from SSplines.helper_functions import coefficients_linear, coefficients_quadratic, barycentric_coordinates, \
    determine_sub_triangle, evaluate_non_zero_basis_splines, evaluate_non_zero_basis_derivatives, \
    directional_coordinates


class SplineFunction(object):
    """
    Represents a single callable spline function of degree 0/1/2 over the PS12-split of given triangle.
    """

    def __init__(self, triangle, degree, coefficients):
        self.triangle = np.array(triangle)
        self.degree = int(degree)
        self.coefficients = np.array(coefficients)

    def _non_zero_coefficients(self, k):
        """
        Returns the indices of the non-zero coefficients at sub-triangle(s) k.
        :param k: sub-triangle number
        :return : indices
        """
        if self.degree == 0:
            return np.take(self.coefficients, k)
        elif self.degree == 1:
            return np.take(self.coefficients, coefficients_linear(k))
        elif self.degree == 2:
            return np.take(self.coefficients, coefficients_quadratic(k))

    def __call__(self, x):
        """
        Evaluates the spline function at point(s) x.
        :param x: set of points
        :return: f(x)
        """

        b = barycentric_coordinates(self.triangle, x)
        k = determine_sub_triangle(b)
        z = evaluate_non_zero_basis_splines(b=b, d=self.degree, k=k)
        c = self._non_zero_coefficients(k)

        return np.einsum('...i,...i->...', z, c)  # broadcast the dot product to compute all values at once.

    def D(self, x, u, r):
        """
        Evaluates the r'th directional derivative of the function at the point(s) x in direction u.
        :param x: set of points
        :param u: direction
        :param r: order of derivative
        :return: f^(r)(x)
        """

        b = barycentric_coordinates(self.triangle, x)
        k = determine_sub_triangle(b)
        a = directional_coordinates(self.triangle, u)
        z = evaluate_non_zero_basis_derivatives(a=a, b=b, k=k, r=r, d=self.degree)
        c = self._non_zero_coefficients(k)

        return np.einsum('...i,...i->...', z, c)  # broadcast the dot product to compute all values at once.
