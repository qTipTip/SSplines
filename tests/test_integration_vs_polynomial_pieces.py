import numpy as np
import quadpy

from SSplines import ps12_sub_triangles, SplineSpace


def test_integration_second_basis_second_triangle():
    k = 2

    triangle = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])
    b_num = SplineSpace(triangle, 2).basis()[2]
    triangle_2 = ps12_sub_triangles(triangle)[k]
    expected_integral = 0.0117187

    def f(p):
        return b_num(p.T)

    computed_integral = quadpy.triangle.integrate(f, triangle_2, quadpy.triangle.SevenPoint())

    np.testing.assert_almost_equal(computed_integral, expected_integral)


def test_integration_second_basis_model_problem_rhs():
    k = 2

    triangle = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])
    b_num = SplineSpace(triangle, 2).basis()[2]
    triangle_2 = ps12_sub_triangles(triangle)[k]
    expected_integral = 188.32704191224434

    def f(p):
        """
        The exact source term for the model problem for the biharmonic equation.
        """
        x, y = p[:, 0], p[:, 1]
        return 8 * np.pi ** 4 * (
            np.sin(2 * np.pi * x) - 8 * np.sin(4 * np.pi * x) + 25 * np.sin(2 * np.pi * (x - 2 * y)) -
            32 * np.sin(4 * np.pi * (x - y)) - np.sin(2 * np.pi * y) + 8 * (np.sin(4 * np.pi * y) +
                                                                            np.sin(2 * np.pi * (-x + y))) + 25 * np.sin(
                4 * np.pi * x - 2 * np.pi * y))

    def integrand(p):
        return b_num(p.T) * f(p.T)

    computed_integral = quadpy.triangle.integrate(integrand, triangle_2, quadpy.triangle.XiaoGimbutas(20))
    np.testing.assert_almost_equal(computed_integral, expected_integral)


def test_integration_second_basis_model_problem_rhs_subdomain_integration():
    k = 2

    triangle = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])
    b_num = SplineSpace(triangle, 2).basis()[2]
    triangle_2 = ps12_sub_triangles(triangle)[k]
    expected_integral = 188.32704191224434

    def f(p):
        """
        The exact source term for the model problem for the biharmonic equation.
        """
        x, y = p[:, 0], p[:, 1]
        return 8 * np.pi ** 4 * (
            np.sin(2 * np.pi * x) - 8 * np.sin(4 * np.pi * x) + 25 * np.sin(2 * np.pi * (x - 2 * y)) -
            32 * np.sin(4 * np.pi * (x - y)) - np.sin(2 * np.pi * y) + 8 * (np.sin(4 * np.pi * y) +
                                                                            np.sin(2 * np.pi * (-x + y))) + 25 * np.sin(
                4 * np.pi * x - 2 * np.pi * y))

    def integrand(p):
        return b_num(p.T) * f(p.T)

    sub_triangles = ps12_sub_triangles(triangle_2)
    computed_integral = 0
    for sub_triangle in sub_triangles:
        computed_integral += quadpy.triangle.integrate(integrand, sub_triangle, quadpy.triangle.XiaoGimbutas(12))
    np.testing.assert_almost_equal(computed_integral, expected_integral)
