import numpy as np

from .constants import PS12_BARYCENTRIC_COORDINATES, PS12_SUB_TRIANGLE_VERTICES


def barycentric_coordinates(triangle, points, tol=1.0E-15):
    """
    Computes the barycentric coordinates of one or more point(s) with respect to the given triangle.
    :param triangle: vertices of the triangle
    :param points: a set of points
    :param tol: a tolerance for round off error
    :return: a set of barycentric coordinates, ndarray of ndim = 2.
    """

    p = np.atleast_2d(points)  # make sure the points are shaped properly
    A = np.concatenate((triangle, np.ones((3, 1))), axis=1).T  # append a column of ones
    b = np.concatenate((p, np.ones((len(p), 1))), axis=1)  # append a column of ones

    x = np.linalg.solve(A[None, :, :], b)  # broadcast A to solve all systems at once

    x[abs(x) < tol] = 0  # remove round off errors around zero
    return x


def points_from_barycentric_coordinates(triangle, b):
    """
    Given a triangle and a set of barycentric coordinates, computes the point(s) corresponding to the barycentric coordinates.
    :param triangle: vertices of triangle
    :param b: barycentric coordinates
    :return: points
    """

    b = np.atleast_2d(b)
    t = np.atleast_2d(triangle)
    p = b.dot(t)  # compute a broadcasted dot product
    return p


def directional_coordinates(triangle, direction):
    """
    Computes the directional coordinates of one or more direction vector(s) with respect to the given triangle.
    :param triangle: vertices of the triangle
    :param direction: a set of directions.
    :return: a set of directional coordinates, ndarray of ndim = 2.
    """

    u = np.atleast_2d(direction)  # make sure the directions are shaped properly
    A = np.concatenate((triangle, np.ones((3, 1))), axis=1).T  # append a column of ones
    b = np.concatenate((u, np.zeros((len(u), 1))), axis=1)  # append a column of zeros

    a = np.linalg.solve(A[None, :, :], b)  # broadcast A to solve all systems at once
    return a


def determine_sub_triangle(bary_coords):
    """
    Determines the integer(s) k such that the point(s) lies in sub-triangle(k) of the Powell--Sabin 12-split
    of the given triangle.
    :param bary_coords: barycentric coordinates of one or several points
    :return: the integer k for one or several points
    """
    index_lookup_table = {38: 0, 46: 0, 39: 1, 19: 2, 17: 3, 8: 4, 25: 4, 12: 5, 6: 6, 7: 7, 3: 8, 1: 9, 0: 10, 4: 11}

    b = np.atleast_2d(bary_coords)
    b1, b2, b3 = b[:, 0], b[:, 1], b[:, 2]
    s = 32 * (b1 > 0.5) + 16 * (b2 >= 0.5) + 8 * (b3 >= 0.5) + 4 * (b1 > b2) + 2 * (b1 > b3) + (b2 >= b3)
    return np.vectorize(index_lookup_table.get)(s).astype(np.int)


def ps12_vertices(triangle):
    """
    Returns the set of ten vertices in the Powell--Sabin 12-split.
    :param triangle: vertices of triangle
    :return: set of vertices in 12-split
    """

    return points_from_barycentric_coordinates(triangle, PS12_BARYCENTRIC_COORDINATES)


def ps12_sub_triangles(triangle):
    """
    Returns a set of vertex triples corresponding to the 12 sub-triangles in the ps12.
    :param triangle: triangle
    :return: vertex triples
    """

    return np.take(ps12_vertices(triangle), PS12_SUB_TRIANGLE_VERTICES)


def r1_single(B):
    """
    Computes the linear evaluation matrix for Splines on the Powell-Sabin
    12-split of the triangle delineated by given vertices, evaluated at x.
    :param B: barycentric coordinates of point of evaluation
    :return: (12x10) linear evaluation matrix.
    """

    R = np.zeros((12, 10))
    b = B[:, None] - B[None, :]  # beta
    g = 2 * B - 1

    R[0:2, 0] = g[0]
    R[2:4, 1] = g[1]
    R[4:6, 2] = g[2]

    R[:, 3] = [0, 2 * b[1, 2], 2 * b[0, 2], 0, 0, 0, 0, 2 * b[1, 2], 2 * b[0, 2], 0, 0, 0]
    R[:, 4] = [0, 0, 0, 2 * b[2, 0], 2 * b[1, 0], 0, 0, 0, 0, 2 * b[2, 0], 2 * b[1, 0], 0]
    R[:, 5] = [2 * b[2, 1], 0, 0, 0, 0, 2 * b[0, 1], 2 * b[2, 1], 0, 0, 0, 0, 2 * b[0, 1]]
    R[:, 6] = [4 * B[1], 4 * B[2], 0, 0, 0, 0, 4 * b[0, 2], 4 * b[0, 1], 0, 0, 0, 0]
    R[:, 7] = [0, 0, 4 * B[2], 4 * B[0], 0, 0, 0, 0, 4 * b[1, 0], 4 * b[1, 2], 0, 0]
    R[:, 8] = [0, 0, 0, 0, 4 * B[0], 4 * B[1], 0, 0, 0, 0, 4 * b[2, 1], 4 * b[2, 0]]
    R[:, 9] = [0, 0, 0, 0, 0, 0, -3 * g[0], -3 * g[0], -3 * g[1], -3 * g[1], -3 * g[2], -3 * g[2]]

    return R


def r2_single(B):
    """
    Computes the quadratic evaluation matrix for Splines on the Powell-Sabin
    12-split of the triangle delineated by given vertices, evaluated at x.
    :param B: barycentric coordinates of point of evaluation
    :return: (10x12) quadratic evaluation matrix.
    """

    R = np.zeros((10, 12))
    g = 2 * B - 1  # gamma
    b = B[:, None] - B[None, :]  # beta

    R[0, :] = [g[0], 2 * B[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * B[2]]
    R[1, 3:6] = [2 * B[0], g[1], 2 * B[2]]
    R[2, 7:10] = [2 * B[1], g[2], 2 * B[0]]
    R[3, 1:4] = [b[0, 2], 3 * B[2], b[1, 2]]
    R[4, 5:8] = [b[1, 0], 3 * B[0], b[2, 0]]
    R[5, 9:] = [b[2, 1], 3 * B[1], b[0, 1]]
    R[6, :] = [0, 0.5 * b[0, 2], 1.5 * B[1], 0, 0, 0, 0, 0, 0, 0, 1.5 * B[2], 0.5 * b[0, 1]]
    R[7, :] = [0, 0, 1.5 * B[0], 0.5 * b[1, 2], 0, 0.5 * b[1, 0], 1.5 * B[2], 0, 0, 0, 0, 0]
    R[8, :] = [0, 0, 0, 0, 0, 0, 1.5 * B[1], 0.5 * b[2, 0], 0, 0.5 * b[2, 1], 1.5 * B[0], 0]
    R[9, :] = [0, 0, -g[2], 0, 0, 0, -g[0], 0, 0, 0, -g[1], 0]

    return R


def u1_single(A):
    """
    Computes the linear derivative matrix for Splines on the Powell-Sabin
    12-split of the triangle delineated by given vertices in the direction u.
    :param A: directional coordinates w.r.t some triangle
    :return: (12x10) linear derivative matrix.
    """

    U = np.zeros((12, 10))
    a = A[:, None] - A[None, :]

    U[0:2, 0] = [2 * A[0], 2 * A[0]]
    U[2:4, 1] = [2 * A[1], 2 * A[1]]
    U[4:6, 2] = [2 * A[2], 2 * A[2]]
    U[:, 3] = [0, 2 * a[1, 2], 2 * a[0, 2], 0, 0, 0, 0, 2 * a[1, 2], 2 * a[0, 2], 0, 0, 0]
    U[:, 4] = [0, 0, 0, 2 * a[2, 0], 2 * a[1, 0], 0, 0, 0, 0, 2 * a[2, 0], 2 * a[1, 0], 0]
    U[:, 5] = [2 * a[2, 1], 0, 0, 0, 0, 2 * a[0, 1], 2 * a[2, 1], 0, 0, 0, 0, 2 * a[0, 1]]
    U[:, 6] = [4 * A[1], 4 * A[2], 0, 0, 0, 0, 4 * a[0, 2], 4 * a[0, 1], 0, 0, 0, 0]
    U[:, 7] = [0, 0, 4 * A[2], 4 * A[0], 0, 0, 0, 0, 4 * a[1, 0], 4 * a[1, 2], 0, 0]
    U[:, 8] = [0, 0, 0, 0, 4 * A[0], 4 * A[1], 0, 0, 0, 0, 4 * a[2, 1], 4 * a[2, 0]]
    U[:, 9] = [0, 0, 0, 0, 0, 0, -6 * A[0], -6 * A[0], -6 * A[1], -6 * A[1], -6 * A[2], -6 * A[2]]

    return U


def u2_single(A):
    """
    Computes the quadratic derivative matrix for Splines on the Powell-Sabin
    12-split of the triangle delineated by given vertices in the direction u.
    :param A: directional coordinates wrt to triangle
    :return: (12x10) quadratic derivative matrix.
    """

    U = np.zeros((10, 12))
    a = A[:, None] - A[None, :]

    U[0, :] = [2 * A[0], 2 * A[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * A[2]]
    U[1, 3:6] = [2 * A[0], 2 * A[1], 2 * A[2]]
    U[2, 7:10] = [2 * A[1], 2 * A[2], 2 * A[0]]
    U[3, 1:4] = [a[0, 2], 3 * A[2], a[1, 2]]
    U[4, 5:8] = [a[1, 0], 3 * A[0], a[2, 0]]
    U[5, 9:] = [a[2, 1], 3 * A[1], a[0, 1]]
    U[6, :] = [0, 0.5 * a[0, 2], 1.5 * A[1], 0, 0, 0, 0, 0, 0, 0, 1.5 * A[2], 0.5 * a[0, 1]]
    U[7, :] = [0, 0, 1.5 * A[0], 0.5 * a[1, 2], 0, 0.5 * a[1, 0], 1.5 * A[2], 0, 0, 0, 0, 0]
    U[8, :] = [0, 0, 0, 0, 0, 0, 1.5 * A[1], 0.5 * a[2, 0], 0, 0.5 * a[2, 1], 1.5 * A[0], 0]
    U[9, :] = [0, 0, -2 * A[2], 0, 0, 0, -2 * A[0], 0, 0, 0, -2 * A[1], 0]

    return U


def r1(B):
    """
    Computes R1 matrices for a series of barycentric coordinates.
    :param B: barycentric coordinates
    :return: (len(B), 12, 10) array of matrices
    """
    R = np.empty((len(B), 12, 10))
    for i, b in enumerate(B):
        R[i] = r1_single(b)
    return R


def r2(B):
    """
    Computes R2 matrices for a series of barycentric coordinates.
    :param B: barycentric coordinates
    :return: (len(B), 12, 10) array of matrices
    """
    R = np.empty((len(B), 10, 12))
    for i, b in enumerate(B):
        R[i] = r2_single(b)
    return R


def u1(A):
    """
    Computes U1 matrices for a series of directional coordinates.
    :param A: directional coordinates
    :return: (len(A), 12, 10) array of matrices
    """
    U = np.empty((len(A), 12, 10))
    for i, a in enumerate(A):
        U[i] = u1_single(a)
    return U


def u2(A):
    """
    Computes U2 matrices for a series of directional coordinates.
    :param A: barycentric coordinates
    :return: (len(A), 12, 10) array of matrices
    """
    U = np.empty((len(A), 10, 12))
    for i, a in enumerate(A):
        U[i] = u2_single(a)
    return U


def evaluate_non_zero_basis_splines(d, b, k):
    """
    Evaluates the non-zero basis splines of degree d over a set of point(s) represented by its barycentric coordinates
    over the PS12 split of a triangle.
    :param d: degree of spline
    :param b: barycentric coordinates
    :param k: a list of sub-triangles corresponding to each barycentric coordinate given.
    :return: array, ndarray of non-zero basis splines evaluated at x.
    """

    s = np.ones((len(b), 1))

    matrices = [r1, r2]
    R = [matrices[i](b) for i in range(d)]
    for i in range(d):
        sub = sub_matrix(R[i], i + 1, k)  # extract sub matrices used for evaluation
        s = np.einsum('...ij,...jk->...ik', np.atleast_3d(s), sub)  # compute a broadcast dot product
    return np.squeeze(s)  # squeeze to remove redundant dimension


def evaluate_non_zero_basis_derivatives(d, r, b, a, k):
    """
    Evaluates the r'th directional derivative of the non-zero basis splines of degree d at point x
    over the Powell-Sabin 12 split of the given triangle.
    :param d: spline degree
    :param r: order of derivative
    :param b: barycentric coordinates
    :param a: directional coordinates for which to differentiate
    :param k: sub-triangle(s)
    :return: array of non-zero directional derivatives evaluated at x
    """
    s = np.ones((len(b), 1))
    r_matrices = [r1, r2]
    u_matrices = [u1, u2]
    R = [r_matrices[i](b) for i in range(d)]
    U = [u_matrices[i](a) for i in range(d)]

    for i in range(d - r):
        r_sub = sub_matrix(R[i], i + 1, k)
        s = np.einsum('...ij,...jk->...ik', np.atleast_3d(s), r_sub)  # compute a broadcast dot product

    for j, i in enumerate(range(d - r, d)):
        u_sub = sub_matrix(np.repeat(U[i], len(k), axis=0), i + 1, k)  # in order to extract sub-matrices properly.
        s = (i + 1) * np.einsum('...ij,...jk->...ik', np.atleast_3d(s), u_sub)
    return np.squeeze(s)  # squeeze to remove redundant dimension


def coefficients_quadratic(k):
    """
    Returns the indices of quadratic coefficients corresponding to non-zero S-splines on a set of
    sub-triangles k.
    :param k: array of indices
    :return: array of coefficient indices
    """

    c2 = np.array([
        [0, 1, 2, 9, 10, 11], [0, 1, 2, 3, 10, 11], [1, 2, 3, 4, 5, 6],
        [2, 3, 4, 5, 6, 7], [5, 6, 7, 8, 9, 10], [6, 7, 8, 9, 10, 11],
        [1, 2, 6, 9, 10, 11], [1, 2, 3, 6, 10, 11], [1, 2, 3, 5, 6, 10],
        [2, 3, 5, 6, 7, 10], [2, 5, 6, 7, 9, 10], [2, 6, 7, 9, 10, 11]
    ], dtype=np.int)
    return c2[k]


def coefficients_linear(k):
    """
    Returns the indices of linear coefficients corresponding to non-zero S-splines on a set of
    sub-triangles k
    """
    c1 = np.array([
        [0, 5, 6], [0, 3, 6], [1, 3, 7],
        [1, 4, 7], [2, 4, 8], [2, 5, 8],
        [5, 6, 9], [3, 6, 9], [3, 7, 9],
        [4, 7, 9], [4, 8, 9], [5, 8, 9]
    ], dtype=np.int)

    return c1[k]


def sub_matrix(matrix, d, k):
    """
    Gets the sub-matrix used in evaluation over sub-triangle k for the S-spline matrix or matrices of degree d.
    :param matrix: S-spline matrix(ces) of degree 1 or 2. Note, len(matrix) has to equal(len(k))
    :param d: degree 1 or 2
    :param k: sub triangle(s)
    :return: (1x3) or (3x6) sub-matrix for d = 1, d = 2 respectively.
    """

    c1, c2 = coefficients_linear, coefficients_quadratic
    n = matrix.shape[0]
    if d == 1:
        s = np.zeros((n, 1, 3))
        c = c1(k)
        for i in range(n):
            s[i] = matrix[i, k[i], c[i]]
        return s
    elif d == 2:
        s = np.zeros((n, 3, 6))
        cl = c1(k)
        cq = c2(k)
        for i in range(n):
            s[i] = matrix[np.ix_([i], cl[i], cq[i])]
        return s


def sample_triangle(triangle, d, ret_number=False):
    """
    Returns a set of uniformly spaced points in the triangle. The number of points correspond to the dimension
    of the space of bi-variate polynomials of degree d.
    :param triangle: vertices of triangle
    :param d: sample (1/2)(d+1)(d+2) points
    :param ret_number: whether to return the number of points or not
    :return: sampled points

    TODO / IDEA: Instead of returning the points, return the barycentric coordinates (i/d, j/d, k/d).
    """

    p1, p2, p3 = triangle
    n = int((1 / 2) * (d + 1) * (d + 2))  # total number of domain points
    points = np.zeros((n, 2), dtype=np.float64)

    m = 0
    for i in range(d + 1):
        for j in range(d - i + 1):
            k = (d - i - j)
            p = (i * p1 + j * p2 + k * p3) / d
            points[m] = p
            m += 1

    if ret_number:
        return points, n

    return points


def signed_area(triangle):
    """
    Computes the signed area of a triangle
    :param triangle: vertices of triangle
    :return: the signed area of the triangle
    """

    u = triangle[1, :] - triangle[0, :]
    v = triangle[2, :] - triangle[0, :]

    return 0.5 * np.linalg.det(np.array((u, v)))


def area(triangle):
    """
    Computes the absolute area of a triangle.
    :param triangle: vertices of triangle
    :return: absolute area of triangle
    """

    return np.abs(signed_area(triangle))


def projection_length(u, v):
    """
    Returns the length of the projection of v onto u.
    :param u: vector
    :param v: vector
    :returns: length of projection_u(v)
    """

    return np.einsum('i,i->', u, v) / np.einsum('i,i->', u, u)


def hermite_basis_coefficients(triangle, outward_normal_derivative=False):
    """
    Returns the set of twelve coefficient vectors corresponding to a quadratic Hermite nodal basis
    on the PS12 of given triangle.
    :param triangle: vertices of triangle
    :param outward_normal_derivative: whether to have the normal derivative basis functions use the outward or inward pointing
    unit normal.
    :return: (12 x 12) matrix of coefficients, where each column correspond to a basis function.
    """

    A = np.zeros((12, 12))

    p1, p2, p3 = triangle

    p4 = 0.5 * (p1 + p2)
    p5 = 0.5 * (p2 + p3)
    p6 = 0.5 * (p3 + p1)

    l126 = projection_length(p1 - p2, p2 - p6)
    l134 = projection_length(p1 - p3, p3 - p4)
    l215 = projection_length(p2 - p1, p1 - p5)
    l234 = projection_length(p2 - p3, p3 - p4)
    l326 = projection_length(p3 - p2, p2 - p6)
    l315 = projection_length(p3 - p1, p1 - p5)

    x21, y21 = p2 - p1
    x12, y12 = p1 - p2
    x13, y13 = p1 - p3
    x31, y31 = p3 - p1
    x32, y32 = p3 - p2
    x23, y23 = p2 - p3

    d = signed_area(triangle)

    p12 = 3 * np.linalg.norm(p1 - p2)
    p23 = 3 * np.linalg.norm(p2 - p3)
    p31 = 3 * np.linalg.norm(p3 - p1)

    A[:, 0] = [1, 1, -2 / 3 * l126, 0, 0, 0, 0, 0, 0, 0, -2 / 3 * l134, 1]
    A[:, 1] = [0, 1 / 4 * x21, 1 / 6 * x12 * l126, 0, 0, 0, 0, 0, 0, 0, 1 / 6 * x13 * l134, 1 / 4 * x31]
    A[:, 2] = [0, 1 / 4 * y21, 1 / 6 * y12 * l126, 0, 0, 0, 0, 0, 0, 0, 1 / 6 * y13 * l134, 1 / 4 * y31]
    A[:, 3] = [0, 0, d / p12, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    A[:, 4] = [0, 0, -2 / 3 * l215, 1, 1, 1, -2 / 3 * l234, 0, 0, 0, 0, 0]
    A[:, 5] = [0, 0, 1 / 6 * x21 * l215, 1 / 4 * x12, 0, 1 / 4 * x32, 1 / 6 * x23 * l234, 0, 0, 0, 0, 0]
    A[:, 6] = [0, 0, 1 / 6 * y21 * l215, 1 / 4 * y12, 0, 1 / 4 * y32, 1 / 6 * y23 * l234, 0, 0, 0, 0, 0]
    A[:, 7] = [0, 0, 0, 0, 0, 0, d / p23, 0, 0, 0, 0, 0]
    A[:, 8] = [0, 0, 0, 0, 0, 0, -2 / 3 * l326, 1, 1, 1, -2 / 3 * l315, 0]
    A[:, 9] = [0, 0, 0, 0, 0, 0, 1 / 6 * x32 * l326, 1 / 4 * x23, 0, 1 / 4 * x13, 1 / 6 * x31 * l315, 0]
    A[:, 10] = [0, 0, 0, 0, 0, 0, 1 / 6 * y32 * l326, 1 / 4 * y23, 0, 1 / 4 * y13, 1 / 6 * y31 * l315, 0]
    A[:, 11] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, d / p31, 0]

    if outward_normal_derivative:
        A[:, [3, 7, 11]] *= -1

    return A


def gaussian_quadrature_data(order):
    """
    Computes weights and barycentric coordinates for Gaussian quadrature of the given order.
    :param order: order of integration
    :return: weights and barycentric coordinates
    """

    if order == 1:

        b = np.array([
            [1 / 3, 1 / 3, 1 / 3]
        ])
        w = np.array([1])

    elif order == 2:

        b = np.array([
            [0.5, 0.5, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0.5]
        ])
        w = np.array([1 / 3, 1 / 3, 1 / 3])

    elif order == 3:

        b = np.array([
            [1 / 3, 1 / 3, 1 / 3],
            [2 / 15, 2 / 15, 11 / 15],
            [11 / 15, 2 / 15, 2 / 15],
            [2 / 15, 11 / 15, 2 / 15]
        ])
        w = np.array([-27 / 48, 25 / 48, 25 / 28, 25 / 48])

    else:
        raise NotImplementedError('Higher order weights are not implemented yet')

    return b, w


def gaussian_quadrature(triangle, func, b, w):
    """
    Approximates the integral of f over triangle numerically using a quadrature rule with the given points and weights.
    :param func: function R^2 -> R to integrate
    :param triangle: vertices of triangle
    :param b: barycentric coordinates of quadrature points
    :param w: weights of quadrature points
    :return: numerical integral of f
    """

    p = points_from_barycentric_coordinates(triangle, b)
    a = abs(signed_area(triangle))
    f = func(p)

    return a * (np.dot(w, f))
