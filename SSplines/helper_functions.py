import numpy as np


def barycentric_coordinates(triangle, points, tol=1.0E-15):
    """
    Computes the barycentric coordinates of one or more point(s) with respect to the given triangle.
    :param triangle: vertices of the triangle
    :param points: a set of points
    :param tol: a tolerance for round off error
    :return: a set of barycentric coordinates, ndarray of ndim = 2.
    """

    p = np.atleast_2d(points) # make sure the points are shaped properly
    A = np.concatenate((triangle, np.ones((3, 1))), axis=1).T # append a column of ones
    b = np.concatenate((p, np.ones((len(p), 1))), axis=1) # append a column of ones

    x = np.linalg.solve(A[None, :, :], b) # broadcast A to solve all systems at once

    x[abs(x) < tol] = 0 # remove round off errors around zero
    return x


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


def determine_sub_triangle(triangle, bary_coords):
    """
    Determines the integer(s) k such that the point(s) lies in sub-triangle(k) of the Powell--Sabin 12-split
    of the given triangle.
    :param triangle: vertices of triangle
    :param bary_coords: barycentric coordinates of one or several points
    :return: the integer k for one or several points
    """
    index_lookup_table = {38: 0, 46: 0, 39: 1, 19: 2, 17: 3, 8: 4, 25: 4, 12: 5, 6: 6, 7: 7, 3: 8, 1: 9, 0: 10, 4: 11}

    b = np.atleast_2d(bary_coords)
    b1, b2, b3 = b[:, 0], b[:, 1], b[:, 2]
    s = 32 * (b1 > 0.5) + 16 * (b2 >= 0.5) + 8 * (b3 >= 0.5) + 4 * (b1 > b2) + 2 * (b1 > b3) + (b2 >= b3)
    return np.vectorize(index_lookup_table.get)(s).astype(np.int)


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

    R[:, 3] = [0, 2 * b[1, 2], 2 * b[0, 2], 0, 0, 0, 0, 2 * b[1, 2], 2 * b[0, 2], 0, 0]
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
    R[5, 9:12] = [b[2, 1], 3 * B[1], b[0, 1]]
    R[6, :] = [0, 0.5 * b[0, 2], 1.5 * B[1], 0, 0, 0, 0, 0, 0, 0, 1.5 * B[2], 0.5 * b[0, 1]]
    R[7, :] = [0, 0, 1.5 * B[0], 0.5 * b[1, 2], 0, 0.5 * b[1, 0], 1.5 * B[2], 0, 0, 0, 0, 0]
    R[8, :] = [0, 0, 0, 0, 0, 0, 1.5 * B[1], 0.5 * b[2, 0], 0, 0.5 * b[2, 1], 1.5 * B[0], 0]
    R[9, :] = [0, 1, -g[2], 0, 0, 0, -g[0], 0, 0, 0, 0, -g[1]]

    return R


def r1(B):
    """
    Computes R1 matrices for a series of barycentric coordinates.
    :param B: barycentric coordiantes
    :return: (len(B), 12, 10) array of matrices
    """
    R = np.empty((len(B), 12, 10))
    for i, b in enumerate(B):
        R[i] = r1_single(B)
    return R


def r2(B):
    """
    Computes R2 matrices for a series of barycentric coordinates.
    :param B: barycentric coordiantes
    :return: (len(B), 12, 10) array of matrices
    """
    R = np.empty((len(B), 12, 10))
    for i, b in enumerate(B):
        R[i] = r2_single(B)
    return R
