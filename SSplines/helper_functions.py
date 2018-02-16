import numpy as np


def barycentric_coordinates(triangle, points, tol=1.0E-15):
    """
    Computes the barycentric coordinates of one or more point(s) with respect to the given triangle.
    :param triangle: vertices of the triangle
    :param points: a set of points
    :param tol: a tolerance for round off error
    :return: a set of barycentric coordinates
    """

    p = np.atleast_2d(points) # make sure the points are shaped properly
    A = np.concatenate((triangle, np.ones(3)), axis=0) # append a row of ones
    b = np.concatenate((points, np.ones(len(points), 1)), axis=1) # append a column of ones

    x = np.linalg.solve(A[None, :, :], b) # broadcast A to solve all systems at once

    x[abs(x) < tol] = 0 # remove round off errors around zero
    return x