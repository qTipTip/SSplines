import numpy as np
import sympy as sp
from fractions import Fraction

import matplotlib as matplotlib
import matplotlib.pyplot as plt

from .constants import PS12_BARYCENTRIC_COORDINATES, PS12_SUB_TRIANGLE_VERTICES, \
    PS12_DOMAIN_POINTS_BARYCENTRIC_COORDINATES_QUADRATIC, \
    PS12_DOMAIN_POINTS_BARYCENTRIC_COORDINATES_CUBIC

from SSplines.dicts import KNOT_CONFIGURATION_TO_FACE_INDICES

def barycentric_coordinates_multiple_triangles(triangles, point, tol=1.0E-15):
    """
    Computes the barycentric coordinates of a single point with respect to a set of triangles.
    :param triangles: set of vertices of triangle
    :param point: a single point
    :param tol: a tolerance for round off error
    :return: a set of barycentric coordinates of the point w.r.t each triangle
    """

    raise NotImplementedError('Not implemented yet')

    p = np.atleast_2d(point)
    N = triangles.shape[0]
    A = np.concatenate((triangles, np.ones((N, 3, 1))), axis=2)
    b = np.concatenate((p, np.ones((1, 1))), axis=1)

    x = np.linalg.solve(A, b[None, :])


def barycentric_coordinates(triangle, points, tol=1.0E-15, exact = False):
    """
    Computes the barycentric coordinates of one or more point(s) with respect to the given triangle.
    :param triangle: vertices of the triangle
    :param points: a set of points
    :param tol: a tolerance for round off error
    :return: a set of barycentric coordinates, ndarray of ndim = 2.
    """
    p = np.atleast_2d(points)  # make sure the points are shaped properly
    if exact:
        A = np.concatenate((triangle, np.array(3*[[Fraction(1,1)]], dtype = object)), axis=1).T
        b = np.concatenate((       p, np.array(p.shape[0]*[[Fraction(1,1)]], dtype = object)), axis=1).T

        A = sp.Matrix(A)
        b = sp.Matrix(b)

        x = np.array(A.solve(b)).T   
    else:
        A = np.concatenate((triangle.astype(float), np.ones((3, 1))), axis=1).T  # append a column of ones
        b = np.concatenate((p.astype(float), np.ones((len(p), 1))), axis=1)  # append a column of ones
        x = np.linalg.solve(A[None, :, :], b)  # broadcast A to solve all systems at once

        x[abs(x) < tol] = 0  # remove round off errors around zero
        
    return x


def points_from_barycentric_coordinates(triangle, b):
    """
    Given a triangle(s) and a set of barycentric coordinates, computes the point(s) corresponding to the barycentric coordinates.
    :param triangle: vertices of triangle, or multiple
    :param b: barycentric coordinates
    :return: points
    """

    b = np.atleast_2d(b)
    t = np.atleast_2d(triangle)
    p = np.dot(b, t)  # compute a broadcasted dot product
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
    Determines the integer(s) k such that the point(s) lies in sub-triangle(k) of the Powell-Sabin 12-split
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
    #return points_from_barycentric_coordinates(triangle, PS12_BARYCENTRIC_COORDINATES.astype('float'))


def ps12_sub_triangles(triangle):
    """
    Returns a set of vertex triples corresponding to the 12 sub-triangles in the ps12.
    :param triangle: triangle
    :return: vertex triples
    """

    return np.take(ps12_vertices(triangle), PS12_SUB_TRIANGLE_VERTICES, axis=0)

def alternative_basis_transform_quadratic(exact = False):
    if exact:
        T = np.identity(12, dtype = object)
        T[2:11:4,2:11:4] = np.array([[Fraction(1,2), Fraction(1,2),             0], \
                                     [            0, Fraction(1,2), Fraction(1,2)], \
                                     [Fraction(1,2),             0, Fraction(1,2)]], dtype = object)
    else:
        T = np.identity(12, dtype = float)
        T[2:11:4,2:11:4] = np.array([[0.5, 0.5,   0], \
                                     [0.0, 0.0, 0.5], \
                                     [0.5, 0.0, 0.5]], dtype = float)

    return T

def alternative_basis_transform_cubic(exact = False)
    if exact:
        T = np.identity(16, dtype = object)
        T[12:16,12:16] = np.array([[Fraction(3,4),            0,            0,Fraction(1,4)], \
                                   [            0,Fraction(3,4),            0,Fraction(1,4)], \
                                   [            0,            0,Fraction(3,4),Fraction(1,4)], \
                                   [            0,            0,            0,            1]], dtype = object)
    else:
        T = np.identity(16, dtype = float)
        T[12:16,12:16] = np.array([[0.75,   0,   0,0.25], \
                                   [   0,0.75,   0,0.25], \
                                   [   0,   0,0.75,0.25], \
                                   [   0,   0,   0,   1]], dtype = float)

    return T

def r1_single(B, exact = False):
    """
    Computes the linear evaluation matrix for Splines on the Powell-Sabin
    12-split of the triangle delineated by given vertices, evaluated at x.
    :param B: barycentric coordinates of point of evaluation
    :return: (12x10) linear evaluation matrix.
    """

    if exact:
        R = np.zeros((12, 10), dtype = object)
    else:
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


def r2_single(B, exact = False, alternative_basis = False):
    """
    Computes the quadratic evaluation matrix for Splines on the Powell-Sabin
    12-split of the triangle delineated by given vertices, evaluated at x.
    :param B: barycentric coordinates of point of evaluation
    :return: (10x12) quadratic evaluation matrix.
    """

    if exact:
        R = np.zeros((10, 12), dtype = object)
        f = Fraction(1,2)
    else:
        R = np.zeros((10, 12))
        f = 0.5
    
    g = 2 * B - 1  # gamma
    b = B[:, None] - B[None, :]  # beta

    R[0,  :  ] = [g[0], 2 * B[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * B[2]]
    R[1, 3:6 ] = [2 * B[0], g[1], 2 * B[2]]
    R[2, 7:10] = [2 * B[1], g[2], 2 * B[0]]
    R[3, 1:4 ] = [b[0, 2], 3 * B[2], b[1, 2]]
    R[4, 5:8 ] = [b[1, 0], 3 * B[0], b[2, 0]]
    R[5, 9:  ] = [b[2, 1], 3 * B[1], b[0, 1]]
    R[6,  :  ] = [0, f * b[0, 2], 3 * f * B[1], 0, 0, 0, 0, 0, 0, 0, 3 * f * B[2], f * b[0, 1]]
    R[7,  :  ] = [0, 0, 3*f * B[0], f * b[1, 2], 0, f * b[1, 0], 3*f * B[2], 0, 0, 0, 0, 0]
    R[8,  :  ] = [0, 0, 0, 0, 0, 0, 3 * f * B[1], f * b[2, 0], 0, f * b[2, 1], 3 * f * B[0], 0]
    R[9,  :  ] = [0, 0, -g[2], 0, 0, 0, -g[0], 0, 0, 0, -g[1], 0]
    
    if alternative_basis:
        T = alternative_basis_transform_quadratic(exact = exact)
        R = np.dot(R, T)
    
    return R
    
def r3_single(B, exact = False, alternative_basis = False):
    """
    Computes the cubic evaluation matrix for splines on the Powell-Sabin
    12-split of the triangle delineated by given vertices, evaluated at x.
    :param B: barycentric coordinates of point of evaluation
    :return: (12x16) cubic evaluation matrix.
    """

    if exact:
        R = np.zeros((12, 16), dtype = object)
        f = Fraction(1,3)
    else:
        R = np.zeros((12, 16))
        f = 1.0/3
    
    g = 2 * B - 1  # gamma
    b = B[:, None] - B[None, :]  # beta
    s = B[:, None] + B[None, :]  # sigma
    
    R[0,:] = [g[0],2*B[1],0,0,0,0,0,0,0,0,0,2*B[2],0,0,0,0]
    R[1,:] = [0,b[0,2],B[1],0,0,0,0,0,0,0,0,0,2*B[2],0,0,0]
    R[2,:] = [0,0,f*s[0,1],0,0,0,f*B[2],0,0,0,f*B[2],0,2*f*B[0],2*f*B[1],0,f*B[2]]
    R[3,:] = [0,0,B[0],b[1,2],0,0,0,0,0,0,0,0,0,2*B[2],0,0]
    R[4,:] = [0,0,0,2*B[0],g[1],2*B[2],0,0,0,0,0,0,0,0,0,0]
    R[5,:] = [0,0,0,0,0,b[1,0],B[2],0,0,0,0,0,0,2*B[0],0,0]    
    R[6,:] = [0,0,f*B[0],0,0,0,f*s[1,2],0,0,0,f*B[0],0,0,2*f*B[1],2*f*B[2],f*B[0]]
    R[7,:] = [0,0,0,0,0,0,B[1],b[2,0],0,0,0,0,0,0,2*B[0],0]
    R[8,:] = [0,0,0,0,0,0,0,2*B[1],g[2],2*B[0],0,0,0,0,0,0]
    R[9,:] = [0,0,0,0,0,0,0,0,0,b[2,1],B[0],0,0,0,2*B[1],0]
    R[10,:] = [0,0,f*B[1],0,0,0,f*B[1],0,0,0,f*s[0,2],0,2*f*B[0],0,2*f*B[2],f*B[1]]
    R[11,:] = [0,0,0,0,0,0,0,0,0,0,B[2],b[0,1],2*B[1],0,0,0]
    
    if alternative_basis:
        T = alternative_basis_transform_cubic(exact = exact)
        R = np.dot(R, T)

    return R
    
def u1_single(A, exact = False):
    """
    Computes the linear derivative matrix for Splines on the Powell-Sabin
    12-split of the triangle delineated by given vertices in the direction u.
    :param A: directional coordinates w.r.t some triangle
    :return: (12x10) linear derivative matrix.
    """

    if exact:
        U = np.zeros((12, 10), dtype = object)
    else:
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


def u2_single(A, exact = False, alternative_basis = False):
    """
    Computes the quadratic derivative matrix for Splines on the Powell-Sabin
    12-split of the triangle delineated by given vertices in the direction u.
    :param A: directional coordinates wrt to triangle
    :return: (10x12) quadratic derivative matrix.
    """
    if exact:
        U = np.zeros((10, 12), dtype = object)
        f = Fraction(1,2)
    else:
        U = np.zeros((10, 12))
        f = 0.5

    a = A[:, None] - A[None, :]

    U[0, :] = [2 * A[0], 2 * A[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * A[2]]
    U[1, 3:6] = [2 * A[0], 2 * A[1], 2 * A[2]]
    U[2, 7:10] = [2 * A[1], 2 * A[2], 2 * A[0]]
    U[3, 1:4] = [a[0, 2], 3 * A[2], a[1, 2]]
    U[4, 5:8] = [a[1, 0], 3 * A[0], a[2, 0]]
    U[5, 9:] = [a[2, 1], 3 * A[1], a[0, 1]]
    U[6, :] = [0, f * a[0, 2], 3*f * A[1], 0, 0, 0, 0, 0, 0, 0, 3*f * A[2], f * a[0, 1]]
    U[7, :] = [0, 0, 3*f * A[0], f * a[1, 2], 0, f * a[1, 0], 3*f * A[2], 0, 0, 0, 0, 0]
    U[8, :] = [0, 0, 0, 0, 0, 0, 3*f * A[1], f * a[2, 0], 0, f * a[2, 1], 3*f * A[0], 0]
    U[9, :] = [0, 0, -2 * A[2], 0, 0, 0, -2 * A[0], 0, 0, 0, -2 * A[1], 0]

    if alternative_basis:
        T = alternative_basis_transform_quadratic(exact = exact)
        U = np.dot(U, T)
    
    return U

def u3_single(A, exact = False, alternative_basis = False):
    """
    Computes the cubic derivative matrix for Splines on the Powell-Sabin
    12-split of the triangle delineated by given vertices in the direction u.
    :param A: directional coordinates wrt to triangle
    :return: (12x16) cubic derivative matrix.
    """

    if exact:
        U = np.zeros((12, 16), dtype = object)
        f = Fraction(1,3)
    else:
        U = np.zeros((12, 16))
        f = 1.0/3
    
    a = A[:, None] - A[None, :]  # alpha
    t = A[:, None] + A[None, :]  # tau
    
    U[0,:] = [2*A[0],2*A[1],0,0,0,0,0,0,0,0,0,2*A[2],0,0,0,0]
    U[1,:] = [0,a[0,2],A[1],0,0,0,0,0,0,0,0,0,2*A[2],0,0,0]
    U[2,:] = [0,0,f*t[0,1],0,0,0,f*A[2],0,0,0,f*A[2],0,2*f*A[0],2*f*A[1],0,f*A[2]]
    U[3,:] = [0,0,A[0],a[1,2],0,0,0,0,0,0,0,0,0,2*A[2],0,0]
    U[4,:] = [0,0,0,2*A[0],2*A[1],2*A[2],0,0,0,0,0,0,0,0,0,0]
    U[5,:] = [0,0,0,0,0,a[1,0],A[2],0,0,0,0,0,0,2*A[0],0,0]    
    U[6,:] = [0,0,f*A[0],0,0,0,f*t[1,2],0,0,0,f*A[0],0,0,2*f*A[1],2*f*A[2],f*A[0]]
    U[7,:] = [0,0,0,0,0,0,A[1],a[2,0],0,0,0,0,0,0,2*A[0],0]
    U[8,:] = [0,0,0,0,0,0,0,2*A[1],2*A[2],2*A[0],0,0,0,0,0,0]
    U[9,:] = [0,0,0,0,0,0,0,0,0,a[2,1],A[0],0,0,0,2*A[1],0]
    U[10,:] = [0,0,f*A[1],0,0,0,f*A[1],0,0,0,f*t[0,2],0,2*f*A[0],0,2*f*A[2],f*A[1]]
    U[11,:] = [0,0,0,0,0,0,0,0,0,0,A[2],a[0,1],2*A[1],0,0,0]

    if alternative_basis:
        T = alternative_basis_transform_cubic(exact = exact)
        U = np.dot(U, T)
    
    return U

def r1(B, exact = False, alternative_basis = False):
    """
    Computes R1 matrices for a series of barycentric coordinates.
    :param B: barycentric coordinates
    :return: (len(B), 12, 10) array of matrices
    """
    if exact:
        R = np.empty((len(B), 12, 10), dtype = object)
    else:
        R = np.empty((len(B), 12, 10))

    for i, b in enumerate(B):
        R[i] = r1_single(b, exact = exact)

    return R


def r2(B, exact = False, alternative_basis = False):
    """
    Computes R2 matrices for a series of barycentric coordinates.
    :param B: barycentric coordinates
    :return: (len(B), 10, 12) array of matrices
    """
    if exact:
        R = np.empty((len(B), 10, 12), dtype = object)
    else:
        R = np.empty((len(B), 10, 12))

    for i, b in enumerate(B):
        R[i] = r2_single(b, exact = exact, alternative_basis = alternative_basis)
    
    return R

def r3(B, exact = False, alternative_basis = False):
    """
    Computes R2 matrices for a series of barycentric coordinates.
    :param B: barycentric coordinates
    :return: (len(B), 12, 16) array of matrices
    """
    if exact:
        R = np.empty((len(B), 12, 16), dtype = object)
    else:
        R = np.empty((len(B), 12, 16))

    for i, b in enumerate(B):
        R[i] = r3_single(b, exact = exact, alternative_basis = alternative_basis)

    return R

def u1(A, exact = False):
    """
    Computes U1 matrices for a series of directional coordinates.
    :param A: directional coordinates
    :return: (len(A), 12, 10) array of matrices
    """
    if exact:
        U = np.empty((len(A), 12, 10), dtype = object)
    else:
        U = np.empty((len(A), 12, 10))

    for i, a in enumerate(A):
        U[i] = u1_single(a, exact = exact)
    
    return U


def u2(A, exact = False, alternative_basis = False):
    """
    Computes U2 matrices for a series of directional coordinates.
    :param A: barycentric coordinates
    :return: (len(A), 10, 12) array of matrices
    """
    if exact:
        U = np.empty((len(A), 10, 12), dtype = object)
    else:
        U = np.empty((len(A), 10, 12))

    for i, a in enumerate(A):
        U[i] = u2_single(a, exact = exact, alternative_basis = alternative_basis)
    
    return U

def u3(A, exact = False, alternative_basis = False):
    """
    Computes U3 matrices for a series of directional coordinates.
    :param A: barycentric coordinates
    :return: (len(A), 12, 16) array of matrices
    """
    if exact:
        U = np.empty((len(A), 12, 16), dtype = object)
    else:
        U = np.empty((len(A), 12, 16))

    for i, a in enumerate(A):
        U[i] = u3_single(a, exact = exact, alternative_basis = alternative_basis)

    return U
    
def evaluate_non_zero_basis_splines(d, b, k, exact = False, alternative_basis = False):
    """
    Evaluates the non-zero basis splines of degree d over a set of point(s) represented by its barycentric coordinates
    over the PS12 split of a triangle.
    :param d: degree of spline
    :param b: barycentric coordinates
    :param k: a list of sub-triangles corresponding to each barycentric coordinate given.
    :return: array, ndarray of non-zero basis splines evaluated at x.
    """

    if exact:
        s = np.ones((len(b), 1), dtype = object)
    else:
        s = np.ones((len(b), 1))
    
    matrices = [r1, r2, r3]
    #If an alternative basis is chosen, we modify the recursion matrix of highest degree.
    R = [matrices[i](b, exact = exact, alternative_basis = (alternative_basis and i == d-1)) for i in range(d)]
    
    for i in range(d):
        # extract sub matrices used for evaluation
        # If an alternative basis is chosen, we modify the recursion matrix of highest degree.
        sub = sub_matrix(R[i], i + 1, k, exact = exact, alternative_basis = (alternative_basis and i == d-1) )
        
        if exact:
            s = np.dot(s,sub)
        else:
            s = np.einsum('...ij,...jk->...ik', np.atleast_3d(s), sub)  # compute a broadcast dot product
    
    return np.squeeze(s)  # squeeze to remove redundant dimension

# TODO: Implement evaluation of derivatives in exact arithmetic.
def evaluate_non_zero_basis_derivatives(d, r, b, a, k, exact = False, alternative_basis = False):
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
    r_matrices = [r1, r2, r3]
    u_matrices = [u1, u2] # TODO: add u3
    R = [r_matrices[i](b) for i in range(d)]
    U = [u_matrices[i](a) for i in range(d)]

    for i in range(d - r):
        r_sub = sub_matrix(R[i], i + 1, k, exact = exact, alternative_basis = alternative_basis)
        s = np.einsum('...ij,...jk->...ik', np.atleast_3d(s), r_sub)  # compute a broadcast dot product

    for j, i in enumerate(range(d - r, d)):
        # in order to extract sub-matrices properly.
        u_sub = sub_matrix(np.repeat(U[i], len(k), axis=0), i + 1, k, exact = exact, alternative_basis = alternative_basis)  
        s = (i + 1) * np.einsum('...ij,...jk->...ik', np.atleast_3d(s), u_sub)
    return np.squeeze(s)  # squeeze to remove redundant dimension

def coefficients_cubic(k):
    """
    Returns the indices of the coefficients corresponding to non-zero cubic S-splines on a set of
    sub-triangles k.
    :param k: array of indices
    :return: array of coefficient indices
    """
    
    c3 = np.array([
        [0,1,2,6, 9,10,11,12,13,14,15], [0,1,2,3, 6,10,11,12,13,14,15], [1,2,3,4, 5, 6,10,12,13,14,15],
        [2,3,4,5, 6, 7,10,12,13,14,15], [2,5,6,7, 8, 9,10,12,13,14,15], [2,6,7,8, 9,10,11,12,13,14,15],
        [1,2,6,9,10,11,12,13,14,15,-1], [1,2,3,6,10,11,12,13,14,15,-1], [1,2,3,5, 6,10,12,13,14,15,-1],
        [2,3,5,6, 7,10,12,13,14,15,-1], [2,5,6,7, 9,10,12,13,14,15,-1], [2,6,7,9,10,11,12,13,14,15,-1]
    ], dtype=np.int)
    return c3[k]

    
def coefficients_quadratic_alternative(k):
    """
    Returns the indices of quadratic coefficients corresponding to non-zero S-splines on a set of
    sub-triangles k.
    :param k: array of indices
    :return: array of coefficient indices
    """

    c2 = np.array([
        [0, 1, 2, 6,  9, 10, 11], [0, 1, 2, 3,  6, 10, 11], [1, 2, 3, 4,  5,  6, 10],
        [2, 3, 4, 5,  6,  7, 10], [2, 5, 6, 7,  8,  9, 10], [2, 6, 7, 8,  9, 10, 11],
        [1, 2, 6, 9, 10, 11, -1], [1, 2, 3, 6, 10, 11, -1], [1, 2, 3, 5,  6, 10, -1],
        [2, 3, 5, 6,  7, 10, -1], [2, 5, 6, 7,  9, 10, -1], [2, 6, 7, 9, 10, 11, -1]
    ], dtype=np.int)
    
    return c2[k]

    
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


def sub_matrix(matrix, d, k, exact = False, alternative_basis = False):
    """
    Gets the sub-matrix used in evaluation over sub-triangle k for the S-spline matrix or matrices of degree d.
    :param matrix: S-spline matrix(ces) of degree 1, 2, or 3. Note, len(matrix) has to equal(len(k))
    :param d: degree 1, 2 or 3
    :param k: sub triangle(s)
    :return: (1x3), (3x6), (3x11) sub-matrix for d = 1, d = 2, d = 3 respectively.
    """
    
    c1, c2, c3 = coefficients_linear, coefficients_quadratic, coefficients_cubic
    n = matrix.shape[0]
    if d == 1:
        if exact:
            s = np.zeros((n, 1, 3), dtype = object)
        else:
            s = np.zeros((n, 1, 3))
            
        c = c1(k)
        for i in range(n):
            s[i] = matrix[i, k[i], c[i]]
        return s
    elif d == 2:
        if alternative_basis:
            m = 7
            c2 = coefficients_quadratic_alternative
        else:
            m = 6
            
        if exact:
            s = np.zeros((n, 3, m), dtype = object)
        else:
            s = np.zeros((n, 3, m))
        
        cl = c1(k)
        cq = c2(k)
        for i in range(n):
            s[i] = matrix[np.ix_([i], cl[i], cq[i])]
        return s
    elif d == 3:
        if exact:
            s = np.zeros((n, 6, 11), dtype = object)
        else:
            s = np.zeros((n, 6, 11))
        
        cq = c2(k)
        cc = c3(k)
        for i in range(n):
            M = np.pad(matrix, pad_width = ((0,0),(0,1),(0,0)), mode = 'constant', constant_values = 0)
            M = M[np.ix_([i], cq[i], cc[i])]
            
            s[i] = M
            
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


def signed_area(triangle, exact = False):
    """
    Computes the signed area of a triangle
    :param triangle: vertices of triangle
    :return: the signed area of the triangle
    """

    if triangle.ndim == 2:  # if only one triangle is supplied, pretend the matrix is 3dim
        triangle = np.expand_dims(triangle, axis=0)

    u = triangle[:, 1, :] - triangle[:, 0, :]
    v = triangle[:, 2, :] - triangle[:, 0, :]
    A = np.array((u.T, v.T)).T

    return [Fraction(1/2) * sp.Matrix(A0).det() for A0 in A]
    #if exact:
    #    print("TODO")
    #    return [Fraction(1/2) * sp.Matrix(A0).det() for A0 in A]

    #else:        
    #    return 0.5 * np.linalg.det(A.astype(float))


def area(triangle, exact = False):
    """
    Computes the absolute area of a triangle.
    :param triangle: vertices of triangle
    :return: absolute area of triangle
    """

    return np.abs(signed_area(triangle, exact = exact))


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

    elif order == 4:
        # http://www.cs.rpi.edu/~flaherje/pdf/fea6.pdf
        W1 = 0.109951743655322
        W2 = 0.223381589678011

        b11 = 0.816847572980459
        b12 = 0.091576213509771

        b21 = 0.108103018168070
        b22 = 0.445948490915965

        b = np.array([
            [b11, b12, b12],
            [b12, b11, b12],
            [b12, b12, b11],
            [b21, b22, b22],
            [b22, b21, b22],
            [b22, b22, b21]
        ])

        w = np.array([
            W1, W1, W1, W2, W2, W2
        ])

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


def gaussian_quadrature_ps12(triangle, func, b, w):
    """
    Approximates the integral of f over the PS12-split of triangle, using a quadrature rule with given points and weights.
    :param triangle: vertices of triangle
    :param func: function R^2 -> R to integrate
    :param b: barycentric coordinates of quadrature points
    :param w: weights of quadrature points
    :return: numerical integral of f
    """

    sub_triangle = ps12_sub_triangles(triangle)

    i = 0

    for t in sub_triangle:
        v = gaussian_quadrature(t, func, b, w)

        i += v
    return i


def domain_point_quadrature(triangle, func, degree=2):
    """
    Approximates the integral of func over the PS12-split of a triangle, using the domain points
    as integration points. Uniformly weighted.
    :param triangle: vertices of triangle
    :param func: func to integrate
    :return:
    """
    p = domain_points(triangle, degree)
    w = np.ones(12) / 12
    a = abs(signed_area(triangle))

    f = func(p)

    return a * (np.dot(w, f))


def domain_point_quadrature_ps12(triangle, func, degree=2):
    sub_triangle = ps12_sub_triangles(triangle)

    i = 0

    for t in sub_triangle:
        v = domain_point_quadrature(t, func, degree)
        i += v
    return i

def edge_quadrature_data(order):
    # http://www.karlin.mff.cuni.cz/~dolejsi/Vyuka/FEM-implement.pdf
    if order == 2:
        b = np.array([
            [0.21132486540519, 0.78867513459481],
            [0.78867513459481, 0.21132486540519]
        ])
        w = np.array([0.5, 0.5])

    elif order == 3:
        b = np.array([
            [0.1120166537926, 0.887229833462074],
            [0.5, 0.5],
            [0.887229833462074, 0.1120166537926]
        ])
        w = np.array([0.277777777777778, 0.444444444444444, 0.277777777777778])

    elif order == 4:
        b = np.array([
            [0.06943184420297, 1 - 0.06943184420297],
            [0.33000947820757, 1 - 0.33000947820757],
            [0.66999052179243, 1 - 0.66999052179243],
            [0.93056815579703, 1 - 0.93056815579703]
        ])
        w = np.array([0.17392742256873, 0.32607257743127, 0.32607257743127, 0.17392742256873])

    return b, w


def edge_quadrature(edge, func, b, w):
    """
    Approximates the integral of func over edge using a quadrature rule with specified
    points and weights.
    :param edge: two vertices delineating an edge
    :param func: the integrand
    :param b: barycentric coordinates of quadrature points with respect to end points
    :param w: weights of quadrature points
    :return:
    """

    p = np.dot(b, edge)  # points in edge
    f = func(p)
    length = np.linalg.norm(edge[0, :] - edge[1, :])

    return length * np.dot(f, w)


def edge_quadrature_ps12(edge, func, b, w):
    """
    Approximates the integral of func over edge using a quadrature rule with specified
    points and weights, but split over each of the two triangles that lies on the edge.
    :param edge: two vertices delineating an edge
    :param func: the integrand
    :param b: barycentric coordinates of quadrature points with respect to end points
    :param w: weights of quadrature points
    :return:
    """

    mp = (edge[0, :] + edge[1, :]) / 2

    edge_one = np.array([edge[0, :], mp])
    edge_two = np.array([mp, edge[1, :]])

    return edge_quadrature(edge_one, func, b, w) + edge_quadrature(edge_two, func, b, w)


def domain_points(triangle, degree):
    """
    Returns the domain points for a degree 1/2/3 spline.
    :param triangle: vertices of triangle
    :param degree: 1/2/3
    :return: set of 10/12/16 domain points
    """

    if degree == 1:
        return ps12_vertices(triangle)
    elif degree == 2:
        return points_from_barycentric_coordinates(triangle=triangle,
                                                   b=PS12_DOMAIN_POINTS_BARYCENTRIC_COORDINATES_QUADRATIC)
    elif degree == 3:
        return points_from_barycentric_coordinates(triangle=triangle,
                                                   b=PS12_DOMAIN_POINTS_BARYCENTRIC_COORDINATES_CUBIC)
    else:
        raise NotImplementedError('Domain points not defined for degrees other than 1 and 2')


def simplex_spline_graphic_small(mm, scale = 2, filename = False, is_visible = True):
    """
    Show the graphical notation of a simplex spline with knot multiplicity sequence mm.
    """
    v1, v2, v3 = np.array([(int(-12),int(0)), (int(12),int(0)), (int(0),int(21))], dtype = np.float)
    Lv = ps12_vertices([v1, v2, v3])
    v1, v2, v3, v4, v5, v6, v7, v8, v9, v10 = Lv
    
    # Make a figure
    fig = plt.figure(figsize = [2/scale, np.sqrt(3)/scale], dpi = 300)
    ax = fig.add_subplot(1,1,1, frameon = True)

    # Edges
    x = [v4[0], v2[0], v3[0], v1[0], v4[0]]
    y = [v4[1], v2[1], v3[1], v1[1], v4[1]]
    line = matplotlib.lines.Line2D(x, y, lw = int(2), color='black', zorder = 2)
    ax.add_line(line)
    line = matplotlib.lines.Line2D([int(min(x) - 5), int(max(x) + 5)], \
                                   [int(min(y) - 6), int(max(y) + 5)], \
                                   lw = int(2), color='white', zorder = 0)
    
    ax.add_line(line)
    
    # Vertices
    for i in range(len(mm)):
        if mm[i] != 0:
            ax.add_artist(plt.Circle((Lv[i][0], Lv[i][1], 1), facecolor = "black", zorder = 10))
            t = matplotlib.text.Text(Lv[i][0], Lv[i][1], str(mm[i]), zorder = 20, fontsize = 24/scale, \
                                     ha='center', va='center', family='sans-serif', color='white')

            ax.add_artist(t)
    
    for i in range(6):
        for j in range(i,6):
            x = [Lv[i][0], Lv[j][0]]
            y = [Lv[i][1], Lv[j][1]]
            line = matplotlib.lines.Line2D(x, y, lw=int(1), color='black', zorder = 2)
            ax.add_line(line)
            
    # Color the faces.
    F = [[1,6,7],[1,4,7],[2,4,8],[2,5,8],[3,5,9],[3,6,9],[6,7,10],[4,7,10],[4,8,10],[5,8,10],[5,9,10],[6,9,10]]
    mm0 = tuple([i for i in range(len(mm)) if mm[i] != 0])
    for f in KNOT_CONFIGURATION_TO_FACE_INDICES[mm0]:
        ax.add_artist(matplotlib.patches.Polygon([Lv[i-1] for i in F[f]], color='#8888ff'))

    plt.axis('equal')
    plt.axis('off')

    if filename:
        #bbox = matplotlib.transforms.Bbox([[0.155,0.15], [.87,.74]])
        bbox = matplotlib.transforms.Bbox([[0.145,0.16], [.9,.74]])
        plt.savefig(filename, dpi = 300, bbox_inches = bbox)
    
    if is_visible:
        plt.show()
        
    plt.close('all')
    
