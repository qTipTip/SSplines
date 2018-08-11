import numpy as np
import sympy as sp
from fractions import Fraction

PS12_BARYCENTRIC_COORDINATES = np.array([
    [Fraction(1,1), Fraction(0,1), Fraction(0,1)],
    [Fraction(0,1), Fraction(1,1), Fraction(0,1)],
    [Fraction(0,1), Fraction(0,1), Fraction(1,1)],
    [Fraction(1,2), Fraction(1,2), Fraction(0,1)],
    [Fraction(0,1), Fraction(1,2), Fraction(1,2)],
    [Fraction(1,2), Fraction(0,1), Fraction(1,2)],
    [Fraction(1,2), Fraction(1,4), Fraction(1,4)],
    [Fraction(1,4), Fraction(1,2), Fraction(1,4)],
    [Fraction(1,4), Fraction(1,4), Fraction(1,2)],
    [Fraction(1,3), Fraction(1,3), Fraction(1,3)]
], dtype = object)

PS12_DUAL_POINTS_INDEX_LINEAR = np.array([
    [ 1],
    [ 2],
    [ 3],
    [ 4],
    [ 5],
    [ 6],
    [ 7],
    [ 8],
    [ 9],
    [10],
], dtype = np.int)

PS12_DUAL_POINTS_INDEX_QUADRATIC = np.array([
    [1, 1],
    [1, 4],
    [4,10],
    [4, 2],
    [2, 2],
    [2, 5],
    [5,10],
    [5, 3],
    [3, 3],
    [3, 6],
    [6,10],
    [6, 1]
], dtype = np.int)

PS12_DUAL_POINTS_INDEX_QUADRATIC_ALTERNATIVE = np.array([
    [1, 1],
    [1, 4],
    [1,10],
    [4, 2],
    [2, 2],
    [2, 5],
    [2,10],
    [5, 3],
    [3, 3],
    [3, 6],
    [3,10],
    [6, 1]
], dtype = np.int)


PS12_DUAL_POINTS_INDEX_CUBIC = np.array([
    [1, 1, 1],
    [1, 1, 4],
    [1, 2, 4],
    [2, 2, 4],
    [2, 2, 2],
    [2, 2, 5],
    [2, 3, 5],
    [3, 3, 5],
    [3, 3, 3],
    [3, 3, 6],
    [1, 3, 6],
    [1, 1, 6],
    [1, 4, 6],
    [2, 4, 5],
    [3, 5, 6],
    [1, 2, 3],
], dtype = np.int)

PS12_DUAL_POINTS_INDEX_CUBIC_ALTERNATIVE = np.array([
    [1, 1, 1],
    [1, 1, 4],
    [1, 2, 4],
    [2, 2, 4],
    [2, 2, 2],
    [2, 2, 5],
    [2, 3, 5],
    [3, 3, 5],
    [3, 3, 3],
    [3, 3, 6],
    [1, 3, 6],
    [1, 1, 6],
    [1, 1, 10],
    [2, 2, 10],
    [3, 3, 10],
    [1, 2,  3],
], dtype = np.int)


PS12_DUAL_POINTS_BARYCENTRIC_COORDINATES_LINEAR = np.array(
    [[PS12_BARYCENTRIC_COORDINATES[PS12_DUAL_POINTS_INDEX_LINEAR[i][j]-1] for j in range(1)] for i in range(10)], dtype=object)

PS12_DUAL_POINTS_BARYCENTRIC_COORDINATES_QUADRATIC = np.array(
    [[PS12_BARYCENTRIC_COORDINATES[PS12_DUAL_POINTS_INDEX_QUADRATIC[i][j]-1] for j in range(2)] for i in range(12)], dtype = object)

PS12_DUAL_POINTS_BARYCENTRIC_COORDINATES_QUADRATIC_ALTERNATIVE = np.array(
    [[PS12_BARYCENTRIC_COORDINATES[PS12_DUAL_POINTS_INDEX_QUADRATIC_ALTERNATIVE[i][j]-1] for j in range(2)] for i in range(12)], dtype = object)
    
PS12_DUAL_POINTS_BARYCENTRIC_COORDINATES_CUBIC = np.array(
    [[PS12_BARYCENTRIC_COORDINATES[PS12_DUAL_POINTS_INDEX_CUBIC[i][j]-1] for j in range(3)] for i in range(16)], dtype = object)

PS12_DUAL_POINTS_BARYCENTRIC_COORDINATES_CUBIC_ALTERNATIVE = np.array(
    [[PS12_BARYCENTRIC_COORDINATES[PS12_DUAL_POINTS_INDEX_CUBIC_ALTERNATIVE[i][j]-1] for j in range(3)] for i in range(16)], dtype = object)    
    
PS12_DUAL_POINTS_BARYCENTRIC_COORDINATES = [
    PS12_DUAL_POINTS_BARYCENTRIC_COORDINATES_LINEAR,
    PS12_DUAL_POINTS_BARYCENTRIC_COORDINATES_QUADRATIC,
    PS12_DUAL_POINTS_BARYCENTRIC_COORDINATES_CUBIC,
]
   
PS12_DOMAIN_POINTS_BARYCENTRIC_COORDINATES_LINEAR = \
    np.average(PS12_DUAL_POINTS_BARYCENTRIC_COORDINATES_LINEAR, axis = 1)

PS12_DOMAIN_POINTS_BARYCENTRIC_COORDINATES_QUADRATIC = \
    np.average(PS12_DUAL_POINTS_BARYCENTRIC_COORDINATES_QUADRATIC, axis = 1)
    
PS12_DOMAIN_POINTS_BARYCENTRIC_COORDINATES_QUADRATIC_ALTERNATIVE = \
    np.average(PS12_DUAL_POINTS_BARYCENTRIC_COORDINATES_QUADRATIC_ALTERNATIVE, axis = 1)
    
PS12_DOMAIN_POINTS_BARYCENTRIC_COORDINATES_CUBIC = \
    np.average(PS12_DUAL_POINTS_BARYCENTRIC_COORDINATES_CUBIC, axis = 1)

PS12_DOMAIN_POINTS_BARYCENTRIC_COORDINATES_CUBIC_ALTERNATIVE = \
    np.average(PS12_DUAL_POINTS_BARYCENTRIC_COORDINATES_CUBIC_ALTERNATIVE, axis = 1)


PS12_DOMAIN_POINTS_BARYCENTRIC_COORDINATES = [
    PS12_DOMAIN_POINTS_BARYCENTRIC_COORDINATES_LINEAR,
    PS12_DOMAIN_POINTS_BARYCENTRIC_COORDINATES_QUADRATIC,
    PS12_DOMAIN_POINTS_BARYCENTRIC_COORDINATES_CUBIC,
]

PS12_SUB_TRIANGLE_VERTICES = np.array([
    [0, 6, 5],
    [0, 3, 6],
    [3, 1, 7],
    [1, 4, 7],
    [4, 2, 8],
    [2, 5, 8],
    [5, 6, 9],
    [3, 9, 6],
    [3, 7, 9],
    [4, 9, 7],
    [4, 8, 9],
    [5, 9, 8]
], dtype=np.int)

UX = np.array([1, 0], dtype = np.int)
UY = np.array([0, 1], dtype = np.int)

DELETED_KNOT_TO_TRIANGLE = {
    (0, 1, 3, 4): (0, 3, 4),  #
    (0, 1, 3, 5): (1, 3, 5),  #
    (0, 2, 3, 5): (2, 3, 5),  #
    (0, 2, 4, 5): (0, 4, 5),  #
    (1, 2, 3, 4): (2, 3, 4),  #
    (1, 2, 4, 5): (1, 4, 5),  #
    (0, 3, 4, 5): (0, 3, 4),  #
    (1, 3, 4, 5): (1, 4, 5),  #
    (2, 3, 4, 5): (2, 3, 5),  #
    (0, 1, 4, 5): (0, 1, 5),  #
    (1, 2, 3, 5): (1, 2, 3),  #
    (0, 2, 3, 4): (0, 2, 3),  #
    (0, 1, 2, 5): (0, 1, 2),  #
    (0, 1, 2, 3): (0, 1, 2),  #
    (0, 1, 2, 4): (0, 1, 2),  #
    (1, 2, 3, 4, 5): (1, 2, 3),  #
    (0, 2, 3, 4, 5): (0, 2, 3),  #
    (0, 1, 3, 4, 5): (0, 1, 4),  #
    (0, 1, 2, 4, 5): (0, 1, 4),  #
    (0, 1, 2, 3, 5): (0, 2, 3),  #
    (0, 1, 2, 3, 4): (0, 1, 4),  #
    (0, 1, 2, 3, 4, 5): (0, 1, 2)  #
}

KNOT_CONFIGURATION_TO_FACE_INDICES = {
    (0, 3, 5): [0, 1],
    (0, 3, 8, 5): [0, 1, 6, 7, 11],
    (0, 2, 3): [0, 1, 5, 6, 7, 11],
    (1, 2, 3): [2, 3, 4, 8, 9, 10],
    (1, 3, 4): [2, 3],
    (2, 4, 5): [4, 5],
    (0, 1, 5): [0, 1, 2, 6, 7, 8],
    (0, 1, 3, 5): [0, 1, 2, 6, 7, 8],
    (1, 2, 5): [3, 4, 5, 9, 10, 11],
    (0, 1, 4): [1, 2, 3, 7, 8, 9],
    (0, 2, 4): [0, 4, 5, 6, 10, 11],
    (0, 2, 4, 3): [0, 1, 4, 5, 6, 7, 8, 9, 10, 11],
    (0, 1, 4, 5): [0, 1, 2, 3, 6, 7, 8, 9, 10, 11],
    (1, 2, 5, 3): [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    (0, 1, 2): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    (3, 6, 9, 7): [7, 8],
    (0, 1, 9): [1, 2, 7, 8],
    (0, 3, 9, 5): [0, 1, 6, 7],
    (3, 4, 5): [6, 7, 8, 9, 10, 11],
    (2, 4, 3, 5): [4, 5, 6, 7, 8, 9, 10, 11],
    (0, 5, 6): [0],
    (0, 3, 6): [1],
    (1, 3, 7): [2],
    (1, 4, 7): [3],
    (2, 4, 8): [4],
    (2, 5, 8): [5],
    (5, 6, 9): [6],
    (3, 6, 9): [7],
    (3, 7, 9): [8],
    (4, 7, 9): [9],
    (4, 8, 9): [10],
    (5, 8, 9): [11],
    (0, 3, 9): [1, 7],
    (1, 3, 9): [2, 8],
    (1, 4, 9): [3, 9],
    (2, 4, 9): [4, 10],
    (2, 5, 9): [5, 11],
    (0, 5, 9): [0, 6],
    (0, 3, 4): [1, 7, 8, 9],
    (1, 3, 5): [2, 6, 7, 8],
    (1, 4, 5): [3, 9, 10, 11],
    (2, 3, 4): [4, 8, 9, 10],
    (2, 3, 5): [5, 6, 7, 11],
    (0, 4, 5): [0, 6, 10, 11],
    (0, 1, 2, 3): range(0, 12),
    (0, 1, 2, 3, 4): range(0, 12),
    (0, 1, 2, 3, 5): range(0, 12),
    (0, 1, 3, 4, 5): range(0, 12),
    (0, 1, 2, 3, 4, 5): range(0, 12),
    (0, 2, 3, 5): [0, 1, 5, 6, 7, 11],
    (1, 2, 3, 4, 5): range(2, 12),
    (0, 1, 2, 5): range(0, 12),
    (0, 1, 3, 9): [1, 2, 7, 8],
    (0, 2, 3, 5, 9): [0, 1, 5, 6, 7, 11],
    (1, 2, 3, 4, 5, 9): range(2, 12),
    (0, 1, 2, 9): range(0, 12),
    (0, 1, 2, 3, 9): range(0, 12),
    (0, 1, 2, 4, 5): range(0, 12),
    (0, 1, 2, 3, 5, 9): range(0, 12),
    (0, 1, 2, 3, 4, 5, 9): range(0, 12),
    (0, 1, 2, 4): range(0, 12),
    (1, 2, 4, 9): [3,4,9,10],
}

WEIGHTS_CONSTANT = {
    0: Fraction(1,8),
    1: Fraction(1,8),
    2: Fraction(1,8),
    3: Fraction(1,8),
    4: Fraction(1,8),
    5: Fraction(1,8),
    6: Fraction(1,24),
    7: Fraction(1,24),
    8: Fraction(1,24),
    9: Fraction(1,24),
    10: Fraction(1,24),
    11: Fraction(1,24),
}

WEIGHTS_LINEAR = {
    0: Fraction(1,4),
    1: Fraction(1,4),
    2: Fraction(1,4),
    3: Fraction(1,3),
    4: Fraction(1,3),
    5: Fraction(1,3),
    6: Fraction(1,3),
    7: Fraction(1,3),
    8: Fraction(1,3),
    9: Fraction(1,4),
}

WEIGHTS_QUADRATIC = {
    0: Fraction(1,4),
    1: Fraction(1,2),
    2: Fraction(3,4),
    3: Fraction(1,2),
    4: Fraction(1,4),
    5: Fraction(1,2),
    6: Fraction(3,4),
    7: Fraction(1,2),
    8: Fraction(1,4),
    9: Fraction(1,2),
    10: Fraction(3,4),
    11: Fraction(1,2),
}

WEIGHTS_QUADRATIC_ALTERNATIVE = WEIGHTS_QUADRATIC

WEIGHTS_CUBIC = {
    0: Fraction(1,4),
    1: Fraction(1,2),
    2: 1,
    3: Fraction(1,2),
    4: Fraction(1,4),
    5: Fraction(1,2),
    6: 1,
    7: Fraction(1,2),
    8: Fraction(1,4),
    9: Fraction(1,2),
    10: 1,
    11: Fraction(1,2),
    12: 1,
    13: 1,
    14: 1,
    15: Fraction(1,4),
}

WEIGHTS_CUBIC_ALTERNATIVE = {
    0: Fraction(1,4),
    1: Fraction(1,2),
    2: 1,
    3: Fraction(1,2),
    4: Fraction(1,4),
    5: Fraction(1,2),
    6: 1,
    7: Fraction(1,2),
    8: Fraction(1,4),
    9: Fraction(1,2),
    10: 1,
    11: Fraction(1,2),
    12: Fraction(3,4),
    13: Fraction(3,4),
    14: Fraction(3,4),
    15: 1,
}

WEIGHTS = [
    WEIGHTS_CONSTANT,
    WEIGHTS_LINEAR,
    WEIGHTS_QUADRATIC,
    WEIGHTS_CUBIC
]

KNOT_MULTIPLICITIES_CONSTANT = {
    0: [1,0,0,0,0,1,1,0,0],
    1: [1,0,0,1,0,0,1,0,0],
    2: [0,1,0,1,0,0,0,1,0],
    3: [0,1,0,0,1,0,0,1,0],
    4: [0,0,1,0,1,0,0,0,1],
    5: [0,0,1,0,0,1,0,0,1],
    6: [0,0,0,0,0,1,1,0,0,1],
    7: [0,0,0,1,0,0,1,0,0,1],
    8: [0,0,0,1,0,0,0,1,0,1],
    9: [0,0,0,0,1,0,0,1,0,1],
    10: [0,0,0,0,1,0,0,0,1,1],
    11: [0,0,0,0,0,1,0,0,1,1],
}

KNOT_MULTIPLICITIES_LINEAR = {
    0: [2, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    1: [0, 2, 0, 1, 1, 0, 0, 0, 0, 0],
    2: [0, 0, 2, 0, 1, 1, 0, 0, 0, 0],
    3: [1, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    4: [0, 1, 1, 0, 1, 0, 0, 0, 0, 1],
    5: [1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    6: [1, 0, 0, 1, 0, 1, 0, 0, 0, 1],
    7: [0, 1, 0, 1, 1, 0, 0, 0, 0, 1],
    8: [0, 0, 1, 0, 1, 1, 0, 0, 0, 1],
    9: [0, 0, 0, 1, 1, 1, 0, 0, 0, 1]
}

KNOT_MULTIPLICITIES_QUADRATIC = {
    0: [3, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    1: [2, 1, 0, 1, 0, 1, 0, 0, 0, 0],
    2: [1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
    3: [1, 2, 0, 1, 1, 0, 0, 0, 0, 0],
    4: [0, 3, 0, 1, 1, 0, 0, 0, 0, 0],
    5: [0, 2, 1, 1, 1, 0, 0, 0, 0, 0],
    6: [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    7: [0, 1, 2, 0, 1, 1, 0, 0, 0, 0],
    8: [0, 0, 3, 0, 1, 1, 0, 0, 0, 0],
    9: [1, 0, 2, 0, 1, 1, 0, 0, 0, 0],
    10: [1, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    11: [2, 0, 1, 1, 0, 1, 0, 0, 0, 0]
}

KNOT_MULTIPLICITIES_QUADRATIC_ALTERNATIVE = {
    0: [3, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    1: [2, 1, 0, 1, 0, 1, 0, 0, 0, 0],
    2: [1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
    3: [1, 2, 0, 1, 1, 0, 0, 0, 0, 0],
    4: [0, 3, 0, 1, 1, 0, 0, 0, 0, 0],
    5: [0, 2, 1, 1, 1, 0, 0, 0, 0, 0],
    6: [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    7: [0, 1, 2, 0, 1, 1, 0, 0, 0, 0],
    8: [0, 0, 3, 0, 1, 1, 0, 0, 0, 0],
    9: [1, 0, 2, 0, 1, 1, 0, 0, 0, 0],
    10: [1, 1, 1, 0, 1, 1, 0, 0, 0, 0],
    11: [2, 0, 1, 1, 0, 1, 0, 0, 0, 0]
}

KNOT_MULTIPLICITIES_CUBIC = {
    0: [4,0,0,1,0,1,0,0,0,0],
    1: [3,1,0,1,0,1,0,0,0,0],
    2: [2,2,1,1,0,0,0,0,0,0],
    3: [1,3,0,1,1,0,0,0,0,0],
    4: [0,4,0,1,1,0,0,0,0,0],
    5: [0,3,1,1,1,0,0,0,0,0],
    6: [1,2,2,0,1,0,0,0,0,0],
    7: [0,1,3,0,1,1,0,0,0,0],
    8: [0,0,4,0,1,1,0,0,0,0],
    9: [1,0,3,0,1,1,0,0,0,0],
    10: [2,1,2,0,0,1,0,0,0,0],
    11: [3,0,1,1,0,1,0,0,0,0],
    12: [2,1,1,1,0,1,0,0,0,0],
    13: [1,2,1,1,1,0,0,0,0,0],
    14: [1,1,2,0,1,1,0,0,0,0],
    15: [1,1,1,1,1,1,0,0,0,0]
}

KNOT_MULTIPLICITIES_CUBIC_ALTERNATIVE = {
    0: [4,0,0,1,0,1,0,0,0,0],
    1: [3,1,0,1,0,1,0,0,0,0],
    2: [2,2,1,1,0,0,0,0,0,0],
    3: [1,3,0,1,1,0,0,0,0,0],
    4: [0,4,0,1,1,0,0,0,0,0],
    5: [0,3,1,1,1,0,0,0,0,0],
    6: [1,2,2,0,1,0,0,0,0,0],
    7: [0,1,3,0,1,1,0,0,0,0],
    8: [0,0,4,0,1,1,0,0,0,0],
    9: [1,0,3,0,1,1,0,0,0,0],
    10: [2,1,2,0,0,1,0,0,0,0],
    11: [3,0,1,1,0,1,0,0,0,0],
    12: [2,1,1,1,0,1,0,0,0,0],
    13: [1,2,1,1,1,0,0,0,0,0],
    14: [1,1,2,0,1,1,0,0,0,0],
    15: [2,2,2,0,0,0,0,0,0,0]
}

KNOT_MULTIPLICITIES = [
    KNOT_MULTIPLICITIES_CONSTANT,
    KNOT_MULTIPLICITIES_LINEAR,
    KNOT_MULTIPLICITIES_QUADRATIC,
    KNOT_MULTIPLICITIES_CUBIC,    
]

QI_INDICES_CUBIC = [
    [1, 1,   1,  1,  1,  1,  1],
    [1, 1,   3,  1, 17, 17,  2],
    [1, 5,   3,  3, 17, 18,  3],
    [5, 5,   3,  5, 18, 18,  4],
    [5, 5,   5,  5,  5,  5,  5],
    [5, 5,   7,  5, 19, 19,  6],
    [5, 9,   7,  7, 19, 20,  7],
    [9, 9,   7,  9, 20, 20,  8],
    [9, 9,   9,  9,  9,  9,  9],
    [9, 9,  11,  9, 21, 21, 10],
    [9, 1,  11, 11, 21, 22, 11],
    [1, 1,  11,  1, 22, 22, 12],
    [1,  3, 11, 17, 22, 23, 13],
    [5,  7,  3, 19, 18, 24, 14],
    [9, 11,  7, 21, 20, 25, 15],
    [1,  5,  9,  3, 11,  7, 16]
]

QI_POINTS_BARYCENTRIC_CUBIC = np.append(PS12_DOMAIN_POINTS_BARYCENTRIC_COORDINATES_CUBIC, [
    [Fraction(3,4), Fraction(1,4), Fraction(0,1)],
    [Fraction(1,4), Fraction(3,4), Fraction(0,1)],
    [Fraction(0,1), Fraction(3,4), Fraction(1,4)],
    [Fraction(0,1), Fraction(1,4), Fraction(3,4)],
    [Fraction(1,4), Fraction(0,1), Fraction(3,4)],
    [Fraction(3,4), Fraction(0,1), Fraction(1,4)],
    [Fraction(1,2), Fraction(1,4), Fraction(1,4)],
    [Fraction(1,4), Fraction(1,2), Fraction(1,4)],
    [Fraction(1,4), Fraction(1,4), Fraction(1,2)],
], axis = 0)

