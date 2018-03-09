DELETED_KNOT_TO_TRIANGLE = {
    (0, 1, 3, 4): (0, 3, 4),
    (0, 1, 3, 5): (1, 3, 5),
    (0, 2, 3, 5): (2, 3, 5),
    (0, 2, 4, 5): (0, 4, 5),
    (1, 2, 3, 4): (2, 3, 4),
    (1, 2, 4, 5): (1, 4, 5),
    (0, 3, 4, 5): (0, 3, 4),
    (1, 3, 4, 5): (1, 4, 5),
    (2, 3, 4, 5): (2, 3, 5),
    (0, 1, 4, 5): (0, 1, 5),
    (1, 2, 3, 5): (1, 2, 3),
    (0, 2, 3, 4): (0, 2, 3),
    (0, 1, 2, 5): (0, 1, 2),
    (0, 1, 2, 3): (0, 1, 2),
    (0, 1, 2, 4): (0, 1, 2),
    (1, 2, 3, 4, 5): (1, 2, 3),
    (0, 2, 3, 4, 5): (0, 2, 3),
    (0, 1, 3, 4, 5): (0, 1, 4),
    (0, 1, 2, 4, 5): (0, 1, 4),
    (0, 1, 2, 3, 5): (0, 2, 3),
    (0, 1, 2, 3, 4): (0, 1, 4),
    (0, 1, 2, 3, 4, 5): (0, 1, 2)
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
    (0, 1, 2, 3, 4, 5, 9): range(0, 12)
}

KNOT_MULTIPLICITIES_QUADRATIC = {
    0: [3, 0, 0, 1, 0, 1],
    1: [2, 1, 0, 1, 0, 1],
    2: [1, 1, 0, 1, 1, 1],
    3: [1, 2, 0, 1, 1, 0],
    4: [0, 3, 0, 1, 1, 0],
    5: [0, 2, 1, 1, 1, 0],
    6: [0, 1, 1, 1, 1, 1],
    7: [0, 1, 2, 0, 1, 1],
    8: [0, 0, 3, 0, 1, 1],
    9: [1, 0, 2, 0, 1, 1],
    10: [1, 0, 1, 1, 1, 1],
    11: [2, 0, 1, 1, 0, 1]
}

KNOT_CONFIGURATION_TO_SUBTRIANGLES = {
    (0, 3, 5): [[0, 3, 5]],
    (0, 1, 3, 5): [[0, 1, 5]],
    (0, 1, 3, 4, 5): [[0, 4, 5], [0, 1, 4]],
    (0, 1, 3, 4): [[0, 1, 4]],
    (1, 3, 4): [[1, 3, 4]],
    (1, 2, 3, 4): [[1, 2, 3]],
    (1, 2, 3, 4, 5): [[1, 3, 2], [3, 2, 5]],
    (1, 2, 4, 5): [[1, 2, 5]],
    (2, 4, 5): [[2, 4, 5]],
    (0, 2, 4, 5): [[0, 2, 4]],
    (0, 2, 3, 4, 5): [[0, 3, 2], [3, 4, 2]],
    (0, 2, 3, 5): [[0, 3, 2]]
}