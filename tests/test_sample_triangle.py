import numpy as np

from SSplines.helper_functions import sample_triangle, barycentric_coordinates


def test_sample_triangle():
    """
    We sample a triangle, and assert that the barycentric coordinates of the resulting points are all
    positive, hence the points lie inside the triangle.
    """
    triangle = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])
    d = 30

    points = sample_triangle(triangle, d)
    bary_coords = barycentric_coordinates(triangle, points)

    assert np.all(bary_coords[bary_coords > 0])
