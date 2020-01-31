import pytest
import numpy as np

from SSplines import SplineFunction, SplineSpace


def test_spline_function_add_validation():
    t1 = np.array([
        [0, 0],
        [1, 0],
        [0.5, np.sqrt(3) / 2]
    ])
    t2 = np.array([
        [0, 0],
        [1, 0],
        [0.6, np.sqrt(3) / 2]
    ])

    c1 = np.ones(12)
    c2 = np.ones(12)
    c3 = np.ones(10)
    d1 = 2
    d2 = 2
    d3 = 1

    s1 = SplineFunction(t1, d1, c1)
    s2 = SplineFunction(t2, d2, c2)
    s3 = SplineFunction(t2, d3, c3)

    with pytest.raises(ValueError):
        s1 + s2

    with pytest.raises(ValueError):
        s2 + s3

    try:
        s1 + s1
    except ValueError:
        pytest.fail('An error is raised when the splines are compatible.')
