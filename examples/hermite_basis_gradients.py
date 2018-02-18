import matplotlib.pyplot as plt
import numpy as np

from SSplines.helper_functions import sample_triangle
from SSplines.spline_space import SplineSpace

triangle = np.array([
    [0, 0],
    [1, 0],
    [0, 1]
])
d = 2
S = SplineSpace(triangle, d)
B = S.hermite_basis()
p = sample_triangle(triangle, 50)
for b in B:
    z = b.grad(p)
    plt.quiver(p[:, 0], p[:, 1], z[:, 0], z[:, 1])
    plt.show()
