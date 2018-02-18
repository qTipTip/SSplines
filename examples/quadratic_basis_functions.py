import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from SSplines.helper_functions import sample_triangle
from SSplines.spline_space import SplineSpace

triangle = np.array([
    [0, 0],
    [1, 0],
    [0.5, np.sqrt(3) / 2]
])
d = 2
S = SplineSpace(triangle, d)
B = S.basis()
p = sample_triangle(triangle, 50)

for b in B:
    z = b(p)

    fig = plt.figure()
    axs = Axes3D(fig)
    axs.set_zlim3d(0, 1)
    axs.plot_trisurf(p[:, 0], p[:, 1], z)
    plt.show()
