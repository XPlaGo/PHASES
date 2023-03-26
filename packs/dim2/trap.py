import numpy as np
from solve.solve_2dim import Solve2Dim


FILENAME = "trap2dim.mp4"


def init_pocket(x, y, xoffset, yoffset):
    xm, ym = np.meshgrid(x, y)
    return np.exp(-((xm - xoffset) ** 2) / 2 - ((ym - yoffset) ** 2) / 2, dtype=complex)


def potencial(x, y, xoffset, yoffset):
    xm, ym = np.meshgrid(x, y)
    return ((xm - xoffset) ** 2 + (ym - yoffset) ** 2) * 0.5


print("Using 2-dimensional TRAP template")
solve = Solve2Dim(5.0, 2 ** 7, 5.0, 2 ** 7, 0.1, 360, s=0, ylim=(0, 2))
solve.load(init_pocket(solve.x, solve.y, -1, -1), potencial(solve.x, solve.y, 0, 0))
solve.solve()