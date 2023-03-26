import numpy as np
from solve.solve_1dim import Solve1Dim


FILENAME = "coshdiv1dim.mp4"


def psi_init(x, n):
    return 1 / np.cosh(x * n + 0j, dtype=complex)


def potential(x, width, height, offset):
    maxx = x.max()
    minx = x.min()
    p = []
    for i in x:
        p.append(0 if (minx + width / 2 < i - offset < -width / 2 or width / 2 < i - offset < maxx - width / 2) else height)
    return np.array(p)


print("Using 1-dimensional COSHDIV template")
solve = Solve1Dim(10.0, 2**10, 0.01, 400, s=1, ylim=(-0.25, 2))
solve.load(psi_init(solve.x, 0.4), potential(solve.x, 1, 1, 0))
solve.solve()
