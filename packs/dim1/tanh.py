import numpy as np
from solve.solve_1dim import Solve1Dim


FILENAME = "tanh1dim.mp4"


def psi_init(x, n):
    return np.tanh(x * n + 0j, dtype=complex)


def potential(x, height):
    maxx = x.max()
    minx = x.min()
    p = []
    for i in x:
        p.append(
            0 if (minx <= i <= maxx) else height)
    return np.array(p)


print("Using 1-dimensional TANH template")
solve = Solve1Dim(5.0, 2**7, 0.0005, 200, ylim=(-0.25, 2))
solve.load(psi_init(solve.x, 0.4), potential(solve.x, 1E6))
solve.solve()