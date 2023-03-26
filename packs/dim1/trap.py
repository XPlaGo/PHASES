import numpy as np
from solve.solve_1dim import Solve1Dim


FILENAME = "trap1dim.mp4"


def psi_init(x, offset):
    return np.exp(-((x - offset) ** 2) / 2)


def potential(x, offset):
    return 0.5 * (x - offset) ** 2


print("Using 1-dimensional TRAP template")
solve = Solve1Dim(5.0, 256, 0.05, 360, s=0, ylim=(-0.25, 2))
solve.load(psi_init(solve.x, -1), potential(solve.x, 0))
solve.solve()
