import numpy as np
from solve.solve_1dim import Solve1Dim


FILENAME = "barrier1dim.mp4"


def psi_init(x, mean, std, k0):
    return np.exp(-((x-mean)**2)/(4*std**2) + 1j*x*k0)/(2*np.pi*std**2)**0.25


def potential(x, width, height, offset):
    maxx = x.max()
    minx = x.min()
    p = []
    for i in x:
        p.append(0 if (minx + width / 2 < i - offset < -width / 2 or width / 2 < i - offset < maxx - width / 2) else height)
    return np.array(p)


print("Using 1-dimensional BARRIER template")
solve = Solve1Dim(50.0, 256, 0.05, 100)
solve.load(psi_init(solve.x, -5, 1, 5), potential(solve.x, 0.5, 1E6, 0))
solve.solve()