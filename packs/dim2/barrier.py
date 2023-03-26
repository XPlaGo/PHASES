import numpy as np
from solve.solve_2dim import Solve2Dim


FILENAME = "barrier2dim.mp4"


def init_pocket(x, y, xoffset, yoffset, std, kx, ky):
    xm, ym = np.meshgrid(x, y)
    return np.exp((-((xm - xoffset) ** 2) / (4 * std ** 2)), dtype=complex) * (
           np.exp(-1j * xm * kx, dtype=complex)) * (
           np.exp(-((ym - yoffset) ** 2) / (4 * std ** 2), dtype=complex)) * (
           np.exp(-1j * ym * ky, dtype=complex)) / (2 * np.pi * std ** 2) ** 0.25


def potencial(x, y, xoffset, width, height):
    p = []
    for _ in x:
        line = []
        for i in y:
            if y.min() + width < i < -width / 2 or y.max() - width > i > width / 2:
                line.append(0)
            else:
                line.append(height)
        p.append(np.array(line))
    return np.array(p)


print("Using 2-dimensional BARRIER template")
solve = Solve2Dim(10.0, 2 ** 7, 10.0, 2 ** 7, 0.01, 360, ylim=(0, 1))
solve.load(init_pocket(solve.x, solve.y, 5, 0, 1, 5, 0), potencial(solve.x, solve.y, 0, 0.2, 20))
solve.solve()
