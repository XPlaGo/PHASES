import numpy as np
from solve.solve_2dim import Solve2Dim


FILENAME = "diffraction2dim.mp4"


def init_pocket(x, y, xoffset, yoffset, std, kx, ky):
    xm, ym = np.meshgrid(x, y)
    return np.exp((-((xm - xoffset) ** 2) / (4 * std ** 2)), dtype=complex) * (
        np.exp(-1j * xm * kx, dtype=complex)) * (
        np.exp(-((ym - yoffset) ** 2) / (4 * std ** 2), dtype=complex)) * (
        np.exp(-1j * ym * ky, dtype=complex)) / (2 * np.pi * std ** 2) ** 0.25


def potencial(x, y, xoffset, width, height, yoffset1, yoffset2, gap1, gap2):
    p = []
    for j in x:
        line = []
        for i in y:
            if (x.min() < j < yoffset1 - gap1 / 2 or yoffset1 + gap1 / 2 < j < yoffset2 - gap2 / 2 or yoffset2 + gap2 / 2 < j < x.max()) and (-width / 2 < i < width / 2):
                line.append(height)
            else:
                line.append(0)
        p.append(np.array(line))
    return np.array(p)


print("Using 2-dimensional DIFFRACTION template")
solve = Solve2Dim(20.0, 2 ** 8, 20.0, 2 ** 8, 0.05, 360, ylim=(0, 1))
solve.load(init_pocket(solve.x, solve.y, 5, 0, 1, 5, 0), potencial(solve.x, solve.y, 0, 0.2, 1E6, -1, 1, 0.2, 0.2))
solve.solve()
