import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tools.console import print_progress
import sys
import warnings
warnings.filterwarnings("ignore")


class Solve1Dim:
    def __init__(self, xam, xn, dt, tn, s=0, ylim=(-0.25, 1)):
        self.xam = xam
        self.xn = xn
        self.dt = dt
        self.tn = tn
        self.ylim = ylim
        self.s = s

        self.dx = 2 * xam / xn
        self.x = np.arange(-xam + xam / xn, xam, self.dx)
        self.dk = np.pi / xam
        self.k = np.concatenate((np.arange(0, xn / 2), np.arange(-xn / 2, 0))) * self.dk

        self.potential = np.empty(self.xn, dtype=complex)
        self.N = np.empty(self.xn, dtype=complex)
        self.D = np.empty(self.xn, dtype=complex)
        self.u = np.empty(self.xn, dtype=complex)
        self.p = np.empty(self.xn, dtype=complex)
        self.ur = np.empty((self.tn, self.xn), dtype=complex)

    def load(self, init_u, potential):
        self.potential = potential
        self.u = init_u
        self.p = np.zeros(self.xn, dtype=complex)
        self.D = np.exp(-0.5 * (self.k ** 2) * self.dt * 1j)
        self.N = np.exp(-0.5 * potential * self.dt * 1j)

    def solve(self):
        ur = []
        ur.append(self.u)
        for i in range(1, self.tn):
            exp1 = np.exp(-0.5 * self.s * abs(self.u) ** 2 * self.dt * 1j)
            self.p = np.fft.ifft(self.D * np.fft.fft(self.N * exp1 * self.u))
            exp2 = np.exp(-0.5 * self.s * abs(self.p) ** 2 * self.dt * 1j)
            self.u = self.N * exp2 * self.p
            ur.append(np.array(abs(self.u) ** 2, dtype=complex))

            print_progress(i + 1, self.tn, "Computing", "Computed")

        self.ur = np.array(ur)
        return self.ur

    def animate(self, save=False, filename=None, fps=60, dpi=400):
        fig = plt.figure()
        ax = plt.axes(xlim=(-self.xam, self.xam), ylim=self.ylim)
        vx, = ax.plot(self.x, abs(self.potential), lw=1)
        ln, = ax.plot(self.x, self.ur[0], lw=1)

        def save_progress(i, frames):
            print_progress(i + 1, frames, "Rendering", f"Saved as \"{filename}\"")

        def init():
            ln.set_data(self.x, self.ur[0])
            return ln,

        def update(frame):
            ln.set_data(self.x, self.ur[frame])
            return ln,

        anim = FuncAnimation(fig, update, frames=self.tn, interval=10, init_func=init, blit=True)

        if save and filename is not None:
            anim.save(filename, fps=fps, dpi=dpi, progress_callback=save_progress)

        plt.show()
