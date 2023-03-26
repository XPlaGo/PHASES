import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from tools.console import print_progress
import warnings

warnings.filterwarnings("ignore")


class Solve2Dim:
    def __init__(self, xam, xn, yam, yn, dt, tn, s=0, ylim=(-0.25, 1)):
        self.xam = xam
        self.xn = xn
        self.yam = yam
        self.yn = yn
        self.dt = dt
        self.tn = tn
        self.ylim = ylim
        self.s = s

        self.dx = 2 * xam / xn
        self.x = np.arange(-xam + xam / xn, xam, self.dx)
        self.dy = 2 * yam / yn
        self.y = np.arange(-yam + yam / yn, yam, self.dy)
        self.dkx = np.pi / xam
        self.kx = np.concatenate((np.arange(0, xn / 2), np.arange(-xn / 2, 0))) * self.dkx
        self.dky = np.pi / yam
        self.ky = np.concatenate((np.arange(0, yn / 2), np.arange(-yn / 2, 0))) * self.dky

        self.potential = np.empty((self.xn, self.yn), dtype=complex)
        self.N = np.empty((self.xn, self.yn), dtype=complex)
        self.D = np.empty((self.xn, self.yn), dtype=complex)
        self.u = np.empty((self.xn, self.yn), dtype=complex)
        self.p = np.empty((self.xn, self.yn), dtype=complex)
        self.ur = np.empty((self.tn, self.xn, self.yn), dtype=float)

    def load(self, init_u, potential):
        self.potential = potential
        self.u = init_u
        self.p = np.zeros((self.xn, self.yn), dtype=complex)
        self.D = np.array(
            [np.array([np.exp(-0.5 * (x**2 + y**2) * self.dt * 1j) for y in self.ky], dtype=complex) for x in self.kx])
        self.N = np.array(
            [np.array([np.exp(-0.5 * y * self.dt * 1j) for y in x], dtype=complex) for x in self.potential])

    def solve(self):
        ur = []
        ur.append(self.u)
        for i in range(1, self.tn):
            exp1 = np.exp(-0.5 * self.s * abs(self.u) ** 2 * self.dt * 1j)
            self.p = np.fft.ifftn(self.D * np.fft.fftn(self.N * exp1 * self.u))
            exp2 = np.exp(-0.5 * self.s * abs(self.p) ** 2 * self.dt * 1j)
            self.u = self.N * exp2 * self.p
            ur.append(np.array(abs(self.u) ** 2, dtype=complex))

            print_progress(i + 1, self.tn, "Computing", "Computed")

        self.ur = np.array(ur)
        return self.ur

    def animate(self, save=False, filename=None, fps=60, dpi=400, is_rotate=False):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        x, y = np.meshgrid(self.x, self.y)
        plot = [ax.plot_surface(x, y, self.ur[0], cmap=cm.coolwarm)]
        ax.plot_surface(x, y, self.potential, linewidth=0, antialiased=False, color=[0, 0, 0, 0.05])
        ax.set_zlim(self.ylim[0], self.ylim[1])

        def save_progress(i, frames):
            print_progress(i + 1, frames, "Rendering", f"Saved as \"{filename}\"")

        def update(frame, plot):
            plot[0].remove()
            plot[0] = ax.plot_surface(x, y, self.ur[frame], cmap=cm.coolwarm)
            if is_rotate:
                ax.view_init(elev=10.0, azim=frame)

        anim = FuncAnimation(fig, update, frames=self.tn, blit=False, fargs=[plot])

        if save and filename is not None:
            anim.save(filename, fps=fps, dpi=dpi, progress_callback=save_progress)

        plt.show()
