import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.linalg.lapack import dgtsv
from time import time


class TSDEquation:
    def __init__(self, domain, airfoil_coords, nx, ny, n_airfoil_refine, convergence_tolerance=1e-10, max_iterations=1e6):
        """
        :param domain: (x_min, x_max, y_min, y_max)
        :param nx: Total number of cells in x direction
        :param ny: Total number of cells in x direction
        :param n_airfoil: Number of points over the airfoil in the x direction
        :param airfoil_coords: Start and end position of the airfoil in the x direction
        :param convergence_tolerance: Tolerance limit for the residual for the linear solve of the system
        :param max_iterations: maximum allowable number of iterations for the linear solve
        """
        self.tol = convergence_tolerance
        self.max_itr = max_iterations
        self.D = domain
        self.airfoil_pos = airfoil_coords
        self.nx = nx
        self.ny = ny
        self.n_ref = n_airfoil_refine
        self.x = None                                               # Location of grid points in the x-direction
        self.y = None                                               # Location of grid points in the y-direction
        self.phi = None                                             # Computational grid to evaluate phi over

        self.Minf = None                                            # Free stream mach
        self.gamma = 1.4                                            # Specific heat for air
        self.R = 287.058                                            # Gas constant for air [J/(kg K)]
        self.Tinf = 293                                             # Free stream temperature [K]
        self.Pinf = 100000                                          # Free stream pressure [Pa]
        self.Cinf = np.sqrt(self.gamma * self.R * self.Tinf)        # Free stream speed of sound
        self.Uinf = None                                            # Free stream velocity

    def solve(self, mach_number, update_method, return_metrics=True):
        t1 = time()
        self.Minf = mach_number
        self.Uinf = self.Minf * self.Cinf
        self.phi = np.zeros((self.nx, self.ny))
        self.set_grid()

        residual = 1
        residual_history = []                                       # Keeps track of the residual at each iteration
        phi_prev = np.zeros(self.phi.shape)                         # Previous iteration solution for residual calculation
        itr = 0                                                     # Current iteration number

        while residual > self.tol and itr < self.max_itr:
            phi_prev[:, :] = self.phi[:, :]
            self.update_BC()
            for i in range(2, self.nx-1):
                for j in range(1, self.ny-1):
                    coeffs = self.set_coeffs(i, j)

                    if update_method == "Gauss-Seidel":
                        self.gauss_seidel_update(i, j, coeffs)
                    elif update_method == "Line Implicit Gauss-Seidel":
                        self.line_implicit_gauss_seidel_update(i, j, coeffs)
                    else:
                        print("Invalid solver option")
                        exit(1)

            residual = la.norm(self.phi - phi_prev, ord=np.inf)
            residual_history.append(residual)

            if not itr % 100 and itr != 0:
                print(f'Iteration number: {itr}, residual: {residual:.2e}')
            itr += 1
        t2 = time()

        if itr == self.max_itr:
            print(f'\n\nReached maximum number of iterations, final residual: {residual_history[-1]}, CPU time: {t2-t1:.1f}s\n')
        else:
            print(f'\n\nDone solving, number of iterations: {itr}, CPU time: {t2-t1:.1f}s\n')

        if return_metrics:
            return residual_history, itr, t2-t1
        else:
            return

    def line_implicit_gauss_seidel_update(self, i, j, coeffs):
        a, b, c, d, e, g = coeffs
        lower_diag = np.ones(2) * c
        diag = np.ones(3) * a
        upper_diag = np.ones(2) * b
        RHS = np.array([-c * ])
        return 1

    def gauss_seidel_update(self, i, j, coeffs):
        a, b, c, d, e, g = coeffs
        self.phi[i, j] = (
            1 / a * (-c * self.phi[i, j-1] - g * self.phi[i-2, j] - d * self.phi[i-1, j] - e * self.phi[i+1, j] - b * self.phi[i, j+1])
        )

    def set_A_mu(self, i, j):
        A = (
                1 - self.Minf ** 2 - (self.gamma + 1) * self.Minf ** 2 / self.Uinf * (self.phi[i + 1, j] -
                self.phi[i - 1, j]) / (self.x[i + 1] - self.x[i - 1])
        )
        if A > 0:
            mu = 0
        elif A < 0:
            mu = 1
        # elif self.Minf > 1:
        #     mu = 0
        # else:
        #     mu = 1
        return A, mu

    def set_coeffs(self, i, j):
        A, mu = self.set_A_mu(i, j)
        A1, mu1 = self.set_A_mu(i-1, j)
        a = (
                -2 * (1 - mu) * A / ((self.x[i + 1] - self.x[i]) * (self.x[i + 1] - self.x[i - 1])) +
                -2 * (1 - mu) * A / ((self.x[i] - self.x[i - 1]) * (self.x[i + 1] - self.x[i - 1])) +
                2 * mu1 * A1 / ((self.x[i] - self.x[i - 1]) * (self.x[i] - self.x[i - 2])) +
                -2 / ((self.y[j+1] - self.y[j]) * (self.y[j+1] - self.y[j-1])) +
                -2 / ((self.y[j] - self.y[j-1]) * (self.y[j+1] - self.y[j-1]))
        )
        b = 2 / ((self.y[j+1] - self.y[j]) * (self.y[j+1] - self.y[j-1]))
        c = 2 / ((self.y[j] - self.y[j-1]) * (self.y[j+1] - self.y[j-1]))
        d = (
                2 * (1 - mu) * A / ((self.x[i] - self.x[i - 1]) * (self.x[i + 1] - self.x[i - 1])) +
                -2 * mu1 * A1 / ((self.x[i] - self.x[i - 1]) * (self.x[i] - self.x[i - 2])) +
                -2 * mu1 * A1 / ((self.x[i - 1] - self.x[i - 2]) * (self.x[i] - self.x[i - 2]))
        )
        e = 2 * (1 - mu) * A / ((self.x[i + 1] - self.x[i]) * (self.x[i + 1] - self.x[i - 1]))
        g = 2 * mu1 * A1 / ((self.x[i - 1] - self.x[i - 2]) * (self.x[i] - self.x[i - 2]))
        return [a, b, c, d, e, g]

    def update_BC(self):
        le_idx = int((self.nx - self.n_ref) / 2)
        for i in range(1, self.nx-1):
            if le_idx <= i < le_idx + self.n_ref:
                self.phi[i, 0] = self.phi[i, 1] - (self.y[1] - self.y[0]) * self.Uinf * self.dydx(self.x[i])
            else:
                self.phi[i, 0] = self.phi[i, 1]

    @staticmethod
    def dydx(x):
        return 0.08 * (-4 * x + 82)

    def set_grid(self):
        print(f'Building the grid with {self.nx:.0f} points in the x-direction, with {self.n_ref:.0f} over the airfoil, '
              f'and {self.ny:.0f} points in the y direction. Total DoF: {self.nx * self.ny:.0f}')
        airfoil = np.linspace(self.airfoil_pos[0], self.airfoil_pos[1], self.n_ref)  # Grid over the airfoil
        n_upstream = int((self.nx - self.n_ref) / 2)
        n_downstream = int(self.nx - self.n_ref - n_upstream)

        # Build the grid in x
        grid = True
        grid_tol = 0.1
        a = 1.1
        while grid:
            upstream = []
            x1 = self.airfoil_pos[0]
            dx = 1 / self.n_ref
            i = n_upstream
            while i > 0:
                x1 -= dx
                i -= 1
                dx *= a
                upstream.append(x1)
            if (-grid_tol < upstream[-1] < grid_tol):
                grid = False
            elif upstream[-1] > grid_tol:
                a *= 1.01
            elif upstream[-1] < -grid_tol:
                a *= 0.09
            else:
                print('Failed to generate grid')
                exit(1)

        grid = True
        grid_tol = 0.1
        a = 1.1
        while grid:
            downstream = []
            x2 = self.airfoil_pos[1]
            dx = 1 / self.n_ref
            i = 0
            while i < n_downstream:
                x2 += dx
                dx *= a
                i += 1
                downstream.append(x2)
            if (self.D[1]-grid_tol < downstream[-1] < self.D[1] + grid_tol):
                grid = False
            elif downstream[-1] > self.D[-1] + grid_tol:
                a *= 0.09
            elif downstream[-1] < self.D[-1]-grid_tol:
                a *= 1.01
            else:
                print('Failed to generate grid')
        self.x = np.concatenate((np.flip(np.array(upstream)), airfoil, np.array(downstream)))

        # Build the grid in y
        grid = True
        grid_tol = 0.1
        a = 1.1
        while grid:
            y_grid = []
            y1 = 0
            dy = 0.04
            i = 0
            while i < self.ny:
                y_grid.append(y1)
                y1 += dy
                i += 1
                dy *= a
            if (self.D[-1]-grid_tol < y_grid[-1] < self.D[-1] + grid_tol):
                grid = False
            elif y_grid[-1] > self.D[-1] + grid_tol:
                a *= 0.09
            elif y_grid[-1] < self.D[-1]-grid_tol:
                a *= 1.01
            else:
                print('Failed to generate grid')
        self.y = np.array(y_grid)
        print("Done building grid\n")

    def get_pressure_velocity_field(self):
        pressure = np.zeros(self.phi.shape)
        velocity = np.zeros(self.phi.shape)
        for i in range(1, self.nx-1):
            for j in range(1, self.ny-1):
                phi_x = (self.phi[i+1, j] - self.phi[i-1, j]) / (self.x[i+1] - self.x[i-1])
                u = phi_x + self.Uinf
                v = (self.phi[i, j+1] - self.phi[i, j-1]) / (self.y[j+1] - self.y[j-1])
                velocity[i, j] = np.sqrt(u**2 + v**2)
                pressure[i, j] = (
                        self.Pinf * (1 + (self.gamma - 1) / 2 * self.Minf**2 *
                                     (1 - (u**2 + v**2) / self.Uinf**2))**(self.gamma / (self.gamma - 1))
                )
        pressure[-1, :] = self.Pinf * np.ones(self.ny)
        pressure[0, :] = self.Pinf * np.ones(self.ny)
        pressure[:, -1] = self.Pinf * np.ones(self.nx)

        velocity[-1, :] = self.Uinf * np.ones(self.ny)
        velocity[0, :] = self.Uinf * np.ones(self.ny)
        velocity[:, -1] = self.Uinf * np.ones(self.nx)

        le_idx = int((self.nx - self.n_ref) / 2)
        for i in range(1, self.nx-1):
            if le_idx <= i < le_idx + self.n_ref:
                phi_x = (self.phi[i+1, 0] - self.phi[i-1, 0]) / (self.x[i+1] - self.x[i-1])
                u = phi_x + self.Uinf
                v = self.Uinf * self.dydx(self.x[i])

                velocity[i, 0] = np.sqrt(u**2 + v**2)
                pressure[i, 0] = (
                        self.Pinf * (1 + (self.gamma - 1) / 2 * self.Minf ** 2 *
                                     (1 - (u ** 2 + v ** 2) / self.Uinf ** 2)) ** (self.gamma / (self.gamma - 1))
                )
            else:
                phi_x = (self.phi[i+1, 0] - self.phi[i-1, 0]) / (self.x[i+1] - self.x[i-1])
                u = phi_x + self.Uinf
                velocity[i, 0] = np.sqrt(u**2)
                pressure[i, 0] = (
                        self.Pinf * (1 + (self.gamma - 1) / 2 * self.Minf ** 2 *
                                     (1 - (u ** 2 + v ** 2) / self.Uinf ** 2)) ** (self.gamma / (self.gamma - 1))
                )

        return pressure.transpose(), velocity.transpose()

    def get_airfoil_Cp(self):
        le_idx = int((self.nx - self.n_ref) / 2)
        Cp = np.zeros(self.n_ref)
        k = 0
        for i in range(le_idx, le_idx + self.n_ref):
            Cp[k] = -2 / self.Uinf * (self.phi[i+1, 0] - self.phi[i-1, 0]) / (self.x[i+1] - self.x[i-1])
            k += 1
        return Cp


if __name__ == "__main__":
    domain = (0, 50, 0, 50)
    airfoil_coords = (20, 21)
    nx, ny = 60, 20
    airfoil_refinements = 30
    max_itr = 1e6
    tol = 1

    TSD = TSDEquation(
        domain=domain,
        airfoil_coords=airfoil_coords,
    nx=nx,
    ny=ny,
    n_airfoil_refine=airfoil_refinements,
    max_iterations=max_itr,
    convergence_tolerance=tol)

    mach = 0.8
    residual, iterations, cpu_time = TSD.solve(mach, "Gauss-Seidel")

    # Plot the grid
    x = TSD.x
    y = TSD.y
    X, Y = np.meshgrid(x, y)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.pcolor(x, y, np.zeros((len(y) - 1, len(x) - 1)), edgecolor='k', cmap='RdBu', vmin=-1, vmax=1)
    fig1.savefig("Q1_grid_far.png", dpi=300)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.pcolor(x, y, np.zeros((len(y) - 1, len(x) - 1)), edgecolor='k', cmap='RdBu', vmin=-1, vmax=1)
    ax2.set_xlim((16, 25))
    ax2.set_ylim((0, 5))
    fig2.savefig("Q1_grid_close.png", dpi=300)



