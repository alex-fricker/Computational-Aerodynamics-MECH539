import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
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
        self.A = None                                               # Values of Aij at each grid point
        self.mu = None                                              # Switch value at each grid point
        self.coeffs = None                                          # Values of aij, bij, cij, dij, eij, gij

        self.Minf = 0.5                                             # Free stream mach
        self.gamma = 1.4                                            # Specific heat for air
        self.R = 287.058                                            # Gas constant for air [J/(kg K)]
        self.Tinf = 293                                             # Free stream temperature [K]
        self.Pinf = 100000                                          # Free stream pressure [Pa]
        self.Cinf = np.sqrt(self.gamma * self.R * self.Tinf)        # Free stream speed of sound
        self.Uinf = self.Minf * self.Cinf                           # Free stream velocity

    def solve(self, update_method, return_metrics=True):
        t1 = time()
        self.phi = np.zeros((self.nx, self.ny))
        self.set_grid()
        residual = 2
        residual_history = []                                       # Keeps track of the residual at each iteration
        phi_prev = np.zeros(self.phi.shape)                         # Previous iteration solution for residual calculation
        itr = 0                                                     # Current iteration number
        while residual > self.tol and itr < self.max_itr:
            self.update_BC()
            self.set_A_mu_coeffs()
            phi_prev[:, :] = self.phi[:, :]

            for i in range(2, self.nx-1):
                for j in range(1, self.ny-1):
                    if update_method == "Gauss-Seidel":
                        self.gauss_seidel_update(i, j)
                    elif update_method == "Line Implicit Gauss-Seidel":
                        self.line_implicit_gauss_seidel_update(i, j)
                    else:
                        print("Invalid solver option")
                        exit(1)

            residual = la.norm(self.phi - phi_prev, ord=np.inf)
            residual_history.append(residual)
            if not itr % 100 or itr == 0:
                print(f'Iteration number: {itr}, residual: {residual:.2e}')
            itr += 1
        t2 = time()
        if itr == self.max_itr:
            print(f'\n\nReached maximum number of iterations, final residual: {residual_history[-1]}, CPU time: {t2-t1:.2e}s\n')
        else:
            print(f'\n\nDone solving, number of iterations: {itr}, CPU time: {t2-t1:.2e}s\n')

        if return_metrics:
            return residual_history, itr, t2-t1
        else:
            return

    def line_implicit_gauss_seidel_update(self, i, j):
        return None

    def gauss_seidel_update(self, i, j):
        a, b, c, d, e, g = self.coeffs[i, j]
        self.phi[i, j] = (
            1 / a * (-c * self.phi[i, j-1] - g * self.phi[i-2, j] - d * self.phi[i-1, j] - e * self.phi[i+1, j] - b * self.phi[i, j+1])
        )

    def set_A_mu_coeffs(self):  # Compute the matrix of Aij coefficients
        self.A = np.zeros((self.nx, self.ny))
        self.mu = np.zeros((self.nx, self.ny))
        self.coeffs = np.empty((self.nx, self.ny), dtype=list)
        for i in range(1, self.nx - 1):
            for j in range(1, self.ny - 1):
                self.A[i, j] = (
                        1 - self.Minf ** 2 - (self.gamma + 1) * self.Minf ** 2 / self.Uinf *
                        (self.phi[i+1, j] - self.phi[i-1, j]) / (self.x[i+1] - self.x[i-1])
                )

                if self.A[i, j] > 0:
                    self.mu[i, j] = 0
                elif self.A[i, j] < 0:
                    self.mu[i, j] = 1
                elif self.Minf > 1:
                    self.mu[i, j] = 0
                else:
                    self.mu[i, j] = 1

                if i == 1:  # Done need coefficient terms for i = 1
                    continue
                else:
                    self.coeffs[i, j] = self.set_coeffs(i, j)

    def set_coeffs(self, i, j):  # Compute the coefficients aij, bij, cij, dij, eij, gij for Murman-Cole updates
        a = (
            -2 * (1 - self.mu[i, j]) * self.A[i, j] / ((self.x[i+1] - self.x[i]) * (self.x[i+1] - self.x[i-1])) +
            -2 * (1 - self.mu[i, j]) * self.A[i, j] / ((self.x[i] - self.x[i-1]) * (self.x[i+1] - self.x[i-1])) +
            2 * self.mu[i-1, j] * self.A[i-1, j] / ((self.x[i] - self.x[i-1]) * (self.x[i] - self.x[i-2])) +
            -2 / ((self.y[j+1] - self.y[j]) * (self.y[j+1] - self.y[j-1])) +
            -2 / ((self.y[j] - self.y[j-1]) * (self.y[j+1] - self.y[j-1]))
        )
        b = 2 / ((self.y[j+1] - self.y[j]) * (self.y[j+1] - self.y[j-1]))
        c = 2 / ((self.y[j] - self.y[j-1]) * (self.y[j+1] - self.y[j-1]))
        d = (
            2 * (1 - self.mu[i, j]) * self.A[i, j] / ((self.x[i] - self.x[i-1]) * (self.x[i+1] - self.x[i-1])) +
            -2 * self.mu[i-1, j] * self.A[i-1, j] / ((self.x[i] - self.x[i-1]) * (self.x[i] - self.x[i-2])) +
            -2 * self.mu[i-1, j] * self.A[i-1, j] / ((self.x[i-1] - self.x[i-2]) * (self.x[i] - self.x[i-2]))
        )
        e = 2 * (1 - self.mu[i, j]) * self.A[i, j] / ((self.x[i+1] - self.x[i]) * (self.x[i+1] - self.x[i-1]))
        g = 2 * self.mu[i-1, j] * self.A[i-1, j] / ((self.x[i-1] - self.x[i-2]) * (self.x[i] - self.x[i-2]))
        return [a, b, c, d, e, g]

    def update_BC(self):
        le_idx = int((self.nx - self.n_ref) / 2)
        te_idx = le_idx + self.n_ref
        self.phi[1:le_idx, 0] = self.phi[1:le_idx, 1]
        self.phi[te_idx+1: self.nx-1, 0] = self.phi[te_idx+1: self.nx-1, 1]
        self.phi[le_idx:te_idx, 0] = self.phi[le_idx:te_idx, 1] - (self.y[1] - self.y[0]) * self.dydx(self.x[le_idx:te_idx])

    @staticmethod
    def dydx(x):
        return 0.8 * (-4 * x + 82)

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


if __name__ == "__main__":
    domain = (0, 50, 0, 50)
    airfoil_coords = (20, 21)
    nx, ny = 60, 40
    airfoil_refinements = 20
    max_itr = 1e6
    tol = 1e-10

    TSD = TSDEquation(
        domain=domain,
        airfoil_coords=airfoil_coords,
    nx=nx,
    ny=ny,
    n_airfoil_refine=airfoil_refinements,
    max_iterations=max_itr,
    convergence_tolerance=tol)

    TSD.Minf = 0.8
    residual, iterations, cpu_time = TSD.solve("Gauss-Seidel")

    # Plot the grid
    x = TSD.x
    y = TSD.y
    X, Y = np.meshgrid(x, y)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.pcolor(x, y, np.zeros((len(y) - 1, len(x) - 1)), edgecolor='k', cmap='RdBu', vmin=-1, vmax=1)
    fig1.show()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.pcolor(x, y, np.zeros((len(y) - 1, len(x) - 1)), edgecolor='k', cmap='RdBu', vmin=-1, vmax=1)
    ax2.set_xlim((16, 25))
    ax2.set_ylim((0, 5))
    fig2.show()



