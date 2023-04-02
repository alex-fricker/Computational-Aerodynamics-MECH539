import numpy as np
import matplotlib.pyplot as plt

class quasi1DEuler:
    """
    Author: Alexander Fricker - 260773670
    Class: Mech 539 - Computational Aerodynamis (Winter 2023)
    """
    def __init__(self, n, epsilon, convergence_tolerance=1e-14, max_iterations=1e5):
        self.x0 = 0                                 # Right boundary
        self.x1 = 1                                 # Left boundary
        self.x = None                               # Computational grid
        self.W = None                               # Solution vector
        self.W1 = None                              # Solution vector at next timestep
        self.dx = 0                                 # Grid spacing
        self.tol = convergence_tolerance            # Tolerance for the density residual
        self.max_itr = max_iterations               # Maximum number of iterations
        self.CFL = 1

        self.h = 0.15                               # Max bump height
        self.t1 = 0.8                               # Location of max bump height
        self.t2 = 3                                 # Width of bump in channel

        self.g = 1.4                                # Specific heat ratio
        self.Tt = 531.2                             # Total inlet temperature [R]
        self.Pt = 2117                              # Total inlet pressure [lb/ft^2]
        self.R = 1716                               # Gas constant [ft lb / (slug R)

        self.n = n                                  # Number of grid points
        self.e = epsilon                            # Epsilon
        self.Pe = self.Pt                           # Total pressure ratio at the exit

    def solve(self, exit_pressure_ratio):  # Solve the flow for a given exit pressure ratio
        self.Pe = exit_pressure_ratio * self.Pt
        convergence = []
        current_residual = np.zeros(self.n-2)
        itr = 0
        while current_residual > self.tol and itr < self.max_itr:
            for i in range(1, self.n-1):
                rho, u, p = self.primitive(i)
                c = np.sqrt(self.g * p / rho)
                dt = self.CFL * self.dx / max(np.abs(np.array([u, u + c, u - c])))
                V = self.S(self.x[i]) * self.dx
                Ri = self.residual(i)
                current_residual[i-1] = Ri
                self.W1[i, :] = self.W[i, :] - dt / V * Ri
            convergence.append(np.linalg.norm(current_residual, ord=np.inf))
            self.update_BC()
        return convergence

    def update_BC(self):  # Updates the boundary conditions
        return None

    def residual(self, i):  # Computes the residual at the ith grid point
        rho, u, p = self.primitive(i)
        Ri = self.flux(i) * self.S(self.x[i] + 0.5 * self.dx) - self.flux(i - 1) * self.S(self.x[i] - 0.5 * self.dx) - self.Q(i, p)
        return Ri

    def Q(self, i, p):  # Computes the momentum source term
        return np.array([
            0,
            p * (-self.t1 * self.t2 * self.h * self.x[i]**(self.t1 - 1) *
            np.sin(np.pi * self.x[i]**self.t1)**(self.t2 - 1) * np.cos(np.pi * self.x[i]**self.t1)) /
            (1 - self.h * (np.sin(np.pi * self.x[i]**self.t1))**self.t2),
            0
        ])

    def flux(self, i):  # Computes the numerical flux at the F_{i+0.5} cell
        rho, u, p = self.primitive(i)
        f = np.array([
            rho * u,
            rho * u**2 + p,
            (self.W[i, 2] + p) * u
        ])

        rho1, u1, p1 = self.primitive(i + 1)
        f1 = np.array([
            rho1 * u1,
            rho1 * u1**2 + p1,
            (self.W[i + 1, 2] + p1) * u1
        ])
        c = np.sqrt(self.g * p / rho)
        L = max(u, u + c, u - c)
        return 0.5 * (f + f1) - 0.5 * self.e * L * (self.W[i+1, :] - self.W[i, :])

    def primitive(self, i):  # Computes the primitive variables at a specific point (rho, u , P)
        return np.array([
            self.W[i, 0],
            self.W[i, 1] / self.W[i, 0],
            (self.g - 1) * self.W[i, 0] * (self.W[i, 2] / self.W[i, 0] - 0.5 * (self.W[i, 1] / self.W[i, 0])**2)
        ])

    def S(self, xi):  # Computes area of the nozzle
        return 1 - self.h * (np.sin(np.pi * xi**self.t1))**self.t2

    def init_grid(self):  # Initializes the grid and sets initial conditions
        self.x = np.linspace(self.x0, self.x1, self.n)
        self.W = np.zeros((self.n, 3))
        self.W1 = np.zeros((self.n, 3))
        self.dx = self.x[1] - self.x[0]

        # Initialize state vector as stationary flow
        self.W[:-1, :] = (self.Pt, 0, self.Pt)

        # Initialize exit conditions
        Me = (((self.Pe / self.Pt)**((1 - self.g) / self.g) - 1) * 2 / (self.g - 1))**0.5
        Te = self.Tt * (1 + (self.g - 1) / 2 * Me**2)**-1
        rho_e = self.Pe / (self.R * Te)
        ue = Me * np.sqrt(self.g * self.R * Te)
        self.W[-1, :] = (rho_e,
                         rho_e * ue,
                         self.Pe / (self.g - 1) * + 0.5 * rho_e * ue**2)
