import numpy as np
import matplotlib.pyplot as plt
import time


class quasi1DEuler:
    """
    Author: Alexander Fricker - 260773670
    Class: Mech 539 - Computational Aerodynamics (Winter 2023)
    """
    def __init__(self, n, epsilon, CFL=0.8, convergence_tolerance=1e-12, max_iterations=1e5):
        self.x0 = 0                                 # Right boundary
        self.x1 = 1                                 # Left boundary
        self.x = None                               # Computational grid
        self.W = None                               # Solution vector
        self.W1 = None                              # Solution vector at next timestep
        self.dx = 0                                 # Grid spacing
        self.tol = convergence_tolerance            # Tolerance for the density residual
        self.max_itr = max_iterations               # Maximum number of iterations
        self.CFL = CFL                              # CFL number for determining time steps

        self.h = 0.15                               # Max bump height
        self.t1 = 0.8                               # Location of max bump height
        self.t2 = 3                                 # Width of bump in channel

        self.g = 1.4                                # Specific heat ratio
        self.Tt = 295.1                             # Total inlet temperature [K]
        self.Pt = 101362.5                          # Total inlet pressure [pa]
        self.R_gas = 287.05                         # Gas constant [J / (kg K)]
        self.cv = self.R_gas / (self.g - 1)         # Specific heat at constant volume
        self.mach_in = None                         # Inlet mach number, specified if supersonic inlet

        self.n = n                                  # Number of grid points
        self.e = epsilon                            # Epsilon
        self.Pe = None                              # Total pressure ratio at the exit

    def solve(self, exit_pressure_ratio, mach_in=0.2):  # Solve the flow for a given exit pressure ratio
        self.Pe = exit_pressure_ratio * self.Pt
        self.mach_in = mach_in
        self.init_grid()
        convergence = []  # Convergence history of the density residual
        error = 1
        itr = 0
        t1 = time.time()
        while error > self.tol and itr < self.max_itr:
            timesteps = []  # Storage for the timesteps in the computation for computing the residual
            for i in range(1, self.n-1):
                rho, u, p = self.primitive(i)
                c = np.sqrt(self.g * p / rho)
                dt = self.CFL * self.dx / (u + c)
                timesteps.append(dt)
                V = self.S(self.x[i]) * self.dx
                Ri = self.R(i)
                self.W1[i, :] = self.W[i, :] - dt / V * Ri

            self.update_BC(mach_in)
            iteration_residual = self.assemble_residual(timesteps)
            error = np.linalg.norm(iteration_residual[:, 0], ord=np.inf)
            convergence.append(error)
            if not (itr % 1000):
                print(f"Iteration: {itr} - Density residual: {convergence[-1]}")
            self.W[:, :] = self.W1[:, :]
            itr += 1
        t2 = time.time()

        print(f"Done solving:\n\tNumber of iterations: {itr}\n\tFinal density residual: {convergence[-1]}\n\tSolver CPU time: {t2-t1:.2f}s")
        return np.array(convergence)

    def assemble_residual(self, dt):
        residual = []
        for i in range(1, self.n-1):
            Ri = (self.W1[i, :] - self.W[i, :]) / dt[i-1] + dt[i-1] / (self.S(self.x[i]) * self.dx) * self.R(i)
            residual.append(Ri)
        return np.array(residual)

    def R(self, i):
        # Compute numerical F_{i+0.5}
        rho_avg_p, u_avg_p, p_avg_p = 0.5 * (self.primitive(i) + self.primitive(i+1))
        c_avg_p = np.sqrt(self.g * p_avg_p / rho_avg_p)
        L1 = max(u_avg_p, u_avg_p + c_avg_p, u_avg_p - c_avg_p)
        F_plus = 0.5 * (self.flux(i) + self.flux(i+1)) - 0.5 * self.e * L1 * (self.W[i+1, :] - self.W[i, :])

        # Compute numerical flux F_{i-0.5}
        rho_avg_m, u_avg_m, p_avg_m = 0.5 * (self.primitive(i) + self.primitive(i-1))
        c_avg_m = np.sqrt(self.g * p_avg_m / rho_avg_m)
        L2 = max(u_avg_m, u_avg_m + c_avg_m, u_avg_m - c_avg_m)
        F_minus = 0.5 * (self.flux(i-1) + self.flux(i)) - 0.5 * self.e * L2 * (self.W[i, :] - self.W[i-1, :])

        Ri = F_plus * self.S(self.x[i] + 0.5 * self.dx) - F_minus * self.S(self.x[i] - 0.5 * self.dx) - self.Q(i)
        return Ri

    def flux(self, i):
        rho, u, p = self.primitive(i)
        return np.array([
            rho * u,
            rho * u**2 + p,
            (self.W[i, 2] + p) * u
        ])

    def primitive(self, i):  # Computes the primitive variables at a specific point (rho, u , P)
        if i == self.n - 1:
            return np.array([
                self.W[i, 0],
                self.W[i, 1] / self.W[i, 0],
                self.Pe
            ])
        else:
            return np.array([
                self.W[i, 0],
                self.W[i, 1] / self.W[i, 0],
                (self.g - 1) * (self.W[i, 2] - 0.5 * self.W[i, 1]**2 / self.W[i, 0])
            ])

    def update_BC(self, mach_in):  # Updates the boundary conditions
        # Updating inlet boundary conditions
        if mach_in < 1:  # Case for the subsonic inlet condition
            rho0, u0, p0 = self.primitive(0)
            rho1, u1, p1 = self.primitive(1)
            c0 = np.sqrt(self.g * p0 / rho0)
            c1 = np.sqrt(self.g * p1 / rho1)

            a_star2 = 2 * self.g * (self.g - 1) / (self.g + 1) * self.cv * self.Tt
            dpdu = (
                self.Pt * self.g / (self.g - 1) * (1 - (self.g - 1) / (self.g + 1) * u0**2 / a_star2**2)**(1 / (self.g - 1)) *
                (-2 * (self.g - 1) / (self.g + 1) * u0 / a_star2)
            )
            dt1 = self.CFL * self.dx / (u0 + c0)
            L = ((u1 + u0) / 2 - (c1 + c0) / 2) * dt1 / self.dx
            delta_u = -L * (p1 - p0 - rho0 * c0 * (u1 - u0)) / (dpdu - rho0 * c0)

            u_new = u0 + delta_u
            T_new = self.Tt * (1 - (self.g - 1) / (self.g + 1) * u_new**2 / a_star2)
            p_new = self.Pt * (T_new / self.Tt)**(self.g / (self.g - 1))
            rho_new = p_new / (self.R_gas * T_new)
            c_new = np.sqrt(self.g * p_new / rho_new)
            M_new = u_new / c_new
            e_new = rho_new * (self.cv * T_new + 0.5 * u_new**2)
            self.W1[0, :] = np.array([
                rho_new,
                rho_new * u_new,
                e_new
            ])

            if M_new >= 1:
                self.mach_in = M_new

        else:  # Case for the supersonic inlet
            p_in = self.Pt * (1 + (self.g - 1) / 2 * self.mach_in ** 2) ** (-self.g / (self.g - 1))
            T_in = self.Tt * (1 + (self.g - 1) / 2 * self.mach_in ** 2) ** (-1)
            rho_in = p_in / (self.R_gas * T_in)
            c_in = np.sqrt(self.g * p_in / rho_in)
            u_in = self.mach_in * c_in
            self.W1[0, :] = np.array([
                rho_in,
                rho_in * u_in,
                p_in / (self.g - 1) + 0.5 * rho_in * u_in ** 2
            ])

        # Updating outlet boundary conditions
        rho2, u2, p2 = self.primitive(self.n - 1)
        c2 = np.sqrt(self.g * p2 / rho2)
        rho3, u3, p3 = self.primitive(self.n - 2)
        c3 = np.sqrt(self.g * p3 / rho3)
        dt2 = self.CFL * self.dx / (u2 + c2)
        Me = ((u2 + u3) / 2) / ((c2 + c3) / 2)

        L1 = ((u2+ u3) / 2 * dt2 / self.dx)
        L2 = ((u2 + u3) / 2 + (c2 + c3) / 2) * dt2 / self.dx
        L3 = ((u2 + u3) / 2 - (c2 + c3) / 2) * dt2 / self.dx

        R1 = -L1 * (rho2 - rho3 - 1 / c2**2 * (p2 - p3))
        R2 = -L2 * (p2 - p3 + rho2 * c2 * (u2 - u3))
        R3 = -L3 * (p2 - p3 - rho2 * c2 * (u2 - u3))

        if Me > 1:  # Supersonic exit
            delta_p = 0.5 * (R2 + R3)
        else:  # Subsonic exit
            delta_p = 0

        rho_new = rho2 + R1 + delta_p / c2**2
        u_new = u2 + (R2 - delta_p) / (rho2 * c2)
        p_new = p2 + delta_p
        T_new = p_new / (rho_new * self.R_gas)
        e_new = rho_new * (self.cv * T_new + 0.5 * u_new**2)
        self.W1[-1, :] = np.array([
            rho_new,
            rho_new * u_new,
            e_new
        ])
        # self.W1[-1, :] = self.W1[-2, :]

    def init_grid(self):  # Initializes the grid and sets initial conditions
        self.x = np.linspace(self.x0, self.x1, self.n)
        self.W = np.zeros((self.n, 3))
        self.W1 = np.zeros((self.n, 3))
        self.dx = self.x[1] - self.x[0]

        for i in range(self.n-1):
            Pi = self.Pt * (1 + (self.g - 1) / 2 * self.mach_in**2)**(-self.g / (self.g - 1))
            Ti = self.Tt * (1 + (self.g - 1) / 2 * self.mach_in**2)**(-1)
            rhoi = Pi / (self.R_gas * Ti)
            ui = self.mach_in * np.sqrt(self.g * self.R_gas * Ti)
            ei = Pi / (self.g - 1) + 0.5 * rhoi * ui**2
            self.W[i, :] = (rhoi, rhoi * ui, ei)

        # Initialize exit conditions
        Me = (((self.Pe / self.Pt)**((1 - self.g) / self.g) - 1) * 2 / (self.g - 1))**0.5
        Te = self.Tt * (1 + (self.g - 1) / 2 * Me**2)**(-1)
        rho_e = self.Pe / (self.R_gas * Te)
        ue = Me * np.sqrt(self.g * self.R_gas * Te)
        ee = self.Pe / (self.g - 1) + 0.5 * rho_e * ue**2
        self.W[-1, :] = (rho_e, rho_e * ue, ee)

    def Q(self, i):
        rho, u, p = self.primitive(i)
        return np.array([0, p * (self.S(self.x[i]) - self.S(self.x[i-1])), 0])

    def S(self, xi):  # Computes area of the nozzle
        return 1 - self.h * (np.sin(np.pi * xi ** self.t1)) ** self.t2

    # def S(self, xi):
    #     return 0.2 * xi + 1




if __name__ == '__main__':
    n_grid = 50
    pressure_ratio = 0.8
    M_in = 0.1
    epsilon = 0.5
    max_itr = 5000
    tol = 1e-6

    euler_solver = quasi1DEuler(n=n_grid, epsilon=epsilon, max_iterations=max_itr, convergence_tolerance=tol)
    residual = euler_solver.solve(exit_pressure_ratio=pressure_ratio, mach_in=M_in)

    velocity = []
    pressure = []
    mach = []
    for i in range(euler_solver.n):
        rho, u, p = euler_solver.primitive(i)
        c = np.sqrt(euler_solver.g * p / rho)
        pressure.append(p)
        velocity.append(u)
        mach.append(u / c)

    pressure = np.array([pressure]).reshape(-1) / 1e3
    velocity = np.array(velocity)

    # # Plotting the residual
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot()
    # ax1.plot(np.arange(1, len(residual) + 1), residual)
    # ax1.set_title("Convergence plot for the density residual")
    # ax1.set_yscale('log')
    # ax1.set_xlabel('x')
    # ax1.set_ylabel(r'$||(R_i)_\rho||_\infty$')
    # fig1.show()

    # Plotting the pressure distribution
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.plot(euler_solver.x, pressure, color="tab:blue")
    ax2.tick_params(axis='y', colors="tab:blue")
    ax2.set_ylabel("Pressure [kPa]", color="tab:blue")

    ax21 = ax2.twinx()
    ax21.plot(euler_solver.x, mach, 'tab:red')
    ax21.tick_params(axis='y', colors='tab:red')
    ax21.set_ylabel('Mach Number', color='tab:red')

    ax2.set_title("Pressure distribution through the nozzle")
    ax2.set_xlabel("x")
    fig2.show()






