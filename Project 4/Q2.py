import numpy as np
import matplotlib.pyplot as plt
from quasi_1D_euler import quasi1DEuler

n_grid = 50
epsilon = 0.15
CFL = 0.3
max_itr = 1e6
tol = 1e-12

fig1 = plt.figure()
ax1 = fig1.add_subplot()

fig2 = plt.figure()
ax2 = fig2.add_subplot()

fig3 = plt.figure()
ax3 = fig3.add_subplot()

for pr in [0.76, 0.72, 0.68, 0.60]:
    euler_solver = quasi1DEuler(n=n_grid, epsilon=epsilon, CFL=CFL, max_iterations=max_itr, convergence_tolerance=tol)
    residual = euler_solver.solve(exit_pressure_ratio=pr)

    pressure = []
    mach = []
    for i in range(euler_solver.n):
        rho, u, p = euler_solver.primitive(i)
        c = np.sqrt(euler_solver.g * p / rho)
        pressure.append(p)
        mach.append(u / c)
    pressure = np.array([pressure]).reshape(-1) / euler_solver.Pt

    ax1.plot(np.linspace(euler_solver.x0, euler_solver.x1, euler_solver.n), pressure, label=f'{pr}')
    ax2.plot(np.linspace(euler_solver.x0, euler_solver.x1, euler_solver.n), mach, label=f'{pr}')
    ax3.plot(np.arange(1, len(residual) + 1), residual, label=f"{pr}")

ax1.set_xlabel('x')
ax1.set_ylabel(r"$\frac{P}{P_{total, inlet}}$ [kPa]")
ax1.set_title("Pressure distribution through the nozzle")
ax1.legend(title="Pressure ratio")

ax2.set_ylabel('Mach Number')
ax2.set_title("Mach distribution through the nozzle")
ax2.set_xlabel("x")
ax2.legend(title="Pressure ratio")

ax3.set_title("Convergence plot for the density residual")
ax3.set_yscale('log')
ax3.set_xlabel('x')
ax3.set_ylabel(r'$||(R_i)_\rho||_\infty$')
ax3.legend(title="Pressure ratio")

fig1.savefig("Q2_pressure.png", dpi=300)
fig2.savefig("Q2_mach.png", dpi=300)
fig3.savefig("Q2_convergence.png", dpi=300)

fig1.show()
fig2.show()
fig3.show()


