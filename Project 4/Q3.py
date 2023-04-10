import numpy as np
import matplotlib.pyplot as plt
from quasi_1D_euler import quasi1DEuler


fig1 = plt.figure()
ax1 = fig1.add_subplot()
fig2 = plt.figure()
ax2 = fig2.add_subplot()
fig3 = plt.figure()
ax3 = fig3.add_subplot()

pressure_ratio = 0.72
epsilon = [0.17, 0.18, 0.18, 0.3]  # 0.3 for 200, 0.18 for 100 &50
CFL = 0.32
max_itr = 1e5
tol = 1e-16
delta_Pt = np.zeros((4, 2))
k = 0

for n_grid in [25, 50, 100, 200]:
    e = epsilon[k]
    euler_solver = quasi1DEuler(n=n_grid, epsilon=e, CFL=CFL, max_iterations=max_itr, convergence_tolerance=tol)
    residual = euler_solver.solve(exit_pressure_ratio=pressure_ratio)

    pressure = []
    mach = []
    for i in range(euler_solver.n):
        rho, u, p = euler_solver.primitive(i)
        c = np.sqrt(euler_solver.g * p / rho)
        pressure.append(p)
        mach.append(u / c)
    pressure_ratio_distribution = np.array([pressure]).reshape(-1) / euler_solver.Pt
    ax1.plot(np.linspace(euler_solver.x0, euler_solver.x1, euler_solver.n), pressure_ratio_distribution, label=f'{n_grid}')
    ax2.plot(np.linspace(euler_solver.x0, euler_solver.x1, euler_solver.n), mach, label=f'{n_grid}')
    ax3.plot(np.arange(1, len(residual) + 1), residual, label=f"{n_grid}")

    total_pressure_in = pressure[0] * (1 + (euler_solver.g - 1) / 2 * mach[0]**2)**(euler_solver.g / (euler_solver.g - 1))
    total_pressure_out = pressure[-1] * (1 + (euler_solver.g - 1) / 2 * mach[-1] ** 2) ** (euler_solver.g / (euler_solver.g - 1))
    delta_Pt[k, 0] = n_grid
    delta_Pt[k, 1] = abs(total_pressure_in - total_pressure_out) / total_pressure_in * 100
    k += 1

ax1.set_xlabel('x')
ax1.set_ylabel(r"$\frac{P}{P_{total, inlet}}$ [kPa]")
ax1.set_title("Pressure distribution through the nozzle")
ax1.legend(title="Number of grid points")

ax2.set_ylabel('Mach Number')
ax2.set_title("Mach distribution through the nozzle")
ax2.set_xlabel("x")
ax2.legend(title="Number of grid points")

ax3.set_title("Convergence plot for the density residual")
ax3.set_yscale('log')
ax3.set_xlabel('x')
ax3.set_ylabel(r'$||(R_i)_\rho||_\infty$')
ax3.legend(title="Number of grid points")

fig1.savefig("Q3_pressure.png", dpi=300)
fig2.savefig("Q3_mach.png", dpi=300)
fig3.savefig("Q3_convergence.png", dpi=300)

fig1.show()
fig2.show()
fig3.show()

np.savetxt("total_pressure_loss.csv", np.array(delta_Pt), delimiter=',')