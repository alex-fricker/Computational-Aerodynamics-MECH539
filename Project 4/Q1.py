import numpy as np
import matplotlib.pyplot as plt
from quasi_1D_euler import quasi1DEuler

n_grid = 50
pressure_ratio = 0.8
epsilon = 0.2
CFL = 0.34
max_itr = 5e4
tol = 1e-16

euler_solver = quasi1DEuler(n=n_grid, epsilon=epsilon, CFL=CFL, max_iterations=max_itr, convergence_tolerance=tol)
residual = euler_solver.solve(exit_pressure_ratio=pressure_ratio)

velocity = []
pressure = []
mach = []
for i in range(euler_solver.n):
    rho, u, p = euler_solver.primitive(i)
    c = np.sqrt(euler_solver.g * p / rho)
    pressure.append(p)
    velocity.append(u)
    mach.append(u / c)

pressure = np.array([pressure]).reshape(-1) / euler_solver.Pt
velocity = np.array(velocity)
print(max(mach))

fig1 = plt.figure(figsize=(10, 8))
ax11 = plt.subplot2grid((4, 4), (0, 0), rowspan=2, colspan=2)
ax12 = plt.subplot2grid((4, 4), (0, 2), rowspan=2, colspan=2)
ax13 = plt.subplot2grid((4, 4), (2, 0), rowspan=2, colspan=4)

ax13.plot(np.arange(1, len(residual) + 1), residual)
ax13.set_title("Convergence plot for the density residual")
ax13.set_yscale('log')
ax13.set_xlabel('x')
ax13.set_ylabel(r'Density Residual')

# Plotting the pressure distribution
ax11.plot(euler_solver.x, pressure)
ax11.set_ylabel(r"$\frac{P}{P_{total, inlet}}$ [kPa]")
ax11.set_xlabel('x')
ax11.set_title("Pressure distribution through the nozzle")

# Plotting mach distribution
ax12.plot(euler_solver.x, mach)
ax12.set_ylabel('Mach Number')
ax12.set_title("Mach distribution through the nozzle")
ax12.set_xlabel("x")

fig1.suptitle(r"Pressure ratio for $P_{exit} = $" + str(pressure_ratio) + r"$P_{total, inlet}$, $\epsilon$ = " + str(epsilon) + " and CFL= " + str(CFL), fontsize=18)
# fig1.tight_layout(h_pad=0.08, w_pad=0.15, rect=(0, 0, 1, 0.98))
fig1.subplots_adjust(left=0.1, right=0.9, bottom=0.06, top=0.9, wspace=0.5, hspace=0.5)
fig1.savefig("Q1.png", dpi=300)
fig1.show()

print("Done")
