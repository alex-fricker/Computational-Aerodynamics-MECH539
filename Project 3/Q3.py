import numpy as np
import matplotlib.pyplot as plt
from transonic_small_disturbance_equation import TSDEquation


domain = (0, 50, 0, 50)
airfoil_coords = (20, 21)
tol = 1e-4
mach = 0.88

fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)

for grid in [(30, 10, 15), (60, 20, 30), (120, 40, 60)]:
    nx, ny, airfoil_refinement = grid
    TSDE = TSDEquation(
        domain=domain,
        airfoil_coords=airfoil_coords,
        nx=nx,
        ny=ny,
        n_airfoil_refine=airfoil_refinement,
        convergence_tolerance=tol)
    residual, iterations, cpu_time = TSDE.solve(mach, "Gauss-Seidel")
    Cp = TSDE.get_airfoil_Cp()

    le_idx = int((TSDE.nx - TSDE.n_ref) / 2)
    te_idx = le_idx + TSDE.n_ref
    plt3 = ax1.plot(TSDE.x[le_idx: te_idx], Cp, label=f"Grid DoF: {TSDE.nx * TSDE.ny:.0f}")

ax1.set_title(r"Shock evolution for successively refined grids")
ax1.set_xlabel('x')
ax1.set_ylabel('Cp')
ax1.legend()
fig1.gca().invert_yaxis()
fig1.savefig('Q3.png', dpi=300)
