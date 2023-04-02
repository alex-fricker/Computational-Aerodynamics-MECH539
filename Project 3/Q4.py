import numpy as np
import matplotlib.pyplot as plt
from transonic_small_disturbance_equation import TSDEquation


domain = (0, 50, 0, 50)
airfoil_coords = (20, 21)
nx, ny = 60, 20
airfoil_refinements = 30
tol = 1e-14
max_itr = 5e5
mach = 0.86

TSDE = TSDEquation(
    domain=domain,
    airfoil_coords=airfoil_coords,
    nx=nx,
    ny=ny,
    n_airfoil_refine=airfoil_refinements,
    convergence_tolerance=tol,
    max_iterations=max_itr)

GS_residual, GS_iterations, GS_cpu_time = TSDE.solve(mach, "Gauss-Seidel")
# LIGS_residual, LIGS_iterations, LIGS_cpu_time = TSDE.solve(mach, "Line Implicit Gauss-Seidel")

plt.loglog(np.arange(len(GS_residual)), GS_residual)
plt.yscale('log')
plt.title(r"$L_\infty$ Norm convergence VS iterations for 60x40 grid at Mach 0.86")
plt.xlabel('Iterations')
plt.ylabel(r"$|| \phi^{k+1} - \phi^k ||_\infty$")
plt.savefig("Q4a.png", dpi=300)

plt.close()

plt.loglog(np.linspace(0, GS_cpu_time, len(GS_residual)), GS_residual)
plt.yscale('log')
plt.title(r"$L_\infty$ Norm convergence VS CPU time for 60x40 grid at Mach 0.86")
plt.xlabel('CPU time')
plt.ylabel(r"$|| \phi^{k+1} - \phi^k ||_\infty$")
plt.savefig("Q4b.png", dpi=300)



