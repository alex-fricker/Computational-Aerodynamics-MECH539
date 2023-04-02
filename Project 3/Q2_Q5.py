import numpy as np
import matplotlib.pyplot as plt
from transonic_small_disturbance_equation import TSDEquation


domain = (0, 50, 0, 50)
airfoil_coords = (20, 21)
nx, ny = 60, 20
airfoil_refinements = 30
tol = 1e-4
TSDE = TSDEquation(
    domain=domain,
    airfoil_coords=airfoil_coords,
    nx=nx,
    ny=ny,
    n_airfoil_refine=airfoil_refinements,
    convergence_tolerance=tol)

fig4 = plt.figure()
ax4 = fig4.add_subplot()

for mach in np.arange(0.8, 0.91, 0.02):
    print(mach)
    residual, iterations, cpu_time = TSDE.solve(mach, "Gauss-Seidel")
    Cp = TSDE.get_airfoil_Cp()
    pressure, velocity = TSDE.get_pressure_velocity_field()
    x, y = TSDE.x, TSDE.y

    fig = plt.figure(figsize=(9, 6))
    ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=2, rowspan=2, fig=fig)
    ax2 = plt.subplot2grid((4, 4), (0, 2), colspan=2, rowspan=2, fig=fig)
    ax3 = plt.subplot2grid((4, 4), (2, 0), colspan=4, rowspan=2, fig=fig)

    le_idx = int((TSDE.nx - TSDE.n_ref) / 2)
    te_idx = le_idx + TSDE.n_ref
    ax1.plot(TSDE.x[le_idx: te_idx], Cp, label=f"Grid DoF: {TSDE.nx * TSDE.ny:.0f}")
    ax1.set_title("Coefficient of pressure")
    ax1.set_xlabel('x')
    ax1.set_ylabel('Cp')
    ax1.legend()
    ax1.invert_yaxis()

    cf2 = ax2.contourf(x, y, pressure * 1e-3, levels=12, cmap='RdBu_r')
    ax2.contour(x, y, pressure, levels=10, colors='black', linewidths=0.5)
    cbar2 = fig.colorbar(cf2, ax=ax2)
    cbar2.set_label('Pressure [kPa]')
    ax2.set_title("Pressure field")
    ax2.set_xlim(20, 21)
    ax2.set_ylim(0, 1)

    ax3.plot(np.arange(len(residual)), residual)
    ax3.set_title(r"$L_\infty$ norm convergence")
    ax3.set_yscale('log')
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel(r"$|| \phi^{k+1} - \phi^k ||_\infty$")

    fig.tight_layout(h_pad=0.08, w_pad=0.05, rect=(0, 0, 1, 0.9))
    fig.suptitle(f"Mach {mach:.2f}, for a {TSDE.nx:.0f}x{TSDE.ny:.0f} grid")
    fig.savefig(f"Q2_mach{mach:.2f}.png", dpi=300)

    ax4.plot(TSDE.x[le_idx: te_idx], Cp, label=f"Mach {mach:.2f}")

ax4.set_title(r"Cp distribution for varying $M_\infty$ on a {:.0f}x{:.0f} grid".format(nx, ny))
ax4.set_xlabel('x')
ax4.set_ylabel('Cp')
fig4.gca().invert_yaxis()
ax4.legend()
fig4.savefig('Q5.png', dpi=300)
