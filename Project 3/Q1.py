import numpy as np
import matplotlib.pyplot as plt
from transonic_small_disturbance_equation import TSDEquation


domain = (0, 50, 0, 50)
airfoil_coords = (20, 21)
nx, ny = 60, 20
airfoil_refinements = 30
tol = 1e-4
mach = 0.8

TSDE = TSDEquation(
    domain=domain,
    airfoil_coords=airfoil_coords,
    nx=nx,
    ny=ny,
    n_airfoil_refine=airfoil_refinements,
    convergence_tolerance=tol)
residual, iterations, cpu_time = TSDE.solve(mach, "Gauss-Seidel")
Cp = TSDE.get_airfoil_Cp()
pressure, velocity = TSDE.get_pressure_velocity_field()
x, y = TSDE.x, TSDE.y
x, y = np.meshgrid(x, y)

fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
cf1 = ax1.contourf(x, y, velocity, levels=10, cmap='RdBu')
ax1.contour(x, y, velocity, levels=10, linewidths=0.5, colors='black')
ax1.set_title(r"Velocity field at $M_\infty$={:.2f} on a {:.0f}x{:.0f} grid".format(TSDE.Minf, nx, ny))
cbar1 = fig1.colorbar(cf1)
cbar1.set_label('Velocity [m/s]')
ax1.set_xlim(18, 23)
ax1.set_ylim(0, 2)
# fig1.show()
fig1.savefig("Q1_velocity.png", dpi=300)

fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
cf2 = ax2.contourf(x, y, pressure*1e-3, levels=10, cmap='RdBu')
ax2.contour(x, y, pressure, levels=10, colors='black', linewidths=0.5)
cbar2 = fig2.colorbar(cf2)
cbar2.set_label('Pressure [kPa]')
ax2.set_title(r'Pressure field at $M_\infty$={:.2f} on a {:.0f}x{:.0f} grid'.format(TSDE.Minf, nx, ny))
ax2.set_xlim(18, 23)
ax2.set_ylim(0, 2)
# fig2.show()
fig2.savefig("Q1_pressure.png", dpi=300)

le_idx = int((TSDE.nx - TSDE.n_ref) / 2)
te_idx = le_idx + TSDE.n_ref
fig3 = plt.figure()
ax3 = fig3.add_subplot(1, 1, 1)
plt3 = ax3.plot(TSDE.x[le_idx: te_idx], Cp)
ax3.set_title(r"Coefficient of pressure at $M_\infty$={:.2f} on a {:.0f}x{:.0f} grid".format(TSDE.Minf, nx, ny))
ax3.set_xlabel('x')
ax3.set_ylabel('Cp')
fig3.gca().invert_yaxis()
fig3.savefig('Q1_Cp.png', dpi=300)
# fig3.show()

# fig3 = plt.figure()
# ax3 = fig3.add_subplot(1, 1, 1)
# cf3 = ax3.contourf(x, y, Cp, levels=10, cmap='RdBu')
# ax3.contour(x, y, Cp, levels=10, colors='black', linewidths=0.5)
# ax3.set_title(r"Coefficient of pressure at $M_\infty$={:.2f} on a {:.0f}x{:.0f} grid".format(TSDE.Minf, nx, ny))
# cbar3 = fig3.colorbar(cf3)
# cbar3.set_label('Coefficient of pressure [-]')
# ax3.set_xlim(20, 21)
# ax3.set_ylim(0, 1)
# fig3.show()
# fig3.savefig("Q1_Cp.png", dpi=300)