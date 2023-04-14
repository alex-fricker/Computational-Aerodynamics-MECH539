import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd


class postProcessFlow:
    def __init__(self, mesh_params, flow_params, n_variables, filepath=None, data_array=None):
        self.imax, self.jmax, self.TE_start, self.TE_end, self.LE = mesh_params
        self.gamma, self.mach, self.p0, self.alpha, self.Re = flow_params
        self.alpha = np.deg2rad(self.alpha)

        if type(filepath) == str:
            data = np.loadtxt('NACA0012_flowfieldv2.dat')
        elif type(data_array) == np.array:
            data = data_array
        else:
            print("Invalid flow data type. Exiting post processor.")
            exit(1)

        k = 0
        self.x = np.zeros((self.imax, self.jmax))
        self.y = np.zeros((self.imax, self.jmax))

        for j in range(self.jmax):
            for i in range(self.imax):
                self.x[i, j] = data[k, 0]
                self.y[i, j] = data[k, 1]
                k += 1

        k = 0
        self.w = np.zeros((self.imax, self.jmax, n_variables))
        for j in range(self.jmax):
            for i in range(self.imax):
                for n in range(n_variables):
                    self.w[i, j, n] = data[k, 2 + n]
                k += 1

    def compute_uplus_yplus(self, i):
        u_plus = np.zeros(self.jmax)
        y_plus = np.zeros(self.jmax)
        tw = self.compute_wall_shear_stress()[i-self.TE_start]
        u_friction = np.sqrt(abs(tw) / self.w[i, 0, 0])
        nui = (self.w[i, 0, 4] + self.w[i, 0, 5]) / self.w[i, 0, 0]

        k = 0
        for j in range(0, self.jmax):
            dxi = self.x[i, 0] - self.x[i - 1, 0]  # dx/dxi
            dyi = self.y[i, 0] - self.y[i - 1, 0]  # dy/dxi
            dAi = np.sqrt(dxi ** 2 + dyi ** 2)
            cost = dxi / dAi
            sint = dyi / dAi

            uj = self.w[i, j, 1] / self.w[i, j, 0]
            vj = self.w[i, j, 2] / self.w[i, j, 0]
            Vj = np.sqrt((cost * uj)**2 + (sint * vj)**2)

            u_plus[k] = Vj / u_friction
            y_plus[k] = np.sqrt(self.w[i, j, -1]) * u_friction / nui
            k += 1

        return u_plus, y_plus

    def compute_wall_shear_stress(self):
        tw = np.zeros(self.TE_end - self.TE_start)
        k = 0
        for i in range(self.TE_start, self.TE_end):
            dxi = self.x[i, 0] - self.x[i - 1, 0]  # dx/dxi
            dyi = abs(self.y[i, 0]) - abs(self.y[i - 1, 0])  # dy/dxi
            dxj = self.x[i, 1] - self.x[i, 0]  # dx/deta
            dyj = abs(self.y[i, 1]) - abs(self.y[i, 0])  # dy/deta
            dsj = 1 / (dxi * dyj - dyi * dxj)

            dui = self.w[i, 0, 1] / self.w[i, 0, 0] - self.w[i - 1, 0, 1] / self.w[i - 1, 0, 0]  # du/dxi
            dvi = self.w[i, 0, 2] / self.w[i, 0, 0] - self.w[i - 1, 0, 2] / self.w[i - 1, 0, 0]  # dv/dxi
            duj = self.w[i, 1, 1] / self.w[i, 1, 0] - self.w[i, 0, 1] / self.w[i, 0, 0]  # du/deta
            dvj = self.w[i, 1, 2] / self.w[i, 1, 0] - self.w[i, 0, 2] / self.w[i, 0, 0]  # dv/deta

            dux = (dui * dyj - duj * dyi) * dsj  # du/dx
            dvx = (dvi * dyj - dvj * dyi) * dsj  # dv/dx
            duy = (duj * dxi - dui * dxj) * dsj  # du/dy
            dvy = (dvj * dxi - dvi * dxj) * dsj  # dv/dy

            dAi = np.sqrt(dxi**2 + dyi**2)
            cost = dxi / dAi
            sint = dyi / dAi
            mu = self.w[i, 0, 4] + self.w[i, 0, 5]  # mu_L + mu_T

            txx = mu * 2 * dux - 2 / 3 * mu * (dux + dvy)
            tyy = mu * 2 * dvy - 2 / 3 * mu * (dux + dvy)
            txy = mu * (duy + dvx)
            tx = -txx * sint + txy * cost
            ty = -txy * sint + tyy * cost
            tw[k] = tx * cost + ty * sint
            k += 1
        return tw

    def compute_surface_cp(self):
        cp = np.zeros(self.TE_end - self.TE_start)
        k = 0
        j = 0
        for i in range(self.TE_start, self.TE_end):
            pressure = (self.gamma - 1) * (self.w[i, j, 3] - 0.5 * (self.w[i, j, 1] ** 2 + self.w[i, j, 2] ** 2) / self.w[i, j, 0])
            cp[k] = (pressure / self.p0 - 1) / (0.5 * self.gamma * self.mach**2)
            k += 1

        print(f"Peak Cp top surface: {np.max(cp[:self.LE - self.TE_start])}, "
              f"peak Cp lower surface: {np.min(cp[self.LE - self.TE_start:])}")
        return cp

    def compute_performance(self):
        cp = self.compute_surface_cp()
        cf = 2 * self.compute_wall_shear_stress() / (self.p0 * self.gamma * self.mach ** 2)
        cpx, cpy, cfx, cfy = 0, 0, 0, 0

        k = 0
        for i in range(self.TE_start, self.TE_end):
            dxi = self.x[i, 0] - self.x[i - 1, 0]  # dx/dxi
            dyi = self.y[i, 0] - self.y[i - 1, 0]  # dy/dxi
            dAi = np.sqrt(dxi ** 2 + dyi ** 2)
            cost = dxi / dAi
            sint = dyi / dAi

            cpx += -cp[k] * -sint * dAi
            cpy += -cp[k] * cost * dAi
            cfx += cf[k] * cost * dAi
            cfy += cf[k] * sint * dAi
            k += 1

        cl_p = -cpx * np.sin(self.alpha) + cpy * np.cos(self.alpha)
        cl_f = -cfx * np.sin(self.alpha) + cfy * np.cos(self.alpha)
        cd_p = cpx * np.cos(self.alpha) + cpy * np.sin(self.alpha)
        cd_f = cfx * np.cos(self.alpha) + cfy * np.sin(self.alpha)
        data = {"Cl": [cl_p, cl_f, cl_p + cl_f], "Cd": [cd_p, cd_f, cd_p + cd_f]}
        results = pd.DataFrame(data, index=["Pressure", "Friction", "Total"])

        print(tabulate(results))
        results.to_csv("aero_performance.csv")
        print("Done computing aerodynamic performance coefficients")

    def generate_plots(self, cf_plot=False, cp_plot=False, airfoil_surf=False, comp_domain=False,
                       u_plus_VS_y_plus=False, x_point=None, surface=None, momentum_deficit_plot=False,
                       BL_regime_plot=False):
        ### Plotting the momentum deficit in the wake
        if momentum_deficit_plot:
            U_inf = np.sqrt((self.w[self.LE-1, self.jmax-1, 1] / self.w[self.LE-1, self.jmax-1, 0])**2 +
                            (self.w[self.LE-1, self.jmax-1, 2] / self.w[self.LE-1, self.jmax-1, 0])**2)
            fig6 = plt.figure(figsize=(8, 6))
            ax6 = fig6.add_subplot()
            momentum_deficit = np.zeros((2, 4))

            # Collecting points to plot velocity at
            colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
            k = 0
            for xpt in [1.25, 1.5, 2, 3]:
                deficit = 0
                ypts_lower = np.zeros(self.jmax)
                v_lower = np.zeros(self.jmax)
                ypts_upper = np.zeros(self.jmax)
                v_upper = np.zeros(self.jmax)
                # Collect values on lower half of domain
                for j in range(self.jmax):
                    xid_lower = np.argmin(np.abs(self.x[:self.LE, j] - xpt))
                    ypts_lower[j] = self.y[xid_lower, j]
                    v_lower[j] = np.sqrt((self.w[xid_lower, j, 1] / self.w[xid_lower, j, 0])**2 + (self.w[xid_lower, j, 2] / self.w[xid_lower, j, 0])**2)

                    xid_upper = np.argmin(np.abs(self.x[self.LE:, j] - xpt)) + self.LE
                    ypts_upper[j] = self.y[xid_upper, j]
                    v_upper[j] = np.sqrt((self.w[xid_upper, j, 1] / self.w[xid_upper, j, 0])**2 + (self.w[xid_upper, j, 2] / self.w[xid_upper, j, 0])**2)

                    deficit += (U_inf - v_lower[j]) * (ypts_lower[j] - ypts_lower[j-1])
                    deficit += (U_inf - v_upper[j]) * (ypts_upper[j] - ypts_upper[j-1])

                ax6.plot(v_lower, ypts_lower, color=colors[k], label=f"x = {xpt}")
                ax6.plot(v_upper, ypts_upper, color=colors[k])
                momentum_deficit[0, k] = xpt
                momentum_deficit[1, k] = deficit
                k += 1

            ax6.set_xlabel('Speed')
            ax6.set_ylabel('y')
            ax6.legend()
            ax6.set_title("Wake speed profile")
            fig6.savefig("wake_speed_profile.png", dpi=300)

            ax6.set_ylim(-1, 1)
            fig6.savefig("wake_speed_profile_close.png", dpi=300)
            fig6.show()
            np.savetxt("momentum_deficit.csv", momentum_deficit, delimiter=",")

        ### u+ VS y+ at specified point and surface ###
        if u_plus_VS_y_plus:
            if x_point is None or surface is None:
                print("Please specify a point and top/bottom surface to plot the u+ VS y+ at")
                exit(1)
            if surface.lower() == "upper":
                idx = np.argmin(abs(self.x[self.LE:self.TE_end+1, 0] - x_point)) + self.LE
            elif surface.lower() == "lower":
                idx = np.argmin(abs(self.x[self.TE_start:self.LE, 0] - x_point)) + self.TE_start
            u_plus, y_plus = self.compute_uplus_yplus(idx)
            print(f'Minimum y+ value acheived: {np.min(y_plus)}')
            loglaw = 1 / 0.41 * np.log(y_plus) + 5.15

            fig5 = plt.figure()
            ax5 = fig5.add_subplot()
            ax5.plot(y_plus, u_plus, label="Simulated result")
            ax5.plot(y_plus, y_plus, '--k', label=r"$u^+ = y^+$")
            ax5.plot(y_plus, loglaw, '-.k', label=r"$u^+ = \frac{1}{0.41}ln(y^+)+5.15$")
            ax5.set_xlabel(r"$y^+$")
            ax5.set_ylabel(r"$u^+$")
            ax5.set_xscale("log")
            ax5.set_title(f"Velocity profile at x={x_point} on the {surface.lower()} surface")
            ax5.set_ylim(0, 25)
            ax5.set_xlim(0, 1e3)
            ax5.legend()
            fig5.savefig(f"uplus_yplus_x_{x_point}_{surface.lower()}.png", dpi=300)
            fig5.show()

        ### Boundary layer regime plot ###
        # regime = 'laminar'
        if BL_regime_plot:
            tw = self.compute_wall_shear_stress()
            cp = self.compute_surface_cp()
            stag_id = np.argmin(np.abs(cp - 1))
            ids_upper = np.where(tw[stag_id:] < 0)[0] + self.TE_start + stag_id
            ids_lower = np.where(tw[:stag_id] < 0)[0] + self.TE_start
            print('IDs of negative wall shear stress above the stagnation point:\n', ids_upper)
            print('\nIDs of negative wall shear stress below the stagnation point:\n', ids_lower)
            print(f'Stagnation point id: {stag_id + self.TE_start}')
            print(f'\nStagnation point location: x={self.x[stag_id + self.TE_start, 0]}, y={self.y[stag_id + self.TE_start, 0]}')

        #
        #     x = np.zeros((3, np.zeros(self.TE_end - self.TE_start)))  # First row is laminar, 2nd is separated, 3rd turbulent
        #     y = np.zeros((3, np.zeros(self.TE_end - self.TE_start)))
        #     for i in range(self.TE_start, self.TE_end):
        #         if tw > 0 and regime == 'laminar':
        #             x[i, 0] = self.x[i, 0]
        #             y[i, 0] = self.y[i, 0]
        #         elif tw < 0 and regime == 'laminar':
        #             x[i, 1] = self.x[i, 0]
        #             y[i, 1] = self.y[i, 0]
        #             regime = 'separated'
        #         elif tw < 0 and regime == 'separated':
        #             x[i, 1] = self.x[i, 0]
        #             y[i, 1] = self.y[i, 0]
        #         elif tw

            # separation_id_lower = np.max(np.where(tw[:stag_id] < 0)) + self.TE_start
            # reattach_id_lower = np.min(np.where(tw[:stag_id] < 0)) + self.TE_start
            # separation_id_upper = np.min(np.where(tw[stag_id:] < 0)) + self.TE_start
            # reattach_id_upper = np.max(np.where(tw[stag_id:] < 0)) + self.TE_start
            #
            # fig = plt.figure()
            # ax = fig.add_subplot()
            # ax.plot(self.x[self.TE_start:self.TE_end, 0], self.y[self.TE_start:self.TE_end, 0], 'k*-', linewidth=1.5, markersize=1.5)
            # ax.axis('equal')
            # ax.axvline(x=self.x[separation_id_upper, 0])
            # ax.axvline(x=self.x[reattach_id_upper, 0])
            # fig.savefig("airfoil_surf.png", dpi=300)
            # fig.show()

        ### Wall shear stress distribution ###
        if cf_plot:
            tw = self.compute_wall_shear_stress()
            Cf = 2 * tw / (self.p0 * self.gamma * self.mach**2)
            fig2 = plt.figure()
            ax2 = fig2.add_subplot()
            ax2.plot(self.x[self.TE_start: self.LE, 0], Cf[:self.LE - self.TE_start], color="tab:red", label="Pressure side")
            ax2.plot(self.x[self.LE: self.TE_end, 0], Cf[self.LE - self.TE_start:], color="tab:blue", label="Suction side")
            ax2.set_xlabel("x")
            ax2.set_ylabel(r"$C_f$")
            ax2.set_title("Skin friction coefficient distribution")
            ax2.legend()
            ax2.grid('both')
            fig2.savefig("Cf_distribution.png", dpi=300)
            ax2.set_xlim(0, 0.05)
            ax2.set_ylim(-0.002, 0.0025)
            fig2.savefig("FUCK.png", dpi=300)
            fig2.show()

        ### Cp distribution ###
        if cp_plot:
            cp = self.compute_surface_cp()
            fig1 = plt.figure()
            ax1 = fig1.add_subplot()
            ax1.plot(self.x[self.TE_start: self.LE+1, 0], cp[:self.LE - self.TE_start+1], color="tab:red", label="Pressure side")
            ax1.plot(self.x[self.LE: self.TE_end, 0], cp[self.LE - self.TE_start:], color="tab:blue", label="Suction side")
            axins1 = ax1.inset_axes([0.5, 0.5, 0.47, 0.47])
            axins1.plot(self.x[self.LE: self.TE_end, 0], cp[self.LE - self.TE_start:], color="tab:blue")
            axins1.set_xlim(-0.005, 0.05)
            axins1.set_ylim(-2.45, -3.65)
            rect, connects = ax1.indicate_inset_zoom(inset_ax=axins1, edgecolor="black")
            connects[0].set_visible(True)
            connects[1].set_visible(True)
            connects[2].set_visible(False)
            connects[3].set_visible(False)
            ax1.legend()
            ax1.invert_yaxis()
            ax1.set_xlabel("x")
            ax1.set_ylabel("Cp")
            ax1.set_title("Coefficient of pressure distribution")
            ax1.grid('both')
            fig1.savefig("CP_distribution.png", dpi=300)
            fig1.show()

        ### Airfoil surface ###
        if airfoil_surf:
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.plot(self.x[self.TE_start:self.TE_end, 0], self.y[self.TE_start:self.TE_end, 0], 'k*-', linewidth=1.5, markersize=1.5)
            ax.axis('equal')
            fig.savefig("airfoil_surf.png", dpi=300)
            fig.show()

        ### Computational Domain ###
        if comp_domain:
            fig0 = plt.figure()
            ax01 = fig0.add_subplot(211)
            ax02 = fig0.add_subplot(212)
            ax01.axis('equal')
            ax02.axis('equal')
            for j in range(self.jmax):
                ax01.plot(self.x[0:self.imax, j], self.y[0:self.imax, j], 'k-', linewidth=0.2)
            for i in range(self.imax):
                ax01.plot(self.x[i, 0:self.jmax], self.y[i, 0:self.jmax], 'k-', linewidth=0.2)

            for j in range(self.jmax):
                ax02.plot(self.x[0:self.imax, j], self.y[0:self.imax, j], 'k-', linewidth=0.2)
            for i in range(self.imax):
                ax02.plot(self.x[i, 0:self.jmax], self.y[i, 0:self.jmax], 'k-', linewidth=0.2)
            ax02.axis([-0.1, 1.1, -0.2, 0.2])
            fig0.savefig("comp_domain.png", dpi=300)
            fig0.show()

    def get_coords(self, i, j):
        return self.x[i, j], self.y[i, j]


if __name__ == '__main__':
    mesh_params = (513, 257, 65, 449, 257)
    flow_params = (1.4, 0.1, 1.0, 8, 3e6)
    filepath = './NACA0012_flowfieldv2.dat'
    n_vars = 7

    post_processor = postProcessFlow(mesh_params=mesh_params, flow_params=flow_params, n_variables=n_vars, filepath=filepath)
    # post_processor.generate_plots(wall_shear_stress_plot=True,
    #                               cp_plot=True,
    #                               airfoil_surf=True,
    #                               comp_domain=True,
    #                               momentum_deficit_plot=True,
    #                               BL_regime_plot=True)
    post_processor.generate_plots(u_plus_VS_y_plus=True, x_point=0.5, surface='upper')
    post_processor.generate_plots(u_plus_VS_y_plus=True, x_point=0.85, surface='upper')
    post_processor.generate_plots(u_plus_VS_y_plus=True, x_point=0.85, surface='lower')
    post_processor.compute_performance()
