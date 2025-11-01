import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from ode_solver import ODE_Legendre
from Utility import mesh1d


#==================================================================
class ODE_LinAdapt():
#==================================================================
    def __init__(self, app, k=0):
        self.solver = ODE_Legendre(k)
        self.app = app
    def iteration(self, mesh):
        app = self.app
        u_h = self.solver.run(mesh, app.u0, app.a_coef, app.b_coef)
        el2, edisc, err_cell = self.solver.compute_error(mesh, u_h, app.solution)
        eta, eta_cell = self.solver.estimator(mesh, u_h, app.a_coef, app.b_coef)
        return u_h, eta, eta_cell, el2, err_cell
    def interpolate(self, ucoeff, mesh, mesh_new, refinfo):
        u_h, u_T = ucoeff
        # print(f"{mesh=}\n{mesh_new=}")
        refined_map, non_refined_map = refinfo

        n_new = len(mesh_new) - 1
        u_h_new = np.empty((n_new, *u_h.shape[1:]), dtype=u_h.dtype)

        # --- Non-refined intervals: copy directly
        if len(non_refined_map):
            i_old = non_refined_map[:, 0]
            i_new = non_refined_map[:, 1]
            u_h_new[i_new] = u_h[i_old]

        # --- Refined intervals: project onto two subintervals
        if len(refined_map):
            i_old = refined_map[:, 0]
            i_newL = refined_map[:, 1]
            i_newR = refined_map[:, 2]
            M_left, M_right = self.solver.Mprolo

            # use einsum for (n_refine, n_dof, ncomp)
            u_h_old = u_h[i_old]
            u_h_new[i_newL] = np.einsum('jk,ikr->ijr', M_left, u_h_old)
            u_h_new[i_newR] = np.einsum('jk,ikr->ijr', M_right, u_h_old)

        return u_h_new, u_T

#------------------------------------------------------------------
if __name__ == "__main__":
    import ode_examples
    import matplotlib.pyplot as plt
    from Utility import plotting

    # app = ode_examples.Logistic()
    app = ode_examples.PolynomialIntegration(degree=9, ncomp=1)
    # app = ode_examples.Exponential(lam=1.2)
    # app = ode_examples.ExponentialJordan()
    # app = ode_examples.TimeDependentRotation()
    # app = ode_examples.RotScaleForce()
    ode_adapt = ODE_LinAdapt(app, k=3)

    mesh = mesh1d.mesh(app.t_begin, app.t_end, n=4)
    for iter in range(6):
        ucoeff, eta, eta_cell, el2, err_cell = ode_adapt.iteration(mesh)
        t_mp, u_mp = ode_adapt.solver.interpolate_midpoint(mesh, ucoeff)

        mesh_new, refinfo = mesh1d.adapt_mesh(mesh, eta_cell)
        ucoeff_new = ode_adapt.interpolate(ucoeff, mesh, mesh_new, refinfo)
        # print(f"{mesh=}\n{mesh_new=}")
        t_mp_new, u_mp_new = ode_adapt.solver.interpolate_midpoint(mesh_new, ucoeff_new)
        # print(f"{np.linalg.norm(ucoeff[0])=} {np.linalg.norm(ucoeff_new[0])=}")
        mp, up = ode_adapt.solver.evaluate_on_integration_points(mesh, ucoeff[0])
        # print(f"{mp=}")
        for k in range(mp.shape[0]):
            plt.plot(mp[k], up[k], 'b')
        mp, up = ode_adapt.solver.evaluate_on_integration_points(mesh_new, ucoeff_new[0])
        for k in range(mp.shape[0]):
            plt.plot(mp[k], up[k], '--r')

        # for k in range(u_mp.shape[1]):
        #     plt.step(mesh[:-1], u_mp[:, k], '--oy', where='post', label=f"u_mp[{k}]")
        #     plt.step(mesh_new[:-1], u_mp_new[:, k], '--Xk', where='post', label=f"u_mp_new[{k}]")
        plt.grid()
        plt.legend()
        plt.show()

        # plt.step(mesh, np.r_[u_mp[0], u_mp], where='post', label="u_mp")
        # plt.step(mesh_new, np.r_[u_mp_new[0], u_mp_new], where='post', label="u_mp_new")
        # plt.step(t_mp, u_mp, '-o', label="u_mp")
        # plt.step(t_mp_new, u_mp_new, '--x', label="u_mp_new")

        # plt.step(t_mp, eta_cell, label="eta")
        # plt.step(t_mp, err_cell, label="err")
        # plt.grid()
        # plt.legend()
        # plt.show()
        #
        # plt.step(t_mp, 1.0/(mesh[1:]-mesh[:-1]), label="mesh")
        # plt.step(t_mp_new, 1.0/(mesh_new[1:]-mesh_new[:-1]), label="mesh_new")
        # plt.grid()
        # plt.legend()
        # plt.show()

        mesh = mesh_new
