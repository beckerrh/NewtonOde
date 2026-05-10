import linearFEM
from Utility import mesh1d, plotting, printer
from types import SimpleNamespace


class NewtonDriver:
    def __init__(self, app, n0, korder):
        self.mesh = mesh1d.mesh(app.x0, app.x1, n=n0, type='uniform')
        self.solver = linear.LinearElliptic(korder=korder, mesh=self.mesh, app=app)

    def initial_guess(self):
        return self.solver.make_fe_vector()
    def computeMeritFunction(self, u):
        meritvalue, resn = 1, 1
        return SimpleNamespace(meritvalue=meritvalue,
                               norm_X=self.solver.norm(self.mesh, u),
                               resn=resn)
        pass




#------------------------------------------------------------------
if __name__ == "__main__":
    from Newton import newton, newtondata
    import elliptic_examples
    app = elliptic_examples.OscillatoryPoisson(omega=9, alpha=13.0)
    sdata = newtondata.StoppingParamaters(maxiter=500, rtol=1e-6, bt_maxiter=30, bt_c=0.01, bt_omega=0.5, divx=1e20)
    solver = NewtonDriver(app, n0=10, korder=1)
    x0 = solver.initial_guess()
    newton = newton.Newton(nd = solver, verbose=2, sdata=sdata, verbose_bt=False)
    xs, info, printer = newton.solve(x0)
    printer.print_history()


