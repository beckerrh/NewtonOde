#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 15:38:16 2016

@author: becker
"""
try:
    from . import newtondata
except ImportError:
    import newtondata
from Utility import printer
from types import SimpleNamespace
import numpy as np

#----------------------------------------------------------------------
class Newton:
    """
    self.nd: Newton Driver supposed to provide
    """
    def __init__(self, **kwargs):
        self.verbose = kwargs.pop('verbose', True)
        self.verbose_bt = kwargs.pop('verbose_bt', False)
        self.nd = kwargs.pop('nd', None)
        self.sdata = kwargs.pop('sdata', newtondata.StoppingParamaters())
        self.iterdata = kwargs.pop('iterdata', newtondata.IterationData())
        self.name = 'newton   '
        self.name_gs = 'grad step'
        if hasattr(self.sdata,'addname'): self.name += '_' + self.sdata.addname
        types = {'name':'s', "it":'i', '|r|':'e', "|dx|":'e', "|x|":'e','rhodx':'f', 'rhor':'f', 'lin':'i', 'bt':'i', 'r':'f'}
        self.printer = printer.Printer(verbose=self.verbose, types=types)

    #----------------------------------------------------------------------
    def backtracking(self, x0, dx, meritfirst, meritgrad, step=1.0):
        maxiter, omega, c = self.sdata.bt_maxiter, self.sdata.bt_omega, self.sdata.bt_c
        result = SimpleNamespace(success=True)
        for it in range(maxiter):
            if it==0:
                if self.verbose_bt:
                    print("{} {:>3} {:^10} {:^10}  {:^9}".format("bt", "it", "meritnew", "meritfirst", "step"))
            else:
                step *= omega
            x = x0 + step * dx
            residual_result = self.nd.computeResidual(x)
            #     return SimpleNamespace(success=True, r=result.residual, rnorm=result.residual_norm,
            #                            xnorm=result.x_norm, merit=result.meritvalue)
            # residual_result = self.computeResidual(x)
            meritnew = residual_result.meritvalue
            if self.verbose_bt:
                print(f"bt {it:3} {meritnew:10.3e} {meritfirst:10.3e} {step:9.2e}")
            if meritnew <= meritfirst + c*step*meritgrad:
                result.x = x
                result.step = step
                result.iter = it
                result.residual_result = residual_result
                return result
        result.success = False
        result.x = x0
        result.step = step
        result.iter = maxiter
        return result

    # #--------------------------------------------------------------------
    # def computeResidual(self, x):
    #     result = self.nd.computeResidual(x)
    #     return SimpleNamespace(success=True, r=result.residual, rnorm=result.residual_norm,
    #                            xnorm=result.x_norm, merit=result.meritvalue)
    # --------------------------------------------------------------------
    def solve(self, x0, **kwargs):
        """
        Aims to solve F(x) = 0, starting at x0
        with backtracking Newton
        """
        x0 = np.atleast_1d(x0)
        atol, rtol, atoldx, rtoldx = self.sdata.atol, self.sdata.rtol, self.sdata.atoldx, self.sdata.rtoldx
        divx = self.sdata.divx
        maxiter: int = kwargs.pop('maxiter', self.sdata.maxiter)
        x = np.atleast_1d(x0)
        # if not x.ndim == 1:
        #     raise ValueError(f"{x.shape=}")
        self.iterdata.bad_convergence = False
        self.iterdata.success = True
        for iteration in range(maxiter):
            if iteration==0:
                # for iteration>0 this is done in back-tracking
                result = self.nd.computeResidual(x)
                res, resnorm, meritvalue, xnorm = result.residual, result.residual_norm, result.meritvalue, result.x_norm
                self.iterdata.reset(resnorm)
                tol = max(atol, rtol * resnorm)
                toldx = max(atoldx, rtoldx * xnorm)
                if self.iterdata.totaliter == 0:
                    self.printer.print_names()
                self.printer.values['it'] = self.iterdata.iter
                self.printer.values['|r|'] = resnorm
                self.printer.values['name'] = ''
                self.printer.print()
                if resnorm < tol:
                    self.iterdata.success = True
                    return x, self.iterdata
            else:
                self.iterdata.tol_missing = tol / resnorm
            result = self.nd.computeUpdate(r=res, x=x, info=self.iterdata)
            success = getattr(result, 'success', True)
            if not success:
                self.iterdata.success = False
                self.iterdata.failure = 'linear solver did not converge'
                return x, self.iterdata
            liniter = getattr(result, 'liniter', 0)
            dx = result.update
            dxnorm_old = getattr(self, 'dxnorm', xnorm)
            dxnorm = result.update_norm
            gradient_iteration = iteration and (dxnorm>20.0*dxnorm_old)
            btit=0
            if not gradient_iteration:
                meritgrad = result.meritgrad
                btresult = self.backtracking(x, dx, meritvalue, meritgrad)
                x, step = btresult.x, btresult.step
                btit += btresult.iter
                if btresult.success:
                    rr = btresult.residual_result
                    res, resnorm, xnorm, meritvalue = rr.residual, rr.residual_norm, rr.x_norm, rr.meritvalue
                else:
                    gradient_iteration = True
            if gradient_iteration:
                # print(f"=======Gradient step======")
                result = self.nd.computeResidual(x)
                #     return SimpleNamespace(success=True, r=result.residual, rnorm=result.residual_norm,
                #                            xnorm=result.x_norm, merit=result.meritvalue)
                # result = self.computeResidual(x)
                res, resnorm, meritvalue, xnorm = result.residual, result.residual_norm, result.meritvalue, result.x_norm
                result = self.nd.computeUpdateSimple(r=res, x=x, info=self.iterdata)
                dx = result.update
                dxnorm = result.update_norm
                meritgrad = result.meritgrad
                bt_maxiter = self.sdata.bt_maxiter
                self.sdata.bt_maxiter = 50
                step =  getattr(self, 'step_grad', 1.0)
                btresult = self.backtracking(x, dx, meritvalue, meritgrad, step)
                x = btresult.x
                self.step_grad = 2.0*btresult.step
                res = btresult.residual_result.residual
                resnorm = btresult.residual_result.residual_norm

                btit += btresult.iter
                self.sdata.bt_maxiter = bt_maxiter
                if not btresult.success:
                    self.iterdata.success = False
                    self.iterdata.failure = 'backtracking did not converge'
                    return x, self.iterdata
            # dxnorm = np.linalg.norm(dx)
            self.iterdata.newstep(dxnorm, liniter, resnorm, step)
            xnorm = np.linalg.norm(x)
            self.printer.values['it'] = self.iterdata.iter
            self.printer.values['|r|'] = resnorm
            self.printer.values['|dx|'] = self.iterdata.dxnorm[-1]
            self.printer.values['|x|'] = xnorm
            self.printer.values['rhodx'] = self.iterdata.rhodx
            self.printer.values['rhor'] = self.iterdata.rhor
            self.printer.values['lin'] = liniter
            self.printer.values['bt'] = btit
            if gradient_iteration:
                self.printer.values['name'] = self.name_gs
            else:
                self.printer.values['name'] = self.name
            self.printer.print()
            if hasattr(self.nd, 'call_back'):
                self.nd.call_back(x)
            if resnorm<tol:
                return x, self.iterdata
            if xnorm >= divx:
                self.iterdata.success = False
                self.iterdata.failure = 'divx'
                return x, self.iterdata
        self.iterdata.success = False
        self.iterdata.failure = 'maxiter exceded'
        return x, self.iterdata


# ------------------------------------------------------ #

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import test_problems, inspect
    problems = []
    for name, cls in inspect.getmembers(test_problems, inspect.isclass):
        if issubclass(cls, test_problems.TestProblem) and cls is not test_problems.TestProblem:
            problems.append(cls)
    instances = [cls() for cls in problems]
    problems = ["Cubic_Inflection"]
    instances = [getattr(test_problems, name)() for name in problems]
    for instance in instances:
        print(f"\n ---------- {instance.name} ----------")
        nd = test_problems.NewtonDriverNumpy(test_problem=instance)
        newton = Newton(nd=nd)
        # newton.verbose_bt = True
        xs, info = newton.solve(instance.x0, maxiter=50)
        if not info.success:
            print(f"@@@@@@@@@@@@@@@@ {info.failure}")
        if xs.shape[0] ==1:
            x0 = instance.x0
            x = np.linspace(min(xs,x0)-2, max(xs,x0)+2)
            plt.plot(x, nd.F(x), [x[0], x[-1]], [0, 0], '--r')
            plt.plot(xs, nd.F(xs), 'Xk')
            plt.plot(x0, nd.F(x0), 'Xg')
            plt.title(instance.name)
            plt.grid()
            plt.show()
