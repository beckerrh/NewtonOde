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
        self.verbose = kwargs.pop('verbose', 1)
        self.verbose_bt = kwargs.pop('verbose_bt', False)
        self.nd = kwargs.pop('nd', None)
        self.sdata = kwargs.pop('sdata', newtondata.StoppingParamaters())
        self.iterdata = kwargs.pop('iterdata', newtondata.IterationData())
        self.name = 'newton   '
        self.name_gs = 'grad step'
        if getattr(self.sdata, 'addname', None):
            self.name += "_" + self.sdata.addname
        types = {"it":'i', 'bt':'i', 'mer':'e', 'rhomer':'f', "|dx|":'e','rhodx':'f', 'lin':'i', "|x|":'e'}
        self.printer = printer.Printer(verbose=self.verbose, types=types, name=self.name)
        if self.nd is not None and hasattr(self.nd, 'attach_printer'):
            self.nd.attach_printer(self.printer)
        if self.verbose_bt:
            types = {'it': 'i', "merit": 'e', "meritfirst": 'e', "meritgrad": 'e', "step": 'e', 'aimed':'e'}
            self.printer_bt = printer.Printer(types, name='bt')

    #----------------------------------------------------------------------
    def backtracking(self, x0, dx, meritfirst, meritgrad, step=1.0):
        maxiter, omega, c = self.sdata.bt_maxiter, self.sdata.bt_omega, self.sdata.bt_c
        result = SimpleNamespace(success=True)
        if hasattr(self, 'printer_bt'):
            self.printer_bt.values['meritfirst'] = meritfirst
            self.printer_bt.values['meritgrad'] = meritgrad
        for it in range(maxiter):
            if it==0:
                if hasattr(self,'printer_bt'):
                    self.printer_bt.print_names()
            else:
                step *= omega
            if hasattr(self.nd,'add_update'):
                x = self.nd.add_update(x0, step, dx)
            else:
                x = x0 + step * dx
            merit_result = self.nd.computeMeritFunction(x)
            merit = merit_result.meritvalue
            if hasattr(self, 'printer_bt'):
                self.printer_bt.values['it'] = it
                self.printer_bt.values['merit'] = merit
                self.printer_bt.values['step'] = step
                self.printer_bt.values['aimed'] = meritfirst + c*step*meritgrad
                self.printer_bt.print()
            if merit <= meritfirst + c*step*meritgrad:
                result.x = x
                # result.xnorm = xnorm
                result.step = step
                result.iter = it
                result.merit_result = merit_result
                return result
        result.success = False
        result.x = x0
        result.step = step
        result.iter = maxiter
        return result

    def solve(self, x, **kwargs):
        """
        Aims to solve F(x) = 0, starting at x0
        with backtracking Newton
        """
        # x0 = np.atleast_1d(x0)
        atol, rtol, atoldx, rtoldx = self.sdata.atol, self.sdata.rtol, self.sdata.atoldx, self.sdata.rtoldx
        divx = self.sdata.divx
        maxiter: int = kwargs.pop('maxiter', self.sdata.maxiter)
        # x = np.atleast_1d(x0)
        # x = x0
        # if not x.ndim == 1:
        #     raise ValueError(f"{x.shape=}")
        self.iterdata.bad_convergence = False
        self.iterdata.success = True
        for iteration in range(maxiter):
            if iteration==0:
                # for iteration>0 this is done in back-tracking
                if hasattr(self.nd,'computeResidual'):
                    result_merit = self.nd.computeResidual(x)
                else:
                    result_merit = self.nd.computeMeritFunction(x)
                res = getattr(result_merit, 'residual', None)
                meritvalue, xnorm = result_merit.meritvalue, result_merit.x_norm
                self.iterdata.reset(meritvalue)
                tol = max(atol, rtol * meritvalue)
                toldx = max(atoldx, rtoldx * xnorm)
                if self.iterdata.totaliter == 0:
                    self.printer.print_names()
                self.printer.values['it'] = self.iterdata.iter
                self.printer.values['mer'] = meritvalue
                self.printer.print()
                if meritvalue < tol:
                    self.iterdata.success = True
                    return x, self.iterdata
            #else:
            self.iterdata.tol_missing = tol / meritvalue
            self.iterdata.tol_aimed = tol
            result = self.nd.computeUpdate(r=res, x=x, info=self.iterdata, result_merit=result_merit)
            if hasattr(result,'x'): x= result.x
            success = getattr(result, 'success', True)
            if not success:
                self.iterdata.success = False
                self.iterdata.failure = 'linear solver did not converge'
                return x, self.iterdata
            liniter = getattr(result, 'liniter', 0)
            dx = result.update
            dxnorm_old = getattr(self, 'dxnorm', xnorm)
            dxnorm = result.update_norm
            exploding = dxnorm>self.sdata.dxincrease_max*dxnorm_old
            gradient_iteration = iteration and exploding
            btit=0
            if hasattr(self.nd,'update_rule'):
                x,step=self.nd.update_rule(x, dx)
            else:
                if not gradient_iteration:
                    if hasattr(result,'meritgrad'):
                        meritgrad = result.meritgrad
                    else:
                        meritgrad = self.nd.computeMeritGrad(x, dx, result_merit)
                    btresult = self.backtracking(x, dx, meritvalue, meritgrad)
                    x, step = btresult.x, btresult.step
                    btit += btresult.iter
                    if btresult.success:
                        result_merit = btresult.merit_result
                        xnorm, meritvalue = result_merit.x_norm, result_merit.meritvalue
                    else:
                        gradient_iteration = True
                if gradient_iteration:
                    print(f"{btresult.success=}")
                    if hasattr(self.nd, 'call_back_backtrack_failed'):
                        kwargs = {'btresult':btresult, 'dx':dx, 'r':res, 'dxnorm':dxnorm, 'dxnorm_old':dxnorm_old}
                        self.nd.call_back_backtrack_failed(**kwargs)
                    result = self.nd.computeResidual(x)
                    res, resnorm, meritvalue, xnorm = result.residual, result.residual_norm, result.meritvalue, result.norm_X
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
            self.iterdata.newstep(dxnorm, liniter, meritvalue, step)
            # xnorm = np.linalg.norm(x)
            self.printer.update(it=self.iterdata.iter, mer=meritvalue, rhomer=self.iterdata.rhor,
                                rhodx=self.iterdata.rhodx, bt=btit, lin=liniter)
            self.printer.values['|dx|'] = self.iterdata.dxnorm[-1]
            self.printer.values['|x|'] = xnorm
            if gradient_iteration:
                self.printer.name = self.name_gs
            else:
                self.printer.name = self.name
            self.printer.print()
            if hasattr(self.nd, 'call_back'):
                kwargs = {'x':x, 'iterdata':self.iterdata}
                self.nd.call_back(**kwargs)
            if meritvalue<tol:
                return x, self.iterdata, self.printer
            if xnorm >= divx:
                self.iterdata.success = False
                self.iterdata.failure = 'divx'
                return x, self.iterdata, self.printer
        self.iterdata.success = False
        self.iterdata.failure = 'maxiter exceded'
        return x, self.iterdata, self.printer


# ------------------------------------------------------ #

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import test_problems, inspect
    problems = []
    for name, cls in inspect.getmembers(test_problems, inspect.isclass):
        if issubclass(cls, test_problems.TestProblem) and cls is not test_problems.TestProblem:
            problems.append(cls)
    # instances = [cls() for cls in problems]
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
