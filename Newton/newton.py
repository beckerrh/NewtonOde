#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 15:38:16 2016

@author: becker
"""
from backend import np
import newtondata, utils

#----------------------------------------------------------------------
class Newton:
    def __init__(self, **kwargs):
        self.computedx = kwargs.pop('computedx', None)
        self.verbose = kwargs.pop('verbose', True)
        self.verbose_bt = kwargs.pop('verbose_bt', False)
        self.F = kwargs.pop('F', None)
        self.dF = kwargs.pop('dF', None)
        if not self.computedx:  assert self.dF
        self.sdata = kwargs.pop('sdata', newtondata.StoppingParamaters())
        self.iterdata = kwargs.pop('iterdata', newtondata.IterationData())
        self.name = 'newton'
        if hasattr(self.sdata,'addname'): self.name += '_' + self.sdata.addname
        printformat = {self.name:'s', "it":'i', '|r|':'e', "|dx|":'e', "|x|":'e','rhodx':'f', 'rhor':'f', 'lin':'i', 'bt':'i', 'r':'f'}
        self.printvalues = {k:k if v=='s' else 0 for k,v in printformat.items()}
        self.printer = utils.Printer(verbose=self.verbose, types=printformat)

    #----------------------------------------------------------------------
    def backtracking(self, x0, dx, meritvalue, meritgrad):
        maxiter, omega, c = self.sdata.bt_maxiter, self.sdata.bt_omega, self.sdata.bt_c
        step = 1
        x = x0 + step*dx
        res = self.F(x)
        meritnew = 0.5*np.sum(res*res)
        resnorm = np.sqrt(2*meritnew)
        resfirst = np.sqrt(2*meritvalue)
        it = 0
        if self.verbose_bt:
            print("{} {:>3} {:^10} {:^10}  {:^9}".format("bt", "it", "resnorm", "resfirst", "step"))
            print(f"bt {it:3} {resnorm:10.3e} {resfirst:10.3e} {step:9.2e}")
        while meritnew > meritvalue + c*step*meritgrad and it<maxiter:
            it += 1
            step *= omega
            x = x0 + step * dx
            res = self.F(x)
            resnorm = np.linalg.norm(res)
            if self.verbose_bt:
                print(f"bt {it:3} {resnorm:10.3e} {resfirst:10.3e} {step:9.2e}")
        return x, res, resnorm, step, (it<maxiter), it

    #--------------------------------------------------------------------
    def solve(self, x0, maxiter=None):
        """
        Aims to solve f(x) = 0, starting at x0
        computedx: gets dx from f'(x) dx =  -f(x)
        if not given, jac is called and linalg.solve is used
        """
        x0 = np.asarray(x0)
        atol, rtol, atoldx, rtoldx = self.sdata.atol, self.sdata.rtol, self.sdata.atoldx, self.sdata.rtoldx
        divx = self.sdata.divx
        if maxiter is None: maxiter = self.sdata.maxiter
        x = np.asarray(x0)
        if not x.ndim == 1:
            raise ValueError(f"{x.shape=}")
        xnorm = np.linalg.norm(x)
        res = self.F(x)
        meritvalue = 0.5*np.sum(res*res)
        print(f"{meritvalue=} {res=}")
        resnorm = np.sqrt(2*meritvalue)
        self.iterdata.reset(resnorm)

        # print(f"@@@@--------- {np.linalg.norm(x)=} {resnorm=}")
        tol = max(atol, rtol*resnorm)
        toldx = max(atoldx, rtoldx*xnorm)
        # dx, step, resold = None, None, np.zeros_like(res)
        if self.iterdata.totaliter==0:
            self.printer.print_names()
        self.iterdata.bad_convergence = False
        self.iterdata.success = True
        while(resnorm>tol  and self.iterdata.iter < maxiter):
            if resnorm<atol:
                self.iterdata.success = True
                return x, self.iterdata
            self.iterdata.tol_missing = tol/resnorm
            if not self.computedx:
                J = np.atleast_2d(self.dF(x))
                dx, liniter, success = np.linalg.solve(J, -res), 1, True
            else:
                dx, liniter, success = self.computedx(-res, x, self.iterdata)
            if not success:
                self.iterdata.success = False
                self.iterdata.failure = 'linear solver did not converge'
                return x, self.iterdata
            assert dx.shape == x0.shape
            if np.linalg.norm(dx) < self.sdata.atoldx:
                self.iterdata.success = False
                self.iterdata.failure = 'correction too small'
                return x, self.iterdata
            x, res, resnorm, step, btok, btit = self.backtracking(x, dx, meritvalue, -meritvalue)
            if not btok:
                dx = -J.T@res
                meritgrad = 0.5*np.sum(dx*dx)
                x, res, resnorm, step, btok, btit2 = self.backtracking(x, dx, meritvalue, -meritgrad)
                btit += btit2+1
                if not btok:
                    self.iterdata.success = False
                    self.iterdata.failure = 'backtracking did not converge'
                    return x, self.iterdata
            dxnorm = np.linalg.norm(dx)
            self.iterdata.newstep(dxnorm, liniter, resnorm, step)
            xnorm = np.linalg.norm(x)
            matsymb = ''
            self.iterdata.bad_convergence = False
            if self.iterdata.rhodx>self.sdata.rho_aimed:
                self.iterdata.bad_convergence = True
                self.iterdata.bad_convergence_count += 1
                matsymb = 'M'
            self.printvalues['it'] = self.iterdata.iter
            self.printvalues['|r|'] = resnorm
            self.printvalues['|dx|'] = self.iterdata.dxnorm[-1]
            self.printvalues['|x|'] = xnorm
            self.printvalues['rhodx'] = self.iterdata.rhodx
            self.printvalues['rhor'] = self.iterdata.rhor
            self.printvalues['lin'] = liniter
            self.printvalues['bt'] = btit
            self.printer.print(self.printvalues)
            if resnorm<atol:
                self.iterdata.success = True
                # iterdata.failure = 'residual too small'
                return x, self.iterdata
            if self.iterdata.iter == maxiter:
                self.iterdata.success = False
                self.iterdata.failure = 'maxiter exceded'
                return x, self.iterdata
            if xnorm >= divx:
                self.iterdata.success = False
                self.iterdata.failure = 'divx'
                return x, self.iterdata
        return x, self.iterdata


# ------------------------------------------------------ #

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import test_problems, inspect
    problems = []
    for name, cls in inspect.getmembers(test_problems, inspect.isclass):
        if issubclass(cls, test_problems.TestProblem) and cls is not test_problems.TestProblem:
            problems.append(cls)
    instances = [cls() for cls in problems]
    problems = ["Powell_Singular"]
    instances = [getattr(test_problems, name)() for name in problems]
    for instance in instances:
        print(f"\n******* Testing {instance.name} *******\n")
        newton = Newton(F=instance.F, dF=instance.dF)
        info = newton.solve(instance.x0)

    # f = lambda x: 10.0 * np.sin(2.0 * x) + 4.0 - x * x
    # df = lambda x: 20.0 * np.cos(2.0 * x) - 2.0 * x
    # f = lambda x: x**2 -11
    # df = lambda x: 2.0 * x
    # def computedx(r, x, info):
    #     return r/df(x),1, True
    # x0 = [3.]
    # newton = Newton(df=df)
    # info = newton.solve(x0, f)
    # newton = Newton(computedx=computedx)
    # info2 = newton.solve(x0, f)
    # x = np.linspace(-1., 4.0)
    # plt.plot(x, f(x), [x[0], x[-1]], [0,0], '--r')
    # plt.show()
