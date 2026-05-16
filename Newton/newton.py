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
try:
    from . import armijo
except ImportError:
    import armijo
import Utility
from Utility import logger
from types import SimpleNamespace
import numpy as np

#----------------------------------------------------------------------
class Newton:
    """
    self.nd: Newton Driver supposed to provide
    i) xnew = add_update(x, alpha, dx)
    ii) stat = evaluate(x)
        out : state.residual
            state.meritvalue
            state.norm_X
    iii) step = compute_newton_step(x, state, info)
        out: step.dx
            step.update_norm
            step.liniter
            step.meritgrad
    iv) (for regularized Newton) step = compute_regularized_newton_step(x, state, info)
    v) (for gradient step step = compute_gradient_step(x, state, info)
    """

    #======================================================================
    def __init__(self, **kwargs):
        self.verbose = kwargs.pop('verbose', 1)
        self.nd = kwargs.pop('nd')
        self.sdata = kwargs.pop('sdata', newtondata.StoppingParamaters())
        if hasattr(self.nd, 'globalization') and 'globalization' in kwargs:
            raise KeyError('globalization in kwargs and in newton driver')
        if hasattr(self.nd, 'globalization'):
            self.globalization = self.nd.globalization
        elif "globalization" in kwargs:
            self.globalization = kwargs.pop("globalization")
        else:
            verbose_bt = kwargs.pop('verbose_bt', False)
            maxiter = kwargs.pop('maxiter_bt', 20)
            omega = kwargs.pop('omega_bt', 0.75)
            c = kwargs.pop('c_bt', 1e-4)
            self.globalization = armijo.ArmijoGlobalization(
                maxiter=maxiter, omega=omega, c=c, verbose=verbose_bt
            )
        self.iterdata = kwargs.pop('iterdata', newtondata.IterationData())
        self.name = 'newton   '
        self.name_gs = 'grad step'
        if getattr(self.sdata, 'addname', None):
            self.name += "_" + self.sdata.addname
        types = {
            "it": "i", "mer": "e", "rhomer": "f",
            "|dx|": "e", "rhodx": "f",
            "lin": "i", "|x|": "e", "type": "s",
        }
        types.update(getattr(self.globalization, "log_types", {}))
        self.logger = Utility.logger.Logger(verbose=self.verbose, types=types, name=self.name)
        if self.nd is not None and hasattr(self.nd, 'attach_logger'):
            self.nd.attach_logger(self.logger)

    #----------------------------------------------------------------------
    def fail(self, x, msg, accepted):
        self.iterdata.success = False
        self.iterdata.failure = msg
        self.iterdata.failure += f"\n{accepted.failure}"
        return x, self.iterdata, self.logger

    def log_iteration(self, it, state, step=None, accepted=None):
        vals = {
            "it": it,
            "mer": state.meritvalue,
            "|x|": state.norm_X,
            "type": getattr(step, "step_type", "?"),
        }

        if step is not None:
            vals["|dx|"] = step.dx_norm
            vals["lin"] = getattr(step, "liniter", 0)

        if accepted is not None:
            for key, typ in getattr(self.globalization, "log_types", {}).items():
                if hasattr(accepted, key):
                    vals[key] = getattr(accepted, key)
                elif typ == "i":
                    vals[key] = 0
                else:
                    vals[key] = np.nan

        self.logger.update(**vals)
        self.logger.print()
    # ----------------------------------------------------------------------
    def acceptable_step(self, state, step, accepted):
        if not accepted.success:
            return False
        actual_dx = accepted.alpha * step.dx_norm
        step_type = getattr(step, "step_type", "newton")
        if step_type == "newton":
            if accepted.alpha < 1e-8:
                return False
            if actual_dx < 1e-10 * max(1.0, state.norm_X):
                if accepted.state.meritvalue > 0.99 * state.meritvalue:
                    return False
        return True

    # ----------------------------------------------------------------------
    def build_step(self, x, state, request=None):
        if request is None:
            request = SimpleNamespace(step_type="newton", lambda_reg=0.0)

        if request.step_type == "newton":
            step = self.nd.compute_newton_step(x, state, self.iterdata)
            if not hasattr(step, "x"):
                step.x = x
            step.step_type = "newton"
            return step

        if request.step_type == "regularized_newton":
            step = self.nd.compute_regularized_newton_step(
                x, state, self.iterdata, lambda_reg=request.lambda_reg
            )
            if not hasattr(step, "x"):
                step.x = x
            step.step_type = "reg-newt"
            step.lambda_reg = request.lambda_reg
            return step

        if request.step_type == "gradient":
            step = self.nd.compute_gradient_step(x, state, self.iterdata)
            if not hasattr(step, "x"):
                step.x = x
            step.step_type = "gradient"
            return step

        raise ValueError(f"unknown step_type {request.step_type}")
    # def build_step(self, x, state, request=None):
    #     if request is None:
    #         request = SimpleNamespace(step_type="newton", lambda_reg=0.0)
    #
    #     if request.step_type == "newton":
    #         step = self.nd.compute_newton_step(x, state, self.iterdata)
    #         step.step_type = "newton"
    #         return step
    #
    #     if request.step_type == "regularized_newton":
    #         step = self.nd.compute_regularized_newton_step(
    #             x, state, self.iterdata, lambda_reg=request.lambda_reg
    #         )
    #         step.step_type = "reg-newt"
    #         step.lambda_reg = request.lambda_reg
    #         return step
    #
    #     if request.step_type == "gradient":
    #         step = self.nd.compute_gradient_step(x, state, self.iterdata)
    #         step.step_type = "gradient"
    #         return step
    #
    #     raise ValueError(f"unknown step_type {request.step_type}")
    # ----------------------------------------------------------------------
    def compute_step(self, x, state):
        tol = max(self.sdata.atol, self.sdata.rtol * state.meritvalue)

        self.iterdata.tol_aimed = tol
        self.iterdata.tol_missing = tol / max(state.meritvalue, 1e-300)

        step0 = self.build_step(x, state)

        accepted = self.globalization.accept(
            state=state, step=step0, solver=self, info=self.iterdata
        )

        step = getattr(accepted, "step", step0)

        if self.acceptable_step(state, step, accepted):
            return step, accepted

        if hasattr(self.nd, "compute_gradient_step"):
            step = self.build_step(
                x, state,
                request=SimpleNamespace(step_type="gradient")
            )
            accepted = self.globalization.accept(
                state=state, step=step, solver=self, info=self.iterdata
            )
            step = getattr(accepted, "step", step)

        return step, accepted
    # ----------------------------------------------------------------------
    def solve(self, x):
        state = self.nd.evaluate(x)
        self.iterdata.reset(state.meritvalue)
        tol = max(self.sdata.atol, self.sdata.rtol * state.meritvalue)
        toldx = max(self.sdata.atoldx, self.sdata.rtoldx * state.norm_X)
        self.logger.print_names()

        for it in range(self.sdata.maxiter):
            self.iterdata.iter = it
            if state.meritvalue < tol:
                self.iterdata.success = True
                self.iterdata.failure = None
                return x, self.iterdata, self.logger

            step, accepted = self.compute_step(x, state)

            if hasattr(self.nd, "call_back"):
                self.nd.call_back(self.iterdata, accepted)

            if not accepted.success:
                return self.fail(x, "globalization failed", accepted)

            x = accepted.x
            state = accepted.state

            self.log_iteration(it, state, step, accepted)

        return self.fail(x, "maxiter exceeded", accepted)

# ------------------------------------------------------ #

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import test_problems, trust_regularization, newton_driver_numpy
    import inspect

    problems = []
    for name, cls in inspect.getmembers(test_problems, inspect.isclass):
        if issubclass(cls, test_problems.TestProblem) and cls is not test_problems.TestProblem:
            problems.append(cls)
    instances = [cls() for cls in problems]
    failed = []
    for instance in instances:
        print(f"\n ---------- {instance.name} ----------")
        globalization = armijo.ArmijoGlobalization()
        # globalization = armijo.WolfeGlobalization()
        # globalization = trust_regularization.TrustRegularizationGlobalization()
        nd = newton_driver_numpy.NewtonDriverNumpy(test_problem=instance)
        newton = Newton(nd=nd, globalization=globalization)
        xs, info, logger = newton.solve(instance.x0)
        if not info.success:
            failed.append([instance.name,info.failure])
            continue
        if xs.shape[0] ==1:
            x0 = instance.x0
            xs0 = float(np.ravel(xs)[0])
            x00 = float(np.ravel(instance.x0)[0])
            x = np.linspace(min(xs0, x00) - 2, max(xs0, x00) + 2)
            plt.plot(x, nd.F(x), [x[0], x[-1]], [0, 0], '--r')
            plt.plot(xs0, nd.F(xs0), 'Xk')
            plt.plot(x00, nd.F(x00), 'Xg')
            plt.title(instance.name)
            plt.grid()
            plt.show()
    for fail in failed:
        print(f"{fail[0]} ---> {fail[1]}")
