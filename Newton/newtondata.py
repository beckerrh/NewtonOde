#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 15:38:16 2016

@author: becker
"""

import numpy as np

class StoppingParamaters:
    def __repr__(self):
        return f"maxiter={self.maxiter} atol={self.atol} rtol={self.rtol}"
    def __init__(self, **kwargs):
        self.maxiter = kwargs.pop('maxiter',50)
        self.atol = kwargs.pop('atol',1e-12)
        self.rtol = kwargs.pop('rtol',1e-8)
        self.atoldx = kwargs.pop('atoldx',1e-12)
        self.rtoldx = kwargs.pop('rtoldx',1e-8)
        self.divx = kwargs.pop('divx',1e8)
        self.firststep = 1.0
        self.steptype = kwargs.pop('steptype','backtracking')
        if 'nbase' in kwargs: self.nbase = kwargs.pop('nbase')
        self.bt_maxiter = kwargs.pop('bt_maxiter',20)
        self.bt_omega = kwargs.pop('bt_omega',0.75)
        self.bt_c = kwargs.pop('bt_c',0.01)
        self.maxter_stepsize = 5
        self.dxincrease_max = 1e10

class IterationData:
    def __repr__(self):
        all = [f"{k}: {v}" for k,v in self.__dict__.items()]
        return ' '.join(all)
    def __init__(self, **kwargs):
        self.calls = 0
        self.totaliter, self.totalliniter = 0, 0
        self.calls = 0
        self.bad_convergence_count = 0
    def niter_mean(self):
        if not self.calls: return 1
        return self.totaliter/self.calls
    def niter_lin_mean(self):
        return np.mean(np.array(self.liniter))
    def reset(self, meritvalue):
        self.calls += 1
        if hasattr(self, 'iter'): self.totaliter += self.iter
        if hasattr(self, 'liniter'): self.totalliniter += np.sum(self.liniter)
        self.liniter, self.dxnorm, self.meritvalue, self.step = [], [], [], []
        self.iter = 0
        self.success = True
        self.meritvalue.append(meritvalue)
    def newstep(self, dxnorm, liniter, resnorm, step):
        self.liniter.append(liniter)
        self.dxnorm.append(dxnorm)
        self.meritvalue.append(resnorm)
        self.step.append(step)
        if len(self.dxnorm)>1:
            self.rhodx = self.dxnorm[-1]/self.dxnorm[-2]
        else:
            self.rhodx = 0
        self.rhor = self.meritvalue[-1] / self.meritvalue[-2]
        self.iter += 1
       