import os
import numpy as np
from dataclasses import dataclass, field

def _check1setinother_(set1, set2, name1="set1", name2="set2"):
    notin2 = set1.difference(set2)
    if notin2:
        raise KeyError(f"in '{name1}' but not in '{name2}': '{notin2}' {set1=} {set2=}")
def _check2setsequal_(set1, set2, name1="set1", name2="set2"):
    _check1setinother_(set1, set2, name1, name2)
    _check1setinother_(set2, set1, name2, name1)



# ---------------------------------------------------------------- #
class TrackedDict(dict):
    def get(self, *args, **kwargs):
        raise RuntimeError("Use Params.lookup(...) instead of dict.get(...)")

class Params:
    def __init__(self):
        self.fct_glob = TrackedDict()
        self.scal_glob = TrackedDict()
        self.scal_cells = TrackedDict()
        self.scal_celllabels = TrackedDict()
        self.data = TrackedDict()
        self._used = {}

    def begin_usage_tracking(self):
        self._used = {
            "fct_glob": set(),
            "scal_glob": set(),
            "scal_cells": set(),
            "scal_celllabels": set(),
            "data": set(),
        }

    def lookup(self, name, categories):
        found = []

        for category in categories:
            d = getattr(self, category)
            if name in d:
                found.append((category, d[name]))

        if len(found) > 1:
            cats = [c for c, _ in found]
            raise ValueError(f"{name!r} given more than once in {cats}")

        if not found:
            return None, None

        category, value = found[0]
        self._used.setdefault(category, set()).add(name)
        return category, value

    def unused_message(self):
        parts = []

        for category in [
            "fct_glob",
            "scal_glob",
            "scal_cells",
            "scal_celllabels",
            "data",
        ]:
            d = getattr(self, category)
            unused = set(d) - self._used.get(category, set())
            if unused:
                parts.append(f"{category} keys={sorted(unused)}")

        return "; ".join(parts)

# ---------------------------------------------------------------- #
class BoundaryConditions(object):
    """
    Information on boundary conditions
    type: dictionary int->string
    fct: dictionary int->callable
    param: dictionary int->float
    Information can be set with the 'set' function
    """
    def __init__(self, colors=None):
        self.ignore = False
        if colors is None:
            self.type = {}
            self.fct = {}
            self.param = {}
        else:
            self.type = {color: None for color in colors}
            self.fct = {color: None for color in colors}
            self.param = {color: None for color in colors}
    def __repr__(self):
        return f"types={self.type}\nfct={self.fct}\nparam={self.param}"
    def clear(self):
        self.type = {}
        self.fct = {}
        self.param = {}
    def convert(self, mesh):
        if not hasattr(mesh,'labeldict_i2s'): return
        self.type = {mesh.labeldict_s2i[k]:v for k,v in self.type.items()}
        self.fct = {mesh.labeldict_s2i[k]: v for k, v in self.fct.items()}
        self.param = {mesh.labeldict_s2i[k]: v for k, v in self.param.items()}

    def colors(self):
        return self.type.keys()
    def types(self):
        return self.type.values()
    def set(self, type, colors, fcts=None):
        if isinstance(colors, (int,str)): colors = [colors]
        assert isinstance(colors, (list,tuple))
        for i,color in enumerate(colors):
            if color in self.type.keys(): raise ValueError(f"Attempt to define {color=} for {type=}, but already defined b.c {color} as {self.type[color]=}")
            self.type[color] = type
            if fcts: self.fct[color] = fcts[i]
    def colorsOfType(self, types):
        if isinstance(types, str): types = [types]
        colors = []
        for color, typeofcolor in self.type.items():
            if typeofcolor in types: colors.append(color)
        return colors
    def check(self, colors):
        if self.ignore: return
        colors = set(colors)
        typecolors = set(self.type.keys())
        if not len(typecolors):
            raise ValueError(f"*** application should define 'BoundaryConditions' in 'defineProblemData(self, problemdata)'")
        if colors != typecolors: 
            raise ValueError(f"*** problem in boundary conditions mesh {colors=} colors with b.c.={typecolors}")
        # _check2setsequal_(colors, typecolors, "mesh colors", "types")

# ---------------------------------------------------------------- #
class PostProcess(object):
    """
    Information on postprocess
    type: dictionary string(name)->string
    color: dictionary string(name)->list(int)
    """
    def __init__(self):
        self.type = {}
        self.color = {}
    def __repr__(self):
        return f"types={self.type}\ncolor={self.color}"
    def clear(self):
        self.type = {}
        self.color = {}
    def set(self, name, type, colors):
        if isinstance(colors, int): colors=[colors]
        self.type[name] = type
        self.color[name] = colors
    def colors(self, name):
        return self.color[name]
    def check(self, colors):
        if self.type.keys() != self.color.keys():
            raise KeyError(f"postprocess keys differ: type={self.type.keys()}, color={self.color.keys()}")
        colors = set(colors)
        usedcolors = set().union(*self.color.values())
        _check1setinother_(usedcolors, colors, "used", "mesh colors")
    def colorsOfType(self, type):
        colors = []
        for n,t in self.type.items():
            if t == type: colors.extend(self.color[n])
        return colors

class ProblemData(object):
    """
    Contains data for definition of a problem:
    - boundary conditions
    - right-hand sides
    - exact solution (if ever)
    - postprocess
    - params: class Params
    """
    def __init__(self, bdrycond=None, rhs=None, rhscell=None, rhspoint = None, postproc=None):
        if bdrycond is None: self.bdrycond = BoundaryConditions()
        else: self.bdrycond = bdrycond
        if postproc is None: self.postproc = PostProcess()
        else: self.postproc = postproc
        self.solexact = None
        self.params = Params()

    def __bool__(self):
        return bool(self.params)
    def _split2string(self, string, sep='\n\t\t'):
        return sep+sep.join(str(string).split('\n'))

    def __repr__(self):
        repr = f"\n{self.__class__}:"
        repr += f"\n\tbdrycond:{self._split2string(self.bdrycond)}"
        repr += f"\n\tpostproc:{self._split2string(self.postproc)}"
        # if self.rhs: repr += f"\n\trhs={self.rhs}"
        # if self.rhscell: repr += f"\n\trhscell={self.rhscell}"
        # if self.rhspoint: repr += f"\n\trhspoint={self.rhspoint}"
        if self.solexact: repr += f"\n\tsolexact={self.solexact}"
        repr += f"\n\tparams:{self._split2string(self.params)}"
        return repr

    def check(self, mesh):
        colors = mesh.labels.boundary.keys()
        self.bdrycond.convert(mesh)
        self.bdrycond.check(colors)
        colors = list(colors)
        colors.extend(list(mesh.labels.vertex.keys()))
        colors.extend(list(mesh.labels.line.keys()))
        if self.postproc: self.postproc.check(colors)
        self.params.check(mesh)

    def clear(self):
        """
        keeps the boundary condition types and parameters !
        """
        self.solexact = None
        self.postproc = None
        for color in self.bdrycond.fct:
            self.bdrycond.fct[color] = None
