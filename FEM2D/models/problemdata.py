import os
import numpy as np

def _check1setinother_(set1, set2, name1="set1", name2="set2"):
    notin2 = set1.difference(set2)
    if notin2:
        raise KeyError(f"in '{name1}' but not in '{name2}': '{notin2}' {set1=} {set2=}")
def _check2setsequal_(set1, set2, name1="set1", name2="set2"):
    _check1setinother_(set1, set2, name1, name2)
    _check1setinother_(set2, set1, name2, name1)

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

# ---------------------------------------------------------------- #
class Params(object):
    """
    Holds all parameters for a problem:
    - fct_glob: dictionary name -> function
    - scal_glob: dictionary name -> float
    - scal_cells dictionary name -> color -> float
    """
    def __init__(self):
        self.fct_glob = {}
        self.scal_glob = {}
        self.scal_cells = {}
        self.data = {}
    def __repr__(self):
        repr = ""
        if len(self.fct_glob): repr += f" fct_glob={self.fct_glob}"
        if len(self.scal_glob): repr += f" scal_glob={self.scal_glob}"
        if len(self.scal_cells): repr += f" scal_cells={self.scal_cells}"
        return repr[1:]
    def set_scal_cells(self, name, colors, value):
        if not name in self.scal_cells: self.scal_cells[name]={}
        for color in colors: self.scal_cells[name][color] = value
    def check(self, mesh):
        for name in self.scal_cells:
            _check2setsequal_(set(self.scal_cells[name]), set(mesh.cellsoflabel.keys()), "scal_cells", "mesh.cellsoflabel")
        for name in self.scal_glob:
            if name in self.scal_cells: raise ValueError(f"key '{name}' given twice")
            if name in self.fct_glob: raise ValueError(f"key '{name}' given twice")
            if not isinstance(self.scal_glob[name], (int,float)):
                raise ValueError(f"in 'scal_glob' key '{name}' doesnt have floats but is {self.scal_glob[name]}")
        for name in self.scal_cells:
            if name in self.scal_glob: raise ValueError(f"key '{name}' given twice")
            if name in self.fct_glob: raise ValueError(f"key '{name}' given twice")
        for name in self.fct_glob:
            if name in self.scal_cells: raise ValueError(f"key '{name}' given twice")
            if name in self.scal_glob: raise ValueError(f"key '{name}' given twice")
    def paramdefined(self, name):
        return name in self.scal_glob or name in self.scal_cells or name in self.fct_glob
    def update(self, p):
        for k,v in p.fct_glob.items(): self.fct_glob[k] = v
        for k,v in p.scal_glob.items(): self.scal_glob[k] = v
        for k,v in p.scal_cells.items(): self.scal_cells[k] = v
        for k,v in p.data.items(): self.data[k] = v



# ---------------------------------------------------------------- #
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
        colors = mesh.bdrylabels.keys()
        self.bdrycond.convert(mesh)
        self.bdrycond.check(colors)
        colors = list(colors)
        colors.extend(list(mesh.verticesoflabel.keys()))
        colors.extend(list(mesh.linesoflabel.keys()))
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

# ---------------------------------------------------------------- #
class Results(object):
    """
    Contains results from an model:
    - point_data, side_data, cell_data, gobal_data
    - info on iteration
    """
    # TODO: data per physical identity
    def __init__(self, nframes=None):
        # self.data = {"point":{}, "side":{}, "cell":{}, "global":{}}
        self.data = {}
        self.info = {}
        if nframes:
            self.time = np.zeros(nframes+1)
            self.iter = np.zeros(nframes+1)
            self.liniter = np.zeros(nframes+1)
            self.postprocscal = {}
    def __repr__(self):
        repr = f"Results info: {self.info}\n"
        repr += f"Results data:\n"
        for k, d in self.data.items():
            repr += f"type({k})\n"
            for p, v in d.items():
                repr += f"{p}: {v}\n"
        return repr
    def setData(self, data, timer=None, iter=None):
        # for static
        self.data.update(data)
        if timer is not None: self.info['timer'] = timer
        if iter is not None: self.info['iter'] = iter
    def addData(self, iframe, data, time, iter, liniter):
        # for dynamic
        # only take scalar data
        self.time[iframe] = time
        self.iter[iframe] = iter
        self.liniter[iframe] = liniter
        if not len(self.postprocscal.keys()):
            nframes = len(self.time)
            for k in data['scalar'].keys():
                self.postprocscal[k] = np.zeros(nframes)
        for k,v in data['scalar'].items():
            self.postprocscal[k][iframe] = v
    def enrolPPscalar(self, keys):
        for k in keys:
            self.postprocscal[k] = []
    def addPPscalar(self, postprocscal):
        for k,v in postprocscal.items():
            self.postprocscal[k].append(v)

    def save(self, dirname):
        np.save(os.path.join(dirname, f"time"), self.time)
        np.save(os.path.join(dirname, f"iter"), self.iter)
        np.save(os.path.join(dirname, f"liniter"), self.liniter)
        for k,v in self.postprocscal.items():
            np.save(os.path.join(dirname, "postproc_"+k), np.array(v))
