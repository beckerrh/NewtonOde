import numpy as np

#=================================================================#
class FemVector():
    def __init__(self, data, vectorview, fems, names='U', visutypes='point'):
        self.data, self.vectorview, self.fems, self.names, self.visutypes = data, vectorview, fems, names, visutypes
        assert len(fems) == vectorview.nparts
        for i in range(vectorview.nparts):
            assert fems[i].nunknowns() == vectorview.ns[i]
    def __repr__(self):
        data, vectorview, fems = self.data, self.vectorview, self.fems
        s = ""
        for i in range(vectorview.nparts):
            s += f"{fems[i]} ncomp={vectorview.ncomps[i]}"
        return s
    def extract(self, name):
        data, vectorview, fems, names = self.data, self.vectorview, self.fems, self.names
        assert len(names) == vectorview.nparts
        n2i = {names[i]:i for i in range(vectorview.nparts)}
        index = n2i[name]
        # dataex = [vectorview.get(index, icomp, data) for icomp in range(vectorview.ncomps[index])]
        return vectorview.get_part(index,data), fems[index], vectorview.stack_storage
    def tovisudata(self, types=None, names=None):
        if types is None: types = self.visutypes
        if names is None: names = self.names
        data, vectorview, fems = self.data, self.vectorview, self.fems
        if isinstance(types, str):
            types = vectorview.nparts*[types]
        else:
            assert len(types) == vectorview.nparts
        if isinstance(names, str):
            names = [names+f"{i:02d}" for i in range(vectorview.nparts)]
        else:
            assert len(names) == vectorview.nparts
        visudata = {'point': {}, 'cell': {}, 'global': {}}
        for i in range(vectorview.nparts):
            if vectorview.ncomps[i]==1:
                visunames = [names[i]]
            else:
                visunames = [f'{names[i]}_{icomp+1:1d}' for icomp in range(vectorview.ncomps[i])]
            if types[i] == 'point':
                for icomp in range(vectorview.ncomps[i]):
                    visudata[types[i]][visunames[icomp]] = fems[i].tonode(vectorview.get(i, icomp, data))
            elif types[i] == 'cell':
                # print(f"{visunames[i]=}")
                for icomp in range(vectorview.ncomps[i]):
                    visudata[types[i]][visunames[icomp]] = fems[i].tocell(vectorview.get(i, icomp, data))
            elif types[i] == 'global':
                for icomp in range(vectorview.ncomps[i]):
                    visudata[types[i]][visunames[icomp]] = vectorview.get(i, icomp, data)
            else:
                raise ValueError(f"unknown type {types[i]}")
        return visudata
