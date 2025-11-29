class Printer:
    def __init__(self, types, name='My nice iteration', verbose=True):
        self.name = name
        self.verbose = verbose
        self.lengths = {}
        self.formats = {}
        self.values = {}
        self.add_types(types)

    def add_types(self, types):
        for k,val in types.items():
            parts = val.split(":")
            v = parts[0].lower()
            if v == 'd': v = 'i'
            width = int(parts[1]) if len(parts) > 1 else None
            if v == 's':
                w = 10 if width is None else width
                self.lengths[k] = max(w, len(k)+2)
                self.formats[k] = f"{{:^{self.lengths[k]}}}"
            elif v == 'i':
                w = 4 if width is None else width
                self.lengths[k] = max(w, len(k)) + 2
                self.formats[k] = f"{{:>{self.lengths[k]}d}}"
            elif v == 'f':
                w = 5 if width is None else width
                self.lengths[k] = max(w, len(k))+2
                self.formats[k] = f"{{:^{self.lengths[k]}.2f}}"
            elif v == 'e':
                w = 9 if width is None else width
                self.lengths[k] = max(w, len(k))+2
                self.formats[k] = f"{{:^{self.lengths[k]}.3e}}"
            else:
                raise ValueError(f"Unknown type: {v} (known types: s,i,f,e)")
            self.values[k] = k if v == 's' else 0

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.values:
                self.values[k] = v
            else:
                raise KeyError(f"Unknown printer key: {k}")
    def print_names(self):
        if not self.verbose: return
        f = ' '*(len(self.name)+2)
        for k, v in self.lengths.items():
            f += f"{k:^{v}}"
        n = len(f)
        print(n*'-'+'\n'+f+'\n'+n*'-')

    def print(self):
        if not self.verbose: return
        width = len(self.name) + 2
        row = f"{self.name:{width}}"
        for k, fmt in self.formats.items():
            row += fmt.format(self.values[k])
        print(row)

