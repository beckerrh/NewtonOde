class Logger:
    def __init__(self, types, name='My nice iteration', verbose=True, history=True, print_name=False):
        self.name = name
        self.verbose = verbose
        self.print_name = print_name
        self.lengths = {}
        self.formats = {}
        self.values = {}
        self.add_types(types)
        if history: self.history = []

    def add_types(self, types):
        for k,val in types.items():
            parts = val.split(":")
            v = parts[0].lower()
            if v == 'd': v = 'i'
            width = int(parts[1]) if len(parts) > 1 else None
            if v == 's':
                w = 10 if width is None else width
                self.lengths[k] = max(w, len(k))+2
                self.formats[k] = f"{{:^{self.lengths[k]}}}"
            elif v == 'i':
                w = 4 if width is None else width
                self.lengths[k] = max(w, len(k)) + 2
                self.formats[k] = f"{{:>{self.lengths[k]}d}}"
            elif v == 'f':
                w = 5 if width is None else width
                self.lengths[k] = max(w, len(k))+2
                self.formats[k] = f"{{:>{self.lengths[k]}.2f}}"
            elif v == 'e':
                w = 9 if width is None else width
                self.lengths[k] = max(w, len(k))+2
                self.formats[k] = f"{{:>{self.lengths[k]}.3e}}"
            else:
                raise ValueError(f"Unknown type: {v} (known types: s,i,f,e)")
            self.values[k] = k if v == 's' else 0

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.values:
                self.values[k] = v
            else:
                raise KeyError(f"Unknown printer key: {k}")
    def header(self):
        if not self.verbose: return
        f = ' '*(len(self.name)+2) if self.print_name else ''
        for k, v in self.lengths.items():
            f += f"{k:>{v}}"
        n = len(f)
        return n*'-'+'\n'+f+'\n'+n*'-'
    def print_names(self, add=""):
         print(self.header()+add)

    def print(self):
        if not self.verbose: return
        if self.print_name:
            width = len(self.name) + 2
            row = f"{self.name:{width}}"
        else:
            width, row = 0, f""
        for k, fmt in self.formats.items():
            row += fmt.format(self.values[k])
        if hasattr(self, 'history'):
            self.history.append(row)
        print(row)

    def print_history(self, filename=None):
        if not hasattr(self, 'history'):
            raise ValueError("No history stored.")
        if filename == None: filename = f"{self.name.strip()}_history.txt"
        with open(filename, 'w') as f:
            f.write(self.header() + '\n')
            for row in self.history:
                f.write(row + '\n')


