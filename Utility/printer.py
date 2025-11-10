class Printer:
    def __init__(self, types, verbose=True):
        self.verbose = verbose
        lengths = {}
        for k,v in types.items():
            if v == 's':
                lengths[k] = max(10, len(k)+2)
            elif v == 'i':
                lengths[k] = max(4, len(k))+2
            elif v == 'f':
                lengths[k] = max(5, len(k))+2
            elif v == 'e':
                lengths[k] = max(9, len(k))+2
            else:
                raise ValueError(f"Unknown type: {v} (known types: s,i,f,e)")
        self.lengths = lengths
        self.formats = {}
        for k, v in lengths.items():
            if types[k] == 's':
                self.formats[k] = f"{v}"
            elif types[k] == 'i' or types[k] == 'd':
                self.formats[k] = f"^{v}d"
            elif types[k] == 'f':
                self.formats[k] = f"^{v}.2f"
            else:
                self.formats[k] = f"^{v}.3e"
        self.values = {k:k if v=='s' else 0 for k,v in types.items()}

    def print_names(self):
        if not self.verbose: return
        f = ""
        for k, v in self.lengths.items():
            f += f"{k:^{v}}"
        n = len(f)
        print(n*'-'+'\n'+f+'\n'+n*'-')
    def print(self):
        values = self.values
        if not self.verbose: return
        fmt = ""
        for k, v in self.formats.items():
            fmt += f"{{{k}:{v}}}"
        print(fmt.format(**values))

