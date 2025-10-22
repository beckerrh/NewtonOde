class Printer:
    def __init__(self, verbose, types):
        self.verbose = verbose
        lengths = {}
        for k,v in types.items():
            if v == 's':
                lengths[k] = len(k)+2
            elif v == 'i':
                lengths[k] = max(4, len(k))+2
            elif v == 'f':
                lengths[k] = max(5, len(k))+2
            elif v == 'e':
                lengths[k] = max(9, len(k))+2
            else:
                assert False
        self.lengths = lengths
        self.formats = {}
        for k, v in lengths.items():
            if types[k] == 's':
                self.formats[k] = f"{v}"
            elif types[k] == 'i':
                self.formats[k] = f"^{v}d"
            elif types[k] == 'f':
                self.formats[k] = f"^{v}.2f"
            else:
                self.formats[k] = f"^{v}.3e"

    def print_names(self):
        if not self.verbose: return
        f = ""
        first = True
        for k, v in self.lengths.items():
            if first: f += f"{k:<{v}}"
            else: f += f"{k:^{v}}"
            first=False
        print(f)
    def print(self, values):
        if not self.verbose: return
        fmt = ""
        first = True
        for k, v in self.formats.items():
            if first:
                fmt += f"{{{k}:<{v}}}"
            else:
                fmt+= f"{{{k}:{v}}}"
            first = False
        print(fmt.format(**values))

