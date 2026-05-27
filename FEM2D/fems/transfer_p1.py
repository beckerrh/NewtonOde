class P1Transfer:
    def __init__(self, P):
        self.P = P.tocsr()
        self.R = self.P.T.tocsr()

    def prolong(self, uc):
        return self.P @ uc

    def restrict(self, rf):
        return self.R @ rf