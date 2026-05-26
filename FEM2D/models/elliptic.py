from FEM2D.models.elliptic_primal import EllipticPrimal

# ================================================================= #
def Elliptic(**kwargs):
    fem = kwargs.pop("fem", "cr1")
    kwargs["fem"] = fem
    if fem == 'rt0': raise NotImplementedError("mixed not implemented")
    else: return EllipticPrimal(**kwargs)



#=================================================================#
if __name__ == '__main__':
    print("Pas de test")
