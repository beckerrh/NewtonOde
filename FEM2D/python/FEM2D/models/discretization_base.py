import numpy as np
from ..linalg.fem_vector import FemVector
from ..fems import mesh_transfer

# ================================================================= #
class DiscretizationBase:
    def __init__(self, mesh, application, fems, block_names, verbose=0, timer=None):
        self.mesh = mesh
        self.application = application
        self.fems = tuple(fems)
        self.block_names = block_names
        assert len(block_names) == len(self.application.ncomps)
        self.verbose = verbose
        self.timer = timer

    @property
    def nsys(self):
        return len(self.fems)

    @property
    def ncomps(self):
        ncomps = self.application.ncomps
        if isinstance(ncomps, int):
            raise TypeError(
                "application.ncomps must be a list/tuple, e.g. (1,), not 1"
            )
        return tuple(int(nc) for nc in ncomps)

    @property
    def ndofs(self):
        return tuple(fem.nunknowns() for fem in self.fems)

    @property
    def fem(self):
        if len(self.fems) != 1:
            raise AttributeError(
                f"disc.fem is only defined for single-FEM discretizations, "
                f"got {len(self.fems)} FEMs"
            )
        return self.fems[0]

    def newVector(self):
        if len(self.ncomps) != len(self.ndofs):
            raise ValueError(
                f"One ncomp value per FEM/system is required: "
                f"{self.ncomps=}, {self.ndofs=}"
            )

        return FemVector.zeros(
            ncomps=self.ncomps,
            ndofs=self.ndofs,
            names=self.block_names,
        )


    def build_transfer_to_refined_mesh(self, info, disc_fine):
        transfers = []

        for fem_coarse, fem_fine in zip(self.fems, disc_fine.fems):
            P = fem_coarse.build_scalar_prolongation_to_refined_mesh(info)
            transfers.append(mesh_transfer.BlockTransfer(P=P))

        return mesh_transfer.MeshTransfer(
            transfers,
            ncomps=self.ncomps,
            names=self.block_names,
        )

    def compute_errors_exact(self, u):
        scalar = {}
        cell = {}

        for iblock, name in enumerate(self.block_names):
            fem = self.fems[iblock]
            U = u.block(name)
            exact_block = self.application.exactsolution[iblock]

            err_L2c2 = 0.0
            err_L2n2 = 0.0
            err_H12 = 0.0
            err_Flux2 = 0.0
            ec_total = None

            for icomp, uexi in enumerate(exact_block):
                ui = U[icomp]

                e_L2c, ec = fem.computeErrorL2Cell(uexi, ui)
                e_L2n, en = fem.computeErrorL2(uexi, ui)
                e_H1 = fem.computeErrorFluxL2(uexi, ui)
                e_Flux = fem.computeErrorFluxL2(uexi, ui, self.diffcell)

                err_L2c2 += e_L2c ** 2
                err_L2n2 += e_L2n ** 2
                err_H12 += e_H1 ** 2
                err_Flux2 += e_Flux ** 2

                ec2 = ec ** 2
                ec_total = ec2 if ec_total is None else ec_total + ec2

            prefix = f"{name}_"

            scalar[prefix + "err_L2c"] = np.sqrt(err_L2c2)
            scalar[prefix + "err_L2n"] = np.sqrt(err_L2n2)
            scalar[prefix + "err_H1"] = np.sqrt(err_H12)
            scalar[prefix + "err_Flux"] = np.sqrt(err_Flux2)

            if ec_total is not None:
                cell[prefix + "err"] = np.sqrt(ec_total)

        return scalar, cell