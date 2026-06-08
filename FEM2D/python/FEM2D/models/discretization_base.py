import numpy as np
from ..linalg.fem_vector import FemVector
from ..fems import mesh_transfer

from dataclasses import dataclass
from .operators.operator import Operator


# ================================================================= #
@dataclass(frozen=True)
class RoutedOperator:
    test_part: str
    trial_part: str
    operator: Operator

# ================================================================= #
class DiscretizationBase:
    def __init__(self, mesh, application, fems, part_names, verbose=0, timer=None):
        self.mesh = mesh
        self.application = application
        self.fems = tuple(fems)
        self.part_names = part_names
        assert len(part_names) == len(self.application.ncomps)
        self.verbose = verbose
        self.timer = timer
        self.operators = []

    def add_operator(self, test_part, trial_part, op):
        self.operators.append(RoutedOperator(test_part, trial_part, op))

    def find_operator(self, cls):
        for rop in self.operators:
            if isinstance(rop.operator, cls):
                return rop.operator
        return None

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

    def get_operator(self, cls):
        found = [
            rop.operator
            for rop in self.operators
            if isinstance(rop.operator, cls)
        ]

        if len(found) == 0:
            raise RuntimeError(f"missing operator {cls.__name__}")

        if len(found) > 1:
            raise RuntimeError(f"multiple operators {cls.__name__}")

        return found[0]

    def new_part_matrix_blocks(self):
        import scipy.sparse as sparse

        Aparts = {}

        for ipart, iname in enumerate(self.part_names):
            ni = self.ncomps[ipart]
            ndi = self.ndofs[ipart]

            for jpart, jname in enumerate(self.part_names):
                nj = self.ncomps[jpart]
                ndj = self.ndofs[jpart]

                Z = sparse.csr_matrix((ndi, ndj))
                Aparts[(iname, jname)] = [
                    [Z.copy() for _ in range(nj)]
                    for _ in range(ni)
                ]

        return Aparts

    def assemble_part_matrix(self, Aparts):
        import scipy.sparse as sparse

        rows = []

        for ipart in self.part_names:
            row = []

            for jpart in self.part_names:
                Aij = Aparts[(ipart, jpart)]
                row.append(sparse.bmat(Aij, format="csr"))

            rows.append(row)

        return sparse.bmat(rows, format="csr")

    # ---------------------------------------------------------

    def computeForm(self, u, coeffmass=None):
        du = u.zeros_like()

        for rop in self.operators:
            U = u.part(rop.trial_part)
            DU = du.part(rop.test_part)
            rop.operator.add_form(self, DU, U)

        if coeffmass is not None:
            U = u.part("U")
            DU = du.part("U")
            for icomp in range(U.shape[0]):
                self.fem.massDot(DU[icomp], U[icomp], coeff=coeffmass)

        self.add_boundary_form(du, u)
        return du

    def computeMatrix(self, u):
        import scipy.sparse as sparse

        Aparts = self.new_part_matrix_blocks()

        for rop in self.operators:
            U = None if u is None else u.part(rop.trial_part)
            Aij = Aparts[(rop.test_part, rop.trial_part)]
            rop.operator.add_matrix(self, Aij, U)

        A = self.assemble_part_matrix(Aparts)
        A = self.add_boundary_matrix(A, u)
        return A

    def add_boundary_form(self, du, u):
        raise NotImplementedError(f"add_boundary_form() is not implemented for {type(self)}")

    def add_boundary_matrix(self, A, u=None):
        raise NotImplementedError(f"add_boundary_matrix() is not implemented for {type(self)}")

    def newVector(self):
        if len(self.ncomps) != len(self.ndofs):
            raise ValueError(
                f"One ncomp value per FEM/system is required: "
                f"{self.ncomps=}, {self.ndofs=}"
            )

        return FemVector.zeros(
            ncomps=self.ncomps,
            ndofs=self.ndofs,
            names=self.part_names,
        )


    def build_transfer_to_refined_mesh(self, info, disc_fine):
        transfers = []

        for fem_coarse, fem_fine in zip(self.fems, disc_fine.fems):
            P = fem_coarse.build_scalar_prolongation_to_refined_mesh(info)
            transfers.append(mesh_transfer.BlockTransfer(P=P))

        return mesh_transfer.MeshTransfer(
            transfers,
            ncomps=self.ncomps,
            names=self.part_names,
        )

    def compute_errors_exact(self, u):
        scalar = {}
        cell = {}

        for iblock, name in enumerate(self.part_names):
            fem = self.fems[iblock]
            U = u.part(name)
            exact_block = self.application.exactsolution[iblock]

            err_L2c2 = 0.0
            err_L2n2 = 0.0
            err_H12 = 0.0

            ec_L2c2_total = None

            for icomp, uexi in enumerate(exact_block):
                ui = U[icomp]

                e_L2c, ec_L2c2 = fem.computeErrorL2Cell(uexi, ui)
                e_L2n, _ = fem.computeErrorL2(uexi, ui)
                e_H1 = fem.computeErrorFluxL2(uexi, ui)

                err_L2c2 += e_L2c ** 2
                err_L2n2 += e_L2n ** 2
                err_H12 += e_H1 ** 2

                if ec_L2c2 is not None:
                    ec_L2c2 = np.asarray(ec_L2c2, dtype=float)

                    m = np.min(ec_L2c2)
                    if m < -1e-12:
                        raise ValueError(
                            f"{name} component {icomp}: negative L2 cell error contribution: min={m}"
                        )

                    ec_L2c2_total = (
                        ec_L2c2.copy()
                        if ec_L2c2_total is None
                        else ec_L2c2_total + ec_L2c2
                    )

            prefix = f"{name}_"

            scalar[prefix + "err_L2c"] = float(np.sqrt(err_L2c2))
            scalar[prefix + "err_L2n"] = float(np.sqrt(err_L2n2))
            scalar[prefix + "err_H1"] = float(np.sqrt(err_H12))

            if ec_L2c2_total is not None:
                cell[prefix + "err_L2c"] = np.sqrt(ec_L2c2_total)

        return scalar, cell
    def solution_to_plotdata(self, u):
        point_data = {}
        cell_data = {}
        quiver_data = {}

        for name, ui in zip(u.names, u.parts):

            if ui.shape[0] == 1:
                vals = ui[0]

                if vals.shape[0] == self.mesh.nnodes:
                    point_data[name] = vals

                elif vals.shape[0] == self.mesh.nfaces:
                    point_data[name] = self.fem.to_p1(vals)

                elif vals.shape[0] == self.mesh.ncells:
                    cell_data[name] = vals

                else:
                    raise ValueError(f"cannot plot scalar field {name}: {vals.shape=}")

            elif ui.shape[0] == self.mesh.dimension:
                comps = []

                for vals in ui:
                    if vals.shape[0] == self.mesh.nfaces:
                        comps.append(self.fem.to_p1(vals))
                    else:
                        comps.append(vals)

                quiver_data[name] = tuple(comps)

            else:
                for k, vals in enumerate(ui):
                    key = f"{name}_{k}"

                    if vals.shape[0] == self.mesh.nnodes:
                        point_data[key] = vals
                    elif vals.shape[0] == self.mesh.nfaces:
                        point_data[key] = self.fem.to_p1(vals)
                    elif vals.shape[0] == self.mesh.ncells:
                        cell_data[key] = vals
                    else:
                        raise ValueError(f"cannot plot field {key}: {vals.shape=}")

        return {
            "point": point_data,
            "cell": cell_data,
            "quiver": quiver_data,
        }

    def cellmean_vector(self, U):
        return self.fem.cellmean_vector(U)