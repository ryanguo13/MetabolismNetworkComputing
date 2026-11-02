"""Exact mass-action algebra from the k-ecoli457 publisher MATLAB package.

No reduced kinetics: negative stoichiometries are the reaction orders, including
the fractional orders in the original mass_balance_ode.m. Initial conditions
follow perturbModel.m and parallel_automated.m for aerobic glucose (no mutant).
"""

import io
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.sparse import csc_matrix, csr_matrix


class KEcoliModel:
    def __init__(self, path, enzyme_factor=2.0):
        self.path = Path(path)
        with zipfile.ZipFile(path) as outer:
            nested = outer.read("Supplementary Software 1/k-ecoli457.zip")
        with zipfile.ZipFile(io.BytesIO(nested)) as inner:
            data = loadmat(io.BytesIO(inner.read("k-ecoli457/Data.mat")), simplify_cells=True)
        self.data = data
        network = data["network_data"]
        self.S = csc_matrix(network["S_f_b"])
        self.k = data["elementary_kinetic_parameter"].copy()
        self.species = list(network["metab"])
        self.y0 = data["metabolite_and_enzyme_conc"].copy()
        self.y0[:337] = 1.0
        # Original parallel_automated.m disables ten alternate carbon enzymes.
        for matlab_reaction in range(446, 456):
            self.y0[337 + matlab_reaction - 1] = 0
            self.y0[network["enz_enzComplex"][:, matlab_reaction - 1] != 0] = 0
        if enzyme_factor < 1:
            raise ValueError("Only original deterministic overexpression path supported")
        self.y0[337] += enzyme_factor - 1
        # Older SciPy may share index buffers for minimum(0); pruning that
        # result must not corrupt the original stoichiometric matrix.
        negative = self.S.copy()
        negative.data = np.minimum(negative.data, 0)
        negative.eliminate_zeros()
        reactants = -negative.tocoo()
        self.ri, self.rj, self.order = reactants.row, reactants.col, reactants.data
        self.count = np.bincount(self.rj, minlength=len(self.k))
        if self.count.min() != 1 or self.count.max() != 2:
            raise ValueError("Unexpected reaction arity: inspect source before extending")
        self.indices = np.zeros((2, len(self.k)), dtype=int)
        self.orders = np.zeros((2, len(self.k)))
        next_slot = np.zeros(len(self.k), dtype=int)
        for i, j, exponent in zip(self.ri, self.rj, self.order):
            slot = next_slot[j]
            self.indices[slot, j], self.orders[slot, j] = i, exponent
            next_slot[j] += 1
        self.has_second = self.count == 2
        self.dependency = csr_matrix((np.ones(len(self.ri)), (self.rj, self.ri)),
                                     shape=(len(self.k), len(self.species)))
        self.sparsity = (abs(self.S) @ self.dependency).astype(bool).tocsc()
        # Enzyme-pool coefficients from the published mechanism: free enzyme + complexes.
        self.pools = csr_matrix((network["enz_enzComplex"] != 0).T.astype(float))
        free = np.arange(337, 794)
        self.pools = self.pools + csr_matrix((np.ones(457), (np.arange(457), free)), shape=self.pools.shape)
        # Biomass_Bulk is a pseudo-reaction, not an enzyme conservation pool.
        valid = np.asarray(abs(self.pools @ self.S).sum(axis=1)).ravel() == 0
        self.pool_indices = np.flatnonzero(valid)
        self.pools = self.pools[valid]
        # Original get_Vnet.m reports the first elementary step for each of
        # the 457 unexpanded reactions, with the published export sign rule.
        map_rows, map_cols, map_values = [], [], []
        for row, info in enumerate(network['unexp_rxn_info'][:457]):
            first = int(info[1])
            columns = np.flatnonzero(network['rxn_f_bInd_rxnInd'] == first)
            reversible = info[5] != 3 and network['rev'][first - 1] == 1
            sign = -1 if info[5] != 3 and not reversible and row + 1 in network['unexpanded_model']['export'] else 1
            if reversible and len(columns) != 2:
                raise ValueError('Invalid reversible mapping')
            selected = columns if reversible else columns[:1]
            for slot, col in enumerate(selected):
                map_rows.append(row); map_cols.append(col)
                map_values.append(1 if reversible and slot == 0 else -1 if reversible else sign)
        self.net_map = csr_matrix((map_values, (map_rows, map_cols)), shape=(457, len(self.k)))

    def flux(self, t, y):
        with np.errstate(invalid="raise", over="raise", divide="raise"):
            return self.k * np.power(y[self.indices[0]], self.orders[0]) * np.power(y[self.indices[1]], self.orders[1])

    def rhs(self, t, y):
        return self.S @ self.flux(t, y)

    def flux_jacobian(self, t, y):
        # Product rule is valid at zero for integer-order species; no v/y division.
        values = []
        for slot in range(2):
            active = self.orders[slot] > 0
            first = self.indices[slot, active]
            second = self.indices[1 - slot, active]
            exponents = self.orders[slot, active]
            with np.errstate(invalid="raise", over="raise", divide="raise"):
                grad = self.k[active] * exponents * np.power(y[first], exponents - 1) * np.power(y[second], self.orders[1 - slot, active])
            values.append((np.flatnonzero(active), first, grad))
        rows = np.concatenate([v[0] for v in values])
        cols = np.concatenate([v[1] for v in values])
        vals = np.concatenate([v[2] for v in values])
        return csr_matrix((vals, (rows, cols)), shape=self.dependency.shape)

    def jacobian(self, t, y):
        return (self.S @ self.flux_jacobian(t, y)).tocsc()

    def source_directory(self, target):
        """Extract only flat, known source members; do not trust ZIP paths."""
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self.path) as outer:
            nested = outer.read("Supplementary Software 1/k-ecoli457.zip")
        with zipfile.ZipFile(io.BytesIO(nested)) as inner:
            for item in inner.infolist():
                p = Path(item.filename)
                if p.parent.as_posix() == "k-ecoli457" and p.suffix in {".m", ".mat"}:
                    (target / p.name).write_bytes(inner.read(item))
