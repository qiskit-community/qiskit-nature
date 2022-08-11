# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test low rank utilities."""

import itertools
from test import QiskitNatureTestCase
from typing import cast

import numpy as np
from ddt import data, ddt, unpack
from qiskit.quantum_info import random_hermitian

import qiskit_nature.settings
from qiskit_nature.hdf5 import load_from_hdf5
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.problems import ElectronicStructureResult
from qiskit_nature.second_q.properties.electronic_energy import ElectronicBasis, ElectronicEnergy
from qiskit_nature.second_q.properties.integrals import (
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
)
from qiskit_nature.utils.low_rank import (
    _low_rank_compressed_two_body_decomposition,
    _low_rank_optimal_core_tensors,
    _low_rank_two_body_decomposition,
    low_rank_decomposition,
)
from qiskit_nature.utils.random import random_two_body_tensor

# TODO delete this when it's no longer needed
qiskit_nature.settings.dict_aux_operators = True

rng = np.random.default_rng(3091)
hamiltonians = {}

for molecule_name in ["H2_sto3g", "BeH_sto3g_reduced"]:
    result = cast(
        ElectronicStructureResult,
        load_from_hdf5(f"test/second_q/transformers/{molecule_name}.hdf5"),
    )
    hamiltonians[molecule_name] = result.get_property("ElectronicEnergy")

for n_modes_ in [3, 5]:
    one_body_tensor_ = np.array(random_hermitian(n_modes_, seed=rng))
    two_body_tensor_ = random_two_body_tensor(n_modes_, seed=rng)
    one_body_integrals = OneBodyElectronicIntegrals(ElectronicBasis.MO, (one_body_tensor_,) * 2)
    two_body_integrals = TwoBodyElectronicIntegrals(ElectronicBasis.MO, (two_body_tensor_,) * 4)
    hamiltonians[f"random_{n_modes_}"] = ElectronicEnergy([one_body_integrals, two_body_integrals])


@ddt
class TestLowRank(QiskitNatureTestCase):
    """Tests for low rank decomposition utilities."""

    @data("H2_sto3g", "random_3")
    def test_equation(self, hamiltonian_name: str):
        """Test Hamiltonian equation in documentation."""
        electronic_energy = hamiltonians[hamiltonian_name]
        expected = electronic_energy.second_q_ops()["ElectronicEnergy"]
        one_body_tensor = electronic_energy.get_electronic_integral(
            ElectronicBasis.MO, 1
        ).get_matrix()
        two_body_tensor = electronic_energy.get_electronic_integral(
            ElectronicBasis.MO, 2
        ).get_matrix()
        n_modes = one_body_tensor.shape[0]

        actual = FermionicOp.zero(register_length=n_modes)
        for p, q in itertools.product(range(n_modes), repeat=2):
            coeff = one_body_tensor[p, q]
            for sigma in range(2):
                actual += FermionicOp(
                    [([("+", p + sigma * n_modes), ("-", q + sigma * n_modes)], coeff)]
                )
        for p, q, r, s in itertools.product(range(n_modes), repeat=4):
            coeff = two_body_tensor[p, q, r, s]
            for sigma, tau in itertools.product(range(2), repeat=2):
                actual += FermionicOp(
                    [
                        (
                            [
                                ("+", p + sigma * n_modes),
                                ("+", r + tau * n_modes),
                                ("-", s + tau * n_modes),
                                ("-", q + sigma * n_modes),
                            ],
                            0.5 * coeff,
                        )
                    ]
                )

        self.assertTrue(actual.normal_ordered().equiv(expected.normal_ordered(), atol=1e-8))

    @unpack
    @data(("H2_sto3g", False), ("random_3", False), ("random_3", True))
    def test_df_hamiltonian_to_fermionic_op(self, hamiltonian_name: str, z_representation: bool):
        """Test converting DoubleFactorizedHamiltonian to FermionicOp."""
        electronic_energy = hamiltonians[hamiltonian_name]
        expected = electronic_energy.second_q_ops()["ElectronicEnergy"]
        df_hamiltonian = low_rank_decomposition(
            electronic_energy, z_representation=z_representation
        )

        actual = df_hamiltonian.to_fermionic_op()

        self.assertTrue(actual.normal_ordered().equiv(expected.normal_ordered(), atol=1e-8))

    @data("H2_sto3g", "random_3")
    def test_low_rank_decomposition(self, hamiltonian_name: str):
        """Test low rank decomposition."""
        electronic_energy = hamiltonians[hamiltonian_name]
        expected = electronic_energy.second_q_ops()["ElectronicEnergy"]
        df_hamiltonian = low_rank_decomposition(electronic_energy)
        n_modes = df_hamiltonian.n_orbitals

        actual = FermionicOp.zero(register_length=n_modes)
        for p, q in itertools.product(range(n_modes), repeat=2):
            coeff = df_hamiltonian.one_body_tensor[p, q]
            for sigma in range(2):
                actual += FermionicOp(
                    [([("+", p + sigma * n_modes), ("-", q + sigma * n_modes)], coeff)]
                )
        for leaf_tensor, core_tensor in zip(
            df_hamiltonian.leaf_tensors, df_hamiltonian.core_tensors
        ):
            num_ops = []
            for sigma in range(2):
                for i in range(n_modes):
                    num_op = FermionicOp.zero(register_length=n_modes)
                    for p, q in itertools.product(range(n_modes), repeat=2):
                        num_op += FermionicOp(
                            [
                                (
                                    [("+", p + sigma * n_modes), ("-", q + sigma * n_modes)],
                                    leaf_tensor[p, i] * leaf_tensor[q, i].conj(),
                                )
                            ]
                        )
                    num_ops.append(num_op)
            for i, j in itertools.product(range(n_modes), repeat=2):
                for sigma, tau in itertools.product(range(2), repeat=2):
                    actual += (
                        0.5
                        * core_tensor[i, j]
                        * num_ops[i + sigma * n_modes]
                        @ num_ops[j + tau * n_modes]
                    )

        self.assertTrue(actual.normal_ordered().equiv(expected.normal_ordered(), atol=1e-8))

    @data("H2_sto3g", "random_3")
    def test_low_rank_decomposition_z_representation(self, hamiltonian_name: str):
        """Test low rank decomposition equation "Z" representation."""
        electronic_energy = hamiltonians[hamiltonian_name]
        expected = electronic_energy.second_q_ops()["ElectronicEnergy"]
        df_hamiltonian = low_rank_decomposition(electronic_energy, z_representation=True)
        n_modes = df_hamiltonian.n_orbitals

        # TODO: this cast should be unnecessary
        actual = cast(
            FermionicOp, df_hamiltonian.constant * FermionicOp.one(register_length=n_modes)
        )
        for p, q in itertools.product(range(n_modes), repeat=2):
            coeff = df_hamiltonian.one_body_tensor[p, q]
            for sigma in range(2):
                actual += FermionicOp(
                    [([("+", p + sigma * n_modes), ("-", q + sigma * n_modes)], coeff)]
                )
        for leaf_tensor, core_tensor in zip(
            df_hamiltonian.leaf_tensors, df_hamiltonian.core_tensors
        ):
            num_ops = []
            for sigma, i in itertools.product(range(2), range(n_modes)):
                num_op = FermionicOp.zero(register_length=n_modes)
                for p, q in itertools.product(range(n_modes), repeat=2):
                    num_op += FermionicOp(
                        [
                            (
                                [("+", p + sigma * n_modes), ("-", q + sigma * n_modes)],
                                leaf_tensor[p, i] * leaf_tensor[q, i].conj(),
                            )
                        ]
                    )
                num_ops.append(num_op)
            for a, b in itertools.combinations(range(2 * n_modes), 2):
                sigma, i = divmod(a, n_modes)
                tau, j = divmod(b, n_modes)
                z1 = (  # pylint: disable=invalid-name
                    FermionicOp.one(register_length=n_modes) - 2 * num_ops[i + sigma * n_modes]
                )
                z2 = (  # pylint: disable=invalid-name
                    FermionicOp.one(register_length=n_modes) - 2 * num_ops[j + tau * n_modes]
                )
                # TODO: this cast should be unnecessary
                actual += cast(
                    FermionicOp, 0.125 * (core_tensor[i, j] + core_tensor[j, i]) * z1 @ z2
                )

        self.assertTrue(actual.normal_ordered().equiv(expected.normal_ordered(), atol=1e-8))

    @unpack
    @data(("H2_sto3g", 2), ("BeH_sto3g_reduced", 5), ("random_5", None))
    def test_low_rank_decomposition_optimal_core_tensors(
        self, hamiltonian_name: str, max_rank: int
    ):
        """Test low rank decomposition optimal core tensors."""
        electronic_energy = hamiltonians[hamiltonian_name]
        df_hamiltonian = low_rank_decomposition(electronic_energy, max_rank=max_rank)
        two_body_tensor = electronic_energy.get_electronic_integral(
            ElectronicBasis.MO, 2
        ).get_matrix()

        core_tensors = _low_rank_optimal_core_tensors(
            two_body_tensor, df_hamiltonian.leaf_tensors, cutoff_threshold=1e-8
        )
        reconstructed = np.einsum(
            "tpk,tqk,tkl,trl,tsl->pqrs",
            df_hamiltonian.leaf_tensors,
            df_hamiltonian.leaf_tensors,
            core_tensors,
            df_hamiltonian.leaf_tensors,
            df_hamiltonian.leaf_tensors,
        )
        np.testing.assert_allclose(reconstructed, two_body_tensor, atol=1e-8)

    @data("H2_sto3g", "random_5")
    def test_low_rank_two_body_decomposition(self, hamiltonian_name: str):
        """Test low rank two-body decomposition."""
        electronic_energy = hamiltonians[hamiltonian_name]
        two_body_tensor = electronic_energy.get_electronic_integral(
            ElectronicBasis.MO, 2
        ).get_matrix()

        leaf_tensors, core_tensors = _low_rank_two_body_decomposition(two_body_tensor)
        reconstructed = np.einsum(
            "tpk,tqk,tkl,trl,tsl->pqrs",
            leaf_tensors,
            leaf_tensors,
            core_tensors,
            leaf_tensors,
            leaf_tensors,
        )
        np.testing.assert_allclose(reconstructed, two_body_tensor, atol=1e-8)

    @unpack
    @data(("H2_sto3g", 2), ("BeH_sto3g_reduced", 4))
    def test_low_rank_compressed_two_body_decomposition(self, hamiltonian_name: str, max_rank: int):
        """Test low rank compressed two-body decomposition."""
        electronic_energy = hamiltonians[hamiltonian_name]
        two_body_tensor = electronic_energy.get_electronic_integral(
            ElectronicBasis.MO, 2
        ).get_matrix()

        leaf_tensors, core_tensors = _low_rank_compressed_two_body_decomposition(
            two_body_tensor, max_rank=max_rank, seed=rng
        )
        reconstructed = np.einsum(
            "tpk,tqk,tkl,trl,tsl->pqrs",
            leaf_tensors,
            leaf_tensors,
            core_tensors,
            leaf_tensors,
            leaf_tensors,
        )
        np.testing.assert_allclose(reconstructed, two_body_tensor, atol=1e-4)

    @unpack
    @data(("H2_sto3g", 3), ("BeH_sto3g_reduced", 6))
    def test_low_rank_compressed_two_body_decomposition_constrained(
        self, hamiltonian_name: str, max_rank: int
    ):
        """Test low rank compressed two-body decomposition."""
        electronic_energy = hamiltonians[hamiltonian_name]
        two_body_tensor = electronic_energy.get_electronic_integral(
            ElectronicBasis.MO, 2
        ).get_matrix()

        n_modes, _, _, _ = two_body_tensor.shape
        core_tensor_mask = np.sum(
            [np.diag(np.ones(n_modes - abs(k)), k=k) for k in range(-1, 2)], axis=0, dtype=bool
        )
        leaf_tensors, core_tensors = _low_rank_compressed_two_body_decomposition(
            two_body_tensor, max_rank=max_rank, core_tensor_mask=core_tensor_mask, seed=rng
        )
        np.testing.assert_allclose(core_tensors, core_tensors * core_tensor_mask)

        reconstructed = np.einsum(
            "tpk,tqk,tkl,trl,tsl->pqrs",
            leaf_tensors,
            leaf_tensors,
            core_tensors,
            leaf_tensors,
            leaf_tensors,
        )
        np.testing.assert_allclose(reconstructed, two_body_tensor, atol=1e-2)

    @unpack
    @data(("H2_sto3g", 2), ("BeH_sto3g_reduced", 4))
    def test_low_rank_decomposition_compressed(self, hamiltonian_name: str, max_rank: int):
        """Test compressed low rank decomposition."""
        electronic_energy = hamiltonians[hamiltonian_name]
        expected = electronic_energy.second_q_ops()["ElectronicEnergy"]

        df_hamiltonian = low_rank_decomposition(
            electronic_energy,
            max_rank=max_rank,
            optimize=True,
            seed=rng,
            options=dict(ftol=1e-12, gtol=1e-12),
        )
        two_body_tensor = electronic_energy.get_electronic_integral(
            ElectronicBasis.MO, 2
        ).get_matrix()
        np.testing.assert_allclose(df_hamiltonian.two_body_tensor, two_body_tensor, atol=1e-4)
        self.assertLessEqual(len(df_hamiltonian.leaf_tensors), max_rank)
        self.assertLessEqual(len(df_hamiltonian.core_tensors), max_rank)

        n_modes = df_hamiltonian.n_orbitals

        actual = FermionicOp.zero(register_length=n_modes)
        for p, q in itertools.product(range(n_modes), repeat=2):
            coeff = df_hamiltonian.one_body_tensor[p, q]
            for sigma in range(2):
                actual += FermionicOp(
                    [([("+", p + sigma * n_modes), ("-", q + sigma * n_modes)], coeff)]
                )
        for leaf_tensor, core_tensor in zip(
            df_hamiltonian.leaf_tensors, df_hamiltonian.core_tensors
        ):
            num_ops = []
            for sigma in range(2):
                for i in range(n_modes):
                    num_op = FermionicOp.zero(register_length=n_modes)
                    for p, q in itertools.product(range(n_modes), repeat=2):
                        num_op += FermionicOp(
                            [
                                (
                                    [("+", p + sigma * n_modes), ("-", q + sigma * n_modes)],
                                    leaf_tensor[p, i] * leaf_tensor[q, i].conj(),
                                )
                            ]
                        )
                    num_ops.append(num_op)
            for i, j in itertools.product(range(n_modes), repeat=2):
                for sigma, tau in itertools.product(range(2), repeat=2):
                    actual += (
                        0.5
                        * core_tensor[i, j]
                        * num_ops[i + sigma * n_modes]
                        @ num_ops[j + tau * n_modes]
                    )

        diff = (actual - expected).normal_ordered().simplify()
        self.assertLess(diff.induced_norm(), 1e-2)
