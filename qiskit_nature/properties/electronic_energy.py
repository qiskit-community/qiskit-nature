# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""TODO."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import cast, Dict, List, Optional, Tuple, Union

import numpy as np

from qiskit_nature import QiskitNatureError
from qiskit_nature.drivers import QMolecule, WatsonHamiltonian
from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from qiskit_nature.problems.second_quantization.electronic.builders.fermionic_op_builder import (
    build_ferm_op_from_ints,
)
from qiskit_nature.results import EigenstateResult, ElectronicStructureResult

from .property import Property


class _ElectronicIntegrals(ABC):
    """TODO."""

    def __init__(
        self,
        num_body_terms: int,
        matrices: Tuple[Optional[np.ndarray], ...],
    ) -> None:
        """TODO."""
        assert num_body_terms >= 1
        self._num_body_terms = num_body_terms
        assert len(matrices) == 2 ** num_body_terms
        assert matrices[0] is not None
        self._matrices = matrices

    @abstractmethod
    def to_spin(self) -> np.ndarray:
        """TODO."""
        raise NotImplementedError("TODO.")


class _1BodyElectronicIntegrals(_ElectronicIntegrals):
    """TODO."""

    def __init__(self, matrices: Tuple[Optional[np.ndarray], ...]) -> None:
        """TODO."""
        super().__init__(1, matrices)

    def to_spin(self) -> np.ndarray:
        """TODO."""
        matrix_a = self._matrices[0]
        matrix_b = self._matrices[1] or matrix_a
        zeros = np.zeros(matrix_a.shape)
        return np.block([[matrix_a, zeros], [zeros, matrix_b]])


class _2BodyElectronicIntegrals(_ElectronicIntegrals):
    """TODO."""

    EINSUM_AO_TO_MO = "pqrs,pi,qj,rk,sl->ijkl"
    EINSUM_CHEM_TO_PHYS = "ijkl->ljik"

    def __init__(self, matrices: Tuple[Optional[np.ndarray], ...]) -> None:
        """TODO."""
        super().__init__(2, matrices)

    def to_spin(self) -> np.ndarray:
        """TODO."""
        so_matrix = np.zeros([2 * s for s in self._matrices[0].shape])
        one_indices = (
            (0, 0, 0, 0),
            (0, 1, 1, 0),
            (1, 0, 0, 1),
            (1, 1, 1, 1),
        )
        for ao_mat, one_idx in zip(self._matrices, one_indices):
            if ao_mat is None:
                ao_mat = self._matrices[0]
            phys_matrix = np.einsum(_2BodyElectronicIntegrals.EINSUM_CHEM_TO_PHYS, ao_mat)
            kron = np.zeros((2, 2, 2, 2))
            kron[one_idx] = 1
            so_matrix -= 0.5 * np.kron(kron, phys_matrix)
        return so_matrix


class ElectronicEnergy(Property):
    """TODO."""

    def __init__(
        self,
        register_length: int,
        electronic_integrals: Dict[int, _ElectronicIntegrals],
        reference_energy: Optional[float] = None,
        energy_shift: Optional[Dict[str, float]] = None,
    ):
        """TODO."""
        super().__init__(self.__class__.__name__, register_length)
        self._electronic_integrals = electronic_integrals
        self._energy_shift = energy_shift
        self._reference_energy = reference_energy

    @classmethod
    def from_driver_result(cls, result: Union[QMolecule, WatsonHamiltonian]) -> ElectronicEnergy:
        """TODO."""
        if isinstance(result, WatsonHamiltonian):
            raise QiskitNatureError("TODO.")

        qmol = cast(QMolecule, result)

        energy_shift = qmol.energy_shift.copy()
        energy_shift["nuclear repulsion"] = qmol.nuclear_repulsion_energy

        return cls(
            qmol.num_molecular_orbitals * 2,
            {
                1: _1BodyElectronicIntegrals((qmol.mo_onee_ints, qmol.mo_onee_ints_b)),
                2: _2BodyElectronicIntegrals(
                    (
                        qmol.mo_eri_ints,
                        qmol.mo_eri_ints_ba.T if qmol.mo_eri_ints_ba is not None else None,
                        qmol.mo_eri_ints_bb,
                        qmol.mo_eri_ints_ba,
                    ),
                ),
            },
            reference_energy=qmol.hf_energy,
            energy_shift=energy_shift,
        )

    def _integrals(self) -> Tuple[np.ndarray, np.ndarray]:
        """TODO."""
        so_onee_ints = self._electronic_integrals[1].to_spin()
        so_twoe_ints = self._electronic_integrals[2].to_spin()
        return (so_onee_ints, so_twoe_ints)

    def second_q_ops(self) -> List[SecondQuantizedOp]:
        """TODO."""
        return [build_ferm_op_from_ints(*self._integrals())]

    def interpret(self, result: EigenstateResult) -> None:
        """TODO."""
        if not isinstance(result, ElectronicStructureResult):
            raise QiskitNatureError("TODO")
        if self._reference_energy is not None:
            result.hf_energy = self._reference_energy
        if "nuclear_repulsion_energy" in self._energy_shift.keys():
            result.nuclear_repulsion_energy = self._energy_shift["nuclear_repulsion_energy"]
