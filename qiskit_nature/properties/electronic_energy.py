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


class ElectronicEnergy(Property):
    """TODO."""

    def __init__(
        self,
        register_length: int,
        onee_ints: Tuple[np.ndarray, Optional[np.ndarray]],
        twoe_ints: Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]],
        fock_ints: Optional[Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]] = None,
        reference_energy: Optional[float] = None,
        energy_shift: Optional[Dict[str, float]] = None,
    ):
        """TODO."""
        super().__init__(self.__class__.__name__, register_length)
        self._onee_ints = onee_ints
        self._twoe_ints = twoe_ints
        self._fock_ints = fock_ints
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
            (qmol.mo_onee_ints, qmol.mo_onee_ints_b),
            (qmol.mo_eri_ints, qmol.mo_eri_ints_ba, qmol.mo_eri_ints_bb),
            reference_energy=qmol.hf_energy,
            energy_shift=energy_shift,
        )

    def _integrals(self) -> Tuple[np.ndarray, np.ndarray]:
        """TODO."""
        so_onee_ints = QMolecule.onee_to_spin(*self._onee_ints)
        so_twoe_ints = QMolecule.twoe_to_spin(*self._twoe_ints)
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
