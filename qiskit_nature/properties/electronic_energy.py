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

from qiskit_nature import QiskitNatureError
from qiskit_nature.drivers import QMolecule, WatsonHamiltonian
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.results import EigenstateResult, ElectronicStructureResult

from .electronic_integrals import (
    _ElectronicIntegrals,
    _1BodyElectronicIntegrals,
    _2BodyElectronicIntegrals,
)
from .property import Property


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

    def second_q_ops(self) -> List[FermionicOp]:
        """TODO."""
        return [
            sum(
                ints.to_second_q_op() for ints in self._electronic_integrals.values()
            ).reduce()  # type: ignore
        ]

    def interpret(self, result: EigenstateResult) -> None:
        """TODO."""
        if not isinstance(result, ElectronicStructureResult):
            raise QiskitNatureError("TODO")
        if self._reference_energy is not None:
            result.hf_energy = self._reference_energy
        if "nuclear_repulsion_energy" in self._energy_shift.keys():
            result.nuclear_repulsion_energy = self._energy_shift["nuclear_repulsion_energy"]
