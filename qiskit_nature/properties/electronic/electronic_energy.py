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

"""The ElectronicEnergy property."""

from typing import cast, Dict, List, Optional, Union

from qiskit_nature import QiskitNatureError
from qiskit_nature.drivers.second_quantization import QMolecule, WatsonHamiltonian
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.results import EigenstateResult

from .bases import ElectronicBasis
from .integrals import (
    ElectronicIntegrals,
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
)
from ..property import Property


class ElectronicEnergy(Property):
    """The ElectronicEnergy property.

    This is the main property of any electronic structure problem. It constructs the Hamiltonian
    whose eigenvalue is the target of a later used Quantum algorithm.
    """

    def __init__(
        self,
        basis: ElectronicBasis,
        electronic_integrals: Dict[int, ElectronicIntegrals],
        reference_energy: Optional[float] = None,
        energy_shift: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            basis: the basis which the integrals in ``electronic_integrals`` are stored in.
            electronic_integrals: a dictionary mapping the ``# body terms`` to the corresponding
                ``ElectronicIntegrals``.
            reference_energy: an optional reference energy (such as the HF energy).
            energy_shift: an optional dictionary of energy shifts.
        """
        super().__init__(self.__class__.__name__)
        self._basis = basis
        self._electronic_integrals = electronic_integrals
        self._energy_shift = energy_shift
        self._reference_energy = reference_energy

    @classmethod
    def from_driver_result(cls, result: Union[QMolecule, WatsonHamiltonian]) -> "ElectronicEnergy":
        """Construct an ElectronicEnergy instance from a QMolecule.

        Args:
            result: the driver result from which to extract the raw data. For this property, a
                QMolecule is required!

        Returns:
            An instance of this property.

        Raises:
            QiskitNatureError: if a WatsonHamiltonian is provided.
        """
        cls._validate_input_type(result, QMolecule)

        qmol = cast(QMolecule, result)

        energy_shift = qmol.energy_shift.copy()
        energy_shift["nuclear repulsion"] = qmol.nuclear_repulsion_energy

        return cls(
            ElectronicBasis.MO,
            {
                1: OneBodyElectronicIntegrals(
                    ElectronicBasis.MO, (qmol.mo_onee_ints, qmol.mo_onee_ints_b)
                ),
                2: TwoBodyElectronicIntegrals(
                    ElectronicBasis.MO,
                    (
                        qmol.mo_eri_ints,
                        qmol.mo_eri_ints_ba,
                        qmol.mo_eri_ints_bb,
                        None,
                    ),
                ),
            },
            reference_energy=qmol.hf_energy,
            energy_shift=energy_shift,
        )

    def second_q_ops(self) -> List[FermionicOp]:
        """Returns a list containing the Hamiltonian constructed by the stored electronic integrals."""
        return [
            sum(  # type: ignore
                ints.to_second_q_op() for ints in self._electronic_integrals.values()
            ).reduce()
        ]

    def interpret(self, result: EigenstateResult) -> None:
        """Interprets an `qiskit_nature.result.EigenstateResult` in the context of this Property.

        This is currently a method stub which may be used in the future.

        Args:
            result: the result to add meaning to.
        """
        raise NotImplementedError()
