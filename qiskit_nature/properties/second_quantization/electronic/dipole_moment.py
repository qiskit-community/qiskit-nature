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

"""The ElectronicDipoleMoment property."""

from typing import Any, Dict, List, Optional, Tuple, Union, cast

from qiskit_nature.drivers import QMolecule
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.results import EigenstateResult

from ...grouped_property import GroupedProperty
from .bases import ElectronicBasis
from .integrals import ElectronicIntegrals, IntegralProperty, OneBodyElectronicIntegrals
from .types import ElectronicProperty

# A dipole moment, when present as X, Y and Z components will normally have float values for all the
# components. However when using Z2Symmetries, if the dipole component operator does not commute
# with the symmetry then no evaluation is done and None will be used as the 'value' indicating no
# measurement of the observable took place
DipoleTuple = Tuple[Optional[float], Optional[float], Optional[float]]


class DipoleMoment(IntegralProperty):
    """The DipoleMoment property.

    This contains the dipole moment along a single Cartesian axis.
    """

    def __init__(
        self,
        axis: str,
        electronic_integrals: List[ElectronicIntegrals],
        shift: Optional[Dict[str, complex]] = None,
    ):
        """
        Args:
            axis: the name of the Cartesian axis.
            dipole: an IntegralProperty property representing the dipole moment operator.
        """
        self._axis = axis
        name = self.__class__.__name__ + axis.upper()
        super().__init__(name, electronic_integrals, shift=shift)

    def integral_operator(self, density: OneBodyElectronicIntegrals) -> OneBodyElectronicIntegrals:
        """Returns the AO 1-electron integrals.

        The operator for a dipole moment is simply given by the 1-electron integrals and does not
        require the density for its construction.

        Args:
            density: the electronic density at which to compute the operator.

        Returns:
            OneBodyElectronicIntegrals: the operator stored as ElectronicIntegrals.

        Raises:
            NotImplementedError: if no AO electronic integrals are available.
        """
        if ElectronicBasis.AO not in self._electronic_integrals.keys():
            raise NotImplementedError(
                "Construction of the DipoleMoment's integral operator without AO integrals is not "
                "yet implemented."
            )

        return cast(OneBodyElectronicIntegrals, self.get_electronic_integral(ElectronicBasis.AO, 1))

    def interpret(self, result: EigenstateResult) -> None:
        """Interprets an :class:~qiskit_nature.result.EigenstateResult in this property's context.

        Args:
            result: the result to add meaning to.
        """
        pass


class ElectronicDipoleMoment(GroupedProperty[DipoleMoment], ElectronicProperty):
    """The ElectronicDipoleMoment property.

    This Property computes **purely** the electronic dipole moment (possibly minus additional shifts
    introduced via e.g. classical transformers). However, for convenience it provides a storage
    location for the nuclear dipole moment. If available, this information will be used during the
    call of `interpret` to provide the electronic, nuclear and total dipole moments in the result
    object.
    """

    def __init__(
        self,
        dipole_axes: List[DipoleMoment],
        dipole_shift: Optional[Dict[str, DipoleTuple]] = None,
        nuclear_dipole_moment: Optional[DipoleTuple] = None,
        reverse_dipole_sign: bool = False,
    ):
        """
        Args:
            dipole_axes: a dictionary mapping Cartesian axes to DipoleMoment properties.
            dipole_shift: an optional dictionary of named dipole shifts.
        """
        super().__init__(self.__class__.__name__)
        self._dipole_shift = dipole_shift
        self._nuclear_dipole_moment = nuclear_dipole_moment
        self._reverse_dipole_sign = reverse_dipole_sign
        for dipole in dipole_axes:
            self.add_property(dipole)

    @property
    def nuclear_dipole_moment(self) -> Optional[DipoleTuple]:
        """Returns the nuclear dipole moment."""
        return self._nuclear_dipole_moment

    @nuclear_dipole_moment.setter
    def nuclear_dipole_moment(self, nuclear_dipole_moment: Optional[DipoleTuple]) -> None:
        """Sets the nuclear dipole moment."""
        self._nuclear_dipole_moment = nuclear_dipole_moment

    @classmethod
    def from_legacy_driver_result(cls, result: Any) -> Optional["ElectronicDipoleMoment"]:
        """Construct a ElectronicDipoleMoment instance from a QMolecule.

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

        if not qmol.has_dipole_integrals():
            return None

        def dipole_along_axis(axis, ao_ints, mo_ints, energy_shift):
            integrals = []
            if ao_ints[0] is not None:
                integrals.append(OneBodyElectronicIntegrals(ElectronicBasis.AO, ao_ints))
            if mo_ints[0] is not None:
                integrals.append(OneBodyElectronicIntegrals(ElectronicBasis.MO, mo_ints))

            return DipoleMoment(axis, integrals, shift=energy_shift)

        nuclear_dipole_moment: DipoleTuple = None
        if qmol.nuclear_dipole_moment is not None:
            nuclear_dipole_moment = cast(
                DipoleTuple, tuple(d_m for d_m in qmol.nuclear_dipole_moment)
            )

        return cls(
            [
                dipole_along_axis(
                    "x",
                    (qmol.x_dip_ints, None),
                    (qmol.x_dip_mo_ints, qmol.x_dip_mo_ints_b),
                    qmol.x_dip_energy_shift,
                ),
                dipole_along_axis(
                    "y",
                    (qmol.y_dip_ints, None),
                    (qmol.y_dip_mo_ints, qmol.y_dip_mo_ints_b),
                    qmol.y_dip_energy_shift,
                ),
                dipole_along_axis(
                    "z",
                    (qmol.z_dip_ints, None),
                    (qmol.z_dip_mo_ints, qmol.z_dip_mo_ints_b),
                    qmol.z_dip_energy_shift,
                ),
            ],
            nuclear_dipole_moment=nuclear_dipole_moment,
            reverse_dipole_sign=qmol.reverse_dipole_sign,
        )

    def second_q_ops(self) -> List[FermionicOp]:
        """Returns a list of dipole moment operators along all Cartesian axes."""
        return [dip.second_q_ops()[0] for dip in self._properties.values()]

    def interpret(self, result: EigenstateResult) -> None:
        """Interprets an :class:~qiskit_nature.result.EigenstateResult in this property's context.

        Args:
            result: the result to add meaning to.
        """
        result.nuclear_dipole_moment = self._nuclear_dipole_moment
        result.reverse_dipole_sign = self._reverse_dipole_sign
        result.computed_dipole_moment = []
        result.extracted_transformer_dipoles = []

        if not isinstance(result.aux_operator_eigenvalues, list):
            aux_operator_eigenvalues = [result.aux_operator_eigenvalues]
        else:
            aux_operator_eigenvalues = result.aux_operator_eigenvalues  # type: ignore
        for aux_op_eigenvalues in aux_operator_eigenvalues:
            if aux_op_eigenvalues is None:
                continue

            if len(aux_op_eigenvalues) >= 6:
                dipole_moment = []
                for moment in aux_op_eigenvalues[3:6]:
                    if moment is not None:
                        dipole_moment += [moment[0].real]  # type: ignore
                    else:
                        dipole_moment += [None]

                result.computed_dipole_moment.append(cast(DipoleTuple, tuple(dipole_moment)))
                dipole_shifts: Dict[str, Dict[str, complex]] = {}
                for prop in self._properties.values():
                    for name, shift in prop._shift.items():
                        if name not in dipole_shifts.keys():
                            dipole_shifts[name] = {}
                        dipole_shifts[name][prop._axis] = shift

                result.extracted_transformer_dipoles.append(
                    {
                        name: cast(DipoleTuple, (shift["x"], shift["y"], shift["z"]))
                        for name, shift in dipole_shifts.items()
                    }
                )
