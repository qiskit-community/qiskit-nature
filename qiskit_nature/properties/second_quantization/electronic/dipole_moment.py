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

from typing import Dict, List, Optional, Tuple, cast

import h5py

from qiskit_nature import ListOrDictType, settings
from qiskit_nature.drivers import QMolecule
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.results import EigenstateResult

from ...grouped_property import GroupedProperty
from ..second_quantized_property import LegacyDriverResult
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
    ) -> None:
        """
        Args:
            axis: the name of the Cartesian axis.
            dipole: an IntegralProperty property representing the dipole moment operator.
            shift: an optional dictionary of dipole moment shifts.
        """
        self._axis = axis
        name = self.__class__.__name__ + axis.upper()
        super().__init__(name, electronic_integrals, shift=shift)

    def to_hdf5(self, parent: h5py.Group):
        """TODO."""
        super().to_hdf5(parent)
        group = parent.require_group(self.name)

        group.attrs["axis"] = self._axis

    @classmethod
    def from_hdf5(cls, h5py_group: h5py.Group) -> "DipoleMoment":
        """TODO."""
        integral_property = super().from_hdf5(h5py_group)

        axis = h5py_group.attrs["axis"]

        return cls(axis, list(integral_property), shift=integral_property._shift)

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
        if ElectronicBasis.AO not in self._electronic_integrals:
            raise NotImplementedError(
                "Construction of the DipoleMoment's integral operator without AO integrals is not "
                "yet implemented."
            )

        return cast(OneBodyElectronicIntegrals, self.get_electronic_integral(ElectronicBasis.AO, 1))

    def interpret(self, result: EigenstateResult) -> None:
        """Interprets an :class:`~qiskit_nature.results.EigenstateResult` in this property's context.

        Args:
            result: the result to add meaning to.
        """
        pass


class ElectronicDipoleMoment(GroupedProperty[DipoleMoment], ElectronicProperty):
    """The ElectronicDipoleMoment property.

    This Property computes **purely** the electronic dipole moment (possibly minus additional shifts
    introduced via e.g. classical transformers). However, for convenience it provides a storage
    location for the nuclear dipole moment. If available, this information will be used during the
    call of ``interpret`` to provide the electronic, nuclear and total dipole moments in the result
    object.
    """

    def __init__(
        self,
        # NOTE: The fact that this first argument is not optional is a bit inconsistent with the
        # other `GroupedProperty` (sub-)classes. Especially in the docstring of
        # `ElectronicStructureDriverResult` we say that group components should be added via the
        # `add_property` method. Why don't we allow the same here?
        # Thus, I suggest to make this first argument optional.
        dipole_axes: Optional[List[DipoleMoment]] = None,
        dipole_shift: Optional[Dict[str, DipoleTuple]] = None,
        nuclear_dipole_moment: Optional[DipoleTuple] = None,
        reverse_dipole_sign: bool = False,
    ) -> None:
        """
        Args:
            dipole_axes: a dictionary mapping Cartesian axes to DipoleMoment properties.
            dipole_shift: an optional dictionary of named dipole shifts.
            nuclear_dipole_moment: the optional nuclear dipole moment.
            reverse_dipole_sign: indicates whether the sign of the electronic dipole components
                needs to be reversed in order to match the nuclear dipole moment direction.
        """
        super().__init__(self.__class__.__name__)
        self._dipole_shift = dipole_shift
        self._nuclear_dipole_moment = nuclear_dipole_moment
        self._reverse_dipole_sign = reverse_dipole_sign
        if dipole_axes is not None:
            for dipole in dipole_axes:
                self.add_property(dipole)

    def to_hdf5(self, parent: h5py.Group):
        """TODO."""
        super().to_hdf5(parent)

        group = parent.require_group(self.name)

        group.attrs["reverse dipole sign"] = self._reverse_dipole_sign

        if self._nuclear_dipole_moment is not None:
            group.attrs["Nuclear Dipole Moment"] = self._nuclear_dipole_moment

        dipole_shift_group = group.create_group("dipole_shift")
        if self._dipole_shift is not None:
            for name, shift in self._dipole_shift.items():
                dipole_shift_group.attrs[name] = shift

    @classmethod
    def from_hdf5(cls, h5py_group: h5py.Group) -> "ElectronicDipoleMoment":
        """TODO."""
        grouped_property = super().from_hdf5(h5py_group)

        ret = cls(list(grouped_property))

        ret.reverse_dipole_sign = h5py_group.attrs["reverse dipole sign"]
        ret.nuclear_dipole_moment = h5py_group.attrs.get("Nuclear Dipole Moment", None)

        for name, shift in h5py_group["dipole_shift"].attrs.items():
            ret._dipole_shift[name] = shift

        return ret

    @property
    def nuclear_dipole_moment(self) -> Optional[DipoleTuple]:
        """Returns the nuclear dipole moment."""
        return self._nuclear_dipole_moment

    @nuclear_dipole_moment.setter
    def nuclear_dipole_moment(self, nuclear_dipole_moment: Optional[DipoleTuple]) -> None:
        """Sets the nuclear dipole moment."""
        self._nuclear_dipole_moment = nuclear_dipole_moment

    @property
    def reverse_dipole_sign(self) -> bool:
        """Returns whether or not the sign of the electronic dipole components needs to be reversed
        in order to match the nuclear dipole moment direction."""
        return self._reverse_dipole_sign

    @reverse_dipole_sign.setter
    def reverse_dipole_sign(self, reverse_dipole_sign: bool) -> None:
        """Sets whether or not the sign of the electronic dipole components needs to be reversed in
        order to match the nuclear dipole moment direction."""
        self._reverse_dipole_sign = reverse_dipole_sign

    @classmethod
    def from_legacy_driver_result(
        cls, result: LegacyDriverResult
    ) -> Optional["ElectronicDipoleMoment"]:
        """Construct an ElectronicDipoleMoment instance from a
        :class:`~qiskit_nature.drivers.QMolecule`.

        Args:
            result: the driver result from which to extract the raw data. For this property, a
                :class:`~qiskit_nature.drivers.QMolecule` is required!

        Returns:
            An instance of this property.

        Raises:
            QiskitNatureError: if a :class:`~qiskit_nature.drivers.WatsonHamiltonian` is provided.
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

        ret = cls()

        ret.add_property(
            dipole_along_axis(
                "x",
                (qmol.x_dip_ints, None),
                (qmol.x_dip_mo_ints, qmol.x_dip_mo_ints_b),
                qmol.x_dip_energy_shift,
            )
        )
        ret.add_property(
            dipole_along_axis(
                "y",
                (qmol.y_dip_ints, None),
                (qmol.y_dip_mo_ints, qmol.y_dip_mo_ints_b),
                qmol.y_dip_energy_shift,
            )
        )
        ret.add_property(
            dipole_along_axis(
                "z",
                (qmol.z_dip_ints, None),
                (qmol.z_dip_mo_ints, qmol.z_dip_mo_ints_b),
                qmol.z_dip_energy_shift,
            )
        )

        ret.nuclear_dipole_moment = nuclear_dipole_moment
        ret.reverse_dipole_sign = qmol.reverse_dipole_sign

        return ret

    def second_q_ops(self) -> ListOrDictType[FermionicOp]:
        """Returns the second quantized dipole moment operators.

        The actual return-type is determined by `qiskit_nature.settings.dict_aux_operators`.

        Returns:
            A `list` or `dict` of `FermionicOp` objects.
        """
        ops: ListOrDictType[FermionicOp]
        if not settings.dict_aux_operators:
            ops = [dip.second_q_ops()[0] for dip in self._properties.values()]
            return ops

        ops = {}
        for prop in iter(self):
            ops.update(prop.second_q_ops())
        return ops

    def interpret(self, result: EigenstateResult) -> None:
        """Interprets an :class:`~qiskit_nature.results.EigenstateResult` in this property's context.

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
            aux_operator_eigenvalues = result.aux_operator_eigenvalues

        for aux_op_eigenvalues in aux_operator_eigenvalues:
            if aux_op_eigenvalues is None:
                continue

            if isinstance(aux_op_eigenvalues, list) and len(aux_op_eigenvalues) < 6:
                continue

            axes_order = {"x": 0, "y": 1, "z": 2}
            dipole_moment = [None] * 3
            for prop in iter(self):
                moment: Optional[Tuple[complex, complex]]
                try:
                    moment = aux_op_eigenvalues[axes_order[prop._axis] + 3]
                except KeyError:
                    moment = aux_op_eigenvalues.get(prop.name, None)
                if moment is not None:
                    dipole_moment[axes_order[prop._axis]] = moment[0].real  # type: ignore

            result.computed_dipole_moment.append(cast(DipoleTuple, tuple(dipole_moment)))
            dipole_shifts: Dict[str, Dict[str, complex]] = {}
            for prop in self._properties.values():
                for name, shift in prop._shift.items():
                    if name not in dipole_shifts:
                        dipole_shifts[name] = {}
                    dipole_shifts[name][prop._axis] = shift

            result.extracted_transformer_dipoles.append(
                {
                    name: cast(DipoleTuple, (shift["x"], shift["y"], shift["z"]))
                    for name, shift in dipole_shifts.items()
                }
            )
