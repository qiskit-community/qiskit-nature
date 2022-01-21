# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The ElectronicStructureDriverResult class."""

from __future__ import annotations

from typing import List, Tuple, cast

import h5py

from qiskit_nature import ListOrDictType, settings
from qiskit_nature.constants import BOHR
from qiskit_nature.drivers import Molecule
from qiskit_nature.drivers import QMolecule
from qiskit_nature.operators.second_quantization import FermionicOp

from ..second_quantized_property import LegacyDriverResult, SecondQuantizedProperty
from ..driver_metadata import DriverMetadata
from .angular_momentum import AngularMomentum
from .bases import ElectronicBasis, ElectronicBasisTransform
from .dipole_moment import ElectronicDipoleMoment
from .electronic_energy import ElectronicEnergy
from .magnetization import Magnetization
from .particle_number import ParticleNumber
from .types import GroupedElectronicProperty


class ElectronicStructureDriverResult(GroupedElectronicProperty):
    """The ElectronicStructureDriverResult class.

    This is a :class:`~qiskit_nature.properties.GroupedProperty` gathering all property objects
    previously stored in Qiskit Nature's :class:`~qiskit_nature.drivers.QMolecule` object.
    """

    def __init__(self) -> None:
        """
        Property objects should be added via ``add_property`` rather than via the initializer.
        """
        super().__init__(self.__class__.__name__)
        self.molecule: Molecule = None

    def to_hdf5(self, parent: h5py.Group) -> None:
        """TODO."""
        super().to_hdf5(parent)
        group = parent.require_group(self.name)
        self.molecule.to_hdf5(group)

    @classmethod
    def from_hdf5(cls, h5py_group: h5py.Group) -> ElectronicStructureDriverResult:
        """TODO."""
        grouped_property = super().from_hdf5(h5py_group)

        ret = cls()
        for prop in grouped_property:
            if isinstance(prop, Molecule):
                ret.molecule = prop
            else:
                ret.add_property(prop)

        return ret

    @classmethod
    def from_legacy_driver_result(
        cls, result: LegacyDriverResult
    ) -> ElectronicStructureDriverResult:
        """Converts a :class:`~qiskit_nature.drivers.QMolecule` into an
        ``ElectronicStructureDriverResult``.

        Args:
            result: the :class:`~qiskit_nature.drivers.QMolecule` to convert.

        Returns:
            An instance of this property.

        Raises:
            QiskitNatureError: if a :class:`~qiskit_nature.drivers.WatsonHamiltonian` is provided.
        """
        cls._validate_input_type(result, QMolecule)

        qmol = cast(QMolecule, result)

        ret = cls()

        ret.add_property(ElectronicEnergy.from_legacy_driver_result(qmol))
        ret.add_property(ParticleNumber.from_legacy_driver_result(qmol))
        ret.add_property(AngularMomentum.from_legacy_driver_result(qmol))
        ret.add_property(Magnetization.from_legacy_driver_result(qmol))
        ret.add_property(ElectronicDipoleMoment.from_legacy_driver_result(qmol))

        ret.add_property(
            ElectronicBasisTransform(
                ElectronicBasis.AO, ElectronicBasis.MO, qmol.mo_coeff, qmol.mo_coeff_b
            )
        )

        geometry: List[Tuple[str, List[float]]] = []
        for atom, xyz in zip(qmol.atom_symbol, qmol.atom_xyz):
            # QMolecule XYZ defaults to Bohr but Molecule requires Angstrom
            geometry.append((atom, xyz * BOHR))

        ret.molecule = Molecule(geometry, qmol.multiplicity, qmol.molecular_charge)

        ret.add_property(
            DriverMetadata(
                qmol.origin_driver_name,
                qmol.origin_driver_version,
                qmol.origin_driver_config,
            )
        )

        return ret

    def second_q_ops(self) -> ListOrDictType[FermionicOp]:
        """Returns the second quantized operators associated with the properties in this group.

        The actual return-type is determined by `qiskit_nature.settings.dict_aux_operators`.

        Returns:
            A `list` or `dict` of `FermionicOp` objects.
        """
        ops: ListOrDictType[FermionicOp]

        if not settings.dict_aux_operators:
            ops = []
            for cls in [
                ElectronicEnergy,
                ParticleNumber,
                AngularMomentum,
                Magnetization,
                ElectronicDipoleMoment,
            ]:
                prop = self.get_property(cls)  # type: ignore
                if prop is None or not isinstance(prop, SecondQuantizedProperty):
                    continue
                ops.extend(prop.second_q_ops())
            return ops

        ops = {}
        for prop in iter(self):
            if not isinstance(prop, SecondQuantizedProperty):
                continue
            ops.update(prop.second_q_ops())
        return ops
