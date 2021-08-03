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

"""The ElectronicStructureDriverResult class."""

from typing import TYPE_CHECKING, Any, List, Tuple, Union, cast

from qiskit_nature.operators.second_quantization import FermionicOp

from ..driver_metadata import DriverMetadata
from .angular_momentum import AngularMomentum
from .bases import ElectronicBasis, ElectronicBasisTransform
from .dipole_moment import ElectronicDipoleMoment
from .electronic_energy import ElectronicEnergy
from .magnetization import Magnetization
from .particle_number import ParticleNumber
from .types import GroupedElectronicProperty

if TYPE_CHECKING:
    from qiskit_nature.molecule import Molecule


class ElectronicStructureDriverResult(GroupedElectronicProperty):
    """The ElectronicStructureDriverResult class.

    This is a :class:~qiskit_nature.properties.GroupedProperty gathering all property objects
    previously stored in Qiskit Nature's `QMolecule` object.
    """

    def __init__(self) -> None:
        """
        Property objects should be added via `add_property` rather than via the initializer.
        """
        super().__init__(self.__class__.__name__)
        self.molecule: "Molecule" = None

    @classmethod
    def from_legacy_driver_result(cls, result: Any) -> "ElectronicStructureDriverResult":
        """Converts a QMolecule into an `ElectronicStructureDriverResult`.

        Args:
            result: the QMolecule to convert.

        Returns:
            An instance of this property.

        Raises:
            QiskitNatureError: if a WatsonHamiltonian is provided.
        """
        # pylint: disable=import-outside-toplevel
        from qiskit_nature.drivers import Molecule
        from qiskit_nature.drivers import QMolecule as LegacyQMolecule
        from qiskit_nature.drivers.second_quantization import QMolecule

        cls._validate_input_type(result, Union[QMolecule, LegacyQMolecule])

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
            geometry.append((atom, xyz))

        ret.molecule = Molecule(geometry, qmol.multiplicity, qmol.molecular_charge)

        ret.add_property(
            DriverMetadata(
                qmol.origin_driver_name,
                qmol.origin_driver_version,
                qmol.origin_driver_config,
            )
        )

        return ret

    def second_q_ops(self) -> List[FermionicOp]:
        """Returns the list of `FermioncOp`s given by the properties contained in this one."""
        ops: List[FermionicOp] = []
        # TODO: make aux_ops a Dict? Then we don't need to hard-code the order of these properties.
        # NOTE: this will also get rid of the hard-coded aux_operator eigenvalue indices in the
        # `interpret` methods of all of these properties
        for cls in [
            ElectronicEnergy,
            ParticleNumber,
            AngularMomentum,
            Magnetization,
            ElectronicDipoleMoment,
        ]:
            prop = self.get_property(cls)  # type: ignore
            if prop is None:
                continue
            ops.extend(prop.second_q_ops())
        return ops
