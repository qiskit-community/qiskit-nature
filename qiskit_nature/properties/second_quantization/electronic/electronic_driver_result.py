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

"""The ElectronicDriverResult class."""

from typing import Dict, List, cast

from qiskit_nature.drivers.second_quantization import QMolecule
from qiskit_nature.operators.second_quantization import FermionicOp

from ..second_quantized_property import DriverResult
from ..second_quantized_property import \
    ElectronicDriverResult as LegacyElectronicDriverResult
from ..second_quantized_property import SecondQuantizedProperty
from .angular_momentum import AngularMomentum
from .bases import ElectronicBasis, ElectronicBasisTransform
from .dipole_moment import DipoleMoment, TotalDipoleMoment
from .electronic_energy import ElectronicEnergy
from .integrals import OneBodyElectronicIntegrals, TwoBodyElectronicIntegrals
from .magnetization import Magnetization
from .particle_number import ParticleNumber


class ElectronicDriverResult(SecondQuantizedProperty):
    """TODO."""

    def __init__(self) -> None:
        """TODO."""
        super().__init__(self.__class__.__name__)
        self._properties: Dict[str, SecondQuantizedProperty] = {}
        self.electronic_basis_transform: ElectronicBasisTransform = None
        # TODO: add origin driver metadata
        # TODO: where to put orbital_energies?
        # TODO: add molecule geometry metadata
        # TODO: where to put kinetic, overlap matrices? Do we want explicit Fock matrix?

    def add_property(self, prop: SecondQuantizedProperty) -> None:
        """TODO."""
        self._properties[prop.name] = prop

    def reduce_system_size(self, active_orbital_indices: List[int]) -> "ElectronicDriverResult":
        """TODO."""
        raise NotADirectoryError()

    @classmethod
    def from_driver_result(cls, result: DriverResult) -> "ElectronicDriverResult":
        """TODO."""
        cls._validate_input_type(result, LegacyElectronicDriverResult)

        ret = cls()

        qmol = cast(QMolecule, result)

        ret.add_property(ElectronicEnergy.from_driver_result(qmol))
        ret.add_property(ParticleNumber.from_driver_result(qmol))
        ret.add_property(AngularMomentum.from_driver_result(qmol))
        ret.add_property(Magnetization.from_driver_result(qmol))
        ret.add_property(TotalDipoleMoment.from_driver_result(qmol))

        ret.electronic_basis_transform = ElectronicBasisTransform(
            ElectronicBasis.AO, ElectronicBasis.MO, qmol.mo_coeff, qmol.mo_coeff_b
        )

        return ret

    def second_q_ops(self) -> List[FermionicOp]:
        """TODO."""
        ops: List[FermionicOp] = []
        ops.extend(self.electronic_energy_mo.second_q_ops())
        ops.extend(self.particle_number.second_q_ops())
        ops.extend(self.angular_momentum.second_q_ops())
        ops.extend(self.magnetization.second_q_ops())
        ops.extend(self.total_dipole_moment.second_q_ops())
        return ops
