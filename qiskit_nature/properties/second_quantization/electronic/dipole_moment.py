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

"""The TotalDipoleMoment property."""

from typing import Dict, List, Optional, Tuple, cast

from qiskit_nature.drivers.second_quantization import QMolecule
from qiskit_nature.operators.second_quantization import FermionicOp

from ..second_quantized_property import (DriverResult, ElectronicDriverResult,
                                         SecondQuantizedProperty)
from .bases import ElectronicBasis
from .integrals import (ElectronicIntegrals, IntegralProperty,
                        OneBodyElectronicIntegrals)

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
        electronic_integrals: Dict[ElectronicBasis, List[ElectronicIntegrals]],
        shift: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            axis: the name of the Cartesian axis.
            dipole: an IntegralProperty property representing the dipole moment operator.
        """
        super().__init__(self.__class__.__name__, electronic_integrals, shift=shift)
        self._axis = axis


class TotalDipoleMoment(SecondQuantizedProperty):
    """The TotalDipoleMoment property."""

    def __init__(
        self,
        dipole_axes: Dict[str, DipoleMoment],
        dipole_shift: Optional[Dict[str, DipoleTuple]] = None,
    ):
        """
        Args:
            dipole_axes: a dictionary mapping Cartesian axes to DipoleMoment properties.
            dipole_shift: an optional dictionary of named dipole shifts.
        """
        super().__init__(self.__class__.__name__)
        self._dipole_axes = dipole_axes
        self._dipole_shift = dipole_shift

    @classmethod
    def from_driver_result(cls, result: DriverResult) -> Optional["TotalDipoleMoment"]:
        """Construct a TotalDipoleMoment instance from a QMolecule.

        Args:
            result: the driver result from which to extract the raw data. For this property, a
                QMolecule is required!

        Returns:
            An instance of this property.

        Raises:
            QiskitNatureError: if a WatsonHamiltonian is provided.
        """
        cls._validate_input_type(result, ElectronicDriverResult)

        qmol = cast(QMolecule, result)

        if not qmol.has_dipole_integrals():
            return None

        def dipole_along_axis(axis, ao_ints, mo_ints, energy_shift):
            return DipoleMoment(
                axis,
                {
                    ElectronicBasis.AO: [OneBodyElectronicIntegrals(ElectronicBasis.AO, ao_ints)],
                    ElectronicBasis.MO: [OneBodyElectronicIntegrals(ElectronicBasis.MO, mo_ints)],
                },
                shift=energy_shift,
            )

        return cls(
            {
                "x": dipole_along_axis(
                    "x",
                    (qmol.x_dip_ints, None),
                    (qmol.x_dip_mo_ints, qmol.x_dip_mo_ints_b),
                    qmol.x_dip_energy_shift,
                ),
                "y": dipole_along_axis(
                    "y",
                    (qmol.y_dip_ints, None),
                    (qmol.y_dip_mo_ints, qmol.y_dip_mo_ints_b),
                    qmol.y_dip_energy_shift,
                ),
                "z": dipole_along_axis(
                    "z",
                    (qmol.z_dip_ints, None),
                    (qmol.z_dip_mo_ints, qmol.z_dip_mo_ints_b),
                    qmol.z_dip_energy_shift,
                ),
            },
            dipole_shift={
                "nuclear dipole moment": cast(
                    DipoleTuple, tuple(d_m for d_m in qmol.nuclear_dipole_moment)
                ),
            },
        )

    def reduce_system_size(self, electronic_density, transform) -> "TotalDipoleMoment":
        """TODO."""
        return TotalDipoleMoment(
            {
                axis: dipole.reduce_system_size(electronic_density, transform)
                for axis, dipole in self._dipole_axes.items()
            },
            dipole_shift=self._dipole_shift,
        )

    def second_q_ops(self) -> List[FermionicOp]:
        """Returns a list of dipole moment operators along all Cartesian axes."""
        return [dip.second_q_ops()[0] for dip in self._dipole_axes.values()]
