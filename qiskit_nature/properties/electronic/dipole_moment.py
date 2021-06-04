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

from typing import cast, Dict, List, Optional, Tuple, Union

from qiskit_nature import QiskitNatureError
from qiskit_nature.drivers import QMolecule, WatsonHamiltonian
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.results import EigenstateResult

from .bases import ElectronicBasis
from .electronic_energy import ElectronicEnergy
from .integrals import OneBodyElectronicIntegrals
from ..property import Property

# A dipole moment, when present as X, Y and Z components will normally have float values for all the
# components. However when using Z2Symmetries, if the dipole component operator does not commute
# with the symmetry then no evaluation is done and None will be used as the 'value' indicating no
# measurement of the observable took place
DipoleTuple = Tuple[Optional[float], Optional[float], Optional[float]]


class DipoleMoment(Property):
    """TODO."""

    def __init__(
        self,
        axis: str,
        dipole: ElectronicEnergy,
    ):
        """TODO."""
        super().__init__(self.__class__.__name__)
        self._axis = axis
        self._dipole = dipole

    @classmethod
    def from_driver_result(cls, result: Union[QMolecule, WatsonHamiltonian]) -> None:
        """TODO."""
        raise NotImplementedError()

    def second_q_ops(self) -> List[FermionicOp]:
        """TODO."""
        return self._dipole.second_q_ops()

    def interpret(self, result: EigenstateResult) -> None:
        """TODO."""
        pass


class TotalDipoleMoment(Property):
    """TODO."""

    def __init__(
        self,
        dipole_axes: Dict[str, DipoleMoment],
        dipole_shift: Optional[Dict[str, DipoleTuple]] = None,
    ):
        """TODO."""
        super().__init__(self.__class__.__name__)
        self._dipole_axes = dipole_axes
        self._dipole_shift = dipole_shift

    @classmethod
    def from_driver_result(
        cls, result: Union[QMolecule, WatsonHamiltonian]
    ) -> Optional["TotalDipoleMoment"]:
        """TODO."""
        if isinstance(result, WatsonHamiltonian):
            raise QiskitNatureError("TODO.")

        qmol = cast(QMolecule, result)

        if not qmol.has_dipole_integrals():
            return None

        return cls(
            {
                "x": DipoleMoment(
                    "x",
                    ElectronicEnergy(
                        ElectronicBasis.MO,
                        {
                            1: OneBodyElectronicIntegrals(
                                ElectronicBasis.MO,
                                (qmol.x_dip_mo_ints, qmol.x_dip_mo_ints_b),
                            )
                        },
                        energy_shift=qmol.x_dip_energy_shift,
                    ),
                ),
                "y": DipoleMoment(
                    "y",
                    ElectronicEnergy(
                        ElectronicBasis.MO,
                        {
                            1: OneBodyElectronicIntegrals(
                                ElectronicBasis.MO,
                                (qmol.y_dip_mo_ints, qmol.y_dip_mo_ints_b),
                            )
                        },
                        energy_shift=qmol.y_dip_energy_shift,
                    ),
                ),
                "z": DipoleMoment(
                    "z",
                    ElectronicEnergy(
                        ElectronicBasis.MO,
                        {
                            1: OneBodyElectronicIntegrals(
                                ElectronicBasis.MO,
                                (qmol.z_dip_mo_ints, qmol.z_dip_mo_ints_b),
                            )
                        },
                        energy_shift=qmol.z_dip_energy_shift,
                    ),
                ),
            },
            dipole_shift={
                "nuclear dipole moment": cast(
                    DipoleTuple, tuple(d_m for d_m in qmol.nuclear_dipole_moment)
                ),
            },
        )

    def second_q_ops(self) -> List[FermionicOp]:
        """TODO."""
        return [dip.second_q_ops()[0] for dip in self._dipole_axes.values()]

    def interpret(self, result: EigenstateResult) -> None:
        """TODO."""
        pass
