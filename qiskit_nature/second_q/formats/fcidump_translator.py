# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Translator methods for the FCIDump."""

from __future__ import annotations

from typing import List, Optional

from ...second_q.properties.bases import ElectronicBasis
from ...second_q.drivers import ElectronicStructureDriver
from ...second_q.properties import (
    ElectronicStructureDriverResult,
    ElectronicEnergy,
    ParticleNumber,
)
from ...second_q.properties.integrals import (
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
)

from ...second_q.problems import ElectronicStructureProblem
from ...second_q.transformers import BaseTransformer

from .fcidump import FCIDump


class FCIDumpDriver(ElectronicStructureDriver):
    """FCIDumpDriver"""

    def __init__(self, fcidump: FCIDump) -> None:
        """
        Args:
            fcidump: FCIDump dataclass
        """
        super().__init__()
        self._fcidump = fcidump

    def run(self) -> ElectronicStructureDriverResult:
        """Returns an ElectronicStructureDriverResult instance out of a FCIDump file."""

        num_beta = (self._fcidump.num_electrons - (self._fcidump.multiplicity - 1)) // 2
        num_alpha = self._fcidump.num_electrons - num_beta

        particle_number = ParticleNumber(
            num_spin_orbitals=self._fcidump.num_orbitals * 2,
            num_particles=(num_alpha, num_beta),
        )

        electronic_energy = ElectronicEnergy(
            [
                OneBodyElectronicIntegrals(
                    ElectronicBasis.MO, (self._fcidump.hij, self._fcidump.hij_b)
                ),
                TwoBodyElectronicIntegrals(
                    ElectronicBasis.MO,
                    (self._fcidump.hijkl, self._fcidump.hijkl_ba, self._fcidump.hijkl_bb, None),
                ),
            ],
            nuclear_repulsion_energy=self._fcidump.nuclear_repulsion_energy,
        )

        driver_result = ElectronicStructureDriverResult()
        driver_result.add_property(electronic_energy)
        driver_result.add_property(particle_number)

        return driver_result


def fcidump_to_problem(
    fcidump: FCIDump,
    *,
    transformers: Optional[List[BaseTransformer]] = None,
) -> ElectronicStructureProblem:
    """Translates FCIDump to Problem."""

    return ElectronicStructureProblem(FCIDumpDriver(fcidump), transformers)
