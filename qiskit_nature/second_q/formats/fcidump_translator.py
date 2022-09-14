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

from qiskit_nature.second_q.properties.bases import ElectronicBasis
from qiskit_nature.second_q.properties.integrals import (
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
)
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.properties import ParticleNumber

from .fcidump import FCIDump


def fcidump_to_problem(fcidump: FCIDump) -> ElectronicStructureProblem:
    """Builds out an :class:`.ElectronicStructureProblem` from a :class:`.FCIDump` instance.

    This method centralizes the construction of an :class:`.ElectronicStructureProblem` from a
    :class:`.FCIDump`.

    Args:
        fcidump: the :class:`.FCIDump` object from which to build the problem.

    Returns:
        An :class:`.ElectronicStructureProblem` instance.
    """

    num_beta = (fcidump.num_electrons - (fcidump.multiplicity - 1)) // 2
    num_alpha = fcidump.num_electrons - num_beta

    particle_number = ParticleNumber(
        num_spin_orbitals=fcidump.num_orbitals * 2,
        num_particles=(num_alpha, num_beta),
    )

    electronic_energy = ElectronicEnergy(
        [
            OneBodyElectronicIntegrals(ElectronicBasis.MO, (fcidump.hij, fcidump.hij_b)),
            TwoBodyElectronicIntegrals(
                ElectronicBasis.MO,
                (fcidump.hijkl, fcidump.hijkl_ba, fcidump.hijkl_bb, None),
            ),
        ],
        nuclear_repulsion_energy=fcidump.constant_energy,
    )

    problem = ElectronicStructureProblem(electronic_energy)
    problem.properties.particle_number = particle_number
    return problem
