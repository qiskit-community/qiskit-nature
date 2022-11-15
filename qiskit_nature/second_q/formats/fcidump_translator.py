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

from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.problems import ElectronicBasis, ElectronicStructureProblem
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

    particle_number = ParticleNumber(fcidump.num_orbitals)

    electronic_energy = ElectronicEnergy.from_raw_integrals(
        fcidump.hij, fcidump.hijkl, fcidump.hij_b, fcidump.hijkl_bb, fcidump.hijkl_ba
    )
    electronic_energy.nuclear_repulsion_energy = fcidump.constant_energy

    problem = ElectronicStructureProblem(electronic_energy)
    problem.basis = ElectronicBasis.MO
    problem.num_particles = (num_alpha, num_beta)
    problem.num_spatial_orbitals = fcidump.num_orbitals
    problem.properties.particle_number = particle_number

    return problem
