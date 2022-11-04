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

"""Translator methods for the Watson Hamiltonian."""

from __future__ import annotations

from collections import defaultdict

from qiskit_nature.second_q.hamiltonians import VibrationalEnergy
from qiskit_nature.second_q.problems import VibrationalStructureProblem, VibrationalBasis
from qiskit_nature.second_q.properties import OccupiedModals

from .watson import WatsonHamiltonian


def watson_to_problem(
    watson: WatsonHamiltonian,
    basis: VibrationalBasis,
) -> VibrationalStructureProblem:
    """Builds out a :class:`.VibrationalStructureProblem` from a :class:`.WatsonHamiltonian`.

    .. note::

        In the process of constructing the :class:`.VibrationalStructureProblem`, the coefficients
        stored in the :class:`.WatsonHamiltonian` need to be mapped to a second-quantization basis.
        For more details about this, please refer to the documentation of
        :meth:`.VibrationalBasis.map`.

    Args:
        watson: the ``WatsonHamiltonian`` object from which to build the problem.
        basis: the ``VibrationalBasis`` into which to map the hamiltonian coefficients.

    Returns:
        A :class:`.VibrationalStructureProblem` instance.
    """
    nbody: dict[tuple[int, ...], complex] = defaultdict(complex)

    for coefficient, modes in watson:
        for integral, modal_index in basis.map(coefficient, modes):
            nbody[modal_index] += integral

    hamiltonian = VibrationalEnergy.from_raw_integrals(nbody)

    problem = VibrationalStructureProblem(hamiltonian)
    problem.basis = basis
    problem.properties.occupied_modals = OccupiedModals(basis.num_modals)

    return problem
