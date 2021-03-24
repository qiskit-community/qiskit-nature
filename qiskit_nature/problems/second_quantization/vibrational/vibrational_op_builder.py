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
""" Vibrational operator builder. """
from typing import Union, List

from qiskit_nature.drivers import WatsonHamiltonian
from qiskit_nature.drivers.bosonic_bases import HarmonicBasis

from qiskit_nature.operators.second_quantization.vibrational_op import VibrationalOp
from qiskit_nature.problems.second_quantization.vibrational.vibrational_label_builder import \
    _create_labels


def build_vibrational_op(watson_hamiltonian: WatsonHamiltonian,
                         num_modals: Union[int, List[int]],
                         truncation_order: int) -> VibrationalOp:
    """
    Builds a VibrationalOp based on a WatsonHamiltonian object.

    Args:
        watson_hamiltonian (WatsonHamiltonian): WatsonHamiltonian instance.
        num_modals: the number of modals per mode.
        truncation_order: order at which an n-body expansion is truncated

    Returns:
        VibrationalOp: VibrationalOp built from a WatsonHamiltonian object.
    """
    num_modes = watson_hamiltonian.num_modes

    if isinstance(num_modals, int):
        num_modals = [num_modals] * num_modes

    # TODO: make HarmonicBasis an argument and support other bases when implemented
    boson_hamilt_harm_basis = HarmonicBasis(watson_hamiltonian,
                                            num_modals, truncation_order).convert()

    all_labels = _create_labels(boson_hamilt_harm_basis)

    return VibrationalOp(all_labels, num_modes, num_modals)
