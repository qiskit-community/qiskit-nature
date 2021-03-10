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
""" Spin operator builder. """

import itertools
from typing import List, Tuple

import numpy as np

from qiskit_nature import WatsonHamiltonian
from qiskit_nature.components.bosonic_bases import HarmonicBasis
from qiskit_nature.operators.second_quantization import BosonicOp


def build_spin_op(watson_hamiltonian: WatsonHamiltonian, basis_size, truncation_order):
    """
    Builds a spin operator based on a WatsonHamiltonian object.

    Args:
        watson_hamiltonian (WatsonHamiltonian): WatsonHamiltonian instance.

    Returns:
        SpinOp: SpinOp built from a QMolecule object.
    """

    num_modes = watson_hamiltonian.num_modes
    basis_size = [basis_size] * num_modes
    boson_hamilt_harm_basis = HarmonicBasis(watson_hamiltonian,  # type: ignore
                                            basis_size, truncation_order).convert()
    all_labels = []
    for num_body in range(truncation_order):
        ind = num_body
        num_body_data = boson_hamilt_harm_basis[ind]
        num_body_labels = _create_num_body_labels(num_body_data)
        all_labels.extend(num_body_labels)

    # bos_op = BosonicOp(boson_hamilt_harm_basis) #TODO switch to Vibrational Spin Op

    return False


def _create_num_body_labels(num_body_data):
    num_body_labels = []
    for indices, coeff in num_body_data:
        for mode, modal_raise, modal_lower in indices:
            raise_label = "".join(['+_', str(mode), '_', str(modal_raise)])
            lower_label = "".join(['+_', str(mode), '_', str(modal_raise)])
            num_body_labels.append((raise_label, coeff))
            num_body_labels.append((lower_label, coeff))
    return num_body_labels

