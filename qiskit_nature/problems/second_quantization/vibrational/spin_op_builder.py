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

from qiskit_nature import WatsonHamiltonian
from qiskit_nature.components.bosonic_bases import HarmonicBasis


# TODO complete once a relevant Vibrational SpinOp class created
def build_spin_op(watson_hamiltonian: WatsonHamiltonian, basis_size, truncation_order):
    """
    Builds a spin operator based on a WatsonHamiltonian object.

    Args:
        watson_hamiltonian (WatsonHamiltonian): WatsonHamiltonian instance.
        basis_size: size of a basis
        truncation_order: order at which an n-body expansion is truncated

    Returns:
        SpinOp: SpinOp built from a QMolecule object.
    """

    num_modes = watson_hamiltonian.num_modes
    basis_size = [basis_size] * num_modes
    boson_hamilt_harm_basis = HarmonicBasis(watson_hamiltonian,  # type: ignore
                                            basis_size, truncation_order).convert()
    all_labels = _create_labels(boson_hamilt_harm_basis, truncation_order)

    # bos_op = BosonicOp(boson_hamilt_harm_basis) #TODO switch to Vibrational Spin Op

    return False


def _create_labels(boson_hamilt_harm_basis, truncation_order):
    all_labels = []
    for num_body in range(truncation_order):
        num_body_data = boson_hamilt_harm_basis[num_body]
        num_body_labels = _create_num_body_labels(num_body_data)
        all_labels.extend(num_body_labels)
    return all_labels


def _create_num_body_labels(num_body_data):
    num_body_labels = []
    for indices, coeff in num_body_data:
        coeff_label = _create_label_for_coeff(indices)
        num_body_labels.append((coeff_label, coeff))
    return num_body_labels


def _create_label_for_coeff(indices):
    raise_labels_list = []
    lower_labels_list = []
    for mode, modal_raise, modal_lower in indices:
        raise_labels_list.append("".join(['+_', str(mode), '*', str(modal_raise)]))
        lower_labels_list.append("".join(['-_', str(mode), '*', str(modal_lower)]))
    complete_labels_list = raise_labels_list + lower_labels_list
    complete_label = " ".join(complete_labels_list)
    return complete_label
