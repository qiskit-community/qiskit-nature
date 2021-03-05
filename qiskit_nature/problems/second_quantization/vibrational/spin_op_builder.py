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


def build_spin_op(watson_hamiltonian: WatsonHamiltonian, basis_size, truncation_order) -> BosonicOp:
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

    bos_op = BosonicOp(boson_hamilt_harm_basis)

    return bos_op
