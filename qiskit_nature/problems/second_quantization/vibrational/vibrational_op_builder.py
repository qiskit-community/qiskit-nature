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
from typing import Union, List, Optional, Tuple

import logging

from qiskit_nature.drivers import WatsonHamiltonian
from qiskit_nature.drivers.bosonic_bases import BosonicBasis, HarmonicBasis

from qiskit_nature.operators.second_quantization.vibrational_op import VibrationalOp
from qiskit_nature.problems.second_quantization.vibrational.vibrational_label_builder import \
    _create_labels


logger = logging.getLogger(__name__)


def build_vibrational_op(watson_hamiltonian: WatsonHamiltonian,
                         num_modals: Union[int, List[int]],
                         truncation_order: int,
                         basis: Optional[BosonicBasis] = None,
                         ) -> VibrationalOp:
    """
    Builds a :class:`VibrationalOp` based on a :class:`WatsonHamiltonian` object.

    Args:
        watson_hamiltonian: :class:`WatsonHamiltonian` instance.
        num_modals: the number of modals per mode.
        truncation_order: order at which an n-body expansion is truncated
        basis: the :class:`BosonicBasis` to which the :class:`WatsonHamiltonian` gets converted.
        Currently, this argument will be ignored until more bases are available. Therefore, it will
        always use the :class:`HarmonicBasis` internally.

    Returns:
        VibrationalOp: VibrationalOp built from a WatsonHamiltonian object.
    """
    if basis is not None:
        logger.warning(
            'The only supported `BosonicBasis` is the `HarmonicBasis`. However you specified '
            '%s as an input, which will be ignored.', str(basis))

    num_modes = watson_hamiltonian.num_modes

    if isinstance(num_modals, int):
        num_modals = [num_modals] * num_modes

    boson_hamilt_harm_basis = HarmonicBasis(watson_hamiltonian,
                                            num_modals, truncation_order).convert()

    return build_vibrational_op_from_ints(boson_hamilt_harm_basis, num_modes, num_modals)


def build_vibrational_op_from_ints(h_mat: List[List[Tuple[List[List[int]], complex]]],
                                   num_modes: int,
                                   num_modals: List[int],
                                   ) -> VibrationalOp:
    """
    Builds a :class:`VibrationalOp` based on an integral list as produced by
    :meth:`HarmonicBasis.convert()`.

    Args:
        h_mat: integral list.
        num_modes: the number of modes.
        num_modals: the number of modals.

    Returns:
        The constructed VibrationalOp.
    """
    all_labels = _create_labels(h_mat)

    return VibrationalOp(all_labels, num_modes, num_modals)
