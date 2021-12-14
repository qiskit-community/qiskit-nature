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

"""The Ising model"""
import logging
from fractions import Fraction
from typing import Optional

import numpy as np

from qiskit_nature.operators.second_quantization import SpinOp

from .lattice_model import LatticeModel

logger = logging.getLogger(__name__)


class IsingModel(LatticeModel):
    """The Ising model."""

    def coupling_matrix(self) -> np.ndarray:
        """Return the coupling matrix."""
        return self.interaction_matrix()

    def second_q_ops(self, display_format: Optional[str] = None) -> SpinOp:
        """Return the Hamiltonian of the Ising model in terms of `SpinOp`.

        Args:
            display_format: Not supported for Spin operators. If specified, it will be ignored.

        Returns:
            SpinOp: The Hamiltonian of the Ising model.
        """
        if display_format is not None:
            logger.warning(
                "Spin operators do not support display-format. Provided display-format "
                "parameter will be ignored."
            )
        ham = []
        weighted_edge_list = self._lattice.weighted_edge_list
        register_length = self._lattice.num_nodes
        # kinetic terms
        for node_a, node_b, weight in weighted_edge_list:
            if node_a == node_b:
                index = node_a
                ham.append((f"X_{index}", weight))

            else:
                index_left = node_a
                index_right = node_b
                coupling_parameter = weight
                ham.append((f"Z_{index_left} Z_{index_right}", coupling_parameter))

        return SpinOp(ham, spin=Fraction(1, 2), register_length=register_length)
