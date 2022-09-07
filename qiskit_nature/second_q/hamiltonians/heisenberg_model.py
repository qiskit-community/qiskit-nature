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

"""The Heisenberg model."""

from __future__ import annotations
from typing import Tuple
from fractions import Fraction
import numpy as np
from qiskit_nature.second_q.operators import SpinOp

from .lattice_model import LatticeModel
from .lattices import Lattice


class HeisenbergModel(LatticeModel):
    """The Heisenberg model."""

    def __init__(
        self,
        lattice: Lattice,
        coupling_constants: Tuple = (1.0, 1.0, 1.0),
        ext_magnetic_field: Tuple = (0.0, 0.0, 0.0),
    ) -> None:
        """
        Args:
            lattice: Lattice on which the model is defined.
            coupling_constants: The coupling constants in each Cartesian axes.
                                Defaults to (1.0, 1.0, 1.0).
            ext_magnetic_field: Represents a magnetic field in Cartesian coordinates.
                                Defaults to (0.0, 0.0, 0.0).
        """
        super().__init__(lattice)
        self.coupling_constants = coupling_constants
        self.ext_magnetic_field = ext_magnetic_field

    @property
    def register_length(self) -> int:
        return self._lattice.num_nodes

    def second_q_op(self) -> SpinOp:
        """Return the Hamiltonian of the Heisenberg model in terms of `SpinOp`.

        Returns:
            SpinOp: The Hamiltonian of the Heisenberg model.
        """
        hamiltonian = []
        weighted_edge_list = self.lattice.weighted_edge_list
        register_length = self.lattice.num_nodes

        for node_a, node_b, _ in weighted_edge_list:

            if node_a == node_b:
                index = node_a
                for axis, coeff in zip("XYZ", self.ext_magnetic_field):
                    if not np.isclose(coeff, 0.0):
                        hamiltonian.append((f"{axis}_{index}", coeff))
            else:
                index_left = node_a
                index_right = node_b
                for axis, coeff in zip("XYZ", self.coupling_constants):
                    if not np.isclose(coeff, 0.0):
                        hamiltonian.append((f"{axis}_{index_left} {axis}_{index_right}", coeff))

        return SpinOp(hamiltonian, spin=Fraction(1, 2), register_length=register_length)
