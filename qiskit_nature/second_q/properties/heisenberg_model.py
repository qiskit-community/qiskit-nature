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
import logging
from typing import Optional, Tuple
from fractions import Fraction
import numpy as np
from qiskit_nature.second_q.operators import SpinOp
from qiskit_nature.second_q.properties.lattice_model import LatticeModel
from qiskit_nature.second_q.properties.lattices.lattice import Lattice

logger = logging.getLogger(__name__)


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

    @classmethod
    def uniform_parameters(
        cls, lattice: Lattice, uniform_interaction: complex, uniform_onsite_potential: complex
    ) -> HeisenbergModel:
        """Set a uniform interaction parameter and on-site potential over the input lattice.

        Args:
            lattice: Lattice on which the model is defined.
            uniform_interaction: The interaction parameter.
            uniform_onsite_potential: The on-site potential.

        Returns:
            HeisenbergModel: The Heisenberg model with uniform parameters.
        """
        return cls(
            cls._generate_lattice_from_uniform_parameters(
                lattice, uniform_interaction, uniform_onsite_potential
            )
        )

    @classmethod
    def from_parameters(cls, interaction_matrix: np.ndarray) -> HeisenbergModel:
        """Return the Hamiltonian of the Heisenberg model
        from the given interaction matrix and on-site interaction.

        Args:
            interaction_matrix: A real or complex valued squared matrix.

        Returns:
            HeisenbergModel: The Heisenberg model generated from the given interaction
                matrix and on-site interaction.
        """
        return cls(cls._generate_lattice_from_parameters(interaction_matrix))

    def second_q_ops(self, display_format: Optional[str] = None) -> SpinOp:
        """Return the Hamiltonian of the Heisenberg model in terms of `SpinOp`.

        Args:
            display_format: Not supported for Spin operators. If specified, it will be ignored.

        Returns:
            SpinOp: The Hamiltonian of the Heisenberg model.
        """
        if display_format is not None:
            logger.warning(
                "Spin operators do not support display-format. Provided display-format "
                "parameter will be ignored."
            )
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
