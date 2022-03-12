# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"The Heisenberg model"
import logging
from typing import Optional
from fractions import Fraction
from qiskit_nature.operators.second_quantization import SpinOp
from qiskit_nature.problems.second_quantization.lattice.lattices import Lattice
from .lattice_model import LatticeModel

logger = logging.getLogger(__name__)


class HeisenbergModel(LatticeModel):
    """The Heisenberg model."""

    def __init__(
        self,
        lattice: Lattice,
        model_constants: Optional[dict] = None,
        ext_magnetic_field: Optional[dict] = None,
    ) -> None:
        """
        Args:
            model_constants: The constants that define the model.
            ext_magnetic_field: Tell us which direction we have the external magnetic field.
        """
        super().__init__(lattice)
        self.model_constants = model_constants
        self.ext_magnetic_field = ext_magnetic_field

    def second_q_ops(
        self,
        display_format: Optional[str] = None,
    ) -> SpinOp:
        """Return the Hamiltonian of the Heisenberg model in terms of 'SpinOp'.
        Args:
            display_format: Not supported for Spin operators. If specified, it will be ignored.
        Raises:
            ValueError: If model_constants or ext_magnectic_field are not None or dict type.
        Returns:
            SpinOp: The Hamiltonian of the Heisenberg model.
        """
        if display_format is not None:
            logger.warning(
                "Spin operators do not support display-format. Provided display-format "
                "parameter will be ignored."
            )

        if self.model_constants is None or isinstance(self.model_constants, dict):
            if self.model_constants is None:
                self.model_constants = {"J_x": 1, "J_y": 1, "J_z": 1, "h": 0}
        else:
            raise ValueError(
                f"model_constants must be None or a dict, but was given a {type(self.model_constants)}"
            )

        if self.ext_magnetic_field is None or isinstance(self.ext_magnetic_field, dict):
            if self.ext_magnetic_field is None:
                self.ext_magnetic_field = {"B_x": False, "B_y": False, "B_z": False}
        else:
            raise ValueError(
                f"ext_magnetic_field must be None or a dict, but was given a {type(self.ext_magnetic_field)}"
            )

        hamiltonian = []
        weighted_edge_list = self.lattice.weighted_edge_list
        register_length = self.lattice.num_nodes

        for node_a, node_b, _ in weighted_edge_list:

            if node_a == node_b:
                index = node_a
                if self.ext_magnetic_field["B_x"]:
                    hamiltonian.append((f"X_{index}", -1 * self.model_constants["h"]))
                if self.ext_magnetic_field["B_y"]:
                    hamiltonian.append((f"Y_{index}", -1 * self.model_constants["h"]))
                if self.ext_magnetic_field["B_z"]:
                    hamiltonian.append((f"Z_{index}", -1 * self.model_constants["h"]))
            else:
                index_left = node_a
                index_right = node_b
                if self.model_constants["J_x"] != 0:
                    hamiltonian.append(
                        (f"X_{index_left} X_{index_right}", -1 * self.model_constants["J_x"])
                    )
                if self.model_constants["J_y"] != 0:
                    hamiltonian.append(
                        (f"Y_{index_left} Y_{index_right}", -1 * self.model_constants["J_y"])
                    )
                if self.model_constants["J_z"] != 0:
                    hamiltonian.append(
                        (f"Z_{index_left} Z_{index_right}", -1 * self.model_constants["J_z"])
                    )

        return SpinOp(hamiltonian, spin=Fraction(1, 2), register_length=register_length)
