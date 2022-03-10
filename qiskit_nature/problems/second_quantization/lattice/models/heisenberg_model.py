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

"The Heisenberg model"
import logging
from types import NoneType
from typing import Optional
from fractions import Fraction

import numpy as np

from qiskit_nature.operators.second_quantization import SpinOp
from qiskit_nature.problems.second_quantization.lattice.lattices import Lattice

from .lattice_model import LatticeModel

logger = logging.getLogger(__name__)


class HeisenbergModel(LatticeModel):
    """The Heisenberg model."""

    def __init__(self, lattice: Lattice) -> None:
        super().__init__(lattice)

    def second_q_ops(
        self,
        model_constants: Optional[dict] = None,
        ext_magnetic_field: Optional[dict] = None,
        display_format: Optional[str] = None,
    ) -> SpinOp:
        """Return the Hamiltonian of the Heisenberg model in terms of 'SpinOp'.
        Args:
            model_constants: The constants that define the model
                    (Default configuration: {"J_x" = 1, "J_y" = 1, "J_z" = 1, "h" = 0}).
            ext_magnetic_field: A dictionary that tells us the directions we have the presence of the
                    external magnetic field (Default configuration: {"B_x" = False, "B_y" = False, "B_z" = False}).
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

        if type(model_constants) == type(None) or type(model_constants) == dict:
            if model_constants == None:
                model_constants = {"J_x": 1, "J_y": 1, "J_z": 1, "h": 0}
        else:
            raise ValueError(
                "The type of model_constants argument is expected to be None or a dict, but was given a {}".format(
                    type(model_constants)
                )
            )

        if type(ext_magnetic_field) == type(None) or type(ext_magnetic_field) == dict:
            if ext_magnetic_field == None:
                ext_magnetic_field = {"B_x": False, "B_y": False, "B_z": False}
        else:
            raise ValueError(
                "The type of ext_magnetic_field argument is expected to be None or a dict, but was given a {}".format(
                    type(ext_magnetic_field)
                )
            )

        hamiltonian = []
        weighted_edge_list = self.lattice.weighted_edge_list
        register_length = self.lattice.num_nodes

        for node_a, node_b, _ in weighted_edge_list:

            if node_a == node_b:
                index = node_a
                if ext_magnetic_field["B_x"]:
                    hamiltonian.append((f"X_{index}", -1 * model_constants["h"]))
                if ext_magnetic_field["B_y"]:
                    hamiltonian.append((f"Y_{index}", -1 * model_constants["h"]))
                if ext_magnetic_field["B_z"]:
                    hamiltonian.append((f"Z_{index}", -1 * model_constants["h"]))
            else:
                index_left = node_a
                index_right = node_b
                if model_constants["J_x"] != 0:
                    hamiltonian.append(
                        (f"X_{index_left} X_{index_right}", -1 * model_constants["J_x"])
                    )
                if model_constants["J_y"] != 0:
                    hamiltonian.append(
                        (f"Y_{index_left} Y_{index_right}", -1 * model_constants["J_y"])
                    )
                if model_constants["J_z"] != 0:
                    hamiltonian.append(
                        (f"Z_{index_left} Z_{index_right}", -1 * model_constants["J_z"])
                    )

        return SpinOp(hamiltonian, spin=Fraction(1, 2), register_length=register_length)
