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
# This code is part of Qiskit.

"""The Base Problem class."""

from __future__ import annotations

from typing import Callable, List, Optional, Union

import numpy as np
from qiskit.opflow import Z2Symmetries

from qiskit_nature.second_q.mappers import QubitConverter
from qiskit_nature.second_q.operators import SecondQuantizedOp
from qiskit_nature.second_q.hamiltonians import Hamiltonian

from .eigenstate_result import EigenstateResult
from .properties_container import PropertiesContainer


class BaseProblem:
    """Base Problem"""

    def __init__(self, hamiltonian: Hamiltonian) -> None:
        """

        Args:
            driver: A driver encoding the molecule information.
            transformers: A list of transformations to be applied to the driver result.
            main_property_name: A main property name for the problem
        """
        self.hamiltonian = hamiltonian
        self.properties = PropertiesContainer()

    @property
    def num_particles(self) -> tuple[int, int] | None:
        """Returns the number of particles, if available."""
        return None

    def second_q_ops(self) -> tuple[SecondQuantizedOp, dict[str, SecondQuantizedOp]]:
        """Returns the second quantized operators associated with this problem.

        Returns:
            A tuple, with the first object being the main operator and the second being a dictionary
            of auxiliary operators.
        """
        main_op = self.hamiltonian.second_q_op()

        aux_ops: dict[str, SecondQuantizedOp] = {}
        for prop in self.properties:
            aux_ops.update(prop.second_q_ops())

        return main_op, aux_ops

    def symmetry_sector_locator(
        self,
        z2_symmetries: Z2Symmetries,
        converter: QubitConverter,
    ) -> Optional[List[int]]:
        # pylint: disable=unused-argument
        """Given the detected Z2Symmetries, it can determine the correct sector of the tapered
        operators so the correct one can be returned

        Args:
            z2_symmetries: the z2 symmetries object.
            converter: the qubit converter instance used for the operator conversion that
                symmetries are to be determined for.

        Returns:
            the sector of the tapered operators with the problem solution
        """
        return None

    def interpret(self, raw_result: EigenstateResult) -> EigenstateResult:
        """Interprets an EigenstateResult in the context of this problem.

        Args:
            raw_result: an eigenstate result object.

        Returns:
            An interpreted `EigenstateResult` in the form of a subclass of it. The actual type
            depends on the problem that implements this method.
        """
        raise NotImplementedError()

    def get_default_filter_criterion(
        self,
    ) -> Optional[Callable[[Union[List, np.ndarray], float, Optional[List[float]]], bool]]:
        """Returns a default filter criterion method to filter the eigenvalues computed by the
        eigen solver. For more information see also
        qiskit.algorithms.eigen_solvers.NumPyEigensolver.filter_criterion.

        In the fermionic case the default filter ensures that the number of particles is being
        preserved.
        """
        return None
