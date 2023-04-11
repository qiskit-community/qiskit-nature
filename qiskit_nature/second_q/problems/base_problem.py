# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2023.
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

from typing import Callable

import numpy as np
from qiskit.algorithms.eigensolvers import EigensolverResult
from qiskit.algorithms.minimum_eigensolvers import MinimumEigensolverResult
from qiskit.opflow import PauliSumOp
from qiskit.opflow.primitive_ops import Z2Symmetries as OpflowZ2Symmetries
from qiskit.quantum_info.analysis.z2_symmetries import Z2Symmetries

from qiskit_nature.deprecation import deprecate_function
from qiskit_nature.second_q.mappers import QubitConverter, QubitMapper, TaperedQubitMapper
from qiskit_nature.second_q.operators import SparseLabelOp
from qiskit_nature.second_q.hamiltonians import Hamiltonian

from .eigenstate_result import EigenstateResult
from .properties_container import PropertiesContainer


class BaseProblem:
    """The base representation of a second-quantization problem.

    If none of the specific subclasses of this class fit your use case, you can instantiate this
    class itself with your custom :class:`.Hamiltonian` instance and pass it into one of the
    available algorithms.

    The following attributes can be read and updated once the ``BaseProblem`` object has been
    constructed.

    Attributes:
        properties (PropertiesContainer): a container for additional observable operator factories.
    """

    def __init__(self, hamiltonian: Hamiltonian) -> None:
        """

        Args:
            driver: A driver encoding the molecule information.
            transformers: A list of transformations to be applied to the driver result.
            main_property_name: A main property name for the problem
        """
        self._hamiltonian = hamiltonian
        self.properties = PropertiesContainer()

    @property
    def hamiltonian(self) -> Hamiltonian:
        """Returns the hamiltonian wrapped by this problem."""
        return self._hamiltonian

    def second_q_ops(self) -> tuple[SparseLabelOp, dict[str, SparseLabelOp]]:
        """Returns the second quantized operators associated with this problem.

        Returns:
            A tuple, with the first object being the main operator and the second being a dictionary
            of auxiliary operators.
        """
        main_op = self.hamiltonian.second_q_op()

        aux_ops: dict[str, SparseLabelOp] = {}
        for prop in self.properties:
            aux_ops.update(prop.second_q_ops())

        return main_op, aux_ops

    # pylint: disable=bad-docstring-quotes
    @deprecate_function(
        "0.6.0",
        additional_msg=(
            ". This function is deprecated because it will be removed from the public API. It is "
            "no longer necessary to be used when working directly with QubitMapper objects outside "
            "a QubitConverter because a TaperedQubitMapper can now be obtained using the new "
            "get_tapered_mapper function provided by the problem classes"
        ),
    )
    def symmetry_sector_locator(
        self,
        z2_symmetries: OpflowZ2Symmetries | Z2Symmetries,
        converter: QubitConverter | QubitMapper,
    ) -> list[int] | None:
        # pylint: disable=unused-argument
        """Given the detected Z2Symmetries, it can determine the correct sector of the tapered
        operators so the correct one can be returned

        Args:
            z2_symmetries: the z2 symmetries object.
            converter: the ``QubitConverter`` or ``QubitMapper`` instance used for the operator
                conversion that symmetries are to be determined for.

        Returns:
            the sector of the tapered operators with the problem solution
        """
        return None

    def _symmetry_sector_locator(
        self,
        z2_symmetries: OpflowZ2Symmetries | Z2Symmetries,
        mapper: QubitConverter | QubitMapper,
    ) -> list[int] | None:
        # pylint: disable=unused-argument
        """Given the detected Z2Symmetries, it can determine the correct sector of the tapered
        operators so the correct one can be returned

        Args:
            z2_symmetries: the z2 symmetries object.
            mapper: the ``QubitMapper`` or ``QubitConverter`` instance (use of the latter is
                deprecated) used for the operator conversion that symmetries are to be determined
                for.

        Returns:
            the sector of the tapered operators with the problem solution
        """
        return None

    def get_tapered_mapper(self, mapper: QubitMapper) -> TaperedQubitMapper:
        """Builds a ``TaperedQubitMapper`` from one of the mappers.
        This simplifies the identification of the Pauli operator symmetries and of the symmetry sector
        in which lies the solution of the problem.

        Args:
            mapper: ``QubitMapper`` object implementing the mapping of second quantized operators to
                Pauli operators.

        Raises:
            ValueError: If the mapper is a ``TaperedQubitMapper``.

        Returns:
            A ``TaperedQubitMapper`` with pre-built symmetry specifications.
        """
        if isinstance(mapper, TaperedQubitMapper):
            raise ValueError(
                "TaperedQubitMapper instance cannot be built from another "
                "TaperedQubitMapper. If you want to update your TaperedQubitMapper "
                "instance please build a new one starting from the standard mappers."
            )

        qubit_op, _ = self.second_q_ops()
        mapped_op = mapper.map(qubit_op)
        if isinstance(mapped_op, PauliSumOp):
            mapped_op = mapped_op.primitive
        z2_symmetries = Z2Symmetries.find_z2_symmetries(mapped_op)
        # pylint: disable=assignment-from-none
        # Known issue for abstract class methods https://github.com/PyCQA/pylint/issues/2559
        tapering_values = self._symmetry_sector_locator(z2_symmetries, mapper)
        z2_symmetries.tapering_values = tapering_values
        return TaperedQubitMapper(mapper, z2_symmetries)

    def interpret(
        self,
        raw_result: EigenstateResult | EigensolverResult | MinimumEigensolverResult,
    ) -> EigenstateResult:
        """Interprets an EigenstateResult in the context of this problem.

        Args:
            raw_result: an eigenstate result object.

        Returns:
            An interpreted `EigenstateResult` in the form of a subclass of it. The actual type
            depends on the problem that implements this method.
        """
        return EigenstateResult.from_result(raw_result)

    def get_default_filter_criterion(
        self,
    ) -> Callable[[list | np.ndarray, float, list[float] | None], bool] | None:
        """Returns a default filter criterion method to filter the eigenvalues computed by the
        eigen solver. For more information see also
        qiskit.algorithms.eigen_solvers.NumPyEigensolver.filter_criterion.

        In the fermionic case the default filter ensures that the number of particles is being
        preserved.
        """
        return None
