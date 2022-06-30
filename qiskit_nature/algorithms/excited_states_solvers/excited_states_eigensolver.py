# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The calculation of excited states via an Eigensolver algorithm"""

from typing import Union, Optional, Tuple

from qiskit.algorithms import Eigensolver
from qiskit.opflow import PauliSumOp

from qiskit_nature import ListOrDictType, QiskitNatureError
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.converters.second_quantization.utils import ListOrDict
from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from qiskit_nature.problems.second_quantization import BaseProblem
from qiskit_nature.properties.second_quantization.electronic import ElectronicStructureDriverResult
from qiskit_nature.results import EigenstateResult

from .excited_states_solver import ExcitedStatesSolver
from .eigensolver_factories import EigensolverFactory


class ExcitedStatesEigensolver(ExcitedStatesSolver):
    """The calculation of excited states via an Eigensolver algorithm"""

    def __init__(
        self,
        qubit_converter: QubitConverter,
        solver: Union[Eigensolver, EigensolverFactory],
    ) -> None:
        """

        Args:
            qubit_converter: the `QubitConverter` to use for mapping and symmetry reduction. The
                             Z2 symmetries stored in this instance are the basis for the
                             commutativity information returned by this method.
            solver: Minimum Eigensolver or MESFactory object.
        """
        self._qubit_converter = qubit_converter
        self._solver = solver

    @property
    def solver(self) -> Union[Eigensolver, EigensolverFactory]:
        """Returns the minimum eigensolver or factory."""
        return self._solver

    @solver.setter
    def solver(self, solver: Union[Eigensolver, EigensolverFactory]) -> None:
        """Sets the minimum eigensolver or factory."""
        self._solver = solver

    def get_qubit_operators(
        self,
        problem: BaseProblem,
        aux_operators: Optional[ListOrDictType[Union[SecondQuantizedOp, PauliSumOp]]] = None,
        driver_result: Optional[ElectronicStructureDriverResult] = None,
    ) -> Tuple[PauliSumOp, Optional[ListOrDictType[PauliSumOp]]]:
        """Gets the operator and auxiliary operators, and transforms the provided auxiliary operators"""
        # Note that ``aux_ops`` contains not only the transformed ``aux_operators`` passed by the
        # user but also additional ones from the transformation
        second_q_ops = problem.second_q_ops(driver_result)
        aux_second_q_ops: ListOrDictType[SecondQuantizedOp]
        if isinstance(second_q_ops, list):
            main_second_q_op = second_q_ops[0]
            aux_second_q_ops = second_q_ops[1:]
        elif isinstance(second_q_ops, dict):
            name = problem.main_property_name
            main_second_q_op = second_q_ops.pop(name, None)
            if main_second_q_op is None:
                raise ValueError(
                    f"The main `SecondQuantizedOp` associated with the {name} property cannot be "
                    "`None`."
                )
            aux_second_q_ops = second_q_ops

        main_operator = self._qubit_converter.convert(
            main_second_q_op,
            num_particles=problem.num_particles,
            sector_locator=problem.symmetry_sector_locator,
        )
        aux_ops = self._qubit_converter.convert_match(aux_second_q_ops)

        if aux_operators is not None:
            wrapped_aux_operators: ListOrDict[Union[SecondQuantizedOp, PauliSumOp]] = ListOrDict(
                aux_operators
            )
            for name_aux, aux_op in iter(wrapped_aux_operators):
                if isinstance(aux_op, SecondQuantizedOp):
                    converted_aux_op = self._qubit_converter.convert_match(aux_op, True)
                else:
                    converted_aux_op = aux_op
                if isinstance(aux_ops, list):
                    aux_ops.append(converted_aux_op)
                elif isinstance(aux_ops, dict):
                    if name_aux in aux_ops.keys():
                        raise QiskitNatureError(
                            f"The key '{name_aux}' is already taken by an internally constructed "
                            "auxiliary operator! Please use a different name for your custom "
                            "operator."
                        )
                    aux_ops[name_aux] = converted_aux_op

        if isinstance(self._solver, EigensolverFactory):
            # this must be called after transformation.transform
            self._solver = self._solver.get_solver(problem)

        # if the eigensolver does not support auxiliary operators, reset them
        if not self._solver.supports_aux_operators():
            aux_ops = None
        return main_operator, aux_ops

    def solve(
        self,
        problem: BaseProblem,
        aux_operators: Optional[ListOrDictType[Union[SecondQuantizedOp, PauliSumOp]]] = None,
        driver_result: Optional[ElectronicStructureDriverResult] = None,
    ) -> EigenstateResult:
        """Compute Ground and Excited States properties.

        Args:
            problem: a class encoding a problem to be solved.
            aux_operators: Additional auxiliary operators to evaluate.
            driver_result: Previously calculated driver result.

        Raises:
            ValueError: if the grouped property object returned by the driver does not contain a
                main property as requested by the problem being solved (`problem.main_property_name`)
            QiskitNatureError: if the user-provided `aux_operators` contain a name which clashes
                with an internally constructed auxiliary operator. Note: the names used for the
                internal auxiliary operators correspond to the `Property.name` attributes which
                generated the respective operators.

        Returns:
            An interpreted :class:`~.EigenstateResult`. For more information see also
            :meth:`~.BaseProblem.interpret`.
        """
        # get the operator and auxiliary operators, and transform the provided auxiliary operators
        # note that ``aux_operators`` contains not only the transformed ``aux_operators`` passed
        # by the user but also additional ones from the transformation

        main_operator, aux_ops = self.get_qubit_operators(problem, aux_operators)
        raw_es_result = self._solver.compute_eigenvalues(main_operator, aux_ops)  # type: ignore

        eigenstate_result = EigenstateResult()
        eigenstate_result.raw_result = raw_es_result
        eigenstate_result.eigenenergies = raw_es_result.eigenvalues
        eigenstate_result.eigenstates = raw_es_result.eigenstates
        eigenstate_result.aux_operator_eigenvalues = raw_es_result.aux_operator_eigenvalues
        result = problem.interpret(eigenstate_result)
        return result
