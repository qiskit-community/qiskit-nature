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

"""The orbital optimization VQE algorithm"""

from typing import Union, Optional
import copy
import logging
from functools import partial

import numpy as np

from qiskit.opflow import PauliSumOp
from qiskit.algorithms.minimum_eigen_solvers.minimum_eigen_solver import (
    MinimumEigensolver,
)

from qiskit_nature import ListOrDictType, QiskitNatureError
from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from qiskit_nature.converters.second_quantization.utils import ListOrDict
from qiskit_nature.problems.second_quantization import BaseProblem
from qiskit_nature.results import EigenstateResult

from qiskit_nature.algorithms import GroundStateEigensolver
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.algorithms.ground_state_solvers.minimum_eigensolver_factories import (
    MinimumEigensolverFactory,
)
from qiskit_nature.properties.second_quantization.electronic.bases import (
    ElectronicBasis,
    ElectronicBasisTransform,
)

from qiskit_nature.algorithms.ground_state_solvers.custom_problem import CustomProblem
from qiskit_nature.algorithms.ground_state_solvers.orbital_rotation import OrbitalRotation
from qiskit_nature.algorithms.ground_state_solvers.oovqe_solver import compute_minimum_eigenvalue_oo

logger = logging.getLogger(__name__)

class OrbitalOptimizationVQE(GroundStateEigensolver):
    """Solver for ooVQE"""

    def __init__(
        self,
        qubit_converter: QubitConverter,
        solver: Union[MinimumEigensolver, MinimumEigensolverFactory],
    ) -> None:
        super().__init__(qubit_converter, solver)

        # Store problem to have access during energy eval. function.
        self.problem: CustomProblem = (
            None  # I am using temporarily the CustomProblem class, that avoids
        )
        # running the driver every time .second_q_ops() is called

        self.initial_point = None  # in the future: set by user
        self.bounds_oo: np.array = None  # in the future: set by user
        self.bounds: np.array = None  # ansatz + oo

        self.orbital_rotation = OrbitalRotation(
            num_qubits=self.solver.ansatz.num_qubits, qubit_converter=qubit_converter
        )
        self.num_parameters_oovqe = (
            self.solver.ansatz.num_parameters + self.orbital_rotation.num_parameters
        )

        # the initial point of the full ooVQE alg.
        if self.initial_point is None:
            self.set_initial_point()
        else:
            # this will never really happen with the current code
            # but is kept for the future
            if len(self.initial_point) is not self.num_parameters_oovqe:
                raise QiskitNatureError(
                    f"Number of parameters of OOVQE ({self.num_parameters_oovqe,}) "
                    f"does not match the length of the "
                    f"intitial_point ({len(self.initial_point)})"
                )

        if self.bounds is None:
            # set bounds sets both ansatz and oo bounds
            # do we want to change the ansatz bounds here??
            self.set_bounds(self.orbital_rotation.parameter_bound_value)

    def set_initial_point(self, initial_pt_scalar: float = 1e-1) -> None:
        """Initializes the initial point for the algorithm if the user does not provide his own.
        Args:
            initial_pt_scalar: value of the initial parameters for wavefunction and orbital rotation
        """
        self.initial_point = np.asarray(
            [initial_pt_scalar for _ in range(self.num_parameters_oovqe)]
        )

    def set_bounds(
        self,
        bounds_ansatz_value: tuple = (-2 * np.pi, 2 * np.pi),
        bounds_oo_value: tuple = (-2 * np.pi, 2 * np.pi),
    ) -> None:
        """Doctstring"""
        bounds_ansatz = [bounds_ansatz_value for _ in range(self.solver.ansatz.num_parameters)]
        self.bounds_oo = [bounds_oo_value for _ in range(self.orbital_rotation.num_parameters)]
        bounds = bounds_ansatz + self.bounds_oo
        self.bounds = np.array(bounds)

    def get_operators(self, problem, aux_operators):
        """Doctstring"""
        second_q_ops = problem.second_q_ops()

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
            for name, aux_op in iter(wrapped_aux_operators):
                if isinstance(aux_op, SecondQuantizedOp):
                    converted_aux_op = self._qubit_converter.convert_match(aux_op, True)
                else:
                    converted_aux_op = aux_op
                if isinstance(aux_ops, list):
                    aux_ops.append(converted_aux_op)
                elif isinstance(aux_ops, dict):
                    if name in aux_ops.keys():
                        raise QiskitNatureError(
                            f"The key '{name}' is already taken by an internally constructed "
                            "auxliliary operator! Please use a different name for your custom "
                            "operator."
                        )
                    aux_ops[name] = converted_aux_op

        # if the eigensolver does not support auxiliary operators, reset them
        if not self._solver.supports_aux_operators():
            aux_ops = None

        return main_operator, aux_ops

    def rotate_orbitals(self, matrix_a, matrix_b):
        """Doctstring"""
        problem = copy.copy(self.problem)
        grouped_property_transformed = problem.grouped_property_transformed

        # use ElectronicBasisTransform
        transform = ElectronicBasisTransform(
            ElectronicBasis.MO, ElectronicBasis.MO, matrix_a, matrix_b
        )

        # only 1 & 2 body integrals have the "transform_basis" method,
        # so I access them through the electronic energy
        e_energy = grouped_property_transformed.get_property("ElectronicEnergy")
        one_body_integrals = e_energy.get_electronic_integral(ElectronicBasis.MO, 1)
        two_body_integrals = e_energy.get_electronic_integral(ElectronicBasis.MO, 2)

        # the basis transform should be applied in place, but it's not???
        # unless I manually add the integrals, the result of second_q_ops()
        # doesn't change.
        # I have to look further into this.
        e_energy.add_electronic_integral(one_body_integrals.transform_basis(transform))
        e_energy.add_electronic_integral(two_body_integrals.transform_basis(transform))

        # after applying the rotation, recompute operator
        rotated_main_second_q_op = e_energy.second_q_ops()
        rotated_operator = self._qubit_converter.convert(
            rotated_main_second_q_op,
            num_particles=problem.num_particles,
            sector_locator=problem.symmetry_sector_locator,
        )
        return rotated_operator

    def solve(
        self,
        problem: BaseProblem,
        aux_operators: Optional[ListOrDictType[Union[SecondQuantizedOp, PauliSumOp]]] = None,
    ) -> EigenstateResult:
        """Compute Ground State properties.

        Args:
            problem: a class encoding a problem to be solved.
            aux_operators: Additional auxiliary operators to evaluate.

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
        # note that ``aux_ops`` contains not only the transformed ``aux_operators`` passed by the
        # user but also additional ones from the transformation

        self.problem = problem

        # override VQE's compute_minimum_eigenvalue, giving it access to the problem data
        # contained in self.problem
        self.solver.compute_minimum_eigenvalue = partial(
            compute_minimum_eigenvalue_oo, self, self.solver
        )

        main_operator, aux_ops = self.get_operators(problem, aux_operators)
        raw_mes_result = self.solver.compute_minimum_eigenvalue(main_operator, aux_ops)

        result = problem.interpret(raw_mes_result)

        return result
