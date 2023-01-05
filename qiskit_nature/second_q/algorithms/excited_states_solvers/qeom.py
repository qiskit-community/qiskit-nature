# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The calculation of excited states via the qEOM algorithm."""

from __future__ import annotations

from typing import Any, Callable, Sequence, cast
from enum import Enum
import itertools
import logging
import sys


import numpy as np
from scipy import linalg

from qiskit.algorithms.eigensolvers import EigensolverResult
from qiskit.algorithms.list_or_dict import ListOrDict as ListOrDictType
from qiskit.algorithms.minimum_eigensolvers import MinimumEigensolver
from qiskit.algorithms.observables_evaluator import estimate_observables

from qiskit.circuit import QuantumCircuit
from qiskit.opflow import Z2Symmetries, commutator, double_commutator, PauliSumOp
from qiskit.tools import parallel_map
from qiskit.tools.events import TextProgressBar
from qiskit.utils import algorithm_globals
from qiskit.utils.deprecation import deprecate_function

from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import BaseEstimator

from qiskit_nature.second_q.algorithms.ground_state_solvers import GroundStateSolver
from qiskit_nature.second_q.algorithms.ground_state_solvers.ground_state_solver import QubitOperator
from qiskit_nature.second_q.algorithms.ground_state_solvers.minimum_eigensolver_factories import (
    MinimumEigensolverFactory,
)
from qiskit_nature.second_q.algorithms.excited_states_solvers.excited_states_solver import (
    ExcitedStatesSolver,
)
from qiskit_nature.second_q.mappers import QubitConverter, QubitMapper
from qiskit_nature.second_q.operators import SparseLabelOp
from qiskit_nature.second_q.problems import (
    BaseProblem,
    ElectronicStructureProblem,
    VibrationalStructureProblem,
    EigenstateResult,
    ElectronicStructureResult,
)

from .qeom_electronic_ops_builder import build_electronic_ops
from .qeom_vibrational_ops_builder import build_vibrational_ops


logger = logging.getLogger(__name__)


class EvaluationRule(Enum):
    """An enumeration of the available evaluation rules for the excited states solvers.

    This ``Enum`` simply names the available evaluation rules.
    """

    ALL = "all"
    DIAG = "diag"


class QEOM(ExcitedStatesSolver):
    """The calculation of excited states via the qEOM algorithm.

    This algorithm approximates the excited-state properties of a problem using additional measurements
    on the ground state provided by a ``GroundStateSolver`` object.
    The precision of the ``GroundStateSolver.solve`` method for the ground state approximate directly
    affects the precision of the qEOM algorithm for the same problem.
    The ``excitations`` are used to build a linear subspace in which an eigenvalue problem for the
    projected Hamiltonian will be solved. This method typically works well for calculating the
    lowest-lying excited states of a problem.
    The excited-state energies are calculated by default in this algorithm for all excited states.
    Auxiliary observables can be specified to the ``solve`` method along with auxiliary evaluation
    rules of the ``QEOM`` object.

    The following attributes can be read and updated once the ``QEOM`` object has been
    constructed.

    Attributes:
        excitations: The excitations to be included in the eom pseudo-eigenvalue problem.
        aux_eval_rules: The rules determining how observables should be evaluated on excited states.
        tol: The tolerance threshold for the qEOM eigenvalues.
    """

    def __init__(
        self,
        ground_state_solver: GroundStateSolver,
        estimator: BaseEstimator,
        excitations: str
        | int
        | list[int]
        | Callable[
            [int, tuple[int, int]],
            list[tuple[tuple[int, ...], tuple[int, ...]]],
        ] = "sd",
        aux_eval_rules: EvaluationRule | dict[str, list[tuple[int, int]]] | None = None,
        *,
        tol: float = 1e-6,
    ) -> None:
        """
        Args:
            ground_state_solver: A ``GroundStateSolver`` object. The qEOM algorithm
                will use this ground state to compute the EOM matrix elements.
            estimator: The ``BaseEstimator`` to use for the evaluation of
                the qubit operators at the ground state ansatz. If the internal solver provided to
                the ``GroundStateSolver`` also uses a ``BaseEstimator`` primitive, you can provide the
                same estimator instance here.
            excitations: The excitations to be included in the eom pseudo-eigenvalue problem.

                :`str`: which contains the types of excitations. Allowed characters are
                    + `s` for singles
                    + `d` for doubles
                    + `t` for triples
                    + `q` for quadruples
                :`int`: a single, positive integer which denotes the number of excitations
                    (1 == `s`, etc.)
                :`list[int]`: a list of positive integers generalizing the above
                :`Callable`: a function which is used to generate the excitations.
                    The callable must take the __keyword__ arguments `num_spin_orbitals` and
                    `num_particles` (with identical types to those explained above) and must return
                    a `list[tuple[tuple[int, ...], tuple[int, ...]]]`. For more information on how
                    to write such a callable refer to the default method
                    :meth:`~qiskit_nature.circuit.library.ansatzes.utils.generate_fermionic_excitations`.
            aux_eval_rules: The rules determining how observables should be evaluated on excited states.
                By default, none of the auxiliary operators are evaluated on none of the excited states.

                :`Enum`: specific predefined rules. Allowed rules are:
                    + ALL to compute all expectation values and all transition amplitudes.
                    + DIAG to only compute expectation values.
                :`dict[str, list[tuple[int, int]]]`: Dictionary mapping valid auxiliary operator's name
                    to lists of tuple (i, j) specifying the indices of the excited states to be evaluated
                    on.
            tol: Tolerance threshold for the qEOM eigenvalues. This plays a role when one
                excited state approaches the ground state, in which case it is best to avoid manipulating
                very small absolute values.
        """
        self._gsc = ground_state_solver
        self._estimator = estimator
        self.excitations = excitations
        self.aux_eval_rules = aux_eval_rules
        self.tol = tol

        self._untapered_qubit_op_main: QubitOperator | None = None

    @property
    def qubit_converter(self) -> QubitConverter | QubitMapper:
        """Returns the qubit_converter object defined in the ground state solver."""
        return self._gsc.qubit_converter

    @property
    def solver(self) -> MinimumEigensolver | MinimumEigensolverFactory:
        """Returns the solver object defined in the ground state solver."""
        return self._gsc.solver

    def get_qubit_operators(
        self,
        problem: BaseProblem,
        aux_operators: dict[str, SparseLabelOp | QubitOperator] | None = None,
    ) -> tuple[QubitOperator, dict[str, QubitOperator] | None]:
        """
        Gets the operator and auxiliary operators, and transforms the provided auxiliary operators.
        If the user-provided ``aux_operators`` contain a name which clashes with an internally
        constructed auxiliary operator, then the corresponding internal operator will be overridden by
        the user-provided operator.

        Note that this methods performs a specific treatment of the symmetries required by the qEOM
        calculation.

        Args:
            problem: A class encoding a problem to be solved.
            aux_operators: Additional auxiliary operators to evaluate.

        Returns:
            Tuple of the form (Qubit operator, Auxiliary operators).
        """

        main_operator, aux_second_q_operators = problem.second_q_ops()

        # 1. Convert the main operator (hamiltonian) to QubitOperator and apply two qubit reduction
        num_particles = getattr(problem, "num_particles", None)

        if isinstance(self.qubit_converter, QubitConverter):
            main_op = self.qubit_converter.convert_only(
                main_operator,
                num_particles=num_particles,
            )
        else:
            main_op = self.qubit_converter.map(main_operator)

        # aux_ops set to None if the solver does not support auxiliary operators.
        aux_ops = None

        if self.solver.supports_aux_operators():
            if isinstance(self.qubit_converter, QubitConverter):
                self.qubit_converter.force_match(num_particles=num_particles)
                aux_ops = self.qubit_converter.convert_match(aux_second_q_operators)

            else:
                aux_ops = self.qubit_converter.map(aux_second_q_operators)

            cast(ListOrDictType[QubitOperator], aux_ops)
            if aux_operators is not None:
                for name, op in aux_operators.items():
                    if isinstance(op, (SparseLabelOp)):
                        if isinstance(self.qubit_converter, QubitConverter):
                            converted_aux_op = self.qubit_converter.convert_match(op)
                        else:
                            converted_aux_op = self.qubit_converter.map(op)

                    else:
                        converted_aux_op = op
                    if name in aux_ops.keys():
                        logger.warning(
                            "The key '%s' was already taken by an internally constructed auxiliary"
                            "operator!"
                            "The internal operator was overridden by the one provided manually."
                            "If this was not the intended behavior, please consider renaming"
                            "this operator.",
                            name,
                        )
                    # The custom op overrides the default op if the key is already taken.
                    aux_ops[name] = converted_aux_op

        # 2. Find the z2symmetries, set them in the qubit_converter, and apply the first step of the
        # tapering.
        if isinstance(self.qubit_converter, QubitConverter):
            _, z2symmetries = self.qubit_converter.find_taper_op(
                main_op, problem.symmetry_sector_locator
            )
            self.qubit_converter.force_match(z2symmetries=z2symmetries)
            untap_main_op = self.qubit_converter.convert_clifford(main_op)
            untap_aux_ops = self.qubit_converter.convert_clifford(aux_ops)
        else:
            # TODO: Issue #974 sketches the construction of a Tapered Qubit Mapper which would implement
            # the logic of the symmetries. Here, there should be a check for a Tapered Qubit Mapper and
            # a similar logic that used above.
            untap_main_op = main_op
            untap_aux_ops = aux_ops

        # 4. If a MinimumEigensolverFactory was provided, then an additional call to get_solver() is
        # required.
        if isinstance(self.solver, MinimumEigensolverFactory):
            self._gsc._solver = self.solver.get_solver(problem, self.qubit_converter)  # type: ignore

        return untap_main_op, untap_aux_ops

    def solve(
        self,
        problem: BaseProblem,
        aux_operators: dict[str, SparseLabelOp | QubitOperator] | None = None,
    ) -> EigenstateResult:
        """Run the excited-states calculation.

        Construct and solves the EOM pseudo-eigenvalue problem to obtain the excitation energies
        and the excitation operators expansion coefficients.

        Args:
            problem: A class encoding a problem to be solved.
            aux_operators: Additional auxiliary operators to evaluate.
        Returns:
            An interpreted :class:`~.EigenstateResult`. For more information see also
            :meth:`~.BaseProblem.interpret`.
        """

        # 1. Prepare all operators
        (
            untap_main_op,  # Hamiltonian
            untap_aux_ops,  # Auxiliary observables
        ) = self.get_qubit_operators(problem, aux_operators)

        # 2. Run ground state calculation with fully tapered custom auxiliary operators
        # Note that the solve() method includes the `second_q' auxiliary operators
        if isinstance(self.qubit_converter, QubitConverter):
            tap_aux_operators = self.qubit_converter.symmetry_reduce_clifford(untap_aux_ops)
        else:
            tap_aux_operators = untap_aux_ops

        groundstate_result = self._gsc.solve(problem, tap_aux_operators)
        ground_state = groundstate_result.eigenstates[0]

        # 3. Prepare the expansion operators for the excited state calculation
        expansion_basis_data = self._prepare_expansion_basis(problem)

        # 4. Obtain the representation of the Hamiltonian in the linear subspace
        h_mat, s_mat, h_mat_std, s_mat_std = self._build_qeom_pseudoeigenvalue_problem(
            untap_main_op, expansion_basis_data, ground_state
        )

        # 5. Solve the pseudo-eigenvalue problem
        energy_gaps, expansion_coefs, commutator_metric = self._compute_excitation_energies(
            h_mat, s_mat
        )
        gammas_square: np.ndarray = np.abs(np.diagonal(commutator_metric))
        logger.info("Gamma square = %s", gammas_square)
        scaling_matrix: np.ndarray = np.diag(
            np.divide(np.ones_like(gammas_square), np.sqrt(gammas_square))
        )
        expansion_coefs_rescaled: np.ndarray = expansion_coefs @ scaling_matrix

        # 6. Evaluate auxiliary operators on the excited states
        (
            aux_operators_eigenvalues,
            transition_amplitudes,
        ) = self._evaluate_observables_excited_states(
            untap_aux_ops,
            expansion_basis_data,
            ground_state,
            expansion_coefs_rescaled,
        )

        result = self._build_qeom_result(
            problem,
            groundstate_result,
            expansion_coefs,
            energy_gaps,
            h_mat,
            s_mat,
            h_mat_std,
            s_mat_std,
            aux_operators_eigenvalues,
            transition_amplitudes,
            gammas_square,
        )

        return result

    def _build_hopping_ops(
        self, problem: BaseProblem
    ) -> tuple[
        dict[str, QubitOperator],
        dict[str, list[bool]],
        dict[str, tuple[tuple[int, ...], tuple[int, ...]]],
    ]:
        """Builds the product of raising and lowering operators for a given problem.

        Args:
            problem: The problem for which to build out the operators.

        Raises:
            NotImplementedError: For an unsupported problem type.

        Returns:
            Dict of hopping operators, dict of commutativity types and dict of excitation indices.
        """
        if isinstance(problem, ElectronicStructureProblem):
            return build_electronic_ops(
                problem.num_spatial_orbitals,
                (problem.num_alpha, problem.num_beta),
                self.excitations,
                self._gsc.qubit_converter,
            )
        elif isinstance(problem, VibrationalStructureProblem):
            return build_vibrational_ops(
                problem.num_modals,
                self.excitations,
                self._gsc.qubit_converter,
            )
        else:
            raise NotImplementedError(
                "The building of QEOM hopping operators is not yet implemented for a problem of "
                f"type {type(problem)}"
            )

    def _build_all_eom_operators(
        self,
        untap_operator: QubitOperator,
        expansion_basis_data: tuple[dict[str, QubitOperator], dict[str, list[bool]], int],
    ) -> dict:
        """Building all commutators for Q, W, M, V matrices.

        Args:
            untap_operator: Not yet tapered Hamiltonian operator
            expansion_basis_data: all hopping operators based on excitations_list,
                key is the string of single/double excitation;
                value is corresponding operator.

        Returns:
            A dictionary that contains the operators for each matrix element.
        """

        untap_hopping_ops, type_of_commutativities, size = expansion_basis_data
        to_be_computed_list = []
        all_matrix_operators = {}

        mus, nus = np.triu_indices(size)

        def _build_one_sector(available_hopping_ops):
            for idx, m_u in enumerate(mus):
                n_u = nus[idx]
                left_op_1 = available_hopping_ops.get(f"E_{m_u}")
                right_op_1 = available_hopping_ops.get(f"E_{n_u}")
                right_op_2 = available_hopping_ops.get(f"Edag_{n_u}")
                to_be_computed_list.append((m_u, n_u, left_op_1, right_op_1, right_op_2))

        if isinstance(self.qubit_converter, QubitConverter):
            try:
                z2_symmetries = self.qubit_converter.z2symmetries
            except AttributeError:
                z2_symmetries = Z2Symmetries([], [], [])
        else:
            z2_symmetries = Z2Symmetries([], [], [])

        if not z2_symmetries.is_empty():
            combinations = itertools.product([1, -1], repeat=len(z2_symmetries.symmetries))
            for targeted_tapering_values in combinations:
                logger.info(
                    "In sector: (%s)",
                    ",".join([str(x) for x in targeted_tapering_values]),
                )
                # remove the excited operators which are not suitable for the sector

                available_hopping_ops = {}
                targeted_sector = np.asarray(targeted_tapering_values) == 1
                for key, value in type_of_commutativities.items():
                    if np.all(np.asarray(value) == targeted_sector):
                        available_hopping_ops[key] = untap_hopping_ops[key]
                _build_one_sector(available_hopping_ops)

        else:
            _build_one_sector(untap_hopping_ops)

        if logger.isEnabledFor(logging.INFO):
            logger.info("Building all commutators:")
            TextProgressBar(sys.stderr)
        results = parallel_map(
            self._build_commutator_routine,
            to_be_computed_list,
            task_args=(untap_operator, z2_symmetries),
            num_processes=algorithm_globals.num_processes,
        )
        for result in results:
            m_u, n_u, eom_operators = result

            for index_op, op in eom_operators.items():
                if op is not None:
                    all_matrix_operators[f"{index_op}_{m_u}_{n_u}"] = op

        return all_matrix_operators

    @staticmethod
    def _build_commutator_routine(
        params: list, operator: QubitOperator, z2_symmetries: Z2Symmetries
    ) -> tuple[int, int, dict[str, QubitOperator]]:
        """Numerically computes the commutator / double commutator between operators.

        Args:
            params: list containing the indices of matrix element and the corresponding
                excitation operators.
            operator: The hamiltonian.
            z2_symmetries: z2_symmetries in case of tapering.

        Returns:
            The indices of the matrix element and the corresponding qubit
            operator for each of the EOM matrices.
        """
        m_u, n_u, left_op_1, right_op_1, right_op_2 = params
        if left_op_1 is None or right_op_1 is None and right_op_2 is None:
            q_mat_op = None
            w_mat_op = None
            m_mat_op = None
            v_mat_op = None
        else:

            if right_op_1 is not None:
                # The sign which we use in the case of the double commutator is arbitrary. In
                # theory, one would choose this according to the nature of the problem (i.e.
                # whether it is fermionic or bosonic), but in practice, always choosing the
                # anti-commutator has proven to be more robust.
                q_mat_op = -double_commutator(left_op_1, operator, right_op_1, sign=False)
                # In the case of the single commutator, we are always interested in the energy
                # difference of two states. Thus, regardless of the problem's nature, we will
                # always use the commutator.
                w_mat_op = -commutator(left_op_1, right_op_1)
                q_mat_op = None if len(q_mat_op) == 0 else q_mat_op
                w_mat_op = None if len(w_mat_op) == 0 else w_mat_op
            else:
                q_mat_op = None
                w_mat_op = None

            if right_op_2 is not None:
                # For explanations on the choice of commutation relation, please refer to the
                # comments above.
                m_mat_op = double_commutator(left_op_1, operator, right_op_2, sign=False)
                v_mat_op = commutator(left_op_1, right_op_2)
                m_mat_op = None if len(m_mat_op) == 0 else m_mat_op
                v_mat_op = None if len(v_mat_op) == 0 else v_mat_op
            else:
                m_mat_op = None
                v_mat_op = None

        eom_operators = {"q": q_mat_op, "w": w_mat_op, "m": m_mat_op, "v": v_mat_op}

        if not z2_symmetries.is_empty():
            for index_op, eom_op in eom_operators.items():
                if eom_op is not None and len(eom_op) > 0:
                    eom_operators[index_op] = z2_symmetries.taper_clifford(eom_op)

        return m_u, n_u, eom_operators

    def _build_eom_matrices(
        self, gs_results: dict[str, tuple[complex, dict[str, Any]]], size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Constructs the H and S matrices from the results on the ground state.

        Args:
            gs_results: A ground state result object.
            size: Size of eigenvalue problem.

        Returns:
            H and S matrices and their standard deviation
        """

        mus, nus = np.triu_indices(size)

        m_mat = np.zeros((size, size), dtype=complex)
        v_mat = np.zeros((size, size), dtype=complex)
        q_mat = np.zeros((size, size), dtype=complex)
        w_mat = np.zeros((size, size), dtype=complex)
        m_mat_std, v_mat_std, q_mat_std, w_mat_std = 0.0, 0.0, 0.0, 0.0

        # evaluate results
        for m_u, n_u in zip(mus, nus):

            if gs_results.get(f"q_{m_u}_{n_u}") is not None:
                q_mat[m_u][n_u] = gs_results[f"q_{m_u}_{n_u}"][0]
                q_mat_std += gs_results[f"q_{m_u}_{n_u}"][1].get("variance", 0)

            if gs_results.get(f"w_{m_u}_{n_u}") is not None:
                w_mat[m_u][n_u] = gs_results[f"w_{m_u}_{n_u}"][0]
                w_mat_std += gs_results[f"w_{m_u}_{n_u}"][1].get("variance", 0)

            if gs_results.get(f"m_{m_u}_{n_u}") is not None:
                m_mat[m_u][n_u] = gs_results[f"m_{m_u}_{n_u}"][0]
                m_mat_std += gs_results[f"m_{m_u}_{n_u}"][1].get("variance", 0)

            if gs_results.get(f"v_{m_u}_{n_u}") is not None:
                v_mat[m_u][n_u] = gs_results[f"v_{m_u}_{n_u}"][0]
                v_mat_std += gs_results[f"v_{m_u}_{n_u}"][1].get("variance", 0)

        # these matrices are numpy arrays and therefore have the ``shape`` attribute
        # Matrix building rules
        # M.adjoint() = M, V.adjoint() = V
        # Q.T = Q, W.T = -W
        q_mat = q_mat + q_mat.T - np.identity(q_mat.shape[0]) * q_mat
        w_mat = w_mat - w_mat.T - np.identity(w_mat.shape[0]) * w_mat
        m_mat = m_mat + m_mat.T.conj() - np.identity(m_mat.shape[0]) * m_mat
        v_mat = v_mat + v_mat.T.conj() - np.identity(v_mat.shape[0]) * v_mat

        q_mat = np.real(q_mat)
        w_mat = np.real(w_mat)
        m_mat = np.real(m_mat)
        v_mat = np.real(v_mat)

        q_mat_std = q_mat_std / float(size**2)
        w_mat_std = w_mat_std / float(size**2)
        m_mat_std = m_mat_std / float(size**2)
        v_mat_std = v_mat_std / float(size**2)

        logger.debug("\nQ:=========================\n%s", q_mat)
        logger.debug("\nW:=========================\n%s", w_mat)
        logger.debug("\nM:=========================\n%s", m_mat)
        logger.debug("\nV:=========================\n%s", v_mat)

        # Matrix building rules
        # h_mat = [[M, Q], [P, N]] and s_mat = [[V, W], [T, U]]
        # N = M.conj() = M.T
        # P = Q.adjoint() = Q.conj() because Q = Q.T
        # U = -V.conj() = -V.T
        # T = W.adjoint()
        h_mat: np.ndarray = np.block([[m_mat, q_mat], [q_mat.T.conj(), m_mat.T]])
        s_mat: np.ndarray = np.block([[v_mat, w_mat], [w_mat.T.conj(), -v_mat.T]])

        h_mat_std: np.ndarray = np.array([[m_mat_std, q_mat_std], [q_mat_std, m_mat_std]])
        s_mat_std: np.ndarray = np.array([[v_mat_std, w_mat_std], [w_mat_std, v_mat_std]])

        return h_mat, s_mat, h_mat_std, s_mat_std

    def _prepare_expansion_basis(
        self, problem: BaseProblem
    ) -> tuple[dict[str, QubitOperator], dict[str, list[bool]], int]:
        """Prepares the basis expansion operators by calling the builder for second quantized operator
        and applying transformations (Mapping, Reduction, First step of the tapering).

        Args:
            problem: the problem for which to build out the operators.

        Returns:
            Dict of transformed hopping operators, dict of commutativity types, size of the qEOM problem
        """

        logger.debug("Building expansion basis data...")

        data = self._build_hopping_ops(problem)
        hopping_operators, type_of_commutativities, excitation_indices = data
        size = int(len(list(excitation_indices.keys())) // 2)

        # Small workaround to apply two_qubit_reduction to a list with convert_match()
        if isinstance(self.qubit_converter, QubitConverter):
            num_particles = self.qubit_converter.num_particles

            reduced_hopping_ops = {}
            for hopping_name, hopping_op in hopping_operators.items():
                reduced_hopping_ops[hopping_name] = self.qubit_converter._two_qubit_reduce(
                    hopping_op, num_particles
                )
            untap_hopping_ops = self.qubit_converter.convert_clifford(reduced_hopping_ops)
        else:
            untap_hopping_ops = hopping_operators

        return untap_hopping_ops, type_of_commutativities, size

    def _build_qeom_pseudoeigenvalue_problem(
        self,
        untap_operator: QubitOperator,
        expansion_basis_data: tuple[dict[str, QubitOperator], dict[str, list[bool]], int],
        reference_state: tuple[QuantumCircuit, Sequence[float]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Builds the matrices for the qEOM pseudo-eigenvalue problem

        Args:
            untap_operator: Not yet tapered hamiltonian
            expansion_basis_data: Dict of transformed hopping operators, dict of commutativity types,
            size of the qEOM problem
            reference_state: Reference state (often the VQE ground state) to be used for the evaluation
            of EOM operators.

        Returns:
            Matrices of the Pseudo-eigenvalue problem H @ X = S @ X @ E with the associated standard
            deviation errors.
        """

        logger.debug("Build QEOM pseudoeigenvalue problem...")

        # 1. Build all EOM operators to evaluate on the ground state
        tap_eom_matrix_ops = self._build_all_eom_operators(
            untap_operator,
            expansion_basis_data,
        )

        # 2. Evaluate all EOM operators on the ground state
        measurement_results = estimate_observables(
            self._estimator,
            reference_state[0],
            tap_eom_matrix_ops,
            reference_state[1],
        )

        # 4. Post-process the measurement results to construct eom matrices
        _, _, size = expansion_basis_data

        h_mat, s_mat, h_mat_std, s_mat_std = self._build_eom_matrices(measurement_results, size)

        return h_mat, s_mat, h_mat_std, s_mat_std

    def _compute_excitation_energies(
        self, h_mat: np.ndarray, s_mat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Diagonalizing H, S matrices for excitation energies.

        Args:
            h_mat : H matrix
            s_mat : S matrix

        Returns:
            1-D vector stores all energy gap to reference state.
            2-D array storing the X and Y expansion coefficients.
        """
        logger.debug("Diagonalizing qeom matrices for excited states...")

        res = linalg.eig(h_mat, s_mat)
        # convert nan value into 0
        res[0][np.where(np.isnan(res[0]))] = 0.0
        # Only the positive eigenvalues are physical. We need to take care
        # though of very small values
        # should an excited state approach ground state. Here the small values
        # may be both negative or
        # positive. We should take just one of these pairs as zero.
        # Since we may now have
        # small values (positive or negative) take the absolute and then threshold zero.
        logger.debug("... %s", res[0])
        # We keep only the negative half of the eigenvalues in decreasing order. Negative eigenvalues
        # correspond to the excitations whereas positive eigenvalues correspond to de-excitations
        order = np.argsort(np.real(res[0]))[::-1][len(res[0]) // 2 : :]
        w = np.real(res[0])[order]
        logger.debug("Sorted real parts %s", w)
        w = np.abs(w)
        w[w < self.tol] = 0
        excitation_energies_gap = w
        expansion_coefs = res[1][:, order]

        commutator_metric = expansion_coefs.T.conj() @ s_mat @ expansion_coefs

        return excitation_energies_gap, expansion_coefs, commutator_metric

    def _build_excitation_operators(
        self,
        expansion_basis_data: tuple[
            dict[str, QubitOperator],
            dict[str, list[bool]],
            int,
        ],
        reference_state: tuple[QuantumCircuit, Sequence[float]],
        expansion_coefs_rescaled: np.ndarray,
    ) -> list[QubitOperator]:
        """Build the excitation operators O_k such that O_k applied on the reference ground state gives
        the k-th excited state.

        Args:
            expansion_basis_data: Dict of transformed hopping operators, dict of commutativity types,
            size of the qEOM problem.
            reference_state : Reference ground state
            expansion_coefs_rescaled: Expansion coefficient matrix X such that H @ X = S @ X @ E and
            X^dag @ S @ X is the identity

        Returns:
            list of excitation operators [Identity, O_1, O_2, ...]
        """

        untap_hopping_ops, _, size = expansion_basis_data
        if isinstance(self.qubit_converter, QubitConverter):
            tap_hopping_ops = self.qubit_converter.symmetry_reduce_clifford(untap_hopping_ops)
        else:
            tap_hopping_ops = untap_hopping_ops

        additionnal_measurements = estimate_observables(
            self._estimator, reference_state[0], tap_hopping_ops, reference_state[1]
        )

        num_qubits = list(untap_hopping_ops.values())[0].num_qubits
        identity_op = PauliSumOp(SparsePauliOp(["I" * num_qubits], [1.0]))

        ordered_keys = [f"E_{k}" for k in range(size)] + [f"Edag_{k}" for k in range(size)]
        ordered_signs = [1 for k in range(size)] + [-1 for k in range(size)]

        translated_hopping_ops = {}
        for key, sign in zip(ordered_keys, ordered_signs):
            tap_hopping_ops_eval = (
                additionnal_measurements.get(key)[0]
                if additionnal_measurements.get(key) is not None
                else 0.0
            )
            translated_hopping_ops[key] = sign * (
                untap_hopping_ops[key] - identity_op * tap_hopping_ops_eval
            )

        # From the matrix of coefficients and the vector of basis operators, we create the vector of
        # excitation operators. An alternative with list comprehension is provided below as reference.
        #
        # excitations_ops = [
        #     SparsePauliOp.sum(
        #         [
        #             expansion_coefs_rescaled[k, i] * hopping_ops_vector[i]
        #             for i in range(expansion_coefs_rescaled.shape[1])
        #         ]
        #     )
        #     for k in range(expansion_coefs_rescaled.shape[0])
        # ]
        hopping_ops_vector = list(translated_hopping_ops.values())
        excitations_ops = np.array(hopping_ops_vector, dtype=object) @ expansion_coefs_rescaled
        excitations_ops_reduced = [identity_op] + [op.reduce() for op in excitations_ops]

        return excitations_ops_reduced

    def _prepare_excited_states_observables(
        self,
        untap_aux_ops: dict[str, QubitOperator],
        operators_reduced: list[QubitOperator],
        size: int,
    ) -> dict[tuple[str, int, int], QubitOperator]:
        """Prepare the operators O_k^dag @ Aux @ O_l associated to properties of the excited states k,l
        defined in the aux_eval_rules. By default, the expectation value of all observables on all
        excited states are evaluated while no transition amplitudes are computed.

        Args:
            untap_aux_ops: Dict of auxiliary operators for which properties will be computed.
            expansion_basis_data: Dict of transformed hopping operators, dict of commutativity types,
            size of the qEOM problem.
            operators_reduced: list of excitation operators [Identity, O_1, O_2, ...]
            size: size of the qEOM problem.

        Raises:
            ValueError: For when the aux_eval_rules do not correspond to any previously defined
            observable and excited state.

        Returns:
            Dict of operators of the form O_k^dag @ Aux @ O_l as specified in the constraints.
        """

        indices = np.diag_indices(size + 1)
        eval_rules: dict[str, list[tuple[Any, Any]]]

        if self.aux_eval_rules is None:
            eval_rules = {}
        elif isinstance(self.aux_eval_rules, dict):
            eval_rules = self.aux_eval_rules
        elif self.aux_eval_rules == EvaluationRule.ALL:
            indices = np.triu_indices(size + 1)
            aux_names = untap_aux_ops.keys()
            indices_list = list(zip(indices[0], indices[1]))
            eval_rules = {aux_name: indices_list for aux_name in aux_names}
        elif self.aux_eval_rules == EvaluationRule.DIAG:
            indices = np.diag_indices(size + 1)
            aux_names = untap_aux_ops.keys()
            indices_list = list(zip(indices[0], indices[1]))
            eval_rules = {aux_name: indices_list for aux_name in aux_names}
        else:
            raise ValueError("Aux evaluation rules are ill-defined")

        op_aux_op_dict: dict[tuple[str, int, int], QubitOperator] = {}

        for op_name, indices_constraint in eval_rules.items():
            if op_name not in untap_aux_ops.keys():
                raise ValueError("Evaluation constrains cannot be satisfied")
            aux_op = untap_aux_ops[op_name]

            for i, j in indices_constraint:
                if i >= len(operators_reduced) or j >= len(operators_reduced):
                    raise ValueError("Evaluation constrains cannot be satisfied")

                opi, opj = operators_reduced[i], operators_reduced[j]
                op_aux_op_dict[(op_name, i, j)] = (opi.adjoint() @ aux_op @ opj).reduce()

        return op_aux_op_dict

    def _evaluate_observables_excited_states(
        self,
        untap_aux_ops: dict[str, QubitOperator],
        expansion_basis_data: tuple[dict[str, QubitOperator], dict[str, list[bool]], int],
        reference_state: tuple[QuantumCircuit, Sequence[float]],
        expansion_coefs_rescaled: np.ndarray,
    ) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[tuple[int, int], dict[str, Any]]]:
        """Evaluate the expectation values and transition amplitudes of the auxiliary operators on the
        excited states. Custom rules can be used to define which expectation values and transition
        amplitudes to compute. A typical rule is specified in the form of a nary
        {'hamiltonian':[(1,1)]}

        Args:
            untap_aux_ops: Dict of auxiliary operators for which properties will be computed.
            expansion_basis_data: Dict of transformed hopping operators, dict of commutativity types,
            size of the qEOM problem.
            reference_state: Reference ground state.
            expansion_coefs_rescaled: Expansion coefficient matrix X such that H @ X = S @ X @ E and
            X^dag @ S @ X is the identity.

        Returns:
            list of excitation operators [Identity, O_1, O_2, ...]
        """

        aux_operators_eigenvalues: dict[tuple[int, int], dict[str, Any]] = {}
        transition_amplitudes: dict[tuple[int, int], dict[str, Any]] = {}

        _, _, size = expansion_basis_data

        if untap_aux_ops is not None:

            # 1. Build excitation operators O_l such that O_l |0> = |l>
            excitations_ops_reduced = self._build_excitation_operators(
                expansion_basis_data, reference_state, expansion_coefs_rescaled
            )

            # 2. Prepare observables O_k^\dag @ Aux @ O_l
            op_aux_op_dict = self._prepare_excited_states_observables(
                untap_aux_ops, excitations_ops_reduced, size
            )

            # 3. Measure observables
            if isinstance(self.qubit_converter, QubitConverter):
                tap_op_aux_op_dict = self.qubit_converter.symmetry_reduce_clifford(op_aux_op_dict)
            else:
                tap_op_aux_op_dict = op_aux_op_dict

            aux_measurements = estimate_observables(
                self._estimator, reference_state[0], tap_op_aux_op_dict, reference_state[1]
            )

            # 4. Format aux_operators_eigenvalues
            indices_diag = np.diag_indices(size + 1)
            indices_diag_as_list = list(zip(indices_diag[0], indices_diag[1]))
            for indice in indices_diag_as_list:
                aux_operators_eigenvalues[indice] = {}
                for aux_name in untap_aux_ops.keys():
                    aux_operators_eigenvalues[indice][aux_name] = aux_measurements.get(
                        (aux_name, indice[0], indice[1]), (0.0, {})
                    )

            # 5. Format transition_amplitudes
            indices_offdiag = np.triu_indices(size + 1, k=1)
            indices_offdiag_as_list = list(zip(indices_offdiag[0], indices_offdiag[1]))
            for indice in indices_offdiag_as_list:
                transition_amplitudes[indice] = {}
                for aux_name in untap_aux_ops.keys():
                    transition_amplitudes[indice][aux_name] = aux_measurements.get(
                        (aux_name, indice[0], indice[1]), (0.0, {})
                    )

        return aux_operators_eigenvalues, transition_amplitudes

    def _build_qeom_result(
        self,
        problem,
        groundstate_result,
        expansion_coefs,
        energy_gaps,
        h_mat,
        s_mat,
        h_mat_std,
        s_mat_std,
        aux_operators_eigenvalues,
        transition_amplitudes,
        gammas_square,
    ) -> ElectronicStructureResult:

        qeom_result = QEOMResult()
        qeom_result.ground_state_raw_result = groundstate_result.raw_result
        qeom_result.expansion_coefficients = expansion_coefs
        qeom_result.excitation_energies = energy_gaps

        qeom_result.h_matrix = h_mat
        qeom_result.s_matrix = s_mat
        qeom_result.h_matrix_std = h_mat_std
        qeom_result.s_matrix_std = s_mat_std
        qeom_result.gamma_square = gammas_square

        qeom_result.aux_operators_evaluated = list(aux_operators_eigenvalues.values())
        qeom_result.transition_amplitudes = transition_amplitudes

        groundstate_energy_reference = groundstate_result.eigenvalues[0]
        excited_eigenenergies = energy_gaps + groundstate_energy_reference
        qeom_result.eigenvalues = np.append(
            groundstate_result.eigenvalues[0], excited_eigenenergies
        )

        qeom_result.eigenstates = np.array([])

        eigenstate_result = EigenstateResult.from_result(qeom_result)
        result = problem.interpret(eigenstate_result)

        return result


class QEOMResult(EigensolverResult):
    """The results class for the QEOM algorithm."""

    def __init__(self) -> None:
        super().__init__()
        self.ground_state_raw_result = None
        self.excitation_energies: np.ndarray | None = None
        self.expansion_coefficients: np.ndarray | None = None
        self.eigenvalues: np.ndarray | None = None
        self.eigenstates: np.ndarray | None = None
        self.h_matrix: np.ndarray | None = None
        self.s_matrix: np.ndarray | None = None
        self.h_matrix_std: np.ndarray = np.zeros((2, 2))
        self.s_matrix_std: np.ndarray = np.zeros((2, 2))

        self._m_matrix: np.ndarray | None = None
        self._v_matrix: np.ndarray | None = None
        self._q_matrix: np.ndarray | None = None
        self._w_matrix: np.ndarray | None = None
        self._m_matrix_std: float = 0.0
        self._v_matrix_std: float = 0.0
        self._q_matrix_std: float = 0.0
        self._w_matrix_std: float = 0.0

        self.aux_operators_evaluated: list[
            ListOrDictType[tuple[complex, dict[str, Any]]]
        ] | None = None
        self.transition_amplitudes: list[
            ListOrDictType[tuple[complex, dict[str, Any]]]
        ] | None = None
        self.gamma_square: np.ndarray = None

    @property
    def m_matrix(self) -> np.ndarray | None:
        """returns the M matrix"""
        if self.h_matrix is not None and self._m_matrix is None:
            return self.h_matrix[: len(self.h_matrix) // 2, : len(self.h_matrix) // 2]
        else:
            return self._m_matrix

    @m_matrix.setter
    @deprecate_function(
        "The M matrix is now computed from the H matrix. "
        "This setter will be deprecated in a future release and subsequently "
        "removed after that.",
        category=PendingDeprecationWarning,
    )
    def m_matrix(self, value: np.ndarray) -> None:
        """sets the M matrix"""
        self._m_matrix = value

    @property
    def v_matrix(self) -> np.ndarray | None:
        """returns the V matrix"""
        if self.s_matrix is not None and self._v_matrix is None:
            return self.s_matrix[: len(self.s_matrix) // 2, : len(self.s_matrix) // 2]
        else:
            return self._v_matrix

    @v_matrix.setter
    @deprecate_function(
        "The V matrix is now computed from the S matrix. "
        "This setter will be deprecated in a future release and subsequently "
        "removed after that.",
        category=PendingDeprecationWarning,
    )
    def v_matrix(self, value: np.ndarray) -> None:
        """sets the V matrix"""
        self._v_matrix = value

    @property
    def q_matrix(self) -> np.ndarray | None:
        """returns the Q matrix"""
        q_mat: np.ndarray | None = None
        if self.h_matrix is not None:
            q_mat = self.h_matrix[len(self.h_matrix) // 2 :, : len(self.h_matrix) // 2]
        if self._q_matrix is not None:
            q_mat = self._q_matrix
        return q_mat

    @q_matrix.setter
    @deprecate_function(
        "The Q matrix is now computed from the H matrix. "
        "This setter will be deprecated in a future release and subsequently "
        "removed after that.",
        category=PendingDeprecationWarning,
    )
    def q_matrix(self, value: np.ndarray) -> None:
        """sets the Q matrix"""
        self._q_matrix = value

    @property
    def w_matrix(self) -> np.ndarray | None:
        """returns the W matrix"""
        if self.s_matrix is not None and self._w_matrix is None:
            return self.s_matrix[len(self.s_matrix) // 2 :, : len(self.s_matrix) // 2]
        else:
            return self._w_matrix

    @w_matrix.setter
    @deprecate_function(
        "The W matrix is now computed from the S matrix. "
        "This setter will be deprecated in a future release and subsequently "
        "removed after that.",
        category=PendingDeprecationWarning,
    )
    def w_matrix(self, value: np.ndarray) -> None:
        """sets the W matrix"""
        self._w_matrix = value

    @property
    def m_matrix_std(self) -> float:
        """returns the M matrix standard deviation"""
        if not np.isclose(self.h_matrix_std[0, 0], 0.0) and np.isclose(self._m_matrix_std, 0.0):
            return self.h_matrix_std[0, 0]
        else:
            return self._m_matrix_std

    @m_matrix_std.setter
    def m_matrix_std(self, value: float) -> None:
        """sets the M matrix standard deviation"""
        self._m_matrix_std = value

    @property
    def v_matrix_std(self) -> float:
        """returns the V matrix standard deviation"""
        if not np.isclose(self.s_matrix_std[0, 0], 0.0) and np.isclose(self._v_matrix_std, 0.0):
            return self.s_matrix_std[0, 0]
        else:
            return self._v_matrix_std

    @v_matrix_std.setter
    def v_matrix_std(self, value: float) -> None:
        """sets the V matrix standard deviation"""
        self._v_matrix_std = value

    @property
    def q_matrix_std(self) -> float:
        """returns the Q matrix standard deviation"""
        if not np.isclose(self.h_matrix_std[0, 1], 0.0) and np.isclose(self._q_matrix_std, 0.0):
            return self.h_matrix_std[0, 0]
        else:
            return self._q_matrix_std

    @q_matrix_std.setter
    def q_matrix_std(self, value: float) -> None:
        """sets the Q matrix standard deviation"""
        self._q_matrix_std = value

    @property
    def w_matrix_std(self) -> float:
        """returns the W matrix standard deviation"""
        if not np.isclose(self.s_matrix_std[0, 1], 0.0) and np.isclose(self._w_matrix_std, 0.0):
            return self.s_matrix_std[0, 0]
        else:
            return self._w_matrix_std

    @w_matrix_std.setter
    def w_matrix_std(self, value: float) -> None:
        """sets the W matrix standard deviation"""
        self._w_matrix_std = value
