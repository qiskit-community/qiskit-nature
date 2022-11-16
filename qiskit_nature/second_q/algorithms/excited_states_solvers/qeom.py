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

"""The calculation of excited states via the qEOM algorithm."""

from __future__ import annotations

from typing import Any, Callable, List, Sequence, Union, Optional, Tuple, Dict
import itertools
import logging
import sys

import numpy as np
from scipy import linalg
from qiskit.algorithms.observables_evaluator import estimate_observables
from qiskit.tools import parallel_map
from qiskit.tools.events import TextProgressBar
from qiskit.utils import algorithm_globals
from qiskit.algorithms.eigensolvers import EigensolverResult
from qiskit.circuit import QuantumCircuit

from qiskit.opflow import (
    Z2Symmetries,
    commutator,
    double_commutator,
    PauliSumOp,
)
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import BaseEstimator
from qiskit_nature.converters.second_quantization.utils import ListOrDict
from qiskit_nature.second_q.operators import SparseLabelOp
from qiskit_nature.second_q.problems import (
    BaseProblem,
    ElectronicStructureProblem,
    VibrationalStructureProblem,
)
from qiskit_nature.second_q.problems import EigenstateResult, ElectronicStructureResult

from .qeom_electronic_ops_builder import build_electronic_ops
from .qeom_vibrational_ops_builder import build_vibrational_ops
from .excited_states_solver import ExcitedStatesSolver
from ..ground_state_solvers import GroundStateSolver
from ..ground_state_solvers.ground_state_solver import QubitOperator
from ..ground_state_solvers.minimum_eigensolver_factories import MinimumEigensolverFactory

logger = logging.getLogger(__name__)


class QEOM(ExcitedStatesSolver):
    """The calculation of excited states via the qEOM algorithm."""

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
        aux_eval_rules: str | dict[str, list[tuple[int, int]]] | None = None,
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
                :`int`: a single, positive integer which denotes the number of excitations
                    (1 == `s`, 2 == `d`, etc.)
                :`list[int]`: a list of positive integers generalizing the above to multiple numbers
                    of excitations ([1, 2] == `sd`, etc.)
                :`Callable`: a function which can be used to specify a custom list of excitations.
                    For more details on how to write such a function refer to one of the default
                    methods, :meth:`generate_fermionic_excitations` or
                    :meth:`generate_vibrational_excitations`, when solving an
                    :class:`.ElectronicStructureProblem` or a :class:`.VibrationalStructureProblem`,
                    respectively.

            aux_eval_rules: The rules determining how observables should be evaluated on excited states.

                :`str`: specific predefined rules. Allowed strings are:
                    + `all` to compute all expectation values and all transition amplitudes
                    + `diag` to only compute expectation values
                :`dict[str, list[tuple]]`: Dictionary mapping valid auxiliary operator's name to lists
                of tuple (i, j) specifying the indices of the excited states to be evaluated on. By
                default, none of the auxiliary operators are evaluated on none of the excited states.
        """
        self._gsc = ground_state_solver
        self._estimator = estimator
        self.excitations = excitations
        self._aux_eval_rules = aux_eval_rules

        self._untapered_qubit_op_main: PauliSumOp = None

    @property
    def excitations(
        self,
    ) -> str | int | list[int] | Callable[
        [int, tuple[int, int]], list[tuple[tuple[int, ...], tuple[int, ...]]]
    ]:
        """Returns the excitations to be included in the eom pseudo-eigenvalue problem."""
        return self._excitations

    @excitations.setter
    def excitations(
        self,
        excitations: str
        | int
        | list[int]
        | Callable[[int, tuple[int, int]], list[tuple[tuple[int, ...], tuple[int, ...]]]],
    ) -> None:
        """The excitations to be included in the eom pseudo-eigenvalue problem. If a string then
        all excitations of given type will be used. Otherwise, a list of custom excitations can
        directly be provided."""
        if isinstance(excitations, str) and excitations not in ["s", "d", "sd"]:
            raise ValueError(
                "Excitation type must be s (singles), d (doubles) or sd (singles and doubles)"
            )
        self._excitations = excitations

    @property
    def qubit_converter(self):
        """Returns the qubit_converter object defined in the ground state solver."""
        return self._gsc.qubit_converter

    @property
    def solver(self):
        """Returns the solver object defined in the ground state solver."""
        return self._gsc.solver

    def get_qubit_operators(
        self,
        problem: BaseProblem,
        aux_operators: Optional[dict[str, Union[SparseLabelOp, PauliSumOp]]] = None,
    ) -> Tuple[PauliSumOp, Optional[dict[str, PauliSumOp]]]:
        """Construct qubit operators by getting the second quantized operators from the problem
        (potentially running a driver in doing so [can be computationally expensive])
        and using a QubitConverter to map and reduce the operators to qubit operators.

        Args:
            problem: A class encoding a problem to be solved.
            aux_operators: Additional auxiliary operators to evaluate.

        Returns:
            Qubit operator.
            Additional auxiliary operators.
        """
        return self._gsc.get_qubit_operators(problem, aux_operators)

    def get_qeom_qubit_operators(
        self,
        problem: BaseProblem,
        aux_operators: Optional[dict[str, Union[SparseLabelOp, PauliSumOp]]] = None,
    ) -> Tuple[
        QubitOperator, Optional[dict[str, QubitOperator]], Optional[dict[str, QubitOperator]]
    ]:
        """Gets the operator and auxiliary operators, and transforms the provided auxiliary operators.
        Note that contrary to the method :meth:`get_qubit_oprators` from the
        :class:`GroundStateEigensolver`, this returns three outputs: the hamiltonian, second quantization
        default auxiliaries, and user defined auxiliaries.
        Also note that this methods performs a specific treatment of the symmetries required by the qEOM
        calculation."""

        main_second_q_op, aux_second_q_ops = problem.second_q_ops()

        # 1. Convert to PauliSumOp and apply two qubit reduction
        # We apply the meth:`convert()` with the symmetries deliberately set to None
        num_particles = getattr(problem, "num_particles", None)
        main_operator = self.qubit_converter.convert_only(
            main_second_q_op,
            num_particles=num_particles,
        )
        self.qubit_converter.force_match(num_particles=num_particles)

        aux_default_ops = self.qubit_converter.convert_match(aux_second_q_ops)
        aux_custom_ops = {}
        if aux_operators is not None:
            for name_aux, aux_op in aux_operators.items():
                if isinstance(aux_op, (SparseLabelOp)):
                    converted_aux_op = self.qubit_converter.convert_match(aux_op)
                else:
                    converted_aux_op = aux_op

                aux_custom_ops[name_aux] = converted_aux_op

        # 2. Find the z2symmetries, set them in the qubit_converter, and apply the first step of the
        # tapering.
        _, z2symmetries = self.qubit_converter._find_taper_op(
            main_operator, problem.symmetry_sector_locator
        )
        self.qubit_converter.force_match(z2symmetries=z2symmetries)

        pre_tap_main_operator = self.qubit_converter._convert_clifford(main_operator)
        pre_tap_default_aux_ops = self.qubit_converter._convert_clifford(aux_default_ops)
        pre_tap_custom_aux_ops = self.qubit_converter._convert_clifford(aux_custom_ops)

        # 3. If the eigensolver does not support auxiliary operators, reset them
        if not self.solver.supports_aux_operators():
            pre_tap_default_aux_ops = None
            pre_tap_custom_aux_ops = None

        # 4. If a MinimumEigensolverFactory was provided, then an additional call to get_solver() is
        # required.
        if isinstance(self.solver, MinimumEigensolverFactory):
            self._gsc._solver = self.solver.get_solver(problem, self.qubit_converter)

        return pre_tap_main_operator, pre_tap_default_aux_ops, pre_tap_custom_aux_ops

    def solve(
        self,
        problem: BaseProblem,
        aux_operators: Optional[dict[str, SparseLabelOp]] = None,
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
            pre_tap_qubit_op_main,  # Hamiltonian
            pre_tap_default_aux_ops,  # Default auxiliary observables
            pre_tap_custom_aux_ops,  # User-defined auxiliary observables
        ) = self.get_qeom_qubit_operators(problem, aux_operators)

        # 2. Run ground state calculation with fully tapered custom auxiliary operators
        # Note that the solve() method includes the `second_q' auxiliary operators
        tap_custom_aux_ops = self.qubit_converter._symmetry_reduce_clifford(
            ListOrDict(pre_tap_custom_aux_ops), True
        )

        groundstate_result = self._gsc.solve(problem, tap_custom_aux_ops)
        ground_state = groundstate_result.eigenstates[0]

        # 3. Prepare the expansion operators for the excited state calculation
        expansion_basis_data = self._prepare_expansion_basis(problem)

        # 4. Obtain the representation of the Hamiltonian in the linear subspace
        h_mat, s_mat, h_mat_std, s_mat_std = self._build_qeom_pseudoeigenvalue_problem(
            pre_tap_qubit_op_main, expansion_basis_data, ground_state
        )

        # 5. Solve the pseudo-eigenvalue problem
        energy_gaps, expansion_coefs, commutator_metric = self._compute_excitation_energies(
            h_mat, s_mat
        )
        gammas_square: np.ndarray = np.abs(np.diagonal(commutator_metric))
        logger.debug("Gamma square... %s", gammas_square)
        scaling_matrix: np.ndarray = np.diag(
            np.divide(np.ones_like(gammas_square), np.sqrt(gammas_square))
        )
        expansion_coefs_rescaled: np.ndarray = expansion_coefs @ scaling_matrix

        # 6. Evaluate auxiliary operators on the excited states
        if pre_tap_custom_aux_ops is not None:
            pre_tap_aux_observables = {**pre_tap_default_aux_ops, **pre_tap_custom_aux_ops}
        else:
            pre_tap_aux_observables = pre_tap_default_aux_ops

        (
            aux_operators_eigenvalues,
            transition_amplitudes,
        ) = self._evaluate_observables_excited_states(
            pre_tap_aux_observables,
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
    ) -> Tuple[
        Dict[str, PauliSumOp],
        Dict[str, List[bool]],
        Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]],
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
        pre_tap_operator: PauliSumOp,
        expansion_basis_data: Tuple[dict[str, PauliSumOp], dict[str, List[bool]], int],
    ) -> dict:
        """Building all commutators for Q, W, M, V matrices.

        Args:
            pre_tap_operator: Pre tapered Hamiltonian operator
            expansion_basis_data: all hopping operators based on excitations_list,
                key is the string of single/double excitation;
                value is corresponding operator.

        Returns:
            A dictionary that contains the operators for each matrix element.
        """

        pre_tap_hopping_ops, type_of_commutativities, size = expansion_basis_data
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

        try:
            z2_symmetries = self._gsc.qubit_converter.z2symmetries
        except AttributeError:
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
                        available_hopping_ops[key] = pre_tap_hopping_ops[key]
                _build_one_sector(available_hopping_ops)

        else:
            _build_one_sector(pre_tap_hopping_ops)

        if logger.isEnabledFor(logging.INFO):
            logger.info("Building all commutators:")
            TextProgressBar(sys.stderr)
        results = parallel_map(
            self._build_commutator_routine,
            to_be_computed_list,
            task_args=(pre_tap_operator, z2_symmetries),
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
        params: List, operator: PauliSumOp, z2_symmetries: Z2Symmetries
    ) -> Tuple[int, int, dict[str, PauliSumOp]]:
        """Numerically computes the commutator / double commutator between operators.

        Args:
            params: List containing the indices of matrix element and the corresponding
                excitation operators.
            operator: The hamiltonian.
            z2_symmetries: z2_symmetries in case of tapering.

        Returns:
            The indices of the matrix element and the corresponding qubit
            operator for each of the EOM matrices.
        """
        m_u, n_u, left_op1, right_op_1, right_op_2 = params
        if left_op1 is None or right_op_1 is None and right_op_2 is None:
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
                q_mat_op = -double_commutator(left_op1, operator, right_op_1, sign=False)
                # In the case of the single commutator, we are always interested in the energy
                # difference of two states. Thus, regardless of the problem's nature, we will
                # always use the commutator.
                w_mat_op = -commutator(left_op1, right_op_1)
                q_mat_op = None if len(q_mat_op) == 0 else q_mat_op
                w_mat_op = None if len(w_mat_op) == 0 else w_mat_op
            else:
                q_mat_op = None
                w_mat_op = None

            if right_op_2 is not None:
                # For explanations on the choice of commutation relation, please refer to the
                # comments above.
                m_mat_op = double_commutator(left_op1, operator, right_op_2, sign=False)
                v_mat_op = commutator(left_op1, right_op_2)
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        # P = Q.adjoint() = Q.conj()
        # U = -V.conj() = -V.T
        # T = W.adjoint()
        h_mat: np.ndarray = np.array(
            np.matrixlib.bmat([[m_mat, q_mat], [q_mat.T.conj(), m_mat.T]])
        )
        s_mat: np.ndarray = np.array(
            np.matrixlib.bmat([[v_mat, w_mat], [w_mat.T.conj(), -v_mat.T]])
        )

        h_mat_std: np.ndarray = np.array([[m_mat_std, q_mat_std], [q_mat_std, m_mat_std]])
        s_mat_std: np.ndarray = np.array([[v_mat_std, w_mat_std], [w_mat_std, v_mat_std]])

        return h_mat, s_mat, h_mat_std, s_mat_std

    def _prepare_expansion_basis(
        self, problem: BaseProblem
    ) -> Tuple[Dict[str, PauliSumOp], Dict[str, List[bool]], int]:
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

        reduced_hopping_ops = self.qubit_converter.two_qubit_reduce(hopping_operators)
        pre_tap_hopping_ops = self.qubit_converter._convert_clifford(reduced_hopping_ops)

        return pre_tap_hopping_ops, type_of_commutativities, size

    def _build_qeom_pseudoeigenvalue_problem(
        self,
        pre_tap_operator: PauliSumOp,
        expansion_basis_data: Tuple[Dict[str, PauliSumOp], Dict[str, List[bool]], int],
        reference_state: Tuple[QuantumCircuit, Sequence[float]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Builds the matrices for the qEOM pseudo-eigenvalue problem

        Args:
            pre_tap_operator: Partially tapered hamiltonian
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
            pre_tap_operator,
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

    @staticmethod
    def _compute_excitation_energies(
        h_mat: np.ndarray, s_mat: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        # positive. We should take just one of these pairs as zero. So to get the values we want we
        # sort the real parts and then take the upper half of the sorted values.
        # Since we may now have
        # small values (positive or negative) take the absolute and then threshold zero.
        logger.debug("... %s", res[0])
        order = np.argsort(np.real(res[0]))[::-1][len(res[0]) // 2 : :]
        w = np.real(res[0])[order]
        logger.debug("Sorted real parts %s", w)
        w = np.abs(w)
        w[w < 1e-06] = 0
        excitation_energies_gap = w
        expansion_coefs = res[1][:, order]

        commutator_metric = expansion_coefs.T.conj() @ s_mat @ expansion_coefs

        return excitation_energies_gap, expansion_coefs, commutator_metric

    def _build_excitation_operators(
        self,
        expansion_basis_data: Tuple[
            dict[str, PauliSumOp],
            dict[str, List[bool]],
            int,
        ],
        reference_state: Tuple[QuantumCircuit, Sequence[float]],
        expansion_coefs_rescaled: np.ndarray,
    ) -> list[PauliSumOp]:
        """Build the excitation operators O_k such that O_k applied on the reference ground state gives
        the k-th excited state.

        Args:
            expansion_basis_data: Dict of transformed hopping operators, dict of commutativity types,
            size of the qEOM problem.
            reference_state : Reference ground state
            expansion_coefs_rescaled: Expansion coefficient matrix X such that H @ X = S @ X @ E and
            X^dag @ S @ X is the identity

        Returns:
            List of excitation operators [Identity, O_1, O_2, ...]
        """

        pre_tap_hopping_ops, _, size = expansion_basis_data
        tap_hopping_ops = self.qubit_converter._symmetry_reduce_clifford(
            ListOrDict(pre_tap_hopping_ops), True
        )

        additionnal_measurements = estimate_observables(
            self._estimator, reference_state[0], tap_hopping_ops, reference_state[1]
        )

        num_qubits = list(pre_tap_hopping_ops.values())[0].num_qubits
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
                pre_tap_hopping_ops[key] - identity_op * tap_hopping_ops_eval
            )

        hopping_ops_vector = list(translated_hopping_ops.values())
        excitations_ops = np.array(hopping_ops_vector, dtype=object) @ expansion_coefs_rescaled
        excitations_ops_reduced = [identity_op] + [op.reduce() for op in excitations_ops]

        return excitations_ops_reduced

    def _prepare_excited_states_observables(
        self,
        pre_tap_aux_observables: dict[str, PauliSumOp],
        operators_reduced: list[PauliSumOp],
        size: int,
    ) -> dict[Tuple[str, int, int], PauliSumOp]:
        """Prepare the operators O_k^dag @ Aux @ O_l associated to properties of the excited states k,l
        defined in the aux_eval_rules. By default, the expectation value of all observables on all
        excited states are evaluated while no transition amplitudes are computed.

        Args:
            pre_tap_aux_observables: Dict of auxiliary operators for which properties will be computed.
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
        eval_rules: Dict[str, List[Tuple[Any, Any]]]

        if self._aux_eval_rules is None:
            eval_rules = {}
        elif isinstance(self._aux_eval_rules, Dict):
            eval_rules = self._aux_eval_rules
        elif self._aux_eval_rules == "all":
            indices = np.triu_indices(size + 1)
            aux_names = pre_tap_aux_observables.keys()
            indices_list = list(zip(indices[0], indices[1]))
            eval_rules = {aux_name: indices_list for aux_name in aux_names}
        elif self._aux_eval_rules == "diag":
            indices = np.diag_indices(size + 1)
            aux_names = pre_tap_aux_observables.keys()
            indices_list = list(zip(indices[0], indices[1]))
            eval_rules = {aux_name: indices_list for aux_name in aux_names}
        else:
            raise ValueError("Aux evaluation rules are ill-defined")

        op_aux_op_dict: dict[Tuple[str, int, int], PauliSumOp] = {}

        for op_name, indices_constraint in eval_rules.items():
            if op_name not in pre_tap_aux_observables.keys():
                raise ValueError("Evaluation constrains cannot be satisfied")
            aux_op = pre_tap_aux_observables[op_name]

            for i, j in indices_constraint:
                if i >= len(operators_reduced) or j >= len(operators_reduced):
                    raise ValueError("Evaluation constrains cannot be satisfied")

                opi, opj = operators_reduced[i], operators_reduced[j]
                op_aux_op_dict[(op_name, i, j)] = (opi.adjoint() @ aux_op @ opj).reduce()

        return op_aux_op_dict

    def _evaluate_observables_excited_states(
        self,
        pre_tap_aux_observables: dict[str, PauliSumOp],
        expansion_basis_data: Tuple[dict[str, PauliSumOp], dict[str, List[bool]], int],
        reference_state: Tuple[QuantumCircuit, Sequence[float]],
        expansion_coefs_rescaled: np.ndarray,
    ) -> Tuple[Dict[Tuple[int, int], Dict[str, Any]], Dict[Tuple[int, int], Dict[str, Any]]]:
        """Evaluate the expectation values and transition amplitudes of the auxiliary operators on the
        excited states. Custom rules can be used to define which expectation values and transition
        amplitudes to compute. A typical rule is specified in the form of a nary
        {'hamiltonian':[(1,1)]}

        Args:
            pre_tap_aux_observables: Dict of auxiliary operators for which properties will be computed.
            expansion_basis_data: Dict of transformed hopping operators, dict of commutativity types,
            size of the qEOM problem.
            reference_state: Reference ground state.
            expansion_coefs_rescaled: Expansion coefficient matrix X such that H @ X = S @ X @ E and
            X^dag @ S @ X is the identity.

        Returns:
            List of excitation operators [Identity, O_1, O_2, ...]
        """

        aux_operators_eigenvalues: Dict[Tuple[int, int], Dict[str, Any]] = {}
        transition_amplitudes: Dict[Tuple[int, int], Dict[str, Any]] = {}

        _, _, size = expansion_basis_data

        if pre_tap_aux_observables is not None:

            # 1. Build excitation operators O_l such that O_l |0> = |l>
            excitations_ops_reduced = self._build_excitation_operators(
                expansion_basis_data, reference_state, expansion_coefs_rescaled
            )

            # 2. Prepare observables O_k^\dag @ Aux @ O_l
            op_aux_op_dict = self._prepare_excited_states_observables(
                pre_tap_aux_observables, excitations_ops_reduced, size
            )

            # 3. Measure observables
            tap_op_aux_op_dict = self.qubit_converter._symmetry_reduce_clifford(
                op_aux_op_dict, True
            )
            aux_measurements = estimate_observables(
                self._estimator, reference_state[0], tap_op_aux_op_dict, reference_state[1]
            )

            # 4. Format aux_operators_eigenvalues
            indices_diag = np.diag_indices(size + 1)
            indices_diag_as_list = list(zip(indices_diag[0], indices_diag[1]))
            for indice in indices_diag_as_list:
                aux_operators_eigenvalues[indice] = {}
                for aux_name in pre_tap_aux_observables.keys():
                    aux_operators_eigenvalues[indice][aux_name] = aux_measurements.get(
                        (aux_name, indice[0], indice[1]), (0.0, {})
                    )

            # 5. Format transition_amplitudes
            indices_offdiag = np.triu_indices(size + 1, k=1)
            indices_offdiag_as_list = list(zip(indices_offdiag[0], indices_offdiag[1]))
            for indice in indices_offdiag_as_list:
                transition_amplitudes[indice] = {}
                for aux_name in pre_tap_aux_observables.keys():
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
        self._ground_state_raw_result = None
        self._excitation_energies: Optional[np.ndarray] = None
        self._expansion_coefficients: Optional[np.ndarray] = None
        self._eigenvalues: Optional[np.ndarray] = None
        self._eigenstates: Optional[np.ndarray] = None
        self._h_matrix: Optional[np.ndarray] = None
        self._s_matrix: Optional[np.ndarray] = None
        self._h_matrix_std: np.ndarray = np.zeros((2, 2))
        self._s_matrix_std: np.ndarray = np.zeros((2, 2))
        self._aux_operators_evaluated: Optional[List[ListOrDict[Tuple[complex, complex]]]] = None
        self._transition_amplitudes: Optional[List[ListOrDict[Tuple[complex, complex]]]] = None
        self._gamma_square: np.ndarray = None

    @property
    def ground_state_raw_result(self):
        """returns ground state raw result"""
        return self._ground_state_raw_result

    @ground_state_raw_result.setter
    def ground_state_raw_result(self, value) -> None:
        """sets ground state raw result"""
        self._ground_state_raw_result = value

    @property
    def excitation_energies(self) -> Optional[np.ndarray]:
        """returns the excitation energies (energy gaps)"""
        return self._excitation_energies

    @excitation_energies.setter
    def excitation_energies(self, value: np.ndarray) -> None:
        """sets the excitation energies (energy gaps)"""
        self._excitation_energies = value

    @property
    def expansion_coefficients(self) -> Optional[np.ndarray]:
        """returns the X and Y expansion coefficients"""
        return self._expansion_coefficients

    @expansion_coefficients.setter
    def expansion_coefficients(self, value: np.ndarray) -> None:
        """sets the X and Y expansion coefficients"""
        self._expansion_coefficients = value

    @property
    def h_matrix(self) -> Optional[np.ndarray]:
        """returns the H matrix"""
        return self._h_matrix

    @h_matrix.setter
    def h_matrix(self, value: np.ndarray) -> None:
        """sets the H matrix"""
        self._h_matrix = value

    @property
    def s_matrix(self) -> Optional[np.ndarray]:
        """returns the S matrix"""
        return self._s_matrix

    @s_matrix.setter
    def s_matrix(self, value: np.ndarray) -> None:
        """sets the S matrix"""
        self._s_matrix = value

    @property
    def h_matrix_std(self) -> Optional[np.ndarray]:
        """returns the H matrix standard deviation"""
        return self._h_matrix_std

    @h_matrix_std.setter
    def h_matrix_std(self, value: Optional[np.ndarray]) -> None:
        """sets the H matrix standard deviation"""
        self._h_matrix_std = value

    @property
    def s_matrix_std(self) -> Optional[np.ndarray]:
        """returns the S matrix standard deviation"""
        return self._s_matrix_std

    @s_matrix_std.setter
    def s_matrix_std(self, value: Optional[np.ndarray]) -> None:
        """sets the S matrix standard deviation"""
        self._s_matrix_std = value

    @property
    def eigenvalues(self) -> Optional[np.ndarray]:
        """returns eigen values"""
        return self._eigenvalues

    @eigenvalues.setter
    def eigenvalues(self, value: np.ndarray) -> None:
        """set eigen values"""
        self._eigenvalues = value

    @property
    def eigenstates(self) -> Optional[np.ndarray]:
        """return eigen states"""
        return self._eigenstates

    @eigenstates.setter
    def eigenstates(self, value: np.ndarray) -> None:
        """set eigen states"""
        self._eigenstates = value

    @property
    def alphas(self) -> Optional[np.ndarray]:
        """return alphas"""
        return self._alphas

    @alphas.setter
    def alphas(self, value: np.ndarray) -> None:
        """set alphas"""
        self._alphas = value

    @property
    def gamma_square(self) -> Optional[np.ndarray]:
        """return gamma_square"""
        return self._gamma_square

    @gamma_square.setter
    def gamma_square(self, value: np.ndarray) -> None:
        """set gamma_square"""
        self._gamma_square = value

    @property
    def aux_operators_evaluated(self) -> Optional[List[ListOrDict[Tuple[complex, complex]]]]:
        """Return aux operator expectation values."""
        return self._aux_operators_evaluated

    @aux_operators_evaluated.setter
    def aux_operators_evaluated(self, value: List[ListOrDict[Tuple[complex, complex]]]) -> None:
        """set aux operator eigen values"""
        self._aux_operators_evaluated = value

    @property
    def transition_amplitudes(self) -> Optional[List[ListOrDict[Tuple[complex, complex]]]]:
        """Return the transition amplitudes."""
        return self._transition_amplitudes

    @transition_amplitudes.setter
    def transition_amplitudes(self, value: List[ListOrDict[Tuple[complex, complex]]]) -> None:
        """set transition amplitudes"""
        self._transition_amplitudes = value
