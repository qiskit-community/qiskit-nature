# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2020, 2024.
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

from typing import Any, Callable, Sequence
from enum import Enum
import itertools
import logging


import numpy as np
from scipy import linalg

from qiskit_algorithms import EigensolverResult, MinimumEigensolver
from qiskit_algorithms.list_or_dict import ListOrDict as ListOrDictType
from qiskit_algorithms.observables_evaluator import estimate_observables
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import BaseEstimator

from qiskit_nature.second_q.algorithms.ground_state_solvers import GroundStateSolver
from qiskit_nature.second_q.algorithms.excited_states_solvers.excited_states_solver import (
    ExcitedStatesSolver,
)
from qiskit_nature.second_q.mappers import QubitMapper, TaperedQubitMapper
from qiskit_nature.second_q.operators import SparseLabelOp
from qiskit_nature.second_q.problems import (
    BaseProblem,
    ElectronicStructureProblem,
    VibrationalStructureProblem,
    EigenstateResult,
    ElectronicStructureResult,
)
from qiskit_nature.utils import _parallel_map

from .qeom_electronic_ops_builder import build_electronic_ops
from .qeom_vibrational_ops_builder import build_vibrational_ops


logger = logging.getLogger(__name__)


class EvaluationRule(Enum):
    """An enumeration of the available evaluation rules for the excited states solvers.

    This ``Enum`` simply names the available evaluation rules.
    """

    ALL = "all"
    DIAG = "diag"


def _commutator(op_a: SparsePauliOp, op_b: SparsePauliOp) -> SparsePauliOp:
    r"""Compute commutator of `op_a` and `op_b`.

    .. math::

        AB - BA.

    Args:
        op_a: Operator A.
        op_b: Operator B.

    Returns:
        The computed commutator.
    """
    return (op_a @ op_b - op_b @ op_a).simplify(atol=0)


def _double_commutator(
    op_a: SparsePauliOp,
    op_b: SparsePauliOp,
    op_c: SparsePauliOp,
    sign: bool = False,
) -> SparsePauliOp:
    r"""Compute symmetric double commutator of `op_a`, `op_b` and `op_c`.

    See also Equation (13.6.18) in [1].
    If `sign` is `False`, it returns

    .. math::
         [[A, B], C]/2 + [A, [B, C]]/2
         = (2ABC + 2CBA - BAC - CAB - ACB - BCA)/2.

    If `sign` is `True`, it returns

    .. math::
         \lbrace[A, B], C\rbrace/2 + \lbrace A, [B, C]\rbrace/2
         = (2ABC - 2CBA - BAC + CAB - ACB + BCA)/2.

    Args:
        op_a: Operator A.
        op_b: Operator B.
        op_c: Operator C.
        sign: False anti-commutes, True commutes.

    Returns:
        The computed double commutator.

    References:
        [1]: R. McWeeny.
            Methods of Molecular Quantum Mechanics.
            2nd Edition, Academic Press, 1992.
            ISBN 0-12-486552-6.
    """
    sign_num = 1 if sign else -1

    op_ab = op_a @ op_b
    op_ba = op_b @ op_a
    op_ac = op_a @ op_c
    op_ca = op_c @ op_a

    op_abc = op_ab @ op_c
    op_cba = op_c @ op_ba
    op_bac = op_ba @ op_c
    op_cab = op_c @ op_ab
    op_acb = op_ac @ op_b
    op_bca = op_b @ op_ca

    res = (
        op_abc
        - sign_num * op_cba
        + 0.5 * (-op_bac + sign_num * op_cab - op_acb + sign_num * op_bca)
    )

    return res.simplify(atol=0)


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

    For more details, please refer to https://arxiv.org/abs/1910.12890.

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

        self._problem_generated_aux_op_names: set[str] = set()

    @property
    def qubit_mapper(self) -> QubitMapper:
        """Returns the qubit_mapper object defined in the ground state solver."""
        return self._gsc.qubit_mapper

    @property
    def solver(self) -> MinimumEigensolver:
        """Returns the solver object defined in the ground state solver."""
        return self._gsc.solver

    def _map_operators(
        self, operators: SparseLabelOp | ListOrDictType[SparseLabelOp]
    ) -> SparsePauliOp | ListOrDictType[SparsePauliOp]:
        if isinstance(self.qubit_mapper, TaperedQubitMapper):
            mapped_ops = self.qubit_mapper.map_clifford(operators)
        else:
            mapped_ops = self.qubit_mapper.map(operators)
        return mapped_ops

    def _taper_operators(
        self, operators: SparsePauliOp | ListOrDictType[SparsePauliOp]
    ) -> SparsePauliOp | ListOrDictType[SparsePauliOp]:
        if isinstance(self.qubit_mapper, TaperedQubitMapper):
            tapered_ops = self.qubit_mapper.taper_clifford(operators, suppress_none=True)
        else:
            tapered_ops = operators
        return tapered_ops

    def get_qubit_operators(
        self,
        problem: BaseProblem,
        aux_operators: dict[str, SparseLabelOp | SparsePauliOp] | None = None,
    ) -> tuple[SparsePauliOp, dict[str, SparsePauliOp] | None]:
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
        self._problem_generated_aux_op_names = set(aux_second_q_operators.keys())

        # 1. Convert the main operator (hamiltonian) to a Qubit Operator and apply two qubit reduction
        if isinstance(self.qubit_mapper, TaperedQubitMapper):
            main_op = self.qubit_mapper.map_clifford(main_operator)
        else:
            main_op = self.qubit_mapper.map(main_operator)

        # 3. Convert the auxiliary operators.
        # aux_ops set to None if the solver does not support auxiliary operators.
        aux_ops = None

        if self.solver.supports_aux_operators():
            aux_ops = self._map_operators(aux_second_q_operators)

            if aux_operators is not None:
                for name, op in aux_operators.items():
                    if isinstance(op, (SparseLabelOp)):
                        converted_aux_op = self._map_operators(op)
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

        untap_main_op = main_op
        untap_aux_ops = aux_ops

        return untap_main_op, untap_aux_ops

    def solve(
        self,
        problem: BaseProblem,
        aux_operators: dict[str, SparseLabelOp | SparsePauliOp] | None = None,
    ) -> EigenstateResult:
        """Run the excited-states calculation.

        Construct and solve the EOM pseudo-eigenvalue problem to obtain the excitation energies
        and the excitation operators expansion coefficients.

        Args:
            problem: A class encoding a problem to be solved.
            aux_operators: Additional auxiliary operators to evaluate.

        Returns:
            An interpreted :class:`~.EigenstateResult`. For more information see also
            :meth:`~.BaseProblem.interpret`. The :class:`~.EigenstateResult` is constructed
            from a :class:`~qiskit_nature.second_q.algorithms.excited_states_solvers.qeom.QEOMResult`
            instance which holds additional information specific to the qEOM problem.
        """

        # 1. Prepare all operators and set the particle number in the qubit mapper
        (
            untap_main_op,  # Hamiltonian
            untap_aux_ops,  # Auxiliary observables
        ) = self.get_qubit_operators(problem, aux_operators)

        # before we taper our operators we filter the ones which come from the problem internally as
        # to not trigger a bunch of warnings being raised about overwritten auxiliary operators
        filtered_aux_ops = {
            k: v
            for k, v in untap_aux_ops.items()
            if k not in self._problem_generated_aux_op_names and k not in aux_operators.keys()
        }

        # 2. Run ground state calculation with fully tapered custom auxiliary operators
        # Note that the solve() method includes the `second_q' auxiliary operators
        tap_aux_operators = self._taper_operators(filtered_aux_ops)

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
        dict[str, SparsePauliOp],
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
                self.qubit_mapper,
            )
        elif isinstance(problem, VibrationalStructureProblem):
            return build_vibrational_ops(
                problem.num_modals,
                self.excitations,
                self.qubit_mapper,
            )
        else:
            raise NotImplementedError(
                "The building of qEOM hopping operators is not yet implemented for a problem of "
                f"type {type(problem)}"
            )

    def _build_all_eom_operators(
        self,
        untap_operator: SparsePauliOp,
        expansion_basis_data: tuple[dict[str, SparsePauliOp], dict[str, list[bool]], int],
    ) -> dict[str, SparsePauliOp]:
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

        mus, nus = np.triu_indices(size)

        def _build_one_sector(available_hopping_ops):
            for idx, m_u in enumerate(mus):
                n_u = nus[idx]
                left_op_1 = available_hopping_ops.get(f"E_{m_u}")
                right_op_1 = available_hopping_ops.get(f"E_{n_u}")
                right_op_2 = available_hopping_ops.get(f"Edag_{n_u}")
                to_be_computed_list.append((m_u, n_u, left_op_1, right_op_1, right_op_2))

        if (
            isinstance(self.qubit_mapper, TaperedQubitMapper)
            and not self.qubit_mapper.z2symmetries.is_empty()
        ):

            combinations = itertools.product(
                [1, -1], repeat=len(self.qubit_mapper.z2symmetries.symmetries)
            )
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

        results = _parallel_map(
            self._build_commutator_routine,
            to_be_computed_list,
            task_args=(untap_operator,),
        )
        all_matrix_operators = {}
        for result in results:
            m_u, n_u, eom_operators = result

            for index_op, op in eom_operators.items():
                if op is not None:
                    all_matrix_operators[f"{index_op}_{m_u}_{n_u}"] = op

        return all_matrix_operators

    @staticmethod
    def _build_commutator_routine(
        params: list, operator: SparsePauliOp
    ) -> tuple[int, int, dict[str, SparsePauliOp]]:
        """Numerically computes the commutator / double commutator between operators.

        Args:
            params: list containing the indices of matrix element and the corresponding
                excitation operators.
            operator: The hamiltonian.

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
                q_mat_op = -_double_commutator(left_op_1, operator, right_op_1, sign=False)
                # In the case of the single commutator, we are always interested in the energy
                # difference of two states. Thus, regardless of the problem's nature, we will
                # always use the commutator.
                w_mat_op = -_commutator(left_op_1, right_op_1)
                q_mat_op = None if len(q_mat_op) == 0 else q_mat_op
                w_mat_op = None if len(w_mat_op) == 0 else w_mat_op
            else:
                q_mat_op = None
                w_mat_op = None

            if right_op_2 is not None:
                # For explanations on the choice of commutation relation, please refer to the
                # comments above.
                m_mat_op = _double_commutator(left_op_1, operator, right_op_2, sign=False)
                v_mat_op = _commutator(left_op_1, right_op_2)
                m_mat_op = None if len(m_mat_op) == 0 else m_mat_op
                v_mat_op = None if len(v_mat_op) == 0 else v_mat_op
            else:
                m_mat_op = None
                v_mat_op = None

        eom_operators = {"q": q_mat_op, "w": w_mat_op, "m": m_mat_op, "v": v_mat_op}

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
    ) -> tuple[dict[str, SparsePauliOp], dict[str, list[bool]], int]:
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

        return hopping_operators, type_of_commutativities, size

    def _build_qeom_pseudoeigenvalue_problem(
        self,
        untap_operator: SparsePauliOp,
        expansion_basis_data: tuple[dict[str, SparsePauliOp], dict[str, list[bool]], int],
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

        logger.debug("Build qEOM pseudoeigenvalue problem...")

        # 1. Build all EOM operators to evaluate on the ground state
        untap_eom_matrix_ops = self._build_all_eom_operators(
            untap_operator,
            expansion_basis_data,
        )

        tap_eom_matrix_ops = self._taper_operators(untap_eom_matrix_ops)

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
            dict[str, SparsePauliOp],
            dict[str, list[bool]],
            int,
        ],
        reference_state: tuple[QuantumCircuit, Sequence[float]],
        expansion_coefs_rescaled: np.ndarray,
    ) -> list[SparsePauliOp]:
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
        tap_hopping_ops = self._taper_operators(untap_hopping_ops)

        additionnal_measurements = estimate_observables(
            self._estimator, reference_state[0], tap_hopping_ops, reference_state[1]
        )

        num_qubits = list(untap_hopping_ops.values())[0].num_qubits
        identity_op = SparsePauliOp(["I" * num_qubits], [1.0])

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
        # excitation operators.

        hopping_ops_vector = list(translated_hopping_ops.values())
        excitations_ops = [
            SparsePauliOp.sum(
                [
                    expansion_coefs_rescaled[k, i] * hopping_ops_vector[k]
                    for k in range(expansion_coefs_rescaled.shape[0])
                ]
            ).simplify()
            for i in range(expansion_coefs_rescaled.shape[1])
        ]

        excitations_ops_reduced = [identity_op] + excitations_ops

        return excitations_ops_reduced

    def _prepare_excited_states_observables(
        self,
        untap_aux_ops: dict[str, SparsePauliOp],
        operators_reduced: list[SparsePauliOp],
        size: int,
    ) -> dict[tuple[str, int, int], SparsePauliOp]:
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

        op_aux_op_dict: dict[tuple[str, int, int], SparsePauliOp] = {}

        for op_name, indices_constraint in eval_rules.items():
            if op_name not in untap_aux_ops.keys():
                raise ValueError("Evaluation constrains cannot be satisfied")
            aux_op = untap_aux_ops[op_name]

            for i, j in indices_constraint:
                if i >= len(operators_reduced) or j >= len(operators_reduced):
                    raise ValueError("Evaluation constrains cannot be satisfied")

                opi, opj = operators_reduced[i], operators_reduced[j]
                op_aux_op_dict[(op_name, i, j)] = (opi.adjoint() @ aux_op @ opj).simplify()

        return op_aux_op_dict

    def _evaluate_observables_excited_states(
        self,
        untap_aux_ops: dict[str, SparsePauliOp],
        expansion_basis_data: tuple[dict[str, SparsePauliOp], dict[str, list[bool]], int],
        reference_state: tuple[QuantumCircuit, Sequence[float]],
        expansion_coefs_rescaled: np.ndarray,
    ) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[tuple[int, int], dict[str, Any]]]:
        """Evaluate the expectation values and transition amplitudes of the auxiliary operators on the
        excited states. Custom rules can be used to define which expectation values and transition
        amplitudes to compute. A typical rule is specified in the form of a dictionary
        {'hamiltonian':[(1,1)]}

        Args:
            untap_aux_ops: Dict of auxiliary operators for which properties will be computed.
            expansion_basis_data: Dict of transformed hopping operators, dict of commutativity types,
            size of the qEOM problem.
            reference_state: Reference ground state.
            expansion_coefs_rescaled: Expansion coefficient matrix X such that H @ X = S @ X @ E and
            X^dag @ S @ X is the identity.

        Returns:
            Auxiliary operators eigenvalues and transition amplitudes, following the evaluation rules
            defined as attributes of the qEOM class.
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
            tap_op_aux_op_dict = self._taper_operators(op_aux_op_dict)

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
        eigenstate_result = EigenstateResult.from_result(qeom_result)
        result = problem.interpret(eigenstate_result)

        return result


class QEOMResult(EigensolverResult):
    """The results class for the qEOM algorithm.

    For more details about the definitions, please refer to https://arxiv.org/abs/1910.12890.

    Attributes:
        ground_state_raw_result (EigenstateResult): The raw results of the ground state eigensolver.
        excitation_energies (np.ndarray): The excitation energies approximated by the qEOM algorithm.
        expansion_coefficients (np.ndarray): The expansion coefficients matrix of the excitation
            operators onto the set of basis operators spanning the linear qEOM subspace.
        h_matrix (np.ndarray): Matrix representing the Hamiltonian in the qEOM subspace. Because of our
            choice for the expansion basis, the two square sub-matrices on the diagonal are related by
            a transposition and the two submatrices on the anti diagonal are hermitian conjugates.
        s_matrix (np.ndarray): Matrix representing the geometry of the qEOM subspace. Because of our
            choice for the expansion basis, the two square submatrices on the diagonal are related by
            a transposition (with a sign) and the two submatrices on the anti diagonal are hermitian
            conjugates.
        h_matrix_std (np.ndarray): 2 by 2 matrix representing the sums of standard deviations in the four
            square submatrices of H.
        s_matrix_std (np.ndarray): 2 by 2 matrix representing the sums of standard deviations in the four
            square submatrices of S.
        transition_amplitudes (list[ListOrDictType[tuple[complex, dict[str, Any]]]): Transition
            amplitudes of the auxiliary operators computed following the evaluation rules specified when
            the qEOM class was created.
    """

    def __init__(self) -> None:
        super().__init__()
        self.ground_state_raw_result = None
        self.excitation_energies: np.ndarray | None = None
        self.expansion_coefficients: np.ndarray | None = None
        self.eigenvalues: np.ndarray | None = None
        self.h_matrix: np.ndarray | None = None
        self.s_matrix: np.ndarray | None = None
        self.h_matrix_std: np.ndarray = np.zeros((2, 2))
        self.s_matrix_std: np.ndarray = np.zeros((2, 2))

        self.transition_amplitudes: list[
            ListOrDictType[tuple[complex, dict[str, Any]]]
        ] | None = None
        self.gamma_square: np.ndarray = None

    @property
    def m_matrix(self) -> np.ndarray | None:
        """returns the M matrix"""
        if self.h_matrix is None:
            return None
        return self.h_matrix[: len(self.h_matrix) // 2, : len(self.h_matrix) // 2]

    @property
    def v_matrix(self) -> np.ndarray | None:
        """returns the V matrix"""
        if self.s_matrix is None:
            return None
        return self.s_matrix[: len(self.s_matrix) // 2, : len(self.s_matrix) // 2]

    @property
    def q_matrix(self) -> np.ndarray | None:
        """returns the Q matrix"""
        if self.h_matrix is None:
            return None
        return self.h_matrix[len(self.h_matrix) // 2 :, : len(self.h_matrix) // 2]

    @property
    def w_matrix(self) -> np.ndarray | None:
        """returns the W matrix"""
        if self.s_matrix is None:
            return None
        return self.s_matrix[len(self.s_matrix) // 2 :, : len(self.s_matrix) // 2]

    @property
    def m_matrix_std(self) -> float:
        """returns the M matrix standard deviation"""
        if np.isclose(self.h_matrix_std[0, 0], 0.0):
            return 0.0
        return self.h_matrix_std[0, 0]

    @property
    def v_matrix_std(self) -> float:
        """returns the V matrix standard deviation"""
        if np.isclose(self.s_matrix_std[0, 0], 0.0):
            return 0.0
        return self.s_matrix_std[0, 0]

    @property
    def q_matrix_std(self) -> float:
        """returns the Q matrix standard deviation"""
        if np.isclose(self.h_matrix_std[0, 1], 0.0):
            return 0.0
        return self.h_matrix_std[0, 0]

    @property
    def w_matrix_std(self) -> float:
        """returns the W matrix standard deviation"""
        if np.isclose(self.s_matrix_std[0, 1], 0.0):
            return 0.0
        return self.s_matrix_std[0, 0]
