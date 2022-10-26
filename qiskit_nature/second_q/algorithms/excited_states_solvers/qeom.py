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

"""The calculation of excited states via the qEOM algorithm"""

from __future__ import annotations
from operator import index
from tkinter.messagebox import NO
from turtle import right

from typing import Any, Callable, List, Union, Optional, Tuple, Dict
import itertools
import logging
import sys

import numpy as np
from scipy import linalg
from qiskit.algorithms.observables_evaluator import estimate_observables
from qiskit.tools import parallel_map
from qiskit.tools.events import TextProgressBar
from qiskit.utils import algorithm_globals
from qiskit.algorithms import AlgorithmResult, VariationalAlgorithm
from qiskit.algorithms.eigensolvers import EigensolverResult

from qiskit.opflow import (
    Z2Symmetries,
    commutator,
    double_commutator,
    PauliSumOp,
)
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import BaseEstimator

from qiskit_nature.converters.second_quantization.utils import ListOrDict
from qiskit_nature.second_q.operators import SecondQuantizedOp
from qiskit_nature.second_q.operators.sparse_label_op import SparseLabelOp
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

logger = logging.getLogger(__name__)


class QEOM(ExcitedStatesSolver):
    """The calculation of excited states via the qEOM algorithm"""

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
    ) -> None:
        """
        Args:
            ground_state_solver: a GroundStateSolver object. The qEOM algorithm
                will use this ground state to compute the EOM matrix elements
            estimator: the :class:`~qiskit.primitives.BaseEstimator` to use for the evaluation of
                the qubit operators at the ground state ansatz. If the internal solver provided to
                the `GroundStateSolver` also uses a `BaseEstimator` primitive, you can provide the
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
                    :meth:`generate_vibrational_excitations`, when solving a
                    :class:`.ElectronicStructureProblem` or :class:`.VibrationalStructureProblem`,
                    respectively.
        """
        self._gsc = ground_state_solver
        self._estimator = estimator
        self.excitations = excitations

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
        all excitations of given type will be used. Otherwise a list of custom excitations can
        directly be provided."""
        if isinstance(excitations, str) and excitations not in ["s", "d", "sd"]:
            raise ValueError(
                "Excitation type must be s (singles), d (doubles) or sd (singles and doubles)"
            )
        self._excitations = excitations

    @property
    def qubit_converter(self):
        return self._gsc.qubit_converter

    @property
    def solver(self):
        return self._gsc.solver

    def get_qubit_operators(
        self,
        problem: BaseProblem,
        aux_operators: Optional[dict[str, Union[SecondQuantizedOp, PauliSumOp]]] = None,
    ) -> Tuple[PauliSumOp, Optional[dict[str, PauliSumOp]]]:
        return self._gsc.get_qubit_operators(problem, aux_operators)

    def solve(
        self,
        problem: BaseProblem,
        aux_operators: Optional[dict[str, SecondQuantizedOp]] = None,
        eval_second_q_eigenvalues: Optional[list(str)] = True,
    ) -> EigenstateResult:
        """Run the excited-states calculation.

        Construct and solves the EOM pseudo-eigenvalue problem to obtain the excitation energies
        and the excitation operators expansion coefficients.

        Args:
            problem: a class encoding a problem to be solved.
            aux_operators: Additional auxiliary operators to evaluate.

        Returns:
            An interpreted :class:`~.EigenstateResult`. For more information see also
            :meth:`~.BaseProblem.interpret`.
        """

        # 1. Prepare all operators
        (
            pre_tap_qubit_op_main,  # Hamiltonian
            pre_tap_second_q_aux_ops,  # Default auxiliary observables
            pre_tap_custom_aux_ops,  # User-defined auxiliary observables
        ) = self._prepare_all_operators(problem, aux_operators)

        # 2. Run ground state calculation with fully tapered custom auxiliary operators
        # Note that the solve() method natively includes the `second_q' auxiliary operators
        tap_custom_aux_ops = self.qubit_converter._symmetry_reduce_clifford(
            pre_tap_custom_aux_ops, True
        )
        groundstate_result = self._gsc.solve(problem, tap_custom_aux_ops)
        ground_state = groundstate_result.eigenstates[0]

        # 3. Prepare the expansion operators for the excited state calculation
        expansion_basis_data = self._build_expansion_basis(problem)

        # 4. Obtain the representation of the Hamiltonian in the linear subspace
        h_mat, s_mat, h_mat_std, s_mat_std = self._build_qeom_pseudoeigenvalue_problem(
            pre_tap_qubit_op_main, expansion_basis_data, ground_state
        )

        # 5. Solve pseudo-eigenvalue problem
        energy_gaps, expansion_coefs = self._compute_excitation_energies(h_mat, s_mat)

        # print("\n QEOM Hamiltonian")
        # aux_op_h_vals = expansion_coefs.T.conj() @ h_mat @ expansion_coefs
        # aux_op_s_vals = expansion_coefs.T.conj() @ s_mat @ expansion_coefs
        # print(np.real(aux_op_h_vals))
        # print(np.real(aux_op_s_vals))

        # 6. Evaluate auxiliary operators on the excited states

        if eval_second_q_eigenvalues and pre_tap_custom_aux_ops is not None:
            pre_tap_aux_observables = {**pre_tap_second_q_aux_ops, **pre_tap_custom_aux_ops}
        elif eval_second_q_eigenvalues and pre_tap_custom_aux_ops is None:
            pre_tap_aux_observables = pre_tap_second_q_aux_ops
        else:
            pre_tap_aux_observables = pre_tap_custom_aux_ops

        aux_observables_eigenvalues_raw = self._evaluate_observables_excited_states(
            pre_tap_aux_observables,
            expansion_basis_data,
            ground_state,
            expansion_coefs,
            groundstate_result.aux_operators_evaluated,
        )

        aux_observables_eigenvalues = [
            dict(zip(aux_observables_eigenvalues_raw, col))
            for col in zip(*aux_observables_eigenvalues_raw.values())
        ]

        # 7. Evaluate custom auxiliary operators on the excited states

        result = self._build_qeom_result(
            problem,
            groundstate_result,
            expansion_coefs,
            energy_gaps,
            h_mat,
            s_mat,
            h_mat_std,
            s_mat_std,
            aux_observables_eigenvalues,
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
            problem: the problem for which to build out the operators.

        Raises:
            NotImplementedError: for an unsupported problem type.

        Returns:
            Dict of hopping operators, dict of commutativity types and dict of excitation indices
        """
        if isinstance(problem, ElectronicStructureProblem):
            return build_electronic_ops(
                problem.properties.particle_number,
                self._gsc.qubit_converter,
                self.excitations,
            )
        elif isinstance(problem, VibrationalStructureProblem):
            return build_vibrational_ops(
                problem.num_modals,
                self._gsc.qubit_converter,
                self.excitations,
            )
        else:
            raise NotImplementedError(
                "The building of QEOM hopping operators is not yet implemented for a problem of "
                f"type {type(problem)}"
            )

    def _build_all_eom_operators(
        self, pre_tap_operator, expansion_basis_data: tuple(dict, dict, int), commutator: bool
    ) -> dict:
        """Building all commutators for Q, W, M, V matrices.

        Args:
            hopping_operators: all hopping operators based on excitations_list,
                key is the string of single/double excitation;
                value is corresponding operator.
            type_of_commutativities: if tapering is used, it records the commutativities of
                hopping operators with the
                Z2 symmetries found in the original operator.
            size: the number of excitations (size of the qEOM pseudo-eigenvalue problem)

        Returns:
            a dictionary that contains the operators for each matrix element
        """

        build_routine = (
            self._build_commutator_routine if commutator else self._build_product_routine
        )
        pre_tap_hopping_ops, type_of_commutativities, size = expansion_basis_data
        all_matrix_operators = {}

        mus, nus = np.triu_indices(size)

        def _build_one_sector(available_hopping_ops, untapered_op, z2_symmetries):

            to_be_computed_list = []
            for idx, m_u in enumerate(mus):
                n_u = nus[idx]
                left_op_1 = available_hopping_ops.get(f"E_{m_u}")
                left_op_2 = available_hopping_ops.get(f"Edag_{m_u}")
                # left_op2 is only needed for the product metric because it does not have the symmetries
                # of the double commutator.
                right_op_1 = available_hopping_ops.get(f"E_{n_u}")
                right_op_2 = available_hopping_ops.get(f"Edag_{n_u}")
                to_be_computed_list.append((m_u, n_u, left_op_1, left_op_2, right_op_1, right_op_2))

            if logger.isEnabledFor(logging.INFO):
                logger.info("Building all commutators:")
                TextProgressBar(sys.stderr)
            results = parallel_map(
                build_routine,
                to_be_computed_list,
                task_args=(untapered_op, z2_symmetries),
                num_processes=algorithm_globals.num_processes,
            )
            for result in results:
                m_u, n_u, eom_operators = result

                for index_op, op in eom_operators.items():
                    if op is not None:
                        all_matrix_operators[f"{index_op}_{m_u}_{n_u}"] = op

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
                    value = np.asarray(value)
                    if np.all(value == targeted_sector):
                        available_hopping_ops[key] = pre_tap_hopping_ops[key]
                # untapered_qubit_op is a PauliSumOp and should not be exposed.
                _build_one_sector(available_hopping_ops, pre_tap_operator, z2_symmetries)

        else:
            # untapered_qubit_op is a PauliSumOp and should not be exposed.
            _build_one_sector(pre_tap_hopping_ops, pre_tap_operator, z2_symmetries)

        return all_matrix_operators

    @staticmethod
    def _build_commutator_routine(
        params: List, operator: PauliSumOp, z2_symmetries: Z2Symmetries
    ) -> Tuple[int, int, list(PauliSumOp)]:
        """Numerically computes the commutator / double commutator between operators.

        Args:
            params: list containing the indices of matrix element and the corresponding
                excitation operators
            operator: the hamiltonian
            z2_symmetries: z2_symmetries in case of tapering

        Returns:
            The indices of the matrix element and the corresponding qubit
            operator for each of the EOM matrices
        """
        m_u, n_u, left_op1, left_op2, right_op_1, right_op_2 = params
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

    @staticmethod
    def _build_product_routine(
        params: List, operator: PauliSumOp, z2_symmetries: Z2Symmetries
    ) -> Tuple[int, int, PauliSumOp, PauliSumOp, PauliSumOp, PauliSumOp]:
        """Numerically computes the product / double product between operators.

        Args:
            params: list containing the indices of matrix element and the corresponding
                excitation operators
            operator: the hamiltonian
            z2_symmetries: z2_symmetries in case of tapering

        Returns:
            The indices of the matrix element and the corresponding qubit
            operator for each of the EOM matrices
        """
        m_u, n_u, left_op_1, left_op_2, right_op_1, right_op_2 = params
        q_mat_op = None
        # w_mat_op = None
        m_mat_op = None
        # v_mat_op = None
        n_mat_op = None
        # u_mat_op = None
        p_mat_op = None
        # t_mat_op = None

        if left_op_1 is not None and right_op_1 is not None:
            p_mat_op = left_op_2 @ operator @ right_op_2
            # t_mat_op = left_op_2 @ right_op_2
            p_mat_op = None if len(p_mat_op) == 0 else p_mat_op
            # t_mat_op = None if len(t_mat_op) == 0 else t_mat_op

        if left_op_1 is not None and right_op_2 is not None:
            n_mat_op = left_op_2 @ operator @ right_op_1
            # u_mat_op = left_op_2 @ right_op_1
            n_mat_op = None if len(n_mat_op) == 0 else n_mat_op
            # u_mat_op = None if len(u_mat_op) == 0 else u_mat_op

        if left_op_2 is not None and right_op_1 is not None:
            m_mat_op = left_op_1 @ operator @ right_op_2
            # v_mat_op = left_op_1 @ right_op_2
            m_mat_op = None if len(m_mat_op) == 0 else m_mat_op
            # v_mat_op = None if len(v_mat_op) == 0 else v_mat_op

        if left_op_2 is not None and right_op_2 is not None:
            q_mat_op = -left_op_1 @ operator @ right_op_1
            # w_mat_op = - left_op_1 @ right_op_1
            q_mat_op = None if len(q_mat_op) == 0 else q_mat_op
            # w_mat_op = None if len(w_mat_op) == 0 else w_mat_op

        eom_operators = {
            "q": q_mat_op,
            # "w": w_mat_op,
            "m": m_mat_op,
            # "v": v_mat_op,
            "n": n_mat_op,
            # "u": u_mat_op,
            "p": p_mat_op,
            # "t": t_mat_op
        }

        if not (
            (left_op_1 is None and left_op_2 is None)
            or (right_op_1 is None and right_op_2 is None)
            or z2_symmetries.is_empty()
        ):
            for index_op, op in eom_operators.items():
                if op is not None and len(op) > 0:
                    eom_operators[index_op] = z2_symmetries.taper_clifford(op)

        return m_u, n_u, eom_operators

    def _build_eom_matrices(
        self, gs_results: dict[str, tuple[complex, dict[str, Any]]], size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
        """Constructs the M, V, Q and W matrices from the results on the ground state

        Args:
            gs_results: a ground state result object
            size: size of eigenvalue problem

        Returns:
            the matrices and their standard deviation
        """

        mus, nus = np.triu_indices(size)

        m_mat = np.zeros((size, size), dtype=complex)
        v_mat = np.zeros((size, size), dtype=complex)
        q_mat = np.zeros((size, size), dtype=complex)
        w_mat = np.zeros((size, size), dtype=complex)
        m_mat_std, v_mat_std, q_mat_std, w_mat_std = 0.0, 0.0, 0.0, 0.0

        # evaluate results
        for idx, m_u in enumerate(mus):
            n_u = nus[idx]

            q_mat[m_u][n_u] = (
                gs_results[f"q_{m_u}_{n_u}"][0]
                if gs_results.get(f"q_{m_u}_{n_u}") is not None
                else q_mat[m_u][n_u]
            )
            w_mat[m_u][n_u] = (
                gs_results[f"w_{m_u}_{n_u}"][0]
                if gs_results.get(f"w_{m_u}_{n_u}") is not None
                else w_mat[m_u][n_u]
            )
            m_mat[m_u][n_u] = (
                gs_results[f"m_{m_u}_{n_u}"][0]
                if gs_results.get(f"m_{m_u}_{n_u}") is not None
                else m_mat[m_u][n_u]
            )
            v_mat[m_u][n_u] = (
                gs_results[f"v_{m_u}_{n_u}"][0]
                if gs_results.get(f"v_{m_u}_{n_u}") is not None
                else v_mat[m_u][n_u]
            )

            q_mat_std += (
                gs_results[f"q_{m_u}_{n_u}_std"][0].real
                if gs_results.get(f"q_{m_u}_{n_u}_std") is not None
                else 0
            )
            w_mat_std += (
                gs_results[f"w_{m_u}_{n_u}_std"][0].real
                if gs_results.get(f"w_{m_u}_{n_u}_std") is not None
                else 0
            )
            m_mat_std += (
                gs_results[f"m_{m_u}_{n_u}_std"][0].real
                if gs_results.get(f"m_{m_u}_{n_u}_std") is not None
                else 0
            )
            v_mat_std += (
                gs_results[f"v_{m_u}_{n_u}_std"][0].real
                if gs_results.get(f"v_{m_u}_{n_u}_std") is not None
                else 0
            )

        # these matrices are numpy arrays and therefore have the ``shape`` attribute
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

        # N = M.conj() = M.T
        # P = Q.adjoint() = Q.conj()
        # U = -V.conj() = -V.T
        # T = W.adjoint()
        h_mat = np.matrixlib.bmat([[m_mat, q_mat], [q_mat.T.conj(), m_mat.T]])
        s_mat = np.matrixlib.bmat([[v_mat, w_mat], [w_mat.T.conj(), -v_mat.T]])

        h_mat_std = np.array([[m_mat_std, q_mat_std], [q_mat_std, m_mat_std]])
        s_mat_std = np.array([[v_mat_std, w_mat_std], [w_mat_std, v_mat_std]])

        return h_mat, s_mat, h_mat_std, s_mat_std

    def _build_qse_matrices(
        self,
        gs_results: dict[str, tuple[complex, dict[str, Any]]],
        size: int,
        tap_operator_evaluated,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
        """Constructs the M, V, Q and W matrices from the results on the ground state

        Args:
            gs_results: a ground state result object
            size: size of eigenvalue problem

        Returns:
            the matrices and their standard deviation
        """

        H_01 = np.zeros((1, 2 * size), dtype=complex)
        S_01 = np.zeros((1, 2 * size), dtype=complex)

        for m_u in range(2 * size):
            H_01[0, m_u] = (
                gs_results[f"H_0_{m_u}"][0]
                if gs_results.get(f"H_0_{m_u}") is not None
                else H_01[0, m_u]
            )

            S_01[0, m_u] = (
                gs_results[f"S_0_{m_u}"][0]
                if gs_results.get(f"S_0_{m_u}") is not None
                else S_01[0, m_u]
            )

        # print(np.real(H_01))
        # print(np.real(S_01))

        mus, nus = np.triu_indices(size)

        m_mat = np.zeros((size, size), dtype=complex)
        # v_mat = np.zeros((size, size), dtype=complex)
        q_mat = np.zeros((size, size), dtype=complex)
        # w_mat = np.zeros((size, size), dtype=complex)
        m_mat_std, v_mat_std, q_mat_std, w_mat_std = 0.0, 0.0, 0.0, 0.0

        n_mat = np.zeros((size, size), dtype=complex)
        # u_mat = np.zeros((size, size), dtype=complex)
        p_mat = np.zeros((size, size), dtype=complex)
        # t_mat = np.zeros((size, size), dtype=complex)
        n_mat_std, u_mat_std, p_mat_std, t_mat_std = 0.0, 0.0, 0.0, 0.0

        # evaluate results
        for idx, m_u in enumerate(mus):
            n_u = nus[idx]

            q_mat[m_u][n_u] = (
                gs_results[f"q_{m_u}_{n_u}"][0]
                if gs_results.get(f"q_{m_u}_{n_u}") is not None
                else q_mat[m_u][n_u]
            )
            # w_mat[m_u][n_u] = (
            #     gs_results[f"w_{m_u}_{n_u}"][0]
            #     if gs_results.get(f"w_{m_u}_{n_u}") is not None
            #     else w_mat[m_u][n_u]
            # )
            m_mat[m_u][n_u] = (
                gs_results[f"m_{m_u}_{n_u}"][0]
                if gs_results.get(f"m_{m_u}_{n_u}") is not None
                else m_mat[m_u][n_u]
            )
            # v_mat[m_u][n_u] = (
            #     gs_results[f"v_{m_u}_{n_u}"][0]
            #     if gs_results.get(f"v_{m_u}_{n_u}") is not None
            #     else v_mat[m_u][n_u]
            # )
            n_mat[m_u][n_u] = (
                gs_results[f"n_{m_u}_{n_u}"][0]
                if gs_results.get(f"n_{m_u}_{n_u}") is not None
                else n_mat[m_u][n_u]
            )
            # u_mat[m_u][n_u] = (
            #     gs_results[f"u_{m_u}_{n_u}"][0]
            #     if gs_results.get(f"u_{m_u}_{n_u}") is not None
            #     else u_mat[m_u][n_u]
            # )
            p_mat[m_u][n_u] = (
                gs_results[f"p_{m_u}_{n_u}"][0]
                if gs_results.get(f"p_{m_u}_{n_u}") is not None
                else p_mat[m_u][n_u]
            )
            # t_mat[m_u][n_u] = (
            #     gs_results[f"t_{m_u}_{n_u}"][0]
            #     if gs_results.get(f"t_{m_u}_{n_u}") is not None
            #     else t_mat[m_u][n_u]
            # )

            q_mat_std += (
                gs_results[f"q_{m_u}_{n_u}_std"][0].real
                if gs_results.get(f"q_{m_u}_{n_u}_std") is not None
                else 0
            )
            w_mat_std += (
                gs_results[f"w_{m_u}_{n_u}_std"][0].real
                if gs_results.get(f"w_{m_u}_{n_u}_std") is not None
                else 0
            )
            m_mat_std += (
                gs_results[f"m_{m_u}_{n_u}_std"][0].real
                if gs_results.get(f"m_{m_u}_{n_u}_std") is not None
                else 0
            )
            v_mat_std += (
                gs_results[f"v_{m_u}_{n_u}_std"][0].real
                if gs_results.get(f"v_{m_u}_{n_u}_std") is not None
                else 0
            )
            n_mat_std += (
                gs_results[f"n_{m_u}_{n_u}_std"][0].real
                if gs_results.get(f"n_{m_u}_{n_u}_std") is not None
                else 0
            )
            u_mat_std += (
                gs_results[f"u_{m_u}_{n_u}_std"][0].real
                if gs_results.get(f"u_{m_u}_{n_u}_std") is not None
                else 0
            )
            p_mat_std += (
                gs_results[f"p_{m_u}_{n_u}_std"][0].real
                if gs_results.get(f"p_{m_u}_{n_u}_std") is not None
                else 0
            )
            t_mat_std += (
                gs_results[f"t_{m_u}_{n_u}_std"][0].real
                if gs_results.get(f"t_{m_u}_{n_u}_std") is not None
                else 0
            )

        # these matrices are numpy arrays and therefore have the ``shape`` attribute
        m_mat_tot = m_mat + m_mat.T.conj() - np.identity(m_mat.shape[0]) * m_mat
        n_mat_tot = n_mat + n_mat.T.conj() - np.identity(n_mat.shape[0]) * n_mat

        # Q.adjoint() = P
        q_mat_tot = q_mat + p_mat.T.conj() - np.identity(q_mat.shape[0]) * q_mat
        p_mat_tot = p_mat + q_mat.T.conj() - np.identity(p_mat.shape[0]) * p_mat

        q_mat_std = q_mat_std / float(size**2)
        w_mat_std = w_mat_std / float(size**2)
        m_mat_std = m_mat_std / float(size**2)
        v_mat_std = v_mat_std / float(size**2)
        p_mat_std = q_mat_std / float(size**2)
        t_mat_std = w_mat_std / float(size**2)
        n_mat_std = m_mat_std / float(size**2)
        u_mat_std = v_mat_std / float(size**2)

        logger.debug("\nQ:=========================\n%s", q_mat)
        logger.debug("\nM:=========================\n%s", m_mat)

        h_mat = np.matrixlib.bmat([[m_mat_tot, q_mat_tot], [p_mat_tot, n_mat_tot]])
        h_mat_extended = np.block(
            [[tap_operator_evaluated * np.ones((1, 1)), H_01], [H_01.T.conj(), h_mat]]
        )

        h_mat_std = np.array([[m_mat_std, q_mat_std], [p_mat_std, n_mat_std]])

        return h_mat_extended, h_mat_std

    def _build_expansion_basis(self, problem):
        logger.debug("Building expansion basis data...")

        data = self._build_hopping_ops(problem)
        hopping_operators, type_of_commutativities, excitation_indices = data
        size = int(len(list(excitation_indices.keys())) // 2)

        reduced_hopping_ops = self.qubit_converter.two_qubit_reduce(hopping_operators)
        pre_tap_hopping_ops = self.qubit_converter._convert_clifford(reduced_hopping_ops)

        return pre_tap_hopping_ops, type_of_commutativities, size

    def _build_qeom_pseudoeigenvalue_problem(
        self, pre_tap_operator, expansion_basis_data, reference_state
    ):
        logger.debug("Build QEOM pseudoeigenvalue problem...")

        # 1. Build all operators to evaluate on the groundstate to construct the EOM matrices
        tap_eom_matrix_ops = self._build_all_eom_operators(
            pre_tap_operator,
            expansion_basis_data,
            commutator=True,
        )

        # 2. Evaluate all eom operators on the groundstate
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

    def _extract_aux_eigenvalues(self, op_h_mat, op_s_mat, expansion_coefs):
        alphas = op_s_mat[0, 1 + expansion_coefs.shape[1] :]
        # print("alphas", alphas)
        extended_expansion_coefs = np.block([[-alphas], [expansion_coefs]])

        temp_s_vals = extended_expansion_coefs.T.conj() @ op_s_mat @ extended_expansion_coefs
        temp_h_vals = extended_expansion_coefs.T.conj() @ op_h_mat @ extended_expansion_coefs

        ratio = np.divide(np.real(temp_h_vals), np.real(temp_s_vals))
        ratio_formatted = [(ratio_elem, {}) for ratio_elem in np.diag(ratio)]

        return ratio_formatted

    @staticmethod
    def _compute_excitation_energies(
        h_mat: np.ndarray, s_mat: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Diagonalizing H, S matrices for excitation energies.

        Args:
            h_mat : H matrices
            s_mat : S matrices

        Returns:
            1-D vector stores all energy gap to reference state
            2-D array storing the X and Y expansion coefficients
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
        order = np.argsort(np.real(res[0]))[len(res[0]) // 2 : :]  # [::-1]
        w = np.real(res[0])[order]
        logger.debug("Sorted real parts %s", w)
        w = np.abs(w)
        w[w < 1e-06] = 0
        excitation_energies_gap = w
        expansion_coefs = res[1][:, order]

        return excitation_energies_gap, expansion_coefs

    def _prepare_all_operators(self, problem, aux_operators=None):

        logger.debug("Preparing QEOM operators...")

        # 1.1 Setup self.solver if it is a Factory object. `get_qubit_operators' must be called before
        # checking if the solver is variational or not.
        self.get_qubit_operators(problem=problem, aux_operators=None)

        # 1.2 Prepare the Hamiltonian and the auxiliary operators
        main_second_q_op, aux_second_q_ops = problem.second_q_ops()

        # Applies z2symmetries to the Hamiltonian
        # Sets _num_particle and _z2symmetries for the qubit_converter

        untapered_qubit_op_main = self.qubit_converter.convert_only(
            main_second_q_op, num_particles=problem.num_particles
        )

        tapered_qubit_op_main, z2symmetries = self.qubit_converter._find_taper_op(
            untapered_qubit_op_main, problem.symmetry_sector_locator
        )

        # Update the num_particles of the qubit converter to prepare the call of convert_match
        # on the auxiliaries. The z2symmetries are deliberately not set at this stage because we do
        # not want to taper auxiliaries.
        self.qubit_converter.force_match(
            num_particles=problem.num_particles, z2symmetries=Z2Symmetries([], [], [])
        )

        # Apply the Mapping and Two Qubit Reduction as for the Hamiltonian
        untapered_aux_second_q_ops = self.qubit_converter.convert_match(aux_second_q_ops)

        if aux_operators is not None:
            if isinstance(aux_operators, ListOrDict):
                aux_operators = dict(aux_operators)
            custom_aux_ops = self.qubit_converter.convert_match(aux_operators)
            custom_aux_ops = ListOrDict(custom_aux_ops)
        else:
            custom_aux_ops = None

        # Setup the z2symmetries that will be used to taper the qeom matrix element later
        self.qubit_converter.force_match(z2symmetries=z2symmetries)

        # Pre-calculation of the tapering, must come after force_match()
        pre_tap_qubit_op_main = self.qubit_converter._convert_clifford(untapered_qubit_op_main)
        pre_tap_second_q_aux_ops = self.qubit_converter._convert_clifford(
            untapered_aux_second_q_ops
        )
        pre_tap_custom_aux_ops = self.qubit_converter._convert_clifford(custom_aux_ops)

        return (
            pre_tap_qubit_op_main,  # Hamiltonian
            pre_tap_second_q_aux_ops,  # Default auxiliary observables
            pre_tap_custom_aux_ops,  # User-defined auxiliary observables
        )

    def _evaluate_observables_excited_states(
        self,
        pre_tap_aux_observables,
        expansion_basis_data,
        ground_state,
        expansion_coefs,
        aux_operators_evaluated,
    ):

        aux_observables_eigenvalues_raw = {}

        if pre_tap_aux_observables is not None:
            num_qubits = list(pre_tap_aux_observables.values())[0].num_qubits
            identity_operator = PauliSumOp(SparsePauliOp(["I" * num_qubits], [1.0]))

            logger.debug("Prepare subspace representation of the observable: Identity...")

            op_s_mat, op_s_mat_std = self._build_qse_pseudoeigenvalue_problem(
                identity_operator,
                expansion_basis_data,
                ground_state,
                1.0,
            )

            for name, pre_tap_second_q_aux_op in pre_tap_aux_observables.items():
                logger.debug("Prepare subspace representation of the observable: %s...", name)

                aux_operator_evaluated_gs = aux_operators_evaluated[0].get(name, 0.0)
                op_h_mat, op_h_mat_std = self._build_qse_pseudoeigenvalue_problem(
                    pre_tap_second_q_aux_op,
                    expansion_basis_data,
                    ground_state,
                    aux_operator_evaluated_gs,
                )

                logger.debug("Extract eigenvalues for the observable: %s...", name)

                aux_observable_eigenvalues = self._extract_aux_eigenvalues(
                    op_h_mat, op_s_mat, expansion_coefs
                )

                aux_observables_eigenvalues_raw[name] = aux_observable_eigenvalues

        return aux_observables_eigenvalues_raw

    def _build_qse_pseudoeigenvalue_problem(
        self,
        pre_tap_operator,
        expansion_basis_data,
        reference_state,
        tap_operator_evaluated,
    ):
        pre_tap_hopping_ops, type_of_commutativities, size = expansion_basis_data

        # 1. Build all qse operator products
        tap_qse_matrix_ops = self._build_all_eom_operators(
            pre_tap_operator, expansion_basis_data, commutator=False
        )

        overlap_operators = self._build_groundstate_overlaps(
            pre_tap_operator, pre_tap_hopping_ops, size
        )

        tap_qse_operators = {**tap_qse_matrix_ops, **overlap_operators}

        # 3. Evaluate qse operators on the ground state
        measurement_results = estimate_observables(
            self._estimator,
            reference_state[0],
            tap_qse_operators,
            reference_state[1],
        )

        # 4. Post-process ground_state_result to construct qse matrices
        h_mat, h_mat_std = self._build_qse_matrices(
            measurement_results, size, tap_operator_evaluated
        )

        return h_mat, h_mat_std

    def _build_groundstate_overlaps(self, pre_tap_operator, pre_tap_hopping_ops, size):
        overlap_operators = {}
        for mu in range(size):
            overlap_operators[f"H_0_{mu}"] = pre_tap_operator @ pre_tap_hopping_ops[f"E_{mu}"]
            overlap_operators[f"H_0_{mu+size}"] = (
                pre_tap_operator @ pre_tap_hopping_ops[f"Edag_{mu}"]
            )
            # overlap_operators[f'S_0_{mu}'] = pre_tap_hopping_ops[f'E_{mu}']
            # overlap_operators[f'S_0_{mu+size}'] = pre_tap_hopping_ops[f'Edag_{mu}']
        try:
            z2_symmetries = self._gsc.qubit_converter.z2symmetries
        except AttributeError:
            z2_symmetries = Z2Symmetries([], [], [])

        if not z2_symmetries.is_empty():
            for index_op, op in overlap_operators.items():
                if op is not None and len(op) > 0:
                    overlap_operators[index_op] = z2_symmetries.taper_clifford(op)
        return overlap_operators

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
        aux_operator_eigenvalues_excited_states,
    ) -> ElectronicStructureResult:

        qeom_result = QEOMResult()
        qeom_result.ground_state_raw_result = groundstate_result.raw_result
        qeom_result.expansion_coefficients = expansion_coefs
        qeom_result.excitation_energies = energy_gaps
        qeom_result.h_matrix = h_mat
        qeom_result.s_matrix = s_mat
        qeom_result.h_matrix_std = h_mat_std
        qeom_result.s_matrix_std = s_mat_std

        qeom_result.aux_operators_evaluated = [
            groundstate_result.raw_result.aux_operators_evaluated
        ] + aux_operator_eigenvalues_excited_states

        groundstate_energy_reference = groundstate_result.eigenvalues[0]
        excited_eigenenergies = energy_gaps + groundstate_energy_reference
        qeom_result.eigenvalues = np.append(
            groundstate_result.eigenvalues[0], excited_eigenenergies
        )

        qeom_result.eigenstates = []

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
        self._eigenvalues = None
        self._eigenstates = None
        self._h_matrix: Optional[np.ndarray] = None
        self._s_matrix: Optional[np.ndarray] = None
        self._h_matrix_std: np.ndarray = [[0.0, 0.0], [0.0, 0.0]]
        self._s_matrix_std: np.ndarray = [[0.0, 0.0], [0.0, 0.0]]
        self._aux_operators_evaluated = None

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
        return self._m_matrix

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
    def h_matrix_std(self) -> float:
        """returns the H matrix standard deviation"""
        return self._h_matrix_std

    @h_matrix_std.setter
    def h_matrix_std(self, value: float) -> None:
        """sets the H matrix standard deviation"""
        self._h_matrix_std = value

    @property
    def s_matrix_std(self) -> float:
        """returns the S matrix standard deviation"""
        return self._s_matrix_std

    @s_matrix_std.setter
    def s_matrix_std(self, value: float) -> None:
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
    def aux_operators_evaluated(self) -> Optional[List[ListOrDict[Tuple[complex, complex]]]]:
        """Return aux operator expectation values.

        These values are in fact tuples formatted as (mean, standard deviation).
        """
        return self._aux_operators_evaluated

    @aux_operators_evaluated.setter
    def aux_operators_evaluated(self, value: List[ListOrDict[Tuple[complex, complex]]]) -> None:
        """set aux operator eigen values"""
        self._aux_operators_evaluated = value
