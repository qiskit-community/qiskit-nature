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
from qiskit.algorithms import AlgorithmResult
from qiskit.opflow import (
    Z2Symmetries,
    commutator,
    double_commutator,
    PauliSumOp,
)
from qiskit.primitives import BaseEstimator

from qiskit_nature.second_q.operators import SparseLabelOp
from qiskit_nature.second_q.problems import (
    BaseProblem,
    ElectronicStructureProblem,
    VibrationalStructureProblem,
)
from qiskit_nature.second_q.problems import EigenstateResult

from .qeom_electronic_ops_builder import build_electronic_ops
from .qeom_vibrational_ops_builder import build_vibrational_ops
from .excited_states_solver import ExcitedStatesSolver
from ..ground_state_solvers import GroundStateSolver

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
        all excitations of given type will be used. Otherwise, a list of custom excitations can
        directly be provided."""
        if isinstance(excitations, str) and excitations not in ["s", "d", "sd"]:
            raise ValueError(
                "Excitation type must be s (singles), d (doubles) or sd (singles and doubles)"
            )
        self._excitations = excitations

    @property
    def solver(self):
        return self._gsc.solver

    def get_qubit_operators(
        self,
        problem: BaseProblem,
        aux_operators: Optional[dict[str, Union[SparseLabelOp, PauliSumOp]]] = None,
    ) -> Tuple[PauliSumOp, Optional[dict[str, PauliSumOp]]]:
        return self._gsc.get_qubit_operators(problem, aux_operators)

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

        if aux_operators is not None:
            logger.warning(
                "With qEOM the auxiliary operators can currently only be "
                "evaluated on the ground state."
            )

        # 1. Run ground state calculation
        groundstate_result = self._gsc.solve(problem, aux_operators)

        # 2. Prepare the excitation operators
        main_second_q_op, _ = problem.second_q_ops()

        num_particles = getattr(problem, "num_particles", None)

        self._untapered_qubit_op_main = self._gsc.qubit_converter.convert_only(
            main_second_q_op, num_particles
        )
        matrix_operators_dict, size = self._prepare_matrix_operators(problem)

        # 3. Evaluate eom operators
        ground_state = groundstate_result.eigenstates[0]
        measurement_results = estimate_observables(
            self._estimator,
            ground_state[0],
            matrix_operators_dict,
            ground_state[1],
        )

        # 4. Post-process ground_state_result to construct eom matrices
        (
            m_mat,
            v_mat,
            q_mat,
            w_mat,
            m_mat_std,
            v_mat_std,
            q_mat_std,
            w_mat_std,
        ) = self._build_eom_matrices(measurement_results, size)

        # 5. solve pseudo-eigenvalue problem
        energy_gaps, expansion_coefs = self._compute_excitation_energies(m_mat, v_mat, q_mat, w_mat)

        qeom_result = QEOMResult()
        qeom_result.ground_state_raw_result = groundstate_result.raw_result
        qeom_result.expansion_coefficients = expansion_coefs
        qeom_result.excitation_energies = energy_gaps
        qeom_result.m_matrix = m_mat
        qeom_result.v_matrix = v_mat
        qeom_result.q_matrix = q_mat
        qeom_result.w_matrix = w_mat
        qeom_result.m_matrix_std = m_mat_std
        qeom_result.v_matrix_std = v_mat_std
        qeom_result.q_matrix_std = q_mat_std
        qeom_result.w_matrix_std = w_mat_std

        eigenstate_result = EigenstateResult.from_result(groundstate_result)
        eigenstate_result.raw_result = qeom_result

        eigenstate_result.eigenvalues = np.append(
            groundstate_result.eigenvalues,
            np.asarray([groundstate_result.eigenvalues[0] + gap for gap in energy_gaps]),
        )

        result = problem.interpret(eigenstate_result)

        return result

    def _prepare_matrix_operators(self, problem: BaseProblem) -> Tuple[dict, int]:
        """Construct the excitation operators for each matrix element.

        Returns:
            A dictionary of all matrix elements operators and the number of excitations
            (or the size of the qEOM pseudo-eigenvalue problem).
        """
        data = self._build_hopping_ops(problem)
        hopping_operators, type_of_commutativities, excitation_indices = data

        size = int(len(list(excitation_indices.keys())) // 2)

        eom_matrix_operators = self._build_all_commutators(
            hopping_operators, type_of_commutativities, size
        )

        return eom_matrix_operators, size

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

    def _build_all_commutators(
        self, hopping_operators: dict, type_of_commutativities: dict, size: int
    ) -> dict:
        """Building all commutators for Q, W, M, V matrices.

        Args:
            hopping_operators: All hopping operators based on excitations_list,
                key is the string of single/double excitation;
                value is corresponding operator.
            type_of_commutativities: If tapering is used, it records the commutativities of
                hopping operators with the
                Z2 symmetries found in the original operator.
            size: The number of excitations (size of the qEOM pseudo-eigenvalue problem).

        Returns:
            A dictionary that contains the operators for each matrix element.
        """

        all_matrix_operators = {}

        mus, nus = np.triu_indices(size)

        def _build_one_sector(available_hopping_ops, untapered_op, z2_symmetries):

            to_be_computed_list = []
            for idx, m_u in enumerate(mus):
                n_u = nus[idx]
                left_op = available_hopping_ops.get(f"E_{m_u}")
                right_op_1 = available_hopping_ops.get(f"E_{n_u}")
                right_op_2 = available_hopping_ops.get(f"Edag_{n_u}")
                to_be_computed_list.append((m_u, n_u, left_op, right_op_1, right_op_2))

            if logger.isEnabledFor(logging.INFO):
                logger.info("Building all commutators:")
                TextProgressBar(sys.stderr)
            results = parallel_map(
                self._build_commutator_routine,
                to_be_computed_list,
                task_args=(untapered_op, z2_symmetries),
                num_processes=algorithm_globals.num_processes,
            )
            for result in results:
                m_u, n_u, q_mat_op, w_mat_op, m_mat_op, v_mat_op = result

                if q_mat_op is not None:
                    all_matrix_operators[f"q_{m_u}_{n_u}"] = q_mat_op
                if w_mat_op is not None:
                    all_matrix_operators[f"w_{m_u}_{n_u}"] = w_mat_op
                if m_mat_op is not None:
                    all_matrix_operators[f"m_{m_u}_{n_u}"] = m_mat_op
                if v_mat_op is not None:
                    all_matrix_operators[f"v_{m_u}_{n_u}"] = v_mat_op

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
                        available_hopping_ops[key] = hopping_operators[key]
                # untapered_qubit_op is a PauliSumOp and should not be exposed.
                _build_one_sector(
                    available_hopping_ops, self._untapered_qubit_op_main, z2_symmetries
                )

        else:
            # untapered_qubit_op is a PauliSumOp and should not be exposed.
            _build_one_sector(hopping_operators, self._untapered_qubit_op_main, z2_symmetries)

        return all_matrix_operators

    @staticmethod
    def _build_commutator_routine(
        params: List, operator: PauliSumOp, z2_symmetries: Z2Symmetries
    ) -> Tuple[int, int, PauliSumOp, PauliSumOp, PauliSumOp, PauliSumOp]:
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
        m_u, n_u, left_op, right_op_1, right_op_2 = params
        if left_op is None or right_op_1 is None and right_op_2 is None:
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
                q_mat_op = double_commutator(left_op, operator, right_op_1, sign=False)
                # In the case of the single commutator, we are always interested in the energy
                # difference of two states. Thus, regardless of the problem's nature, we will
                # always use the commutator.
                w_mat_op = commutator(left_op, right_op_1)
                q_mat_op = None if len(q_mat_op) == 0 else q_mat_op
                w_mat_op = None if len(w_mat_op) == 0 else w_mat_op
            else:
                q_mat_op = None
                w_mat_op = None

            if right_op_2 is not None:
                # For explanations on the choice of commutation relation, please refer to the
                # comments above.
                m_mat_op = double_commutator(left_op, operator, right_op_2, sign=False)
                v_mat_op = commutator(left_op, right_op_2)
                m_mat_op = None if len(m_mat_op) == 0 else m_mat_op
                v_mat_op = None if len(v_mat_op) == 0 else v_mat_op
            else:
                m_mat_op = None
                v_mat_op = None

            if not z2_symmetries.is_empty():
                if q_mat_op is not None and len(q_mat_op) > 0:
                    q_mat_op = z2_symmetries.taper(q_mat_op)
                if w_mat_op is not None and len(w_mat_op) > 0:
                    w_mat_op = z2_symmetries.taper(w_mat_op)
                if m_mat_op is not None and len(m_mat_op) > 0:
                    m_mat_op = z2_symmetries.taper(m_mat_op)
                if v_mat_op is not None and len(v_mat_op) > 0:
                    v_mat_op = z2_symmetries.taper(v_mat_op)

        return m_u, n_u, q_mat_op, w_mat_op, m_mat_op, v_mat_op

    def _build_eom_matrices(
        self, gs_results: dict[str, tuple[complex, dict[str, Any]]], size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
        """Constructs the M, V, Q and W matrices from the results on the ground state

        Args:
            gs_results: A ground state result object.
            size: Size of eigenvalue problem.

        Returns:
            The matrices and their standard deviation.
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
        q_mat = q_mat + q_mat.T - np.identity(q_mat.shape[0]) * q_mat
        w_mat = w_mat + w_mat.T - np.identity(w_mat.shape[0]) * w_mat
        m_mat = m_mat + m_mat.T - np.identity(m_mat.shape[0]) * m_mat
        v_mat = v_mat + v_mat.T - np.identity(v_mat.shape[0]) * v_mat

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

        return m_mat, v_mat, q_mat, w_mat, m_mat_std, v_mat_std, q_mat_std, w_mat_std

    @staticmethod
    def _compute_excitation_energies(
        m_mat: np.ndarray, v_mat: np.ndarray, q_mat: np.ndarray, w_mat: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Diagonalizing M, V, Q, W matrices for excitation energies.

        Args:
            m_mat : M matrices
            v_mat : V matrices
            q_mat : Q matrices
            w_mat : W matrices

        Returns:
            1-D vector stores all energy gap to reference state.
            2-D array storing the X and Y expansion coefficients.
        """
        logger.debug("Diagonalizing qeom matrices for excited states...")
        a_mat = np.matrixlib.bmat([[m_mat, q_mat], [q_mat.T.conj(), m_mat.T.conj()]])
        b_mat = np.matrixlib.bmat([[v_mat, w_mat], [-w_mat.T.conj(), -v_mat.T.conj()]])
        res = linalg.eig(a_mat, b_mat)
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
        w = np.sort(np.real(res[0]))
        logger.debug("Sorted real parts %s", w)
        w = np.abs(w[len(w) // 2 :])
        w[w < 1e-06] = 0
        excitation_energies_gap = w

        return excitation_energies_gap, res[1]


class QEOMResult(AlgorithmResult):
    """The results class for the QEOM algorithm."""

    def __init__(self) -> None:
        super().__init__()
        self._ground_state_raw_result = None
        self._excitation_energies: Optional[np.ndarray] = None
        self._expansion_coefficients: Optional[np.ndarray] = None
        self._m_matrix: Optional[np.ndarray] = None
        self._v_matrix: Optional[np.ndarray] = None
        self._q_matrix: Optional[np.ndarray] = None
        self._w_matrix: Optional[np.ndarray] = None
        self._v_matrix_std: float = 0.0
        self._q_matrix_std: float = 0.0
        self._w_matrix_std: float = 0.0

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
    def m_matrix(self) -> Optional[np.ndarray]:
        """returns the M matrix"""
        return self._m_matrix

    @m_matrix.setter
    def m_matrix(self, value: np.ndarray) -> None:
        """sets the M matrix"""
        self._m_matrix = value

    @property
    def v_matrix(self) -> Optional[np.ndarray]:
        """returns the V matrix"""
        return self._v_matrix

    @v_matrix.setter
    def v_matrix(self, value: np.ndarray) -> None:
        """sets the V matrix"""
        self._v_matrix = value

    @property
    def q_matrix(self) -> Optional[np.ndarray]:
        """returns the Q matrix"""
        return self._q_matrix

    @q_matrix.setter
    def q_matrix(self, value: np.ndarray) -> None:
        """sets the Q matrix"""
        self._q_matrix = value

    @property
    def w_matrix(self) -> Optional[np.ndarray]:
        """returns the W matrix"""
        return self._w_matrix

    @w_matrix.setter
    def w_matrix(self, value: np.ndarray) -> None:
        """sets the W matrix"""
        self._w_matrix = value

    @property
    def m_matrix_std(self) -> float:
        """returns the M matrix standard deviation"""
        return self._m_matrix_std

    @m_matrix_std.setter
    def m_matrix_std(self, value: float) -> None:
        """sets the M matrix standard deviation"""
        self._m_matrix_std = value

    @property
    def v_matrix_std(self) -> float:
        """returns the V matrix standard deviation"""
        return self._v_matrix_std

    @v_matrix_std.setter
    def v_matrix_std(self, value: float) -> None:
        """sets the V matrix standard deviation"""
        self._v_matrix_std = value

    @property
    def q_matrix_std(self) -> float:
        """returns the Q matrix standard deviation"""
        return self._q_matrix_std

    @q_matrix_std.setter
    def q_matrix_std(self, value: float) -> None:
        """sets the Q matrix standard deviation"""
        self._q_matrix_std = value

    @property
    def w_matrix_std(self) -> float:
        """returns the W matrix standard deviation"""
        return self._w_matrix_std

    @w_matrix_std.setter
    def w_matrix_std(self, value: float) -> None:
        """sets the W matrix standard deviation"""
        self._w_matrix_std = value
