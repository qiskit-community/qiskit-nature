# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Logarithmic Mapper for Bosons."""

from __future__ import annotations
import operator
import math
import logging

from functools import reduce, lru_cache

import numpy as np

from qiskit.quantum_info import SparsePauliOp

from qiskit_nature.second_q.operators import BosonicOp
from .bosonic_mapper import BosonicMapper

logger = logging.getLogger(__name__)


class BosonicLogarithmicMapper(BosonicMapper):
    """The Logarithmic boson-to-qubit Mapper.

    This mapper generates a logarithmic encoding of the Bosonic operator :math:`b_k^\\dagger, b_k` to
    qubit operators (linear combinations of pauli strings).
    In this logarithmic encoding the number of qubits necessary to represent a bosonic mode is
    determined by the max occupation :math:`n_k^{max}` of the mode (meaning the number of states used
    in the expansion of the mode, or equivalently the state at which the maximum excitation can take
    place). The number of qubits is given by:
    :math:`\\lceil\\log_2(n_k^{max} + 1)\\rceil`.

    .. note::
        A consequence of the rounding up for determining the number of required qubits is that the
        actual max occupation is often larger than the one selected by the user. For example, if the
        user selects a :math:`n_k^{max} = 2`, then the number of required qubits is
        :math:`\\lceil\\log_2(3)\\rceil = 2`. If we now compute the max occupation for 2 qubits, we
        get :math:`2^2 - 1 = 3`, which is larger than the user-selected max occupation. The user should
        expect that the actual max occupation is always larger than or equal to the one selected.
        If the code changes the max occupation, the code will issue a warning in the logs.

    The mode :math:`|k\\rangle` is then mapped to the occupation number vector
    :math:`|0_{n_k^{max}}, 0_{n_k^{max} - 1},..., 0_{n_k + 1}, 1_{n_k}, 0_{n_k - 1},..., 0_{0_k}\\rangle`

    This class implements the equation (34) and (35) of Reference [1].

    .. math::
        b_k^\\dagger = \\sum_{n_k = 0}^{2^{N_q}-2}(\\sqrt{n_k + 1}|n+1\\rangle\\langle n|)

        b_k = \\sum_{n_k = 1}^{2^{N_q}-1}(\\sqrt{n_k}|n-1\\rangle\\langle n|)

    where :math:`N_q` is the number of qubits used to represent each mode
    (given by :math:`\\lceil\\log_2(n_k^{max} + 1)\\rceil`). This implementation first computes each
    :math:`|n+1\\rangle\\langle n|` and :math:`|n-1\\rangle\\langle n|` in a binary representation
    and the uses equation (37) from Reference [1] to map to the Pauli operators.

    The length of the qubit register is:

    .. code-block:: python

        BosonicOp.num_modes * numpy.log2(BosonicLogarithmicMapper.max_occupation + 1)

    To use this mapper one can for example:

    .. code-block:: python

        from qiskit_nature.second_q.mappers import BosonicLogarithmicMapper
        from qiskit_nature.second_q.operators import BosonicOp

        mapper = BosonicLogarithmicMapper(max_occupation=2)
        qubit_op = mapper.map(BosonicOp({'+_0 -_0': 1}, num_modes=1))

    .. note::
        Since this mapper truncates the maximum occupation of a bosonic state as represented in the
        qubit register, the commutation relation after the mapping differ from the standard ones.
        Please refer to Section 4, equation 22 of Reference [2] for more details

    References:
        [1] Bo Peng et al., Quantum Simulation of Boson-Related Hamiltonians: Techniques, Effective
        Hamiltonian Construction, and Error Analysis, Arxiv https://doi.org/10.48550/arXiv.2307.06580

        [2] R. Somma et al., Quantum Simulations of Physics Problems, Arxiv
        https://doi.org/10.48550/arXiv.quant-ph/0304063
    """

    def __init__(self, max_occupation: int) -> None:
        # Compute the actual max occupation from the one selected by the user
        self.number_of_qubits_per_mode: int = (
            1 if max_occupation == 0 else math.ceil(np.log2(max_occupation + 1))
        )
        max_calculated_occupation = 2**self.number_of_qubits_per_mode - 1
        if max_occupation != max_calculated_occupation:
            logger.warning(
                f"The user requested a max occupation of {max_occupation}, but the actual "
                + f"max occupation is {max_calculated_occupation}."
            )
        super().__init__(max_calculated_occupation)

    @property
    def number_of_qubits_per_mode(self) -> int:
        """The minimum number of qubits required to represent a bosonic mode given a max occupation."""
        return self._number_of_qubits_per_mode

    @number_of_qubits_per_mode.setter
    def number_of_qubits_per_mode(self, num_qubits: int) -> None:
        if num_qubits < 1:
            raise ValueError(f"The number of qubits must be at least 1, and not {num_qubits}.")
        self._number_of_qubits_per_mode: int = num_qubits

    def _map_single(
        self, second_q_op: BosonicOp, *, register_length: int | None = None
    ) -> SparsePauliOp:
        """Maps a :class:`~qiskit_nature.second_q.operators.SparseLabelOp` to a ``SparsePauliOp``.

        Args:
            second_q_op: the ``SparseLabelOp`` to be mapped.
            register_length: when provided, this will be used to overwrite the ``register_length``
                attribute of the operator being mapped. This is possible because the
                ``register_length`` is considered a lower bound in a ``SparseLabelOp``.

        Returns:
            The qubit operator corresponding to the problem-Hamiltonian in the qubit space.
        """
        if register_length is None:
            register_length = second_q_op.num_modes

        # The actual register length is the number of qubits per mode times the number of modes
        qubit_register_length: int = register_length * self.number_of_qubits_per_mode
        # Create a Pauli operator, which we will fill in this method
        pauli_op: list[SparsePauliOp] = []
        # Then we loop over all the terms of the bosonic operator
        for terms, coeff in second_q_op.terms():
            # Then loop over each term (terms -> List[Tuple[string, int]])
            bos_op_to_pauli_op = SparsePauliOp(["I" * qubit_register_length], coeffs=[1.0])
            # Loop over the operators in the term
            for op, idx in terms:
                if op not in ("+", "-"):
                    break
                pauli_expansion: list[SparsePauliOp] = []
                # Define the index of the mode in the qubit register
                mode_index_in_register: int = idx * self.number_of_qubits_per_mode
                # Now we start mapping the operator. First, define the range of the sum
                terms_range = (
                    range(2**self.number_of_qubits_per_mode - 1)
                    if op == "+"
                    else range(1, 2**self.number_of_qubits_per_mode)
                )
                for n in terms_range:
                    # In each iteration we deal with a term of the form sqrt(n+1)*|n+1><n| or
                    # sqrt(n)*|n-1><n|. The initial and final states are represented in binary.
                    # Define the prefactor and the initial and final states (which results from the
                    # action of the operator). They vary depending on the operator
                    prefactor = np.sqrt(n + 1) if op == "+" else np.sqrt(n)
                    final_state: str = (
                        # fmt: off
                        f"{(n + 1) if op == "+" else (n - 1):0{self.number_of_qubits_per_mode}b}"
                        # fmt: on
                    )
                    init_state: str = f"{n:0{self.number_of_qubits_per_mode}b}"
                    # Now build the Pauli operators
                    single_mapped_term = SparsePauliOp(["I" * qubit_register_length], coeffs=[1.0])
                    # pylint: disable=consider-using-enumerate
                    for j in range(len(init_state)):
                        # We need to comply to the little endian notation of qiskit.
                        # For the binary string representation of the state, the first element is the
                        # most significant bit. Thus, it needs to be put to the end of the mode in the
                        # qubit register.
                        i: int = len(init_state) - j - 1
                        qubit_operator = f"{final_state[j]}{init_state[j]}"
                        # Case |0><0|: this should be converted to 0.5*(I + Z)
                        if qubit_operator == "00":
                            single_mapped_term = single_mapped_term.compose(
                                self._get_single_qubit_pauli_matrix(
                                    mode_index_in_register + i, qubit_register_length, "I+"
                                )
                            )
                        # Case |1><1|: this should be converted to 0.5*(I - Z)
                        elif qubit_operator == "11":
                            single_mapped_term = single_mapped_term.compose(
                                self._get_single_qubit_pauli_matrix(
                                    mode_index_in_register + i, qubit_register_length, "I-"
                                )
                            )
                        # Case |0><1|: this should be converted to 0.5*(X + iY)
                        elif qubit_operator == "01":
                            single_mapped_term = single_mapped_term.compose(
                                self._get_single_qubit_pauli_matrix(
                                    mode_index_in_register + i, qubit_register_length, "S+"
                                )
                            )
                        # Case |1><0|: this should be converted to 0.5*(X - iY)
                        elif qubit_operator == "10":
                            single_mapped_term = single_mapped_term.compose(
                                self._get_single_qubit_pauli_matrix(
                                    mode_index_in_register + i, qubit_register_length, "S-"
                                )
                            )
                        else:
                            raise ValueError(f"Invalid state {qubit_operator}.")
                    pauli_expansion.append(prefactor * single_mapped_term)
                # Add the Pauli expansion for a single n_k to map of the bosonic operator
                bos_op_to_pauli_op = reduce(operator.add, pauli_expansion).compose(
                    bos_op_to_pauli_op
                )
            # Add the map of the single boson op (e.g. +_0) to the map of the full bosonic operator
            pauli_op.append(coeff * reduce(operator.add, bos_op_to_pauli_op.simplify()))
        # return the lookup table for the transformed XYZI operators
        return reduce(operator.add, pauli_op)

    @lru_cache(maxsize=32)
    def _get_single_qubit_pauli_matrix(
        self, qubit_idx: int, register_length: int, pauli_op: str
    ) -> SparsePauliOp:
        """This method builds the Qiskit Pauli operators of one of the operators:
        I_+ = I + Z, I_- = I - Z, S_+ = X + iY and S_- = X - iY.

        Args:
            qubit_idx: the register index of the qubit on which the operator is acting.
            register_length: the length of the qubit register.
            pauli_op: the operator to be mapped. Possible values are 'I+', 'I-', 'S+' and 'S-'.

        Returns:
            A SparsePauliOp representing the Pauli operator.
        """
        # Build the Pauli strings
        if pauli_op == "I+":
            return SparsePauliOp.from_sparse_list(
                [("", [], 0.5), ("Z", [qubit_idx], 0.5)], num_qubits=register_length
            )
        if pauli_op == "I-":
            return SparsePauliOp.from_sparse_list(
                [("", [], 0.5), ("Z", [qubit_idx], -0.5)], num_qubits=register_length
            )
        if pauli_op == "S+":
            return SparsePauliOp.from_sparse_list(
                [("X", [qubit_idx], 0.5), ("Y", [qubit_idx], 0.5j)], num_qubits=register_length
            )
        if pauli_op == "S-":
            return SparsePauliOp.from_sparse_list(
                [("X", [qubit_idx], 0.5), ("Y", [qubit_idx], -0.5j)], num_qubits=register_length
            )
        raise ValueError(
            f"Invalid operator {pauli_op}. Possible values are 'I+', 'I-', 'S+' and 'S-'."
        )
