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

"""The AngularMomentum property."""

from __future__ import annotations

from typing import Mapping

import itertools

import numpy as np

import qiskit_nature  # pylint: disable=unused-import
from qiskit_nature.second_q.operators import FermionicOp, PolynomialTensor
from qiskit_nature.second_q.operators.tensor_ordering import IndexType, to_physicist_ordering


class AngularMomentum:
    """The AngularMomentum property.

    The following attributes can be set via the initializer but can also be read and updated once
    the ``AngularMomentum`` object has been constructed.

    Attributes:
        num_spatial_orbitals (int): the number of spatial orbitals.
    """

    def __init__(self, num_spatial_orbitals: int) -> None:
        """
        Args:
            num_spatial_orbitals: the number of spatial orbitals in the system.
        """
        self.num_spatial_orbitals = num_spatial_orbitals

    def second_q_ops(self) -> Mapping[str, FermionicOp]:
        """Returns the second quantized angular momentum operator.

        Returns:
            A mapping of strings to `FermionicOp` objects.
        """
        x_h1, x_h2 = _calc_s_x_squared_ints(self.num_spatial_orbitals)
        y_h1, y_h2 = _calc_s_y_squared_ints(self.num_spatial_orbitals)
        z_h1, z_h2 = _calc_s_z_squared_ints(self.num_spatial_orbitals)
        h_1 = x_h1 + y_h1 + z_h1
        h_2 = x_h2 + y_h2 + z_h2

        tensor = PolynomialTensor(
            {"+-": h_1, "++--": to_physicist_ordering(h_2, index_order=IndexType.CHEMIST)}
        )

        op = FermionicOp.from_polynomial_tensor(tensor).simplify()

        return {self.__class__.__name__: op}

    def interpret(
        self, result: "qiskit_nature.second_q.problems.EigenstateResult"  # type: ignore[name-defined]
    ) -> None:
        """Interprets an :class:`~qiskit_nature.second_q.problems.EigenstateResult`
        in this property's context.

        Args:
            result: the result to add meaning to.
        """
        result.total_angular_momentum = []

        if result.aux_operators_evaluated is None:
            return

        for aux_op_eigenvalues in result.aux_operators_evaluated:
            if not isinstance(aux_op_eigenvalues, dict):
                continue

            _key = self.__class__.__name__

            if aux_op_eigenvalues[_key] is not None:
                result.total_angular_momentum.append(aux_op_eigenvalues[_key].real)
            else:
                result.total_angular_momentum.append(None)


def _calc_s_x_squared_ints(num_spatial_orbitals: int) -> tuple[np.ndarray, np.ndarray]:
    return _calc_squared_ints(
        num_spatial_orbitals, _modify_s_x_squared_ints_neq, _modify_s_x_squared_ints_eq
    )


def _calc_s_y_squared_ints(num_spatial_orbitals: int) -> tuple[np.ndarray, np.ndarray]:
    return _calc_squared_ints(
        num_spatial_orbitals, _modify_s_y_squared_ints_neq, _modify_s_y_squared_ints_eq
    )


def _calc_s_z_squared_ints(num_spatial_orbitals: int) -> tuple[np.ndarray, np.ndarray]:
    return _calc_squared_ints(
        num_spatial_orbitals, _modify_s_z_squared_ints_neq, _modify_s_z_squared_ints_eq
    )


def _calc_squared_ints(
    num_spatial_orbitals: int, func_neq, func_eq
) -> tuple[np.ndarray, np.ndarray]:
    # calculates 1- and 2-body integrals for a given angular momentum axis (x or y or z,
    # specified by func_neq and func_eq)
    num_spin_orbitals = 2 * num_spatial_orbitals
    h_1 = np.zeros((num_spin_orbitals, num_spin_orbitals))
    h_2 = np.zeros((num_spin_orbitals, num_spin_orbitals, num_spin_orbitals, num_spin_orbitals))

    # pylint: disable=invalid-name
    for p, q in itertools.product(range(num_spatial_orbitals), repeat=2):
        if p != q:
            h_2 = func_neq(h_2, p, q, num_spatial_orbitals)
        else:
            h_2 = func_eq(h_2, p, num_spatial_orbitals)
            h_1[p, p] += 1.0
            h_1[p + num_spatial_orbitals, p + num_spatial_orbitals] += 1.0
    h_1 *= 0.25
    h_2 *= 0.25
    return h_1, h_2


def _modify_s_x_squared_ints_neq(
    h_2: np.ndarray, p_ind: int, q_ind: int, num_spatial_orbitals: int
) -> np.ndarray:
    indices = [
        (p_ind, p_ind + num_spatial_orbitals, q_ind, q_ind + num_spatial_orbitals),
        (p_ind + num_spatial_orbitals, p_ind, q_ind, q_ind + num_spatial_orbitals),
        (p_ind, p_ind + num_spatial_orbitals, q_ind + num_spatial_orbitals, q_ind),
        (p_ind + num_spatial_orbitals, p_ind, q_ind + num_spatial_orbitals, q_ind),
    ]
    values = [1, 1, 1, 1]
    # adds provided values to values of 2-body integrals (x axis of angular momentum) at given
    # indices in case p not equal to q
    return _add_values_to_s_squared_ints(h_2, indices, values)


def _modify_s_x_squared_ints_eq(
    h_2: np.ndarray, p_ind: int, num_spatial_orbitals: int
) -> np.ndarray:
    indices = [
        (p_ind, p_ind + num_spatial_orbitals, p_ind, p_ind + num_spatial_orbitals),
        (p_ind + num_spatial_orbitals, p_ind, p_ind + num_spatial_orbitals, p_ind),
        (p_ind, p_ind, p_ind + num_spatial_orbitals, p_ind + num_spatial_orbitals),
        (p_ind + num_spatial_orbitals, p_ind + num_spatial_orbitals, p_ind, p_ind),
    ]
    values = [-1, -1, -1, -1]
    # adds provided values to values of 2-body integrals (x axis of angular momentum) at given
    # indices in case p equal to q
    return _add_values_to_s_squared_ints(h_2, indices, values)


def _modify_s_y_squared_ints_neq(
    h_2: np.ndarray, p_ind: int, q_ind: int, num_spatial_orbitals: int
) -> np.ndarray:
    indices = [
        (p_ind, p_ind + num_spatial_orbitals, q_ind, q_ind + num_spatial_orbitals),
        (p_ind + num_spatial_orbitals, p_ind, q_ind, q_ind + num_spatial_orbitals),
        (p_ind, p_ind + num_spatial_orbitals, q_ind + num_spatial_orbitals, q_ind),
        (p_ind + num_spatial_orbitals, p_ind, q_ind + num_spatial_orbitals, q_ind),
    ]
    values = [-1, 1, 1, -1]
    # adds provided values to values of 2-body integrals (y axis of angular momentum) at given
    # indices in case p not equal to q
    return _add_values_to_s_squared_ints(h_2, indices, values)


def _modify_s_y_squared_ints_eq(
    h_2: np.ndarray, p_ind: int, num_spatial_orbitals: int
) -> np.ndarray:
    indices = [
        (p_ind, p_ind + num_spatial_orbitals, p_ind, p_ind + num_spatial_orbitals),
        (p_ind + num_spatial_orbitals, p_ind, p_ind + num_spatial_orbitals, p_ind),
        (p_ind, p_ind, p_ind + num_spatial_orbitals, p_ind + num_spatial_orbitals),
        (p_ind + num_spatial_orbitals, p_ind + num_spatial_orbitals, p_ind, p_ind),
    ]
    values = [1, 1, -1, -1]
    # adds provided values to values of 2-body integrals (y axis of angular momentum) at given
    # indices in case p equal to q
    return _add_values_to_s_squared_ints(h_2, indices, values)


def _modify_s_z_squared_ints_neq(
    h_2: np.ndarray, p_ind: int, q_ind: int, num_spatial_orbitals: int
) -> np.ndarray:
    indices = [
        (p_ind, p_ind, q_ind, q_ind),
        (p_ind + num_spatial_orbitals, p_ind + num_spatial_orbitals, q_ind, q_ind),
        (p_ind, p_ind, q_ind + num_spatial_orbitals, q_ind + num_spatial_orbitals),
        (
            p_ind + num_spatial_orbitals,
            p_ind + num_spatial_orbitals,
            q_ind + num_spatial_orbitals,
            q_ind + num_spatial_orbitals,
        ),
    ]
    values = [1, -1, -1, 1]
    # adds provided values to values of 2-body integrals (z axis of angular momentum) at given
    # indices in case p not equal to q
    return _add_values_to_s_squared_ints(h_2, indices, values)


def _modify_s_z_squared_ints_eq(
    h_2: np.ndarray, p_ind: int, num_spatial_orbitals: int
) -> np.ndarray:
    indices = [
        (p_ind, p_ind + num_spatial_orbitals, p_ind + num_spatial_orbitals, p_ind),
        (p_ind + num_spatial_orbitals, p_ind, p_ind, p_ind + num_spatial_orbitals),
    ]
    values = [1, 1]
    # adds provided values to values of 2-body integrals (z axis of angular momentum) at given
    # indices in case p equal to q
    return _add_values_to_s_squared_ints(h_2, indices, values)


def _add_values_to_s_squared_ints(
    h_2: np.ndarray, indices: list[tuple[int, int, int, int]], values: list[int]
) -> np.ndarray:
    for index, value in zip(indices, values):
        h_2[index] += value
    return h_2
