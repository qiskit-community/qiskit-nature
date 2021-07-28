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

"""The AngularMomentum property."""

from typing import cast, List, Tuple

import itertools

import numpy as np

from qiskit_nature.drivers.second_quantization import QMolecule
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.results import EigenstateResult

from .bases import ElectronicBasis
from .integrals import (
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
)
from .types import ElectronicProperty
from ..second_quantized_property import LegacyDriverResult, LegacyElectronicDriverResult


class AngularMomentum(ElectronicProperty):
    """The AngularMomentum property."""

    def __init__(self, num_spin_orbitals: int):
        """
        Args:
            num_spin_orbitals: the number of spin orbitals in the system.
        """
        super().__init__(self.__class__.__name__)
        self._num_spin_orbitals = num_spin_orbitals
        # TODO: store expected spin?

    def __str__(self) -> str:
        string = [super().__str__() + ":"]
        string += [f"\t{self._num_spin_orbitals} SOs"]
        return "\n".join(string)

    @classmethod
    def from_legacy_driver_result(cls, result: LegacyDriverResult) -> "AngularMomentum":
        """Construct an AngularMomentum instance from a QMolecule.

        Args:
            result: the driver result from which to extract the raw data. For this property, a
                QMolecule is required!

        Returns:
            An instance of this property.

        Raises:
            QiskitNatureError: if a WatsonHamiltonian is provided.
        """
        cls._validate_input_type(result, LegacyElectronicDriverResult)

        qmol = cast(QMolecule, result)

        return cls(
            qmol.num_molecular_orbitals * 2,
        )

    def second_q_ops(self) -> List[FermionicOp]:
        """Returns a list containing the angular momentum operator."""
        x_h1, x_h2 = _calc_s_x_squared_ints(self._num_spin_orbitals)
        y_h1, y_h2 = _calc_s_y_squared_ints(self._num_spin_orbitals)
        z_h1, z_h2 = _calc_s_z_squared_ints(self._num_spin_orbitals)
        h_1 = x_h1 + y_h1 + z_h1
        h_2 = x_h2 + y_h2 + z_h2

        h1_ints = OneBodyElectronicIntegrals(ElectronicBasis.SO, h_1)
        h2_ints = TwoBodyElectronicIntegrals(ElectronicBasis.SO, h_2)
        return [(h1_ints.to_second_q_op() + h2_ints.to_second_q_op()).reduce()]

    def interpret(self, result: EigenstateResult) -> None:
        """Interprets an :class:~qiskit_nature.result.EigenstateResult in this property's context.

        Args:
            result: the result to add meaning to.
        """
        result.total_angular_momentum = []

        if not isinstance(result.aux_operator_eigenvalues, list):
            aux_operator_eigenvalues = [result.aux_operator_eigenvalues]
        else:
            aux_operator_eigenvalues = result.aux_operator_eigenvalues  # type: ignore
        for aux_op_eigenvalues in aux_operator_eigenvalues:
            if aux_op_eigenvalues is None:
                continue

            if aux_op_eigenvalues[1] is not None:
                result.total_angular_momentum.append(aux_op_eigenvalues[1][0].real)  # type: ignore
            else:
                result.total_angular_momentum.append(None)


def _calc_s_x_squared_ints(num_modes: int) -> Tuple[np.ndarray, np.ndarray]:
    return _calc_squared_ints(num_modes, _modify_s_x_squared_ints_neq, _modify_s_x_squared_ints_eq)


def _calc_s_y_squared_ints(num_modes: int) -> Tuple[np.ndarray, np.ndarray]:
    return _calc_squared_ints(num_modes, _modify_s_y_squared_ints_neq, _modify_s_y_squared_ints_eq)


def _calc_s_z_squared_ints(num_modes: int) -> Tuple[np.ndarray, np.ndarray]:
    return _calc_squared_ints(num_modes, _modify_s_z_squared_ints_neq, _modify_s_z_squared_ints_eq)


def _calc_squared_ints(num_modes: int, func_neq, func_eq) -> Tuple[np.ndarray, np.ndarray]:
    # calculates 1- and 2-body integrals for a given angular momentum axis (x or y or z,
    # specified by func_neq and func_eq)
    num_modes_2 = num_modes // 2
    h_1 = np.zeros((num_modes, num_modes))
    h_2 = np.zeros((num_modes, num_modes, num_modes, num_modes))

    # pylint: disable=invalid-name
    for p, q in itertools.product(range(num_modes_2), repeat=2):
        if p != q:
            h_2 = func_neq(h_2, p, q, num_modes_2)
        else:
            h_2 = func_eq(h_2, p, num_modes_2)
            h_1[p, p] += 1.0
            h_1[p + num_modes_2, p + num_modes_2] += 1.0
    h_1 *= 0.25
    h_2 *= 0.25
    return h_1, h_2


def _modify_s_x_squared_ints_neq(
    h_2: np.ndarray, p_ind: int, q_ind: int, num_modes_2: int
) -> np.ndarray:
    indices = [
        (p_ind, p_ind + num_modes_2, q_ind, q_ind + num_modes_2),
        (p_ind + num_modes_2, p_ind, q_ind, q_ind + num_modes_2),
        (p_ind, p_ind + num_modes_2, q_ind + num_modes_2, q_ind),
        (p_ind + num_modes_2, p_ind, q_ind + num_modes_2, q_ind),
    ]
    values = [1, 1, 1, 1]
    # adds provided values to values of 2-body integrals (x axis of angular momentum) at given
    # indices in case p not equal to q
    return _add_values_to_s_squared_ints(h_2, indices, values)


def _modify_s_x_squared_ints_eq(h_2: np.ndarray, p_ind: int, num_modes_2: int) -> np.ndarray:
    indices = [
        (p_ind, p_ind + num_modes_2, p_ind, p_ind + num_modes_2),
        (p_ind + num_modes_2, p_ind, p_ind + num_modes_2, p_ind),
        (p_ind, p_ind, p_ind + num_modes_2, p_ind + num_modes_2),
        (p_ind + num_modes_2, p_ind + num_modes_2, p_ind, p_ind),
    ]
    values = [-1, -1, -1, -1]
    # adds provided values to values of 2-body integrals (x axis of angular momentum) at given
    # indices in case p equal to q
    return _add_values_to_s_squared_ints(h_2, indices, values)


def _modify_s_y_squared_ints_neq(
    h_2: np.ndarray, p_ind: int, q_ind: int, num_modes_2: int
) -> np.ndarray:
    indices = [
        (p_ind, p_ind + num_modes_2, q_ind, q_ind + num_modes_2),
        (p_ind + num_modes_2, p_ind, q_ind, q_ind + num_modes_2),
        (p_ind, p_ind + num_modes_2, q_ind + num_modes_2, q_ind),
        (p_ind + num_modes_2, p_ind, q_ind + num_modes_2, q_ind),
    ]
    values = [-1, 1, 1, -1]
    # adds provided values to values of 2-body integrals (y axis of angular momentum) at given
    # indices in case p not equal to q
    return _add_values_to_s_squared_ints(h_2, indices, values)


def _modify_s_y_squared_ints_eq(h_2: np.ndarray, p_ind: int, num_modes_2: int) -> np.ndarray:
    indices = [
        (p_ind, p_ind + num_modes_2, p_ind, p_ind + num_modes_2),
        (p_ind + num_modes_2, p_ind, p_ind + num_modes_2, p_ind),
        (p_ind, p_ind, p_ind + num_modes_2, p_ind + num_modes_2),
        (p_ind + num_modes_2, p_ind + num_modes_2, p_ind, p_ind),
    ]
    values = [1, 1, -1, -1]
    # adds provided values to values of 2-body integrals (y axis of angular momentum) at given
    # indices in case p equal to q
    return _add_values_to_s_squared_ints(h_2, indices, values)


def _modify_s_z_squared_ints_neq(
    h_2: np.ndarray, p_ind: int, q_ind: int, num_modes_2: int
) -> np.ndarray:
    indices = [
        (p_ind, p_ind, q_ind, q_ind),
        (p_ind + num_modes_2, p_ind + num_modes_2, q_ind, q_ind),
        (p_ind, p_ind, q_ind + num_modes_2, q_ind + num_modes_2),
        (
            p_ind + num_modes_2,
            p_ind + num_modes_2,
            q_ind + num_modes_2,
            q_ind + num_modes_2,
        ),
    ]
    values = [1, -1, -1, 1]
    # adds provided values to values of 2-body integrals (z axis of angular momentum) at given
    # indices in case p not equal to q
    return _add_values_to_s_squared_ints(h_2, indices, values)


def _modify_s_z_squared_ints_eq(h_2: np.ndarray, p_ind: int, num_modes_2: int) -> np.ndarray:
    indices = [
        (p_ind, p_ind + num_modes_2, p_ind + num_modes_2, p_ind),
        (p_ind + num_modes_2, p_ind, p_ind, p_ind + num_modes_2),
    ]
    values = [1, 1]
    # adds provided values to values of 2-body integrals (z axis of angular momentum) at given
    # indices in case p equal to q
    return _add_values_to_s_squared_ints(h_2, indices, values)


def _add_values_to_s_squared_ints(
    h_2: np.ndarray, indices: List[Tuple[int, int, int, int]], values: List[int]
) -> np.ndarray:
    for index, value in zip(indices, values):
        h_2[index] += value
    return h_2
