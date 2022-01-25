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

import logging
from typing import cast, List, Optional, Tuple

import itertools

import h5py
import numpy as np

from qiskit_nature import ListOrDictType, settings
from qiskit_nature.drivers import QMolecule
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.results import EigenstateResult

from ..second_quantized_property import LegacyDriverResult
from .bases import ElectronicBasis
from .integrals import (
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
)
from .types import ElectronicProperty

LOGGER = logging.getLogger(__name__)


class AngularMomentum(ElectronicProperty):
    """The AngularMomentum property."""

    ABSOLUTE_TOLERANCE = 1e-05
    RELATIVE_TOLERANCE = 1e-02

    def __init__(
        self,
        num_spin_orbitals: int,
        spin: Optional[float] = None,
        absolute_tolerance: float = ABSOLUTE_TOLERANCE,
        relative_tolerance: float = RELATIVE_TOLERANCE,
    ) -> None:
        """
        Args:
            num_spin_orbitals: the number of spin orbitals in the system.
            spin: the expected spin of the system. This is only used during result interpretation.
                If the measured value does not match this one, this will be logged on the INFO level.
            absolute_tolerance: the absolute tolerance used for checking whether the measured
                particle number matches the expected one.
            relative_tolerance: the relative tolerance used for checking whether the measured
                particle number matches the expected one.
        """
        super().__init__(self.__class__.__name__)
        self._num_spin_orbitals = num_spin_orbitals
        self._spin = spin
        self._absolute_tolerance = absolute_tolerance
        self._relative_tolerance = relative_tolerance

    @property
    def spin(self) -> Optional[float]:
        """Returns the expected spin."""
        return self._spin

    @spin.setter
    def spin(self, spin: Optional[float]) -> None:
        """Sets the expected spin."""
        self._spin = spin

    def __str__(self) -> str:
        string = [super().__str__() + ":"]
        string += [f"\t{self._num_spin_orbitals} SOs"]
        if self.spin is not None:
            string += [f"\tExpected spin: {self.spin}"]
        return "\n".join(string)

    def to_hdf5(self, parent: h5py.Group) -> None:
        """Stores this instance in a HDF5 group inside of the provided parent group.

        Args:
            parent: the parent HDF5 group.
        """
        super().to_hdf5(parent)
        group = parent.require_group(self.name)

        group.attrs["num_spin_orbitals"] = self._num_spin_orbitals
        if self._spin:
            group.attrs["spin"] = self._spin
        group.attrs["absolute_tolerance"] = self._absolute_tolerance
        group.attrs["relative_tolerance"] = self._relative_tolerance

    @classmethod
    def from_hdf5(cls, h5py_group: h5py.Group) -> AngularMomentum:
        """Constructs a new instance from the data stored in the provided HDF5 group.

        Args:
            h5py_group: the HDF5 group from which to load the data.

        Returns:
            A new instance of this class.
        """
        return AngularMomentum(
            h5py_group.attrs["num_spin_orbitals"],
            h5py_group.attrs.get("spin", None),
            h5py_group.attrs["absolute_tolerance"],
            h5py_group.attrs["relative_tolerance"],
        )

    @classmethod
    def from_legacy_driver_result(cls, result: LegacyDriverResult) -> AngularMomentum:
        """Construct an AngularMomentum instance from a :class:`~qiskit_nature.drivers.QMolecule`.

        Args:
            result: the driver result from which to extract the raw data. For this property, a
                :class:`~qiskit_nature.drivers.QMolecule` is required!

        Returns:
            An instance of this property.

        Raises:
            QiskitNatureError: if a :class:`~qiskit_nature.drivers.WatsonHamiltonian` is provided.
        """
        cls._validate_input_type(result, QMolecule)

        qmol = cast(QMolecule, result)

        return cls(
            qmol.num_molecular_orbitals * 2,
        )

    def second_q_ops(self) -> ListOrDictType[FermionicOp]:
        """Returns the second quantized angular momentum operator.

        The actual return-type is determined by `qiskit_nature.settings.dict_aux_operators`.

        Returns:
            A `list` or `dict` of `FermionicOp` objects.
        """
        x_h1, x_h2 = _calc_s_x_squared_ints(self._num_spin_orbitals)
        y_h1, y_h2 = _calc_s_y_squared_ints(self._num_spin_orbitals)
        z_h1, z_h2 = _calc_s_z_squared_ints(self._num_spin_orbitals)
        h_1 = x_h1 + y_h1 + z_h1
        h_2 = x_h2 + y_h2 + z_h2

        h1_ints = OneBodyElectronicIntegrals(ElectronicBasis.SO, h_1)
        h2_ints = TwoBodyElectronicIntegrals(ElectronicBasis.SO, h_2)

        op = (h1_ints.to_second_q_op() + h2_ints.to_second_q_op()).reduce()

        if not settings.dict_aux_operators:
            return [op]

        return {self.name: op}

    def interpret(self, result: EigenstateResult) -> None:
        """Interprets an :class:`~qiskit_nature.results.EigenstateResult` in this property's context.

        Args:
            result: the result to add meaning to.
        """
        expected = self.spin
        result.total_angular_momentum = []

        if not isinstance(result.aux_operator_eigenvalues, list):
            aux_operator_eigenvalues = [result.aux_operator_eigenvalues]
        else:
            aux_operator_eigenvalues = result.aux_operator_eigenvalues
        for aux_op_eigenvalues in aux_operator_eigenvalues:
            if aux_op_eigenvalues is None:
                continue

            _key = self.name if isinstance(aux_op_eigenvalues, dict) else 1

            if aux_op_eigenvalues[_key] is not None:
                total_angular_momentum = aux_op_eigenvalues[_key][0].real
                result.total_angular_momentum.append(total_angular_momentum)

                if expected is not None:
                    spin = (-1.0 + np.sqrt(1 + 4 * total_angular_momentum)) / 2
                    if not np.isclose(
                        spin,
                        expected,
                        rtol=self._relative_tolerance,
                        atol=self._absolute_tolerance,
                    ):
                        LOGGER.info(
                            "The measured spin %s does NOT match the expected spin %s!",
                            spin,
                            expected,
                        )
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
