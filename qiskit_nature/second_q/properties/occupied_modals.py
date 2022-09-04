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

"""The OccupiedModals property."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import h5py

from qiskit_nature.second_q.operators import VibrationalOp

from .bases import VibrationalBasis
from .property import Property

if TYPE_CHECKING:
    from qiskit_nature.second_q.problems import EigenstateResult


class OccupiedModals(Property):
    """The OccupiedModals property."""

    def __init__(
        self,
        basis: Optional[VibrationalBasis] = None,
    ) -> None:
        """
        Args:
            basis: the
                :class:`~qiskit_nature.second_q.properties.bases.VibrationalBasis`
                through which to map the integrals into second quantization. This attribute **MUST**
                be set before the second-quantized operator can be constructed.
        """
        super().__init__(self.__class__.__name__)
        self._basis: VibrationalBasis = basis

    @property
    def basis(self) -> VibrationalBasis:
        """Returns the basis."""
        return self._basis

    @basis.setter
    def basis(self, basis: VibrationalBasis) -> None:
        """Sets the basis."""
        self._basis = basis

    def __str__(self) -> str:
        string = [super().__str__() + ":"]
        string += [f"\t{line}" for line in str(self.basis).split("\n")]
        return "\n".join(string)

    @staticmethod
    def from_hdf5(h5py_group: h5py.Group) -> OccupiedModals:
        # pylint: disable=unused-argument
        """Constructs a new instance from the data stored in the provided HDF5 group.

        See also :func:`~qiskit_nature.hdf5.HDF5Storable.from_hdf5` for more details.

        Args:
            h5py_group: the HDF5 group from which to load the data.

        Returns:
            A new instance of this class.
        """
        return OccupiedModals()

    def second_q_ops(self) -> dict[str, VibrationalOp]:
        """Returns the second quantized operators indicating the occupied modals per mode.

        Returns:
            A `dict` of `VibrationalOp` objects.
        """
        num_modals_per_mode = self.basis._num_modals_per_mode
        num_modes = len(num_modals_per_mode)

        return {str(mode): self._get_mode_op(mode) for mode in range(num_modes)}

    def _get_mode_op(self, mode: int) -> VibrationalOp:
        """Constructs an operator to evaluate which modal of a given mode is occupied.

        Args:
            mode: the mode index.

        Returns:
            The operator to evaluate which modal of the given mode is occupied.
        """
        num_modals_per_mode = self.basis._num_modals_per_mode

        labels: list[tuple[str, complex]] = []

        for modal in range(num_modals_per_mode[mode]):
            labels.append((f"+_{mode}*{modal} -_{mode}*{modal}", 1.0))

        return VibrationalOp(labels, len(num_modals_per_mode), num_modals_per_mode)

    def interpret(self, result: "EigenstateResult") -> None:
        """Interprets an :class:`~qiskit_nature.second_q.problems.EigenstateResult`
        in this property's context.

        Args:
            result: the result to add meaning to.
        """
        result.num_occupied_modals_per_mode = []

        if not isinstance(result.aux_operator_eigenvalues, list):
            aux_operator_eigenvalues = [result.aux_operator_eigenvalues]
        else:
            aux_operator_eigenvalues = result.aux_operator_eigenvalues

        num_modes = len(self._basis._num_modals_per_mode)

        for aux_op_eigenvalues in aux_operator_eigenvalues:
            occ_modals = []
            for mode in range(num_modes):
                _key = str(mode) if isinstance(aux_op_eigenvalues, dict) else mode
                if aux_op_eigenvalues[_key] is not None:
                    occ_modals.append(aux_op_eigenvalues[_key][0].real)
                else:
                    occ_modals.append(None)
            result.num_occupied_modals_per_mode.append(occ_modals)
