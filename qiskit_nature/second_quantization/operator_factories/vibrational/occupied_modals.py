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

from typing import Optional

import h5py

from qiskit_nature import ListOrDictType, settings
from qiskit_nature.drivers import WatsonHamiltonian
from qiskit_nature.second_quantization.operators import VibrationalOp
from qiskit_nature.second_quantization.problems import EigenstateResult

from ..second_quantized_property import LegacyDriverResult
from .bases import VibrationalBasis
from .types import VibrationalProperty
from ....deprecation import deprecate_method


class OccupiedModals(VibrationalProperty):
    """The OccupiedModals property."""

    def __init__(
        self,
        basis: Optional[VibrationalBasis] = None,
    ) -> None:
        """
        Args:
            basis: the
                :class:`~qiskit_nature.properties.second_quantization.vibrational.bases.VibrationalBasis`
                through which to map the integrals into second quantization. This attribute **MUST**
                be set before the second-quantized operator can be constructed.
        """
        super().__init__(self.__class__.__name__, basis)

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

    @classmethod
    @deprecate_method("0.4.0")
    def from_legacy_driver_result(cls, result: LegacyDriverResult) -> OccupiedModals:
        """Construct an OccupiedModals instance from a
        :class:`~qiskit_nature.drivers.WatsonHamiltonian`.

        Args:
            result: the driver result from which to extract the raw data. For this property, a
                :class:`~qiskit_nature.drivers.WatsonHamiltonian` is required!

        Returns:
            An instance of this property.

        Raises:
            QiskitNatureError: if a :class:`~qiskit_nature.drivers.QMolecule` is provided.
        """
        cls._validate_input_type(result, WatsonHamiltonian)

        return cls()

    def second_q_ops(self) -> ListOrDictType[VibrationalOp]:
        """Returns the second quantized operators indicating the occupied modals per mode.

        The actual return-type is determined by `qiskit_nature.settings.dict_aux_operators`.

        Returns:
            A `list` or `dict` of `VibrationalOp` objects.
        """
        num_modals_per_mode = self.basis._num_modals_per_mode
        num_modes = len(num_modals_per_mode)

        if not settings.dict_aux_operators:
            return [self._get_mode_op(mode) for mode in range(num_modes)]

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

    def interpret(self, result: EigenstateResult) -> None:
        """Interprets an :class:`~qiskit_nature.results.EigenstateResult` in this property's context.

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
