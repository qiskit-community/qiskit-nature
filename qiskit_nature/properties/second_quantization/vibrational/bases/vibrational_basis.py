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

"""The Vibrational basis base class."""

from abc import ABC, abstractmethod
from typing import List, Optional

import h5py

from qiskit_nature.properties import PseudoProperty


class VibrationalBasis(PseudoProperty, ABC):
    """The Vibrational basis base class.

    This class defines the interface which any vibrational basis must implement. A basis must be
    applied to the vibrational integrals in order to map them into a second-quantization form. Refer
    to the documentation of :class:`~qiskit_nature.properties.vibrational.integrals` for more details.
    """

    def __init__(
        self,
        num_modals_per_mode: List[int],
        threshold: float = 1e-6,
    ) -> None:
        """
        Args:
            num_modals_per_mode: the number of modals to be used for each mode.
            threshold: the threshold value below which an integral coefficient gets neglected.
        """
        super().__init__(self.__class__.__name__)
        self._num_modals_per_mode = num_modals_per_mode
        self._threshold = threshold

    @property
    def num_modals_per_mode(self) -> List[int]:
        """Returns the number of modals per mode."""
        return self._num_modals_per_mode

    def __str__(self) -> str:
        string = [self.__class__.__name__ + ":"]
        string += [f"\tModals: {self._num_modals_per_mode}"]
        return "\n".join(string)

    def to_hdf5(self, parent: h5py.Group):
        """TODO."""
        super().to_hdf5(parent)
        group = parent.require_group(self.name)

        group.attrs["threshold"] = self._threshold
        group.create_dataset("num_modals_per_mode", data=self.num_modals_per_mode)

    @classmethod
    def from_hdf5(cls, h5py_group: h5py.Group) -> "VibrationalBasis":
        """TODO."""
        return cls(h5py_group["num_modals_per_mode"][...], h5py_group.attrs.get("threshold", None))

    @abstractmethod
    def eval_integral(
        self,
        mode: int,
        modal_1: int,
        modal_2: int,
        power: int,
        kinetic_term: bool = False,
    ) -> Optional[float]:
        """The integral evaluation method of this basis.

        Args:
            mode: the index of the mode.
            modal_1: the index of the first modal.
            modal_2: the index of the second modal.
            power: the exponent of the coordinate.
            kinetic_term: if this is True, the method should compute the integral of the kinetic
                term of the vibrational Hamiltonian, :math:``d^2/dQ^2``.

        Returns:
            The evaluated integral for the specified coordinate or ``None`` if this integral value
            falls below the threshold.

        Raises:
            ValueError: if an unsupported parameter is supplied.
        """
