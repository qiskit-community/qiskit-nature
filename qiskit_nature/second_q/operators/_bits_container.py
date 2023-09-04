# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A bit-storage container."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Generic, TypeVar

# pylint: disable=invalid-name
T = TypeVar("T")


class _BitsContainer(MutableMapping, Generic[T]):
    """A bit-storage container.

    This is a utility object used during the simplification process of a operators.
    It manages access to an internal data container, which maps from integers to bytes.
    Each integer key corresponds to a vibrational mode of an operator term.
    Each value consists of four bits encoding for the corresponding key:

        1. if a `+` has been applied
        2. if a `-` has been applied
        3. whether a `+` or `-` was applied first
        4. whether the last applied operator was a `+` or `-`.
    """

    def __init__(self) -> None:
        self.data: dict[T, int] = {}

    def get_plus(self, index: T) -> int:
        """Returns the value of the `+`-register.

        Args:
            index: the internal data key (corresponding to the vibrational mode).

        Returns:
            1 if `+` has been applied, 0 otherwise.
        """
        return self.get_bit(index, 3)

    def get_minus(self, index: T) -> int:
        """Returns the value of the `-`-register.

        Args:
            index: the internal data key (corresponding to the vibrational mode).

        Returns:
            1 if `-` has been applied, 0 otherwise.
        """
        return self.get_bit(index, 2)

    def set_plus_or_minus(self, index: T, plus_or_minus: bool, value: bool) -> None:
        """Sets the `+`- or `-`-register of the provided index to the provided value.

        Args:
            index: the internal data key (corresponding to the vibrational mode).
            plus_or_minus: True if the `+`-register is to be set, False for the `-`-register
            value: True if the register is to be set to 1, False for 0.
        """
        if value:
            # plus is stored at index 0, but plus_or_minus is True if it is Plus
            self.set_bit(index, 3 - int(not plus_or_minus))
        else:
            self.clear_bit(index, 3 - int(not plus_or_minus))

    def get_order(self, index: T) -> int:
        """Returns the value of the order-register.

        Note: the order-register is read-only and can only be set during initialization.

        Args:
            index: the internal data key (corresponding to the vibrational mode).

        Returns:
            1 if `+` was applied first, 0 if `-` was applied first.
        """
        return self.get_bit(index, 1)

    def get_last(self, index: T) -> int:
        """Returns the value of the last-register.

        Args:
            index: the internal data key (corresponding to the vibrational mode).

        Returns:
            1 if `+` was applied last, 0 otherwise.
        """
        return self.get_bit(index, 0)

    def set_last(self, index: T, value: bool) -> None:
        """Sets the value of the last-register.

        Args:
            index: the internal data key (corresponding to the vibrational mode).
            value: True if the register is to be set to 1, False for 0.
        """
        if value:
            self.set_bit(index, 0)
        else:
            self.clear_bit(index, 0)

    def get_bit(self, index: T, offset: int) -> int:
        """Returns the value of a requested register.

        Args:
            index: the internal data key (corresponding to the vibrational mode).
            offset: the bit-wise offset for the bit-shift operation to obtain the desired register.

        Returns:
            1 if the register was set, 0 otherwise.
        """
        return (self.data[index] >> offset) & 1

    def set_bit(self, index: T, offset: int) -> None:
        """Sets the provided register to 1.

        Args:
            index: the internal data key (corresponding to the vibrational mode).
            offset: the bit-wise offset for the bit-shift operation to set the desired register.
        """
        self.data[index] = self.data[index] | (1 << offset)

    def clear_bit(self, index: T, offset: int) -> None:
        """Clears the provided register (to 0).

        Args:
            index: the internal data key (corresponding to the vibrational mode).
            offset: the bit-wise offset for the bit-shift operation to set the desired register.
        """
        self.data[index] = self.data[index] & ~(1 << offset)

    def __getitem__(self, __k):
        return self.data.__getitem__(__k)

    def __setitem__(self, __k, __v):
        return self.data.__setitem__(__k, __v)

    def __delitem__(self, __v):
        return self.data.__delitem__(__v)

    def __iter__(self):
        return self.data.__iter__()

    def __len__(self):
        return self.data.__len__()
