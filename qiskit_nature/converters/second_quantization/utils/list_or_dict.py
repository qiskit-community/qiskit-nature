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

"""The ListOrDict utility class."""

from typing import Dict, Generator, Generic, Iterable, Optional, Tuple, TypeVar, Union
from qiskit_nature import ListOrDictType

# pylint: disable=invalid-name
T = TypeVar("T")


class ListOrDict(Dict, Iterable, Generic[T]):
    """The ListOrDict utility class.

    This is a utility which allows seamless iteration of a `list` or `dict` object.
    This is mostly used for dealing with `aux_operators` which support both types since Qiskit Terra
    version 0.19.0
    """

    def __init__(self, values: Optional[ListOrDictType] = None):
        """
        Args:
            values: an optional object of `list` or `dict` type.
        """
        if isinstance(values, list):
            values = dict(enumerate(values))
        elif values is None:
            values = {}
        super().__init__(values)

    def __iter__(self) -> Generator[Tuple[Union[int, str], T], T, None]:
        """Return the generator-iterator method."""
        return self._generator()

    def _generator(self) -> Generator[Tuple[Union[int, str], T], T, None]:
        """Return generator method iterating the contents of this class.

        This generator yields the `(key, value)` pairs of the underlying dictionary. If this object
        was constructed from a list, the keys in this generator are simply the numeric indices.

        This generator also supports overriding the yielded value upon receiving any value other
        than `None` from a `send` [1] instruction.

        [1]: https://docs.python.org/3/reference/expressions.html#generator.send
        """
        for key, value in self.items():
            new_value = yield (key, value)
            if new_value is not None:
                self[key] = new_value
