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

"""Property Type utilities."""

from typing import Dict, Generator, Generic, Iterable, Optional, Tuple, TypeVar, Union
from qiskit_nature import ListOrDictType

# pylint: disable=invalid-name
T = TypeVar("T")


class ListOrDict(Dict, Iterable, Generic[T]):
    """TODO."""

    def __init__(self, values: Optional[ListOrDictType] = None):
        """TODO."""
        if isinstance(values, list):
            values = dict(enumerate(values))
        elif values is None:
            values = {}
        super().__init__(values)

    def __iter__(self) -> Generator[Tuple[Union[int, str], T], T, None]:
        """Return the generator-iterator method."""
        return self._generator()

    def _generator(self) -> Generator[Tuple[Union[int, str], T], T, None]:
        """TODO."""
        for key, value in self.items():
            new_value = yield (key, value)
            if new_value is not None:
                self[key] = new_value
