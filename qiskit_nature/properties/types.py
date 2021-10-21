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

from typing import Dict, Generator, Generic, Iterable, List, Optional, Tuple, TypeVar, Union

# pylint: disable=invalid-name
T = TypeVar("T")
ListOrDictType = Union[List[Optional[T]], Dict[Union[int, str], T]]


class ListOrDict(Dict, Iterable, Generic[T]):
    """TODO."""

    def __init__(self, values: Optional[ListOrDictType] = None):
        """TODO."""
        self._main_key: Union[int, str] = 0
        if isinstance(values, list):
            values = dict(enumerate(values))
        elif values is None:
            values = {}
        super().__init__(values)

    @property
    def main(self) -> Optional[T]:
        """TODO."""
        return self[self._main_key]

    @property
    def main_key(self) -> Union[int, str]:
        """TODO."""
        return self._main_key

    @main_key.setter
    def main_key(self, k: Union[int, str]) -> None:
        """TODO."""
        self._main_key = k

    def __iter__(self) -> Generator[Tuple[Union[int, str], T], T, None]:
        """Return the generator-iterator method."""
        return self._generator()

    def _generator(self) -> Generator[Tuple[Union[int, str], T], T, None]:
        """TODO."""
        for key, value in self.items():
            new_value = yield (key, value)
            if new_value is not None:
                self[key] = new_value
