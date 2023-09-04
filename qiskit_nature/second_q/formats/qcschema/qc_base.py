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

"""The QCSchema Base class."""
# pylint: disable=invalid-name

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import json
import h5py


@dataclass
class _QCBase:
    """A base class for the QCSchema dataclasses.

    This base class is used to implement schema-wide conversion utility methods.
    """

    def to_dict(self) -> dict[str, Any]:
        """Converts the schema object to a dictionary.

        Returns:
            The dictionary representation of the schema object.
        """

        def filter_none(d: list[tuple[str, Any]]) -> dict[str, Any]:
            return {k: v for (k, v) in d if v is not None}

        return asdict(self, dict_factory=filter_none)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _QCBase:
        """Constructs a schema object from a dictionary of data.

        The dictionary provided to this method corresponds to the format as obtained by `json.load`
        from a JSON representation of the schema object according to the latest standard as
        documented [here](https://molssi-qc-schema.readthedocs.io/en/latest/).

        Args:
            data: the data dictionary.

        Returns:
            An instance of the schema object.
        """
        return cls(**data)

    def to_json(self) -> str:
        """Converts the schema object to JSON.

        Returns:
            The JSON representation of the schema object.
        """
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_data: str | bytes | Path) -> _QCBase:
        """Constructs a schema object from JSON.

        The JSON data must match the latest standard as documented
        [here](https://molssi-qc-schema.readthedocs.io/en/latest/).

        Args:
            json_data: can be either the path to a file or the json data directly provided as a `str`.

        Returns:
            An instance of the schema object.
        """
        try:
            return cls.from_dict(json.loads(json_data))  # type: ignore[arg-type]
        except json.JSONDecodeError:
            with open(json_data, "r", encoding="utf8") as file:
                return cls.from_dict(json.load(file))

    def to_hdf5(self, group: h5py.Group) -> None:
        """Converts the schema object to HDF5.

        Args:
            group: the h5py group into which to store the object.
        """
        # we use __dict__ here because we do not want the recursive behavior of asdict()
        for key, value in self.__dict__.items():
            if value is None:
                continue
            if hasattr(value, "to_hdf5"):
                inner_group = group.require_group(key)
                value.to_hdf5(inner_group)
            else:
                group.attrs[key] = value

    @classmethod
    def from_hdf5(cls, h5py_data: str | Path | h5py.Group) -> _QCBase:
        """Constructs a schema object from an HDF5 object.

        While the QCSchema is officially tailored to support JSON, HDF5 is supported as a more
        high-performance alternative and considered the standard within Qiskit Nature. Due to its
        similarities with JSON a 1-to-1 correspondence can be made between the two.

        For more details refer to
        [here](https://molssi-qc-schema.readthedocs.io/en/latest/tech_discussion.html#json-and-hdf5).

        Args:
            h5py_data: can be either the path to a file or an `h5py.Group`.

        Returns:
            An instance of the schema object.
        """
        if isinstance(h5py_data, h5py.Group):
            return cls._from_hdf5_group(h5py_data)

        with h5py.File(h5py_data, "r") as file:
            return cls._from_hdf5_group(file)

    @classmethod
    def _from_hdf5_group(cls, h5py_group: h5py.Group) -> _QCBase:
        """This internal method deals with actually constructing a schema object from an `h5py.Group`.

        Args:
            h5py_group: the actual `h5py.Group`.

        Returns:
            An instance of the schema object.
        """
        data = dict(h5py_group.attrs.items())
        for key, value in h5py_group.items():
            data[key] = value[...]
        return cls(**data)
