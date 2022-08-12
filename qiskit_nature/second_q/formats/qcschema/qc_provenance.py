# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The QCSchema provenance dataclass."""

from __future__ import annotations

from dataclasses import dataclass

from .qc_base import _QCBase


@dataclass
class QCProvenance(_QCBase):
    """A dataclass to store the program information that generated the QCSchema file.

    For more information refer to
    [here](https://molssi-qc-schema.readthedocs.io/en/latest/spec_components.html#provenance).
    """

    creator: str
    """The name of the creator of this object."""
    version: str
    """The version of the creator of this object."""
    routine: str
    """The routine that was used to create this object."""
