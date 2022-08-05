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

"""The QCSchema error dataclass."""

from __future__ import annotations

from dataclasses import dataclass

from .qc_base import _QCBase


@dataclass
class QCError(_QCBase):
    """A dataclass to store the failure information contained in a QCSchema.

    For more information refer to
    [here](https://molssi-qc-schema.readthedocs.io/en/latest/spec_components.html#success).
    """

    error_type: str
    """The type of error that was raised."""
    error_message: str
    """A description of the error that was raised."""
