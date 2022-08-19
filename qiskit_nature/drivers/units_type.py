# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
This module declares the Unit Types.
"""

from ..deprecation import (
    warn_deprecated,
    DeprecatedType,
    NatureDeprecationWarning,
    DeprecatedEnum,
    DeprecatedEnumMeta,
)


class UnitsType(DeprecatedEnum, metaclass=DeprecatedEnumMeta):
    """Units Type Enum"""

    ANGSTROM = "Angstrom"
    BOHR = "Bohr"

    def deprecate(self):
        """show deprecate message"""
        warn_deprecated(
            "0.5.0",
            old_type=DeprecatedType.ENUM,
            old_name="qiskit_nature.drivers.UnitsType",
            new_type=DeprecatedType.ENUM,
            new_name="qiskit_nature.second_q.drivers.UnitsType",
            stack_level=3,
            category=NatureDeprecationWarning,
        )
