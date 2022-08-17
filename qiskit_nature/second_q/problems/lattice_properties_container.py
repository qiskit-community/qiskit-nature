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

"""The lattice properties container."""

from __future__ import annotations

from .properties_container import PropertiesContainer


class LatticePropertiesContainer(PropertiesContainer):
    """The container class for lattice structure properties.

    Right now, this is simply an empty subclass, but lattice-specific properties might be exposed
    as attributes in the future.
    """
