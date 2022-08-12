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

"""The expected water output v3 QCSchema."""

from copy import deepcopy

from .water_output import EXPECTED

EXPECTED = deepcopy(EXPECTED)

EXPECTED.schema_version = 3
EXPECTED.wavefunction.restricted = True
