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

"""General Property base class tests."""

from typing import Any

from test import QiskitNatureTestCase
from ddt import data, ddt, unpack

from qiskit_nature import QiskitNatureError
from qiskit_nature.drivers import QMolecule as LegacyQMolecule
from qiskit_nature.drivers import WatsonHamiltonian as LegacyWatsonHamiltonian
from qiskit_nature.drivers.second_quantization import QMolecule, WatsonHamiltonian
from qiskit_nature.properties.property import (
    DriverResult,
    ElectronicDriverResult,
    Property,
    VibrationalDriverResult,
)


@ddt
class TestProperty(QiskitNatureTestCase):
    """General Property base class tests."""

    @unpack
    @data(
        (QMolecule(), ElectronicDriverResult, False),
        (QMolecule(), VibrationalDriverResult, True),
        (LegacyQMolecule(), ElectronicDriverResult, False),
        (LegacyQMolecule(), VibrationalDriverResult, True),
        (WatsonHamiltonian([], -1), ElectronicDriverResult, True),
        (WatsonHamiltonian([], -1), VibrationalDriverResult, False),
        (LegacyWatsonHamiltonian([], -1), ElectronicDriverResult, True),
        (LegacyWatsonHamiltonian([], -1), VibrationalDriverResult, False),
    )
    def test_validate_input_type(self, result: DriverResult, type_: Any, raises_: bool) -> None:
        """Test input type validation."""
        raised = False
        try:
            Property._validate_input_type(result, type_)
        except QiskitNatureError:
            raised = True
        finally:
            assert raised == raises_
