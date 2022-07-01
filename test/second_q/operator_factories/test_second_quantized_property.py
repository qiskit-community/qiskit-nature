# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""General SecondQuantizedProperty base class tests."""

from typing import Any, Union
import warnings

from test import QiskitNatureTestCase
from ddt import data, ddt, unpack

from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.drivers import QMolecule, WatsonHamiltonian
from qiskit_nature.second_q.operator_factories.second_quantized_property import (
    SecondQuantizedProperty,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)


@ddt
class TestSecondQuantizedProperty(QiskitNatureTestCase):
    """General Property base class tests."""

    LegacyDriverResult = Union[QMolecule, WatsonHamiltonian]

    @unpack
    @data(
        (QMolecule(), QMolecule, False),
        (QMolecule(), WatsonHamiltonian, True),
        (WatsonHamiltonian([], -1), QMolecule, True),
        (WatsonHamiltonian([], -1), WatsonHamiltonian, False),
    )
    def test_validate_input_type(
        self, result: LegacyDriverResult, type_: Any, raises_: bool
    ) -> None:
        """Test input type validation."""
        raised = False
        try:
            SecondQuantizedProperty._validate_input_type(result, type_)
        except QiskitNatureError:
            raised = True
        finally:
            self.assertEqual(raised, raises_)
