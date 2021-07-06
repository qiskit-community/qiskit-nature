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

"""General CompositeProperty base class tests."""

from test import QiskitNatureTestCase

from qiskit_nature.properties import CompositeProperty
from qiskit_nature.properties.second_quantization.electronic.integrals import IntegralProperty


class TestCompositeProperty(QiskitNatureTestCase):
    """General CompositeProperty base class tests."""

    def setUp(self):
        """Setup."""
        super().setUp()
        self.dummy_prop_1 = IntegralProperty("Dummy 1", [])
        self.dummy_prop_2 = IntegralProperty("Dummy 2", [])

        self.prop = CompositeProperty("Dummy Composite")
        self.prop.add_property(self.dummy_prop_1)
        self.prop.add_property(self.dummy_prop_2)

    def test_init(self):
        """Test construction."""
        self.assertEqual(
            self.prop._properties, {"Dummy 1": self.dummy_prop_1, "Dummy 2": self.dummy_prop_2}
        )

    def test_add_property(self):
        """Test add_property."""
        dummy_prop = IntegralProperty("IntegralProperty", [])
        prop = CompositeProperty("Dummy Composite")
        prop.add_property(dummy_prop)
        self.assertEqual(prop._properties, {"IntegralProperty": dummy_prop})

    def test_get_property(self):
        """Test get_property."""
        dummy_prop = IntegralProperty("IntegralProperty", [])
        prop = CompositeProperty("Dummy Composite")
        prop.add_property(dummy_prop)
        self.assertEqual(prop.get_property("IntegralProperty"), dummy_prop)
        self.assertEqual(prop.get_property(IntegralProperty), dummy_prop)

    def test_iter(self):
        """Test iteration.

        This method also asserts that the Iterator is indeed a Generator which supports `send`.
        """
        with self.subTest("Iterator 1"):
            prop_list = list(self.prop.__iter__())
            self.assertEqual(prop_list, [self.dummy_prop_1, self.dummy_prop_2])

        with self.subTest("Iterator 2"):
            prop_list = list(iter(self.prop))
            self.assertEqual(prop_list, [self.dummy_prop_1, self.dummy_prop_2])

        with self.subTest("Generator"):
            expected = iter([self.dummy_prop_1, self.dummy_prop_2])
            iterator = iter(self.prop)

            prop = None
            while True:
                try:
                    prop = iterator.send(prop)
                    exp = next(expected)
                except StopIteration:
                    break

                self.assertEqual(prop, exp)
                prop._electronic_integrals = ["Test"]

            self.assertEqual(self.prop.get_property("Dummy 1")._electronic_integrals, ["Test"])
            self.assertEqual(self.prop.get_property("Dummy 2")._electronic_integrals, ["Test"])
