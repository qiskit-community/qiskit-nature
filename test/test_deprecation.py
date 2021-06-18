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

"""
Test The deprecation methods
"""

import unittest
import warnings
from test import QiskitNatureTestCase
from ddt import data, ddt
from qiskit_nature.deprecation import (
    DeprecatedType,
    warn_deprecated,
    warn_deprecated_same_type_name,
    deprecate_function,
    deprecate_method,
    deprecate_arguments,
)

# pylint: disable=bad-docstring-quotes


@deprecate_function("0.1.1", DeprecatedType.FUNCTION, "some_function1", "and more information", 2)
def func1(arg1):
    """function 1"""
    del arg1
    pass


@deprecate_function("0.2.0", DeprecatedType.FUNCTION, "some_function2")
def func2(arg2):
    """function 2"""
    del arg2
    pass


@deprecate_arguments("0.1.2", {"old_arg": "new_arg"})
def func3(new_arg=None, old_arg=None):
    """function 3"""
    del new_arg, old_arg
    pass


class DeprecatedClass1:
    """Deprecated Test class 1"""

    def __init__(self):
        warn_deprecated(
            "0.3.0", DeprecatedType.CLASS, "DeprecatedClass1", DeprecatedType.CLASS, "NewClass"
        )


class DeprecatedClass2:
    """Deprecated Test class 2"""

    def __init__(self):
        warn_deprecated_same_type_name(
            "0.3.0", DeprecatedType.CLASS, "DeprecatedClass2", "from package test2"
        )


class TestClass:
    """Test class with deprecation"""

    @deprecate_method(
        "0.1.0", DeprecatedType.METHOD, "some_method1", "and additional information", 1
    )
    def method1(self):
        """method 1"""
        pass

    @deprecate_method("0.2.0", DeprecatedType.METHOD, "some_method2")
    def method2(self):
        """method 2"""
        pass

    @deprecate_arguments("0.1.2", {"old_arg": "new_arg"})
    def method3(self, new_arg=None, old_arg=None):
        """method3"""
        del new_arg, old_arg
        pass


@ddt
class TestDeprecation(QiskitNatureTestCase):
    """Test deprecation methods"""

    @data(
        (
            "func1",
            "The func1 function is deprecated as of version 0.1.1 "
            "and will be removed no sooner than 3 months after the release. "
            "Instead use the some_function1 function and more information.",
        ),
        (
            "func2",
            "The func2 function is deprecated as of version 0.2.0 "
            "and will be removed no sooner than 3 months after the release. "
            "Instead use the some_function2 function.",
        ),
    )
    def test_function_deprecation(self, config):
        """test function deprecation"""

        function_name, msg_ref = config

        # emit deprecation the first time it is used
        with warnings.catch_warnings(record=True) as c_m:
            warnings.simplefilter("always")
            globals()[function_name](None)
            msg = str(c_m[0].message)
            self.assertEqual(msg, msg_ref)

        # trying again should not emit deprecation
        with warnings.catch_warnings(record=True) as c_m:
            warnings.simplefilter("always")
            globals()[function_name](None)
            self.assertListEqual(c_m, [])

    def test_class_deprecation1(self):
        """test class deprecation 1"""

        msg_ref = (
            "The DeprecatedClass1 class is deprecated as of version 0.3.0 "
            "and will be removed no sooner than 3 months after the release. "
            "Instead use the NewClass class."
        )

        # emit deprecation the first time it is used
        with warnings.catch_warnings(record=True) as c_m:
            warnings.simplefilter("always")
            DeprecatedClass1()
            msg = str(c_m[0].message)
            self.assertEqual(msg, msg_ref)

        # trying again should not emit deprecation
        with warnings.catch_warnings(record=True) as c_m:
            warnings.simplefilter("always")
            DeprecatedClass1()
            self.assertListEqual(c_m, [])

    def test_class_deprecation2(self):
        """test class deprecation 2"""

        msg_ref = (
            "The DeprecatedClass2 class is deprecated as of version 0.3.0 "
            "and will be removed no sooner than 3 months after the release. "
            "Instead use the DeprecatedClass2 class from package test2."
        )

        # emit deprecation the first time it is used
        with warnings.catch_warnings(record=True) as c_m:
            warnings.simplefilter("always")
            DeprecatedClass2()
            msg = str(c_m[0].message)
            self.assertEqual(msg, msg_ref)

        # trying again should not emit deprecation
        with warnings.catch_warnings(record=True) as c_m:
            warnings.simplefilter("always")
            DeprecatedClass2()
            self.assertListEqual(c_m, [])

    @data(
        (
            "method1",
            "The method1 method is deprecated as of version 0.1.0 "
            "and will be removed no sooner than 3 months after the release. "
            "Instead use the some_method1 method and additional information.",
        ),
        (
            "method2",
            "The method2 method is deprecated as of version 0.2.0 "
            "and will be removed no sooner than 3 months after the release. "
            "Instead use the some_method2 method.",
        ),
    )
    def test_method_deprecation(self, config):
        """test method deprecation"""

        method_name, msg_ref = config
        method = getattr(TestClass(), method_name)

        # emit deprecation the first time it is used
        with warnings.catch_warnings(record=True) as c_m:
            warnings.simplefilter("always")
            method()
            msg = str(c_m[0].message)
            self.assertEqual(msg, msg_ref)

        # trying again should not emit deprecation
        with warnings.catch_warnings(record=True) as c_m:
            warnings.simplefilter("always")
            method()
            self.assertListEqual(c_m, [])

    def test_function_arguments_deprecation(self):
        """test function arguments deprecation"""

        msg_ref = (
            "func3: the old_arg argument is deprecated as of version 0.1.2 "
            "and will be removed no sooner than 3 months after the release. "
            "Instead use the new_arg argument."
        )
        # both arguments at the same time should raise exception
        with self.assertRaises(TypeError):
            func3(new_arg="2222", old_arg="hello")

        with warnings.catch_warnings(record=True) as c_m:
            warnings.simplefilter("always")
            func3(old_arg="hello")
            msg = str(c_m[0].message)
            self.assertEqual(msg, msg_ref)

    def test_method_arguments_deprecation(self):
        """test method arguments deprecation"""

        obj = TestClass()

        msg_ref = (
            "method3: the old_arg argument is deprecated as of version 0.1.2 "
            "and will be removed no sooner than 3 months after the release. "
            "Instead use the new_arg argument."
        )
        # both arguments at the same time should raise exception
        with self.assertRaises(TypeError):
            obj.method3(new_arg="2222", old_arg="hello")

        with warnings.catch_warnings(record=True) as c_m:
            warnings.simplefilter("always")
            obj.method3(old_arg="hello")
            msg = str(c_m[0].message)
            self.assertEqual(msg, msg_ref)


if __name__ == "__main__":
    unittest.main()
