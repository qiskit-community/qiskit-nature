# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.#
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utility functions for tests in operators."""


def str2str(string: str):
    """Construct the string data for SecondQuantizedOp.
    This function does not change the argument."""
    return string


def str2tuple(string: str):
    """Construct the tuple data from string for SecondQuantizedOp."""
    return string, 1


def str2list(string: str):
    """Construct the list data from string for SecondQuantizedOp."""
    return [(string, 1)]
