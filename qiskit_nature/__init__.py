# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2018, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
r"""
=============================================
Qiskit Nature module (:mod:`qiskit_nature`)
=============================================

.. currentmodule:: qiskit_nature

Qiskit Nature provides function to experiment with quantum computing for natural
science problems, such as in chemistry and physics. For example computing the ground state energy
or excited state energies of molecules.

The top-level classes and submodules of qiskit_nature are:

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   QiskitNatureError
   UnsupportMethodError

Globals
=======

.. autosummary::
   :toctree:

   logging
   settings

Submodules
==========

.. autosummary::
   :toctree:

   second_q
   testing
   utils

"""

from qiskit_algorithms.list_or_dict import ListOrDict as ListOrDictType

from .exceptions import QiskitNatureError, UnsupportMethodError
from .logging import logging
from .settings import settings
from .version import __version__

__all__ = [
    "__version__",
    "ListOrDictType",
    "QiskitNatureError",
    "UnsupportMethodError",
    "logging",
    "settings",
]
