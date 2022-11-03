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

"""
Formats (:mod:`qiskit_nature.second_q.formats`)
===================================================

.. currentmodule:: qiskit_nature.second_q.formats

Submodules
==========

.. autosummary::
   :toctree:

   fcidump
   qcschema
   watson


.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   MoleculeInfo
   fcidump_to_problem
   watson_to_problem
   qcschema_to_problem
   get_ao_to_mo_from_qcschema

"""

from .molecule_info import MoleculeInfo
from .fcidump_translator import fcidump_to_problem
from .qcschema_translator import qcschema_to_problem, get_ao_to_mo_from_qcschema
from .watson_translator import watson_to_problem

__all__ = [
    "MoleculeInfo",
    "fcidump_to_problem",
    "watson_to_problem",
    "qcschema_to_problem",
    "get_ao_to_mo_from_qcschema",
]
