# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
DEPRECATED Qiskit Nature Runtime (:mod:`qiskit_nature.runtime`)
===============================================================

.. currentmodule:: qiskit_nature.runtime

.. warning::
    This entire module is deprecated as of version 0.6.0 and will be removed no sooner than 3 months
    after the release. Instead you should update your code to use the Qiskit Runtime Primitives. For
    more details on how to migrate check out this guide, here: https://qisk.it/algo_migration#vqe !

Programs that embed Qiskit Runtime in the algorithmic interfaces and facilitate usage of
algorithms and scripts in the cloud.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   VQEClient
   VQERuntimeResult

"""

from .vqe_client import VQEClient, VQERuntimeResult

__all__ = ["VQEClient", "VQERuntimeResult"]
