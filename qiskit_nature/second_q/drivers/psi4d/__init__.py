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

"""
Psi4 Installation
=================
`Psi4 <http://www.psicode.org/>`__ is an open-source program for computational chemistry.
In order for Qiskit Nature to interface with Psi4, i.e. execute Psi4 to extract
the electronic structure information necessary for the computation of the input to the quantum
algorithm, Psi4 must be `installed <http://www.psicode.org/downloads.html>`__ and discoverable on
the system where Qiskit Nature is also installed.
Therefore, Psi4 must be installed in the same python environment as Qiskit Nature.
"""

from .psi4driver import Psi4Driver

__all__ = ["Psi4Driver"]
