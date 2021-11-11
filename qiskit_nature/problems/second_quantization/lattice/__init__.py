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
Lattice Problems (:mod:`qiskit_nature.problems.second_quantization.lattice`)
=============================================================================
The `Lattice` class is used to generate a general lattice,
which is a graph with complex-valued weights on its edges.
The `HyperCubicLattice` class and others provide standard lattices with certain
translational symmetries.
`Models` include classes that define Hamiltonians on a lattice.
It takes a lattice object and model parameters as input and
generates a Hamiltonian defined on the lattice.

.. currentmodule:: qiskit_nature.problems.second_quantization.lattice

Lattice
==============
.. autosummary::
   :toctree: ../stubs/

   Lattice
   HyperCubicLattice
   LineLattice
   SquareLattice
   TriangularLattice

Models
=======
.. autosummary::
   :toctree: ../stubs/

   FermiHubbardModel

"""
from .lattice import (
    DrawStyle,
    HyperCubicLattice,
    Lattice,
    LineLattice,
    SquareLattice,
    TriangularLattice,
)
from .models import FermiHubbardModel
