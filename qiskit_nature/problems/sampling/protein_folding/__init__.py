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
Protein Folding Problems (:mod:`qiskit_nature.problems.sampling.protein_folding`)
=================================================================================

The protein to be folded is defined in a Peptide class. Each peptide consists of one and only one
main chain and optionally several side chains. Side chains cannot be attached to first, second or
last main bead which is an assumption of the algorithm without loss of generality (see the paper
cited below). A chain consists of beads containing information about their relative position to
other beads:
* main beads reference previous main beads,
* _first_ side beads reference the (branching) main bead,
* other side beads reference previous side beads.
Moreover, each bead is characterized by a letter which encodes its residue sequence which defines
the energy of interactions with other beads (unless interactions are random). Each side chain is
attached to one and only one main bead. Currently, only side chains of length 1 (i.e. with 1
bead) are supported which is a simplifying assumption. A generalization of this approach will be
implemented in the future. Constraints on feasible folds are incorporated in the objective function
using penalty terms whose importance is regulated by parameters in the PenaltyParameters class. For
more details consult the paper Robert et al., npj quantum information 7, 38, 2021
(https://www.nature.com/articles/s41534-021-00368-4).

.. currentmodule:: qiskit_nature.problems.sampling.protein_folding

Peptide Class
=============

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   Peptide

Penalty Parameters Class
========================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   PenaltyParameters

Interactions Class
==================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   MixedInteraction
   MiyazawaJerniganInteraction
   RandomInteraction


"""
from .interactions.mixed_interaction import MixedInteraction
from .interactions.miyazawa_jernigan_interaction import MiyazawaJerniganInteraction
from .interactions.random_interaction import RandomInteraction
from .penalty_parameters import PenaltyParameters
from .peptide.peptide import Peptide
from .protein_folding_problem import ProteinFoldingProblem

__all__ = [
    "ProteinFoldingProblem",
    "Peptide",
    "PenaltyParameters",
    "MixedInteraction",
    "MiyazawaJerniganInteraction",
    "RandomInteraction",
]
