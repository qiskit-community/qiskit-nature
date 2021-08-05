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
cited below). Each chain consists of beads that encode information about the turn that follows
to another main bead (in case of main beads) or into a side bead (in case of side beads).
Moreover, each bead is characterized by a letter which encodes its residue sequence which defines
the energy of interactions with other beads (unless interactions are random). Currently,
only interactions involving first nearest neighbors are supported. Each side chain is
attached to one and only one main bead. Currently, only side chains of length 1 (i.e. with 1
bead) are supported which is a simplifying assumption. A generalization of this approach is
for future investigation. Constraints on feasible folds are incorporated in the objective function
using penalty terms whose importance is regulated by parameters in the PenaltyParameters class.

In the final operator for the problem qubit registers have the following meaning:
(interactions qubits) tensored with (conformation qubits),
which can be further broken down into the following groups:
(main-main beads interactions) tensored with (side-side beads interactions) tensored with
(main-side beads interactions) tensored with (side-main beads interactions) tensored with
(side conformation qubits) tensored with (main conformation qubits).
We build interaction operators according to the following indexing:
lower_bead_position * chain_len + upper_bead_position,
i.e. the position of a block encodes the index of a lower bead and the position in a block
encodes the index of an upper bead.
All qubits are indexed from right to left.

For more details consult the paper Robert et al., npj quantum information 7, 38, 2021
(https://www.nature.com/articles/s41534-021-00368-4).

.. currentmodule:: qiskit_nature.problems.sampling.protein_folding

Protein Folding Problem
=======================
.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   ProteinFoldingProblem

Peptide
=======
.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   Peptide

Main Chain
==========
.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   MainChain

Side Chain
==========
.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   SideChain

Interactions
============
.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   Interaction
   MixedInteraction
   MiyazawaJerniganInteraction
   RandomInteraction

Penalty Parameters
==================
.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   PenaltyParameters

Exceptions
==========
.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   InvalidResidueException
   InvalidSideChainException
   InvalidSizeException
"""
from .exceptions.invalid_residue_exception import InvalidResidueException
from .exceptions.invalid_side_chain_exception import InvalidSideChainException
from .exceptions.invalid_size_exception import InvalidSizeException
from .interactions.interaction import Interaction
from .interactions.mixed_interaction import MixedInteraction
from .interactions.miyazawa_jernigan_interaction import MiyazawaJerniganInteraction
from .interactions.random_interaction import RandomInteraction
from .penalty_parameters import PenaltyParameters
from .peptide.chains.main_chain import MainChain
from .peptide.chains.side_chain import SideChain
from .peptide.peptide import Peptide
from .protein_folding_problem import ProteinFoldingProblem

__all__ = {
    "ProteinFoldingProblem",
    "Peptide",
    "MainChain",
    "SideChain",
    "Interaction",
    "MixedInteraction",
    "MiyazawaJerniganInteraction",
    "RandomInteraction",
    "PenaltyParameters",
    "InvalidResidueException",
    "InvalidSideChainException",
    "InvalidSizeException",
}
