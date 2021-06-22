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
last main beads. Each chain consists of beads that encode information about the turn that follows
to another main bead (in case of main beads) or into a side bead (in case of side beads).
Moreover, each bead is characterized by a letter which encodes its residue sequence that defines
energy of interactions with other beads (unless interactions are random). Each side chain is
attached to one and only one main bead. Currently, only side chains of length 1 (i.e. with 1
bead) are supported. For more details consult the paper https://arxiv.org/pdf/1908.02163.pdf.

.. currentmodule:: qiskit_nature.problems.sampling.protein_folding
"""

from .protein_folding_problem import ProteinFoldingProblem

__all__ = [
    "ProteinFoldingProblem",
]
