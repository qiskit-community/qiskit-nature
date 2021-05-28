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
from problems.sampling.protein_folding.peptide.peptide import Peptide
from qiskit_nature.problems.sampling.folding.folding_qubit_op_builder import _build_qubit_op
from qiskit_nature.problems.sampling.protein_folding.interactions.interaction import Interaction
from qiskit_nature.problems.sampling.protein_folding.penalties import Penalties
from qiskit_nature.problems.sampling.sampling_problem import SamplingProblem


class ProteinFoldingProblem(SamplingProblem):

    def __init__(self, peptide: Peptide, interaction: Interaction, penalties: Penalties):
        self._peptide = peptide
        self._interaction = interaction
        self._penalties = penalties
        self._pair_energies = interaction.calc_energy_matrix(len(peptide.get_main_chain),
                                                             peptide.get_main_chain.get_residue_sequence())

    def qubit_op(self):
        return _build_qubit_op()

    def interpret(self):
        pass
