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
from problems.sampling.protein_folding.builders.qubit_op_builder import _build_qubit_op
from problems.sampling.protein_folding.peptide.peptide import Peptide
from qiskit_nature.problems.sampling.protein_folding.interactions.interaction import Interaction
from qiskit_nature.problems.sampling.protein_folding.penalties import Penalties
from qiskit_nature.problems.sampling.sampling_problem import SamplingProblem


class ProteinFoldingProblem(SamplingProblem):

    def __init__(self, peptide: Peptide, interaction: Interaction, penalty_terms: Penalties):
        self._peptide = peptide
        self._interaction = interaction
        self._penalty_terms = penalty_terms
        self._pair_energies = interaction.calc_energy_matrix(len(peptide.get_main_chain),
                                                             peptide.get_main_chain.get_residue_sequence())
        self._N_contacts = 0  # TODO what is the meaning of this param?

    def qubit_op(self):
        return _build_qubit_op(self._peptide, self._pair_energies,
                               self._penalty_terms.lambda_chiral, self._penalty_terms.lambda_back,
                               self._penalty_terms.lambda_1, self._penalty_terms.lambda_contacts,
                               self._N_contacts)

    def interpret(self):
        pass
