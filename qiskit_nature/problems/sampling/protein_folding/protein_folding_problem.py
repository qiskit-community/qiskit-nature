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
from problems.sampling.protein_folding.qubit_op_builder import _build_qubit_op
from problems.sampling.protein_folding.peptide.peptide import Peptide
from qiskit_nature.problems.sampling.protein_folding.interactions.interaction import Interaction
from qiskit_nature.problems.sampling.protein_folding.penalty_parameters import PenaltyParameters
from qiskit_nature.problems.sampling.sampling_problem import SamplingProblem


class ProteinFoldingProblem(SamplingProblem):

    # TODO add loader here
    def __init__(self, peptide: Peptide, interaction: Interaction,
                 penalty_parameters: PenaltyParameters):
        self._peptide = peptide
        self._interaction = interaction
        self._penalty_parameters = penalty_parameters
        self._pair_energies = interaction.calc_energy_matrix(len(peptide.get_main_chain),
                                                             peptide.get_main_chain.main_chain_residue_sequence)
        self._N_contacts = 0  # TODO what is the meaning of this param?

    def qubit_op(self):
        """
        Builds a qubit operator for the Hamiltonian encoding a protein folding problem.

        Returns:
            qubit_operator: a qubit operator for the Hamiltonian encoding a protein folding problem.
        """
        qubit_operator = _build_qubit_op(self._peptide, self._pair_energies,
                                         self._penalty_parameters,
                                         self._N_contacts)
        return qubit_operator

    def interpret(self):
        pass
