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
from problems.sampling.protein_folding.interactions.miyazawa_jernigan_interaction import \
    MiyazawaJerniganInteraction
from problems.sampling.protein_folding.peptide.peptide import Peptide
from problems.sampling.protein_folding.protein_folding_problem import ProteinFoldingProblem
from qiskit_nature.problems.sampling.protein_folding.penalties import Penalties
from test import QiskitNatureTestCase


class TestProteinFoldingProblem(QiskitNatureTestCase):
    """Tests ProteinFoldingProblem."""

    def test_protein_folding_problem(self):

        lambda_back = 10
        lambda_chiral = 10
        lambda_1 = 10
        lambda_contacts = 10
        penalty_terms = Penalties(lambda_chiral, lambda_back, lambda_1, lambda_contacts)

        main_chain_residue_seq = ["S", "A", "A", "S", "S", "A", "S", "A", "A"]
        main_chain_len = 9
        side_chain_lens = [0, 0, 1, 1, 1, 1, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", "A", "A", "A", "A", "A", None]

        peptide = Peptide(main_chain_len, main_chain_residue_seq, side_chain_lens,
                          side_chain_residue_sequences)


        mj = MiyazawaJerniganInteraction()

        protein_folding_problem = ProteinFoldingProblem(peptide, mj, penalty_terms)
        qubit_op = protein_folding_problem.qubit_op()
        print(qubit_op)