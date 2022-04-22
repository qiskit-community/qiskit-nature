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
"""Tests ProteinFoldingResult."""
from test import QiskitNatureTestCase
from qiskit_nature.problems.sampling.protein_folding.protein_folding_problem import (    ProteinFoldingProblem,)
import numpy as np
from qiskit_nature.results.protein_folding_result import ProteinFoldingResult
from qiskit_nature.problems.sampling.protein_folding.interactions.miyazawa_jernigan_interaction import (MiyazawaJerniganInteraction,)
from qiskit_nature.problems.sampling.protein_folding.peptide.peptide import Peptide

from qiskit_nature.problems.sampling.protein_folding.penalty_parameters import PenaltyParameters
from qiskit_nature.problems.sampling.protein_folding.interactions.random_interaction import (    RandomInteraction,)
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.circuit.library import RealAmplitudes
from qiskit.algorithms.optimizers import COBYLA
from qiskit.algorithms import NumPyMinimumEigensolver, VQE
from qiskit.opflow import PauliExpectation, CVaRExpectation
from qiskit import execute, Aer




def create_protein_folding_result(main_chain,side_chains):
    """
    Creates a protein_folding_problem, solves it and uses the result to create a protein_folding_result instance.
    Args:
        main_chain: The desired main_chain for the moleculed to be optimized
        side_chain: The desired side_chains for the moleculed to be optimized
    Returns:
        Protein Folding Result
    """
    algorithm_globals.random_seed = 23
    
    peptide = Peptide(main_chain, side_chains)

    random_interaction = RandomInteraction()
    mj_interaction = MiyazawaJerniganInteraction()

    penalty_back = 10
    penalty_chiral = 10
    penalty_1 = 10

    penalty_terms = PenaltyParameters(penalty_chiral, penalty_back, penalty_1)

    protein_folding_problem = ProteinFoldingProblem(peptide, mj_interaction, penalty_terms)
    qubit_op = protein_folding_problem.qubit_op()
    # set classical optimizer
    optimizer = COBYLA(maxiter=50)

    # set variational ansatz
    ansatz = RealAmplitudes(reps=1)

    # set the backend
    backend_name = "aer_simulator"
    backend = QuantumInstance(
        Aer.get_backend(backend_name),
        shots=8192,
        seed_transpiler=algorithm_globals.random_seed,
        seed_simulator=algorithm_globals.random_seed,
    )


    # initialize CVaR_alpha objective with alpha = 0.1
    cvar_exp = CVaRExpectation(0.1, PauliExpectation())
    
    
    counts = []
    values = []


    def store_intermediate_result(eval_count, parameters, mean, std):
        counts.append(eval_count)
        values.append(mean)
    
    
    # initialize VQE using CVaR
    vqe = VQE(
        expectation=cvar_exp,
        optimizer=optimizer,
        ansatz=ansatz,
        quantum_instance=backend,
        callback=store_intermediate_result,
    )

    result = vqe.compute_minimum_eigenvalue(qubit_op)
    
    return protein_folding_problem.interpret(result)





class TestProteinFoldingResult(QiskitNatureTestCase):
    """Tests ProteinFoldingResult."""

    test_result_1 = create_protein_folding_result("APRLRFY", [""] * 7)
    test_result_2 = create_protein_folding_result("APRLR", ["","","F","Y",""])
    
    def test_best_sequence(self):
        """Tests if the best sequence obtained is the correct one and if it gets passed to the constructor correctly. """
        #Tests for case 1
        self.assertEqual(self.test_result_1.best_sequence,"101100011")
        #Tests for case 2
        self.assertEqual(self.test_result_2.best_sequence,"0011011")
        
    def test_turns_sequence(self):
        """ Tests if the turn decoding works for the main chain and the side chain."""
        #Tests for case 1
        self.assertEqual(self.test_result_1.protein_decoder.get_main_turns(),[1,0,3,2,0,3])
        self.assertEqual(self.test_result_1.protein_decoder.get_side_turns(),[None]*7)
        #Tests for case 2
        self.assertEqual(self.test_result_2.protein_decoder.get_main_turns(),[1,0,3,2])
        self.assertEqual(self.test_result_2.protein_decoder.get_side_turns(),[None,None,3,3,None])
    
    def test_positions(self):
        """Tests if the coordinates of the main and side chains are correct. """
        #Tests for case 1
        self.assertTrue( np.allclose(self.test_result_1.protein_xyz.main_positions ,  np.array(  [[ 0.          ,0.          ,0.        ],
                                                                                                                    [ 0.57735027  ,0.57735027 ,-0.57735027],
                                                                                                                    [ 1.15470054  ,0.         ,-1.15470054],
                                                                                                                    [ 1.73205081 ,-0.57735027 ,-0.57735027],
                                                                                                                    [ 2.30940108  ,0.         , 0.        ],
                                                                                                                    [ 1.73205081  ,0.57735027 , 0.57735027],
                                                                                                                    [ 1.15470054  ,1.15470054 , 0.        ]] ), atol=1e-6 ))
        self.assertTrue( (self.test_result_1.protein_xyz.side_positions == np.array([None]*7)).all() )
        #Tests for case 2
        self.assertTrue( np.allclose(self.test_result_2.protein_xyz.main_positions ,  np.array([[ 0.          ,0.         , 0.        ],
                                                                                                [ 0.57735027  ,0.57735027 ,-0.57735027],
                                                                                                [ 1.15470054  ,0.         ,-1.15470054],
                                                                                                [ 1.73205081 ,-0.57735027 ,-0.57735027],
                                                                                                [ 2.30940108 , 0.         , 0.        ]] ), atol=1e-6 ))
        for a,b in zip( list(self.test_result_2.protein_xyz.side_positions) , [None, None, np.array([ 0.57735027,  0.57735027, -1.73205081]), np.array([ 2.30940108, -1.15470054,  0.        ]), None]):
            if a is None:
                self.assertTrue(b is None)
            else:
                self.assertTrue(np.allclose(a,b,atol=1e-6))
        
            
            
    def test_get_result_binary_vector(self):
        """Tests if a protein folding result returns a correct expanded best sequence if not
        qubits compressed."""
        lambda_back = 10
        lambda_chiral = 10
        lambda_1 = 10
        penalty_terms = PenaltyParameters(lambda_chiral, lambda_back, lambda_1)

        main_chain_residue_seq = "SAASSASAA"
        side_chain_residue_sequences = ["", "", "A", "A", "A", "A", "A", "A", ""]

        peptide = Peptide(main_chain_residue_seq, side_chain_residue_sequences)

        mj_interaction = MiyazawaJerniganInteraction()

        protein_folding_problem = ProteinFoldingProblem(peptide, mj_interaction, penalty_terms)
        best_sequence = "101110010"

        protein_folding_result = ProteinFoldingResult(protein_folding_problem, best_sequence)

        result = protein_folding_result.get_result_binary_vector()

        self.assertEqual(result, best_sequence)

    def test_get_result_binary_vector_compressed(self):
        """Tests if a protein folding result returns a correct expanded best sequence."""
        lambda_back = 10
        lambda_chiral = 10
        lambda_1 = 10
        penalty_terms = PenaltyParameters(lambda_chiral, lambda_back, lambda_1)

        main_chain_residue_seq = "SAASSASAA"
        side_chain_residue_sequences = ["", "", "A", "A", "A", "A", "A", "A", ""]

        peptide = Peptide(main_chain_residue_seq, side_chain_residue_sequences)

        mj_interaction = MiyazawaJerniganInteraction()

        protein_folding_problem = ProteinFoldingProblem(peptide, mj_interaction, penalty_terms)
        best_sequence = "101110010"
        protein_folding_problem._unused_qubits = [0, 1, 2, 5, 7]

        protein_folding_result = ProteinFoldingResult(protein_folding_problem, best_sequence)

        result = protein_folding_result.get_result_binary_vector()
        expected_sequence = "101110*0*10***"

        self.assertEqual(result, expected_sequence)


 