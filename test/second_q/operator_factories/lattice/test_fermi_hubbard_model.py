# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test FermiHubbardModel."""
from test import QiskitNatureTestCase

import numpy as np
from numpy.testing import assert_array_equal
from retworkx import PyGraph, is_isomorphic

from qiskit_nature.second_q.operator_factories.lattice import FermiHubbardModel
from qiskit_nature.second_q.operator_factories.lattices import Lattice


class TestFermiHubbardModel(QiskitNatureTestCase):
    """TestFermiHubbardModel"""

    def test_init(self):
        """Test init."""
        graph = PyGraph(multigraph=False)
        graph.add_nodes_from(range(3))
        weighted_edge_list = [
            (0, 1, 1.0 + 1.0j),
            (0, 2, -1.0),
            (1, 1, 2.0),
        ]
        graph.add_edges_from(weighted_edge_list)
        lattice = Lattice(graph)
        fhm = FermiHubbardModel(lattice, onsite_interaction=10.0)

        with self.subTest("Check the graph."):
            self.assertTrue(
                is_isomorphic(fhm.lattice.graph, lattice.graph, edge_matcher=lambda x, y: x == y)
            )

        with self.subTest("Check the hopping matrix"):
            hopping_matrix = fhm.hopping_matrix()
            target_matrix = np.array(
                [[0.0, 1.0 + 1.0j, -1.0], [1.0 - 1.0j, 2.0, 0.0], [-1.0, 0.0, 0.0]]
            )
            assert_array_equal(hopping_matrix, target_matrix)

        with self.subTest("Check the second q op representation."):
            hopping = [
                ("+_0 -_2", 1.0 + 1.0j),
                ("-_0 +_2", -(1.0 - 1.0j)),
                ("+_0 -_4", -1.0),
                ("-_0 +_4", 1.0),
                ("+_1 -_3", 1.0 + 1.0j),
                ("-_1 +_3", -(1.0 - 1.0j)),
                ("+_1 -_5", -1.0),
                ("-_1 +_5", 1.0),
                ("+_2 -_2", 2.0),
                ("+_3 -_3", 2.0),
            ]

            interaction = [
                ("+_0 -_0 +_1 -_1", 10.0),
                ("+_2 -_2 +_3 -_3", 10.0),
                ("+_4 -_4 +_5 -_5", 10.0),
            ]

            ham = hopping + interaction

            self.assertSetEqual(set(ham), set(fhm.second_q_ops(display_format="sparse").to_list()))

    def test_uniform_parameters(self):
        """Test uniform_parameters."""
        graph = PyGraph(multigraph=False)
        graph.add_nodes_from(range(3))
        weighted_edge_list = [
            (0, 1, 1.0 + 1.0j),
            (0, 2, -1.0),
            (1, 1, 2.0),
        ]
        graph.add_edges_from(weighted_edge_list)
        lattice = Lattice(graph)
        uniform_fhm = FermiHubbardModel.uniform_parameters(
            lattice,
            uniform_interaction=1.0 + 1.0j,
            uniform_onsite_potential=0.0,
            onsite_interaction=10.0,
        )
        with self.subTest("Check the graph."):
            target_graph = PyGraph(multigraph=False)
            target_graph.add_nodes_from(range(3))
            target_weight = [
                (0, 1, 1.0 + 1.0j),
                (0, 2, 1.0 + 1.0j),
                (0, 0, 0.0),
                (1, 1, 0.0),
                (2, 2, 0.0),
            ]
            target_graph.add_edges_from(target_weight)
            self.assertTrue(
                is_isomorphic(
                    uniform_fhm.lattice.graph, target_graph, edge_matcher=lambda x, y: x == y
                )
            )
        with self.subTest("Check the hopping matrix."):
            hopping_matrix = uniform_fhm.hopping_matrix()
            target_matrix = np.array(
                [[0.0, 1.0 + 1.0j, 1.0 + 1.0j], [1.0 - 1.0j, 0.0, 0.0], [1.0 - 1.0j, 0.0, 0.0]]
            )
            assert_array_equal(hopping_matrix, target_matrix)

        with self.subTest("Check the second q op representation."):
            hopping = [
                ("+_0 -_2", 1.0 + 1.0j),
                ("-_0 +_2", -(1.0 - 1.0j)),
                ("+_0 -_4", 1.0 + 1.0j),
                ("-_0 +_4", -(1.0 - 1.0j)),
                ("+_1 -_3", 1.0 + 1.0j),
                ("-_1 +_3", -(1.0 - 1.0j)),
                ("+_1 -_5", 1.0 + 1.0j),
                ("-_1 +_5", -(1.0 - 1.0j)),
                ("+_0 -_0", 0.0),
                ("+_1 -_1", 0.0),
                ("+_2 -_2", 0.0),
                ("+_3 -_3", 0.0),
                ("+_4 -_4", 0.0),
                ("+_5 -_5", 0.0),
            ]

            interaction = [
                ("+_0 -_0 +_1 -_1", 10.0),
                ("+_2 -_2 +_3 -_3", 10.0),
                ("+_4 -_4 +_5 -_5", 10.0),
            ]

            ham = hopping + interaction

            self.assertSetEqual(
                set(ham), set(uniform_fhm.second_q_ops(display_format="sparse").to_list())
            )

    def test_from_parameters(self):
        """Test from_parameters."""
        hopping_matrix = np.array(
            [[1.0, 1.0 + 1.0j, 2.0 + 2.0j], [1.0 - 1.0j, 0.0, 0.0], [2.0 - 2.0j, 0.0, 1.0]]
        )

        onsite_interaction = 10.0
        fhm = FermiHubbardModel.from_parameters(hopping_matrix, onsite_interaction)
        with self.subTest("Check the graph."):
            target_graph = PyGraph(multigraph=False)
            target_graph.add_nodes_from(range(3))
            target_weight = [(0, 0, 1.0), (0, 1, 1.0 + 1.0j), (0, 2, 2.0 + 2.0j), (2, 2, 1.0)]
            target_graph.add_edges_from(target_weight)
            self.assertTrue(
                is_isomorphic(fhm.lattice.graph, target_graph, edge_matcher=lambda x, y: x == y)
            )

        with self.subTest("Check the hopping matrix."):
            assert_array_equal(fhm.hopping_matrix(), hopping_matrix)

        with self.subTest("Check the second q op representation."):
            hopping = [
                ("+_0 -_2", 1.0 + 1.0j),
                ("-_0 +_2", -(1.0 - 1.0j)),
                ("+_0 -_4", 2.0 + 2.0j),
                ("-_0 +_4", -(2.0 - 2.0j)),
                ("+_1 -_3", 1.0 + 1.0j),
                ("-_1 +_3", -(1.0 - 1.0j)),
                ("+_1 -_5", 2.0 + 2.0j),
                ("-_1 +_5", -(2.0 - 2.0j)),
                ("+_0 -_0", 1.0),
                ("+_1 -_1", 1.0),
                ("+_4 -_4", 1.0),
                ("+_5 -_5", 1.0),
            ]

            interaction = [
                ("+_0 -_0 +_1 -_1", onsite_interaction),
                ("+_2 -_2 +_3 -_3", onsite_interaction),
                ("+_4 -_4 +_5 -_5", onsite_interaction),
            ]

            ham = hopping + interaction

            self.assertSetEqual(set(ham), set(fhm.second_q_ops(display_format="sparse").to_list()))
