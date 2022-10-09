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

"""Test for Lattice."""
from test import QiskitNatureTestCase

import unittest
import numpy as np
from numpy.testing import assert_array_equal

from rustworkx import PyGraph, is_isomorphic

from qiskit.utils import optionals as _optionals

from qiskit_nature.second_q.hamiltonians.lattices import Lattice

if _optionals.HAS_NETWORKX:
    import networkx as nx


class TestLattice(QiskitNatureTestCase):
    """Test Lattice."""

    def test_init(self):
        """Test init."""
        graph = PyGraph(multigraph=False)
        graph.add_nodes_from(range(6))
        weighted_edge_list = [
            (0, 1, 1.0 + 1.0j),
            (0, 2, -1.0),
            (2, 3, 2.0),
            (2, 4, -1.0),
            (4, 4, 3.0),
            (2, 5, -1.0),
        ]
        graph.add_edges_from(weighted_edge_list)
        lattice = Lattice(graph)

        with self.subTest("Check the type of lattice."):
            self.assertIsInstance(lattice, Lattice)

        with self.subTest("Check graph."):
            target_graph = PyGraph(multigraph=False)
            target_graph.add_nodes_from(range(6))
            target_weighted_edge_list = [
                (4, 4, 3.0),
                (0, 1, 1 + 1j),
                (2, 3, 2.0),
                (2, 4, -1.0),
                (2, 5, -1.0),
                (0, 2, -1),
            ]
            target_graph.add_edges_from(target_weighted_edge_list)
            self.assertTrue(
                is_isomorphic(lattice.graph, target_graph, edge_matcher=lambda x, y: x == y)
            )

        with self.subTest("Check the number of nodes."):
            self.assertEqual(lattice.num_nodes, 6)

        with self.subTest("Check the set of nodes."):
            self.assertSetEqual(set(lattice.node_indexes), set(range(6)))

        with self.subTest("Check the set of weights."):
            target_set = {
                (0, 1, 1 + 1j),
                (4, 4, 3),
                (2, 5, -1.0),
                (0, 2, -1.0),
                (2, 3, 2.0),
                (2, 4, -1.0),
            }
            self.assertEqual(set(lattice.weighted_edge_list), target_set)

    def test_copy(self):
        """Test test_copy."""
        graph = PyGraph(multigraph=False)
        graph.add_nodes_from(range(6))
        weighted_edge_list = [
            (0, 1, 1.0 + 1.0j),
            (0, 2, -1.0),
            (2, 3, 2.0),
            (2, 4, -1.0),
            (4, 4, 3.0),
            (2, 5, -1.0),
        ]
        graph.add_edges_from(weighted_edge_list)
        lattice = Lattice(graph)
        lattice_copy = lattice.copy()
        self.assertTrue(is_isomorphic(lattice_copy.graph, graph, edge_matcher=lambda x, y: x == y))

    def test_from_nodes_and_edges(self):
        """Test from_nodes_edges."""
        graph = PyGraph(multigraph=False)
        graph.add_nodes_from(range(6))
        weighted_edge_list = [
            (0, 1, 1.0 + 1.0j),
            (0, 2, -1.0),
            (2, 3, 2.0),
            (4, 2, -1.0),
            (4, 4, 3.0),
            (2, 5, -1.0),
        ]
        graph.add_edges_from(weighted_edge_list)
        lattice = Lattice(graph)
        target_num_nodes = 6
        target_weighted_edge_list = [
            (2, 5, -1.0),
            (4, 4, 3),
            (4, 2, -1.0),
            (2, 3, 2.0),
            (0, 2, -1.0),
            (0, 1, 1.0 + 1.0j),
        ]
        target_lattice = Lattice.from_nodes_and_edges(target_num_nodes, target_weighted_edge_list)

        self.assertTrue(
            is_isomorphic(lattice.graph, target_lattice.graph, edge_matcher=lambda x, y: x == y)
        )

    def test_to_adjacency_matrix(self):
        """Test to_adjacency_matrix."""
        graph = PyGraph(multigraph=False)
        graph.add_nodes_from(range(3))
        weighted_edge_list = [(0, 1, 1.0 + 1.0j), (0, 2, -1.0), (2, 2, 3)]
        graph.add_edges_from(weighted_edge_list)
        lattice = Lattice(graph)

        target_matrix = np.array([[0, 1 + 1j, -1.0], [1 - 1j, 0, 0], [-1.0, 0, 3.0]])
        assert_array_equal(lattice.to_adjacency_matrix(weighted=True), target_matrix)

        target_matrix = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 1]])
        assert_array_equal(lattice.to_adjacency_matrix(), target_matrix)

    @unittest.skipIf(not _optionals.HAS_NETWORKX, "networkx not available.")
    def test_from_networkx(self):
        """Test initialization from a networkx graph."""
        graph = nx.Graph()
        graph.add_nodes_from(range(5))
        graph.add_edges_from([(i, i + 1) for i in range(4)])
        lattice = Lattice(graph)

        target_graph = PyGraph()
        target_graph.add_nodes_from(range(5))
        target_graph.add_edges_from([(i, i + 1, 1) for i in range(4)])

        self.assertTrue(
            is_isomorphic(lattice.graph, target_graph, edge_matcher=lambda x, y: x == y)
        )

    def test_nonnumeric_weight_raises(self):
        """Test the initialization with a graph with non-numeric edge weights raises."""
        graph = PyGraph(multigraph=False)
        graph.add_nodes_from(range(3))
        graph.add_edges_from([(0, 1, 1), (1, 2, "banana")])

        with self.assertRaises(ValueError):
            _ = Lattice(graph)

    def test_edges_removed(self):
        """Test the initialization with a graph where edges have been removed."""
        graph = PyGraph(multigraph=False)
        graph.add_nodes_from(range(3))
        graph.add_edges_from([(0, 1, 1), (1, 2, 1)])
        graph.remove_edge_from_index(0)

        lattice = Lattice(graph)

        target_graph = PyGraph(multigraph=False)
        target_graph.add_nodes_from(range(3))
        target_graph.add_edges_from([(1, 2, 1)])

        self.assertTrue(
            is_isomorphic(lattice.graph, target_graph, edge_matcher=lambda x, y: x == y)
        )
