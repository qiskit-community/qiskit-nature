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

"""Test IsingModel."""

from typing import cast
from test import QiskitNatureTestCase

import numpy as np
from numpy.testing import assert_array_equal
from retworkx import PyGraph, is_isomorphic

from qiskit_nature.problems.second_quantization.lattice import IsingModel, Lattice


class TestIsingModel(QiskitNatureTestCase):
    """TestIsingModel"""

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
        ism = IsingModel(lattice)

        with self.subTest("Check the graph."):
            self.assertTrue(
                is_isomorphic(ism.lattice.graph, lattice.graph, edge_matcher=lambda x, y: x == y)
            )

        with self.subTest("Check the coupling matrix"):
            coupling_matrix = ism.coupling_matrix
            target_matrix = np.array(
                [[0.0, 1.0 + 1.0j, -1.0], [1.0 - 1.0j, 2.0, 0.0], [-1.0, 0.0, 0.0]]
            )
            assert_array_equal(coupling_matrix, target_matrix)

        with self.subTest("Check the second q op representation."):
            coupling = [
                ("Z_0 Z_1", 1.0 + 1.0j),
                ("Z_0 Z_2", -1.0),
                ("X_1", 2.0),
            ]

            ham = coupling

            self.assertSetEqual(set(ham), set(ism.second_q_ops().to_list()))

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
        uniform_ism = cast(
            IsingModel,
            IsingModel.uniform_parameters(
                lattice,
                uniform_interaction=1.0 + 1.0j,
                uniform_onsite_potential=0.0,
            ),
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
                    uniform_ism.lattice.graph, target_graph, edge_matcher=lambda x, y: x == y
                )
            )
        with self.subTest("Check the coupling matrix."):
            coupling_matrix = uniform_ism.coupling_matrix  # pylint: disable=no-member
            target_matrix = np.array(
                [[0.0, 1.0 + 1.0j, 1.0 + 1.0j], [1.0 - 1.0j, 0.0, 0.0], [1.0 - 1.0j, 0.0, 0.0]]
            )
            assert_array_equal(coupling_matrix, target_matrix)

        with self.subTest("Check the second q op representation."):
            coupling = [
                ("Z_0 Z_1", 1.0 + 1.0j),
                ("Z_0 Z_2", 1.0 + 1.0j),
                ("X_0", 0.0),
                ("X_1", 0.0),
                ("X_2", 0.0),
            ]

            ham = coupling

            self.assertSetEqual(set(ham), set(uniform_ism.second_q_ops().to_list()))

    def test_from_parameters(self):
        """Test from_parameters."""
        coupling_matrix = np.array(
            [[1.0, 1.0 + 1.0j, 2.0 - 2.0j], [1.0 - 1.0j, 0.0, 0.0], [2.0 + 2.0j, 0.0, 1.0]]
        )

        ism = cast(IsingModel, IsingModel.from_parameters(coupling_matrix))
        with self.subTest("Check the graph."):
            target_graph = PyGraph(multigraph=False)
            target_graph.add_nodes_from(range(3))
            target_weight = [(0, 0, 1.0), (0, 1, 1.0 + 1.0j), (0, 2, 2.0 - 2.0j), (2, 2, 1.0)]
            target_graph.add_edges_from(target_weight)
            self.assertTrue(
                is_isomorphic(ism.lattice.graph, target_graph, edge_matcher=lambda x, y: x == y)
            )

        with self.subTest("Check the coupling matrix."):
            assert_array_equal(ism.coupling_matrix, coupling_matrix)  # pylint: disable=no-member

        with self.subTest("Check the second q op representation."):
            coupling = [
                ("Z_0 Z_1", 1.0 + 1.0j),
                ("Z_0 Z_2", 2.0 - 2.0j),
                ("X_0", 1.0),
                ("X_2", 1.0),
            ]

            ham = coupling

            self.assertSetEqual(set(ham), set(ism.second_q_ops().to_list()))
