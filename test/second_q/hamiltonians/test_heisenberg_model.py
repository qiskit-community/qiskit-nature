# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test HeisenbergModel."""

from test import QiskitNatureTestCase
from rustworkx import PyGraph, is_isomorphic
from qiskit_nature.second_q.hamiltonians.lattices import Lattice, LineLattice
from qiskit_nature.second_q.hamiltonians import HeisenbergModel, IsingModel


class TestHeisenbergModel(QiskitNatureTestCase):
    """TestHeisenbergModel"""

    def test_init(self):
        """Test init."""
        line = LineLattice(num_nodes=2)
        heisenberg_model = HeisenbergModel(lattice=line)

        with self.subTest("Check the graph."):
            self.assertTrue(
                is_isomorphic(
                    heisenberg_model.lattice.graph, line.graph, edge_matcher=lambda x, y: x == y
                )
            )

        with self.subTest("Check the second q op representation."):
            terms = [("X_0 X_1", 1.0), ("Y_0 Y_1", 1.0), ("Z_0 Z_1", 1.0)]

            hamiltonian = terms

            self.assertSetEqual(set(hamiltonian), set(heisenberg_model.second_q_op().items()))

    def test_triangular(self):
        """Test triangular lattice."""
        triangle_graph = PyGraph(multigraph=False)
        triangle_graph.add_nodes_from(range(3))
        triangle_weighted_edge_list = [
            (0, 1, 1.0),
            (0, 2, 1.0),
            (1, 2, 1.0),
            (0, 0, 1.0),
            (1, 1, 1.0),
            (2, 2, 1.0),
        ]
        triangle_graph.add_edges_from(triangle_weighted_edge_list)
        triangle_lattice = Lattice(triangle_graph)
        ext_magnetic_field_y = (0.0, 1.0, 0.0)
        triangle_y_heisenberg_model = HeisenbergModel(
            triangle_lattice, ext_magnetic_field=ext_magnetic_field_y
        )

        with self.subTest("Check the graph of triangular model."):
            self.assertTrue(
                is_isomorphic(
                    triangle_y_heisenberg_model.lattice.graph,
                    triangle_lattice.graph,
                    edge_matcher=lambda x, y: x == y,
                )
            )

        with self.subTest("Check the second q ops in the triangular lattice with param in y axis."):
            terms = [
                ("X_0 X_1", 1.0),
                ("Y_0 Y_1", 1.0),
                ("Z_0 Z_1", 1.0),
                ("X_0 X_2", 1.0),
                ("Y_0 Y_2", 1.0),
                ("Z_0 Z_2", 1.0),
                ("X_1 X_2", 1.0),
                ("Y_1 Y_2", 1.0),
                ("Z_1 Z_2", 1.0),
                ("Y_0", 1.0),
                ("Y_1", 1.0),
                ("Y_2", 1.0),
            ]

            hamiltonian = terms

            self.assertSetEqual(
                set(hamiltonian), set(triangle_y_heisenberg_model.second_q_op().items())
            )

    def test_ising(self):
        """Test Ising."""
        line = LineLattice(num_nodes=2, onsite_parameter=1)
        ism = IsingModel(lattice=line)
        coupling_constants = (0.0, 0.0, 1.0)
        ext_magnetic_field = (1.0, 0.0, 0.0)
        hm_to_ism = HeisenbergModel(
            lattice=line,
            coupling_constants=coupling_constants,
            ext_magnetic_field=ext_magnetic_field,
        )

        with self.subTest("Check if the HeisenbergModel reproduce IsingModel in a special case."):

            self.assertSetEqual(
                set(ism.second_q_op().items()),
                set(hm_to_ism.second_q_op().items()),
            )

    def test_xy(self):
        """Test x and y directions."""
        line = LineLattice(num_nodes=2)
        xy_coupling = (0.5, 0.5, 0.0)
        xy_ext_magnetic_field = (-0.75, 0.25, 0.0)
        xy_test_hm = HeisenbergModel(
            lattice=line, coupling_constants=xy_coupling, ext_magnetic_field=xy_ext_magnetic_field
        )

        with self.subTest("Check if if x and y params are being applied."):
            terms = [
                ("X_0 X_1", 0.5),
                ("Y_0 Y_1", 0.5),
                ("X_0", -0.75),
                ("Y_0", 0.25),
                ("X_1", -0.75),
                ("Y_1", 0.25),
            ]

            hamiltonian = terms

            self.assertSetEqual(set(hamiltonian), set(xy_test_hm.second_q_op().items()))

    def test_xyz_ext_field(self):
        """Test external field."""
        line = LineLattice(num_nodes=2)
        xyz_ext_magnetic_field = (1.0, 1.0, 1.0)
        xyz_test_hm = HeisenbergModel(lattice=line, ext_magnetic_field=xyz_ext_magnetic_field)

        with self.subTest("Check if if x, y and z params are being applied."):
            terms = [
                ("X_0 X_1", 1.0),
                ("Y_0 Y_1", 1.0),
                ("Z_0 Z_1", 1.0),
                ("X_0", 1.0),
                ("X_1", 1.0),
                ("Y_0", 1.0),
                ("Y_1", 1.0),
                ("Z_0", 1.0),
                ("Z_1", 1.0),
            ]

            hamiltonian = terms

            self.assertSetEqual(set(hamiltonian), set(xyz_test_hm.second_q_op().items()))
