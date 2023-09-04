# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Ising model"""

from fractions import Fraction
import numpy as np
from qiskit_nature.second_q.operators import SpinOp

from .lattice_model import LatticeModel


class IsingModel(LatticeModel):
    r"""The transverse-field Ising model.

    This class implements the following Hamiltonian:

    .. math::
        H = -\sum_{\langle i, j \rangle} J_{ij} Z_{i} Z_{j} -  \sum_{i} g_{i} X_{i},

    where :math:`i,j` refer to lattice nodes. The :math:`\sum_{\langle i, j \rangle}` is performed
    over adjacent lattice nodes. This model assumes spin-:math:`\frac{1}{2}` particles. Thus,
    :math:`X_i` and :math:`Z_i` represent the respective Pauli matrices. :math:`J_{ij}` are constants
    with dimensions of energy and :math:`g_{i}` are coupling parameters that determine the relative
    strength between the external transverse field and the nearest neighbor interactions.

    This model is instantiated using a
    :class:`~qiskit_nature.second_q.hamiltonians.lattices.Lattice`. For example, using a
    :class:`~qiskit_nature.second_q.hamiltonians.lattices.LineLattice`:

    .. code-block:: python

        line_lattice = LineLattice(num_nodes=10, boundary_condition=BoundaryCondition.OPEN)

        ising_model = IsingModel(
            line_lattice.uniform_parameters(
                uniform_interaction=-1.0,
                uniform_onsite_potential=0.0,
            ),
        )
    """

    def coupling_matrix(self) -> np.ndarray:
        """Return the coupling matrix."""
        return self.interaction_matrix()

    @property
    def register_length(self) -> int:
        return self._lattice.num_nodes

    def second_q_op(self) -> SpinOp:
        """Return the Hamiltonian of the Ising model in terms of ``SpinOp``.

        Returns:
            SpinOp: The Hamiltonian of the Ising model.
        """
        ham = {}
        weighted_edge_list = self._lattice.weighted_edge_list
        # kinetic terms
        for node_a, node_b, weight in weighted_edge_list:
            if node_a == node_b:
                index = node_a
                ham[f"X_{index}"] = weight

            else:
                index_left = node_a
                index_right = node_b
                coupling_parameter = weight
                ham[f"Z_{index_left} Z_{index_right}"] = coupling_parameter

        return SpinOp(ham, spin=Fraction(1, 2), num_spins=self._lattice.num_nodes)
