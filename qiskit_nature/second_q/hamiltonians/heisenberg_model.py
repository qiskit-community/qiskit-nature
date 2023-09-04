# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Heisenberg model."""

from __future__ import annotations
from fractions import Fraction
import numpy as np
from qiskit_nature.second_q.operators import SpinOp

from .lattice_model import LatticeModel
from .lattices import Lattice


class HeisenbergModel(LatticeModel):
    r"""The Heisenberg model.

    This class implements the following Hamiltonian:

    .. math::
        H = - \vec{J} \sum_{\langle i, j \rangle} \vec{\sigma}_{i} \otimes \vec{\sigma}_{j}
        - \vec{h} \sum_{i} \vec{\sigma}_{i}

    where :math:`i,j` refer to lattice nodes. The :math:`\sum_{\langle i, j \rangle}` is performed
    over adjacent lattice nodes. This model assumes spin-:math:`\frac{1}{2}` particles. Thus,
    :math:`\vec{\sigma}_{i} = (X_i, Y_i, Z_i)` is a vector containing the Pauli matrices.
    :math:`\vec{J}` is the coupling constant and :math:`\vec{h}` is the external magnetic field,
    both with dimensions of energy.

    This model is instantiated using a
    :class:`~qiskit_nature.second_q.hamiltonians.lattices.Lattice`. For example, using a
    :class:`~qiskit_nature.second_q.hamiltonians.lattices.LineLattice`:

    .. code-block:: python

        line_lattice = LineLattice(num_nodes=10, boundary_condition=BoundaryCondition.OPEN)
        heisenberg_model = HeisenbergModel(line_lattice, (1.0, 1.0, 1.0), (0.0, 0.0, 1.0))

    The transverse-field Ising model can be recovered as a special case of the Heisenberg model
    by limiting the model to spins that are parallel/antiparallel with respect to a transverse
    magnetic field:

    .. code-block:: python

        heisenberg_model = HeisenbergModel(line_lattice, (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
    """

    def __init__(
        self,
        lattice: Lattice,
        coupling_constants: tuple = (1.0, 1.0, 1.0),
        ext_magnetic_field: tuple = (0.0, 0.0, 0.0),
    ) -> None:
        """
        Args:
            lattice: Lattice on which the model is defined.
            coupling_constants: The coupling constants in each Cartesian axis.
                Defaults to ``(1.0, 1.0, 1.0)``.
            ext_magnetic_field: Represents a magnetic field in Cartesian coordinates.
                Defaults to ``(0.0, 0.0, 0.0)``.
        """
        super().__init__(lattice)
        self.coupling_constants = coupling_constants
        self.ext_magnetic_field = ext_magnetic_field

    @property
    def register_length(self) -> int:
        return self._lattice.num_nodes

    def second_q_op(self) -> SpinOp:
        """Return the Hamiltonian of the Heisenberg model in terms of ``SpinOp``.

        Returns:
            SpinOp: The Hamiltonian of the Heisenberg model.
        """
        hamiltonian = {}
        weighted_edge_list = self.lattice.weighted_edge_list

        for node_a, node_b, _ in weighted_edge_list:

            if node_a == node_b:
                index = node_a
                for axis, coeff in zip("XYZ", self.ext_magnetic_field):
                    if not np.isclose(coeff, 0.0):
                        hamiltonian[f"{axis}_{index}"] = coeff
            else:
                index_left = node_a
                index_right = node_b
                for axis, coeff in zip("XYZ", self.coupling_constants):
                    if not np.isclose(coeff, 0.0):
                        hamiltonian[f"{axis}_{index_left} {axis}_{index_right}"] = coeff

        return SpinOp(hamiltonian, spin=Fraction(1, 2), num_spins=self.lattice.num_nodes)
