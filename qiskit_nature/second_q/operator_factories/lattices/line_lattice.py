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

"""The line lattice"""
from .boundary_condition import BoundaryCondition
from .hyper_cubic_lattice import HyperCubicLattice


class LineLattice(HyperCubicLattice):
    """Line lattice."""

    def __init__(
        self,
        num_nodes: int,
        edge_parameter: complex = 1.0,
        onsite_parameter: complex = 0.0,
        boundary_condition: BoundaryCondition = BoundaryCondition.OPEN,
    ) -> None:
        """
        Args:
            num_nodes: The number of sites.
            edge_parameter: Weight on the edges. Defaults to 1.0.
            onsite_parameter: Weight on the self-loops, which are edges connecting a node to itself.
                Defaults to 0.0.
            boundary_condition: Boundary condition.
                The available boundary conditions are:
                BoundaryCondition.OPEN, BoundaryCondition.PERIODIC.
                Defaults to BoundaryCondition.OPEN.
        """
        self._num_nodes = num_nodes

        super().__init__(
            size=(num_nodes,),
            edge_parameter=edge_parameter,
            onsite_parameter=onsite_parameter,
            boundary_condition=boundary_condition,
        )

    @property
    def num_nodes(self) -> int:
        """Number of nodes

        Returns:
            The number of nodes for the line lattice
        """
        return self._num_nodes
