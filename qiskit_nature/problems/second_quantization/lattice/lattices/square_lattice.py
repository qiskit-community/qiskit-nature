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

"""The square lattice"""
from typing import Tuple, Union

from .boundary_condition import BoundaryCondition
from .hyper_cubic_lattice import HyperCubicLattice


class SquareLattice(HyperCubicLattice):
    """Square lattice."""

    def __init__(
        self,
        rows: int,
        cols: int,
        edge_parameter: Union[complex, Tuple[complex, complex]] = 1.0,
        onsite_parameter: complex = 0.0,
        boundary_condition: Union[
            BoundaryCondition, Tuple[BoundaryCondition, BoundaryCondition]
        ] = BoundaryCondition.OPEN,
    ) -> None:
        """
        Args:
            rows: Length of the x direction.
            cols: Length of the y direction.
            edge_parameter: Weights on the edges in x and y direction.
                When it is a single value, it is interpreted as a tuple of length 2
                consisting of the same values.
                Defaults to 1.0.
            onsite_parameter: Weight on the self-loops, which are edges connecting a node to itself.
                Defaults to 0.0.
            boundary_condition: Boundary condition for each direction.
                The available boundary conditions are:
                BoundaryCondition.OPEN, BoundaryCondition.PERIODIC.
                When it is a single value, it is interpreted as a tuple of length 2
                consisting of the same values.
                Defaults to BoundaryCondition.OPEN.
        """
        self._rows = rows
        self._cols = cols
        super().__init__(
            size=(rows, cols),
            edge_parameter=edge_parameter,
            onsite_parameter=onsite_parameter,
            boundary_condition=boundary_condition,
        )

    @property
    def rows(self) -> int:
        """Length of the x direction

        Returns:
            the length
        """
        return self._rows

    @property
    def cols(self) -> int:
        """Length of the y direction

        Returns:
            the length
        """
        return self._cols
