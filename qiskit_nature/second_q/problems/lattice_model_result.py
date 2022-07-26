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

"""The lattice model result class"""

from typing import List, Optional

import numpy as np

from qiskit.algorithms import AlgorithmResult
from .eigenstate_result import EigenstateResult


class LatticeModelResult(EigenstateResult):
    """The lattice model result."""

    def __init__(self) -> None:
        super().__init__()
        self._algorithm_result: Optional[AlgorithmResult] = None
        self._computed_lattice_energies: Optional[np.ndarray] = None
        self._num_occupied_modals_per_mode: Optional[List[List[float]]] = None

    @property
    def algorithm_result(self) -> Optional[AlgorithmResult]:
        """Returns raw algorithm result"""
        return self._algorithm_result

    @algorithm_result.setter
    def algorithm_result(self, value: AlgorithmResult) -> None:
        """Sets raw algorithm result"""
        self._algorithm_result = value

    @property
    def computed_lattice_energies(self) -> Optional[np.ndarray]:
        """Returns computed electronic part of ground state energy"""
        return self._computed_lattice_energies

    @computed_lattice_energies.setter
    def computed_lattice_energies(self, value: np.ndarray) -> None:
        """Sets computed electronic part of ground state energy"""
        self._computed_lattice_energies = value

    def __str__(self) -> str:
        """Printable formatted result"""
        return "\n".join(self._formatted())

    def _formatted(self) -> List[str]:
        """Formatted result as a list of strings"""
        lines = []
        lines.append("=== GROUND STATE ===")
        lines.append(" ")
        lines.append(
            "* Lattice ground state energy " f": {np.round(self.computed_lattice_energies[0], 12)}"
        )

        if self.computed_lattice_energies is not None and len(self.computed_lattice_energies) > 1:
            lines.append(" ")
            lines.append("=== EXCITED STATES ===")
            lines.append(" ")

            for idx, lattice_energy in enumerate(self.computed_lattice_energies[1:]):
                lines.append(
                    f"* {(idx + 1): 3d}: Lattice excited state energy: {np.round(lattice_energy, 12)}"
                )
        return lines
