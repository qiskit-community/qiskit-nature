# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""BOPES Sampler result"""

from typing import List, Dict, Union

from .eigenstate_result import EigenstateResult


class BOPESSamplerResult:
    """The BOPES Sampler result"""

    def __init__(
        self,
        points: List[float],
        energies: List[List[float]],
        raw_results: Dict[float, EigenstateResult],
    ) -> None:
        """
        Creates an new instance of the result.
        Args:
            points: List of points.
            energies: List of energies.
            raw_results: Raw results obtained from the solver.
        """
        super().__init__()
        self._points = points
        self._energies = energies
        self._raw_results = raw_results

    @property
    def points(self) -> List[float]:
        """returns list of points."""
        return self._points

    @property
    def energies(self) -> Union[List[float], List[List[float]]]:
        """returns list of energies."""
        formatted_energies: Union[List[float], List[List[float]]]
        if len(self._energies[0]) == 1:
            formatted_energies = [self._energies[k][0] for k in range(len(self._energies))]
        else:
            formatted_energies = self._energies
        return formatted_energies

    @property
    def raw_results(self) -> Dict[float, EigenstateResult]:
        """returns all results for all points."""
        return self._raw_results

    def point_results(self, point: float) -> EigenstateResult:
        """returns all results for a specific point."""
        return self.raw_results[point]
