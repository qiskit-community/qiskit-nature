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
from problems.sampling.protein_folding.distance_calculator import _create_distance_qubits
from problems.sampling.protein_folding.peptide.peptide import Peptide


class DistanceMap:

    def __init__(self, peptide: Peptide):
        self._peptide = peptide
        self._lower_main_upper_main, self._lower_side_upper_main, self._lower_main_upper_side, \
        self._lower_side_upper_side, self.num_distances = _create_distance_qubits(peptide)

    @property
    def peptide(self):
        return self._peptide

    @property
    def lower_main_upper_main(self):
        return self._lower_main_upper_main

    @property
    def lower_side_upper_main(self):
        return self._lower_side_upper_main

    @property
    def lower_main_upper_side(self):
        return self._lower_main_upper_side

    @property
    def lower_side_upper_side(self):
        return self._lower_side_upper_side



