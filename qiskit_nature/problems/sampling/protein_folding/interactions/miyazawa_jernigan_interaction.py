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
import numpy as np

from problems.sampling.protein_folding.residue_validator import _validate_residue_sequence
from qiskit_nature.problems.sampling.protein_folding.data_loaders.energy_matrix_loader import \
    _load_energy_matrix_file
from qiskit_nature.problems.sampling.protein_folding.interactions.interaction import Interaction


class MiyazawaJerniganInteraction(Interaction):

    def calc_energy_matrix(self, chain_len: int, sequence):
        _validate_residue_sequence(sequence)
        mj, list_aa = _load_energy_matrix_file()
        pair_energies = np.zeros((chain_len + 1, 2, chain_len + 1, 2))
        for i in range(1, chain_len + 1):
            for j in range(i + 1, chain_len + 1):
                aa_i = list_aa.index(sequence[i - 1])
                aa_j = list_aa.index(sequence[j - 1])
                pair_energies[i, 0, j, 0] = mj[min(aa_i, aa_j), max(aa_i, aa_j)]
        return pair_energies
