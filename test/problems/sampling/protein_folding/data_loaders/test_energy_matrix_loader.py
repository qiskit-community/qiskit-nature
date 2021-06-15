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

from test import QiskitNatureTestCase
from qiskit_nature.problems.sampling.protein_folding.data_loaders.energy_matrix_loader import \
    _load_energy_matrix_file


class TestEnergyMatrixLoader(QiskitNatureTestCase):
    """Tests EnergyMatrixLoader."""

    def test_load_energy_matrix_file(self):
        """Test that the energy matrix is loaded from the MJ potential file"""
        energy_matrix, list_aa = _load_energy_matrix_file()
        assert energy_matrix[0][0] == -5.44
        assert energy_matrix[2][3] == -6.84
        assert list_aa == ['C', 'M', 'F', 'I', 'L', 'V', 'W', 'Y', 'A', 'G', 'T', 'S', 'N', 'Q',
                           'D', 'E', 'H', 'R', 'K', 'P']
