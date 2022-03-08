# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test MP2 Info """

import unittest
import numpy as np

from ddt import ddt, file_data

from test import QiskitNatureTestCase
from qiskit_nature.settings import settings
from qiskit_nature.initializers import MP2Initializer


@ddt
class TestMP2Initializer(QiskitNatureTestCase):
    """Test Mp2 Info class - uses PYSCF drive to get molecule.

    Full excitation sequences generated using:

    converter = QubitConverter(JordanWignerMapper()
    ansatz = UCCSD(
        qubit_converter=converter,
        num_particles=num_particles,
        num_spin_orbitals=num_spin_orbitals,
        initializer=mp2_init,
    )
    ansatz._build()

    In practice MP2_Initializer should be passed as an argument to UCC, not used in isolation.
    """

    def setUp(self):
        super().setUp()

        settings.dict_aux_operators = True

    @file_data("../resources/test_data_mp2_initializer.json")
    def test_mp2_initializer(
        self,
        num_spin_orbitals,
        orbital_energies,
        integral_matrix,
        excitations,
        reference_energy,
        mp2_coefficients,
        energy_correction,
        energy_corrections,
        absolute_energy,
    ):

        mp2_init = MP2Initializer(
            num_spin_orbitals,
            np.asarray(orbital_energies),
            np.asarray(integral_matrix),
            reference_energy=reference_energy,
        )

        coeffs, e_deltas = mp2_init.compute_corrections(excitations)

        with self.subTest("test mp2 coefficients"):
            np.testing.assert_array_almost_equal(coeffs, mp2_coefficients, decimal=6)

        with self.subTest("test mp2 energy corrections"):
            np.testing.assert_array_almost_equal(e_deltas, energy_corrections, decimal=6)

        with self.subTest("test mp2 reference energy"):
            np.testing.assert_array_almost_equal(
                mp2_init.reference_energy, reference_energy, decimal=6
            )

        with self.subTest("test overall energy correction"):
            np.testing.assert_array_almost_equal(
                mp2_init.energy_correction, energy_correction, decimal=6
            )

        with self.subTest("test absolute energy"):
            np.testing.assert_array_almost_equal(
                mp2_init.absolute_energy, absolute_energy, decimal=6
            )

        with self.subTest("test num spin orbitals"):
            np.testing.assert_array_almost_equal(
                mp2_init.num_spin_orbitals, num_spin_orbitals, decimal=6
            )

    # TODO test this using Nature approach
    # def test_terms_frozen_core(self):
    #     """ mp2 terms frozen core test """
    #     terms = self.mp2_init.terms(True)
    #     self.assertEqual(16, len(terms.keys()))

    # TODO test this using Nature approach
    # def test_terms_frozen_core_orbital_reduction(self):
    #     """ mp2 terms frozen core orbital reduction test """
    #     terms = self.mp2_init.terms(True, [-3, -2])
    #     self.assertEqual(4, len(terms.keys()))


if __name__ == "__main__":
    unittest.main()
