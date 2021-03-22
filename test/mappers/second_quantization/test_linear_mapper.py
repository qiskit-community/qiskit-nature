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

""" Test Jordan Wigner Mapper """

import unittest

from test import QiskitNatureTestCase

from qiskit.opflow import X, Y, Z, I

from qiskit_nature.problems.second_quantization.vibrational.vibrational_spin_op_builder import \
    build_vibrational_spin_op
from qiskit_nature.mappers.second_quantization import LinearMapper
from qiskit_nature.drivers import GaussianForcesDriver

class TestLinearMapper(QiskitNatureTestCase):
    """ Test Jordan Wigner Mapper """

    REF_OPERATOR = None

    def test_mapping(self):
        """ Test mapping to qubit operator """

        logfile = self.get_resource_path('CO2_freq_B3LYP_ccpVDZ.log')
        driver = GaussianForcesDriver(logfile=logfile)

        watson_hamiltonian = driver.run()
        basis_size = 2
        truncation_order = 3

        vib_spin_op = build_vibrational_spin_op(watson_hamiltonian, basis_size,
                                                        truncation_order)

        mapper = LinearMapper()
        qubit_op = mapper.map(vib_spin_op)

        # Note: The PauliSumOp equals, as used in the test below, use the equals of the
        #       SparsePauliOp which in turn uses np.allclose() to determine equality of
        #       coeffs. So the reference operator above will be matched on that basis so
        #       we don't need to worry about tiny precision changes for any reason.

        self.assertEqual(qubit_op, TestLinearMapper.REF_OPERATOR)

if __name__ == '__main__':
    unittest.main()
