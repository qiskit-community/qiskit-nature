# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test of Eom EE."""

import warnings
import unittest

from test import QiskitNatureTestCase
import numpy as np

from qiskit.algorithms import NumPyEigensolver
from qiskit.opflow import Z2Symmetries
from qiskit_nature import QiskitNatureError
from qiskit_nature.drivers import PySCFDriver, UnitsType
from qiskit_nature.core import Hamiltonian, TransformationType, QubitMappingType
from qiskit_nature.algorithms import QEomEE


class TestEomEE(QiskitNatureTestCase):
    """Test case for Eom EE."""
    def setUp(self):
        """Setup."""
        super().setUp()
        try:
            atom = 'H .0 .0 .7414; H .0 .0 .0'
            pyscf_driver = PySCFDriver(atom=atom,
                                       unit=UnitsType.ANGSTROM, charge=0, spin=0, basis='sto3g')
            self.molecule = pyscf_driver.run()
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            core = Hamiltonian(transformation=TransformationType.FULL,
                               qubit_mapping=QubitMappingType.PARITY,
                               two_qubit_reduction=True,
                               freeze_core=False,
                               orbital_reduction=[])
            warnings.filterwarnings('always', category=DeprecationWarning)
            qubit_op, _ = core.run(self.molecule)
            exact_eigensolver = NumPyEigensolver(k=2 ** qubit_op.num_qubits)
            result = exact_eigensolver.compute_eigenvalues(qubit_op)
            self.reference = result.eigenvalues.real
        except QiskitNatureError:
            self.skipTest('PYSCF driver does not appear to be installed')

    def test_h2_four_qubits(self):
        """Test H2 with jordan wigner."""
        two_qubit_reduction = False
        qubit_mapping = 'jordan_wigner'
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        core = Hamiltonian(transformation=TransformationType.FULL,
                           qubit_mapping=QubitMappingType.JORDAN_WIGNER,
                           two_qubit_reduction=two_qubit_reduction,
                           freeze_core=False,
                           orbital_reduction=[])
        warnings.filterwarnings('always', category=DeprecationWarning)
        qubit_op, _ = core.run(self.molecule)

        num_orbitals = core.molecule_info['num_orbitals']
        num_particles = core.molecule_info['num_particles']

        eom_ee = QEomEE(num_orbitals=num_orbitals, num_particles=num_particles,
                        qubit_mapping=qubit_mapping, two_qubit_reduction=two_qubit_reduction)
        result = eom_ee.compute_minimum_eigenvalue(qubit_op)
        np.testing.assert_array_almost_equal(self.reference, result['energies'])

    def test_h2_two_qubits(self):
        """Test H2 with parity mapping."""

        two_qubit_reduction = True
        qubit_mapping = 'parity'
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        core = Hamiltonian(transformation=TransformationType.FULL,
                           qubit_mapping=QubitMappingType.PARITY,
                           two_qubit_reduction=two_qubit_reduction,
                           freeze_core=False,
                           orbital_reduction=[])
        warnings.filterwarnings('always', category=DeprecationWarning)
        qubit_op, _ = core.run(self.molecule)

        num_orbitals = core.molecule_info['num_orbitals']
        num_particles = core.molecule_info['num_particles']

        eom_ee = QEomEE(num_orbitals=num_orbitals, num_particles=num_particles,
                        qubit_mapping=qubit_mapping, two_qubit_reduction=two_qubit_reduction)

        result = eom_ee.compute_minimum_eigenvalue(qubit_op)
        np.testing.assert_array_almost_equal(self.reference, result['energies'])

    def test_h2_one_qubit(self):
        """Test H2 with tapering."""
        two_qubit_reduction = False
        qubit_mapping = 'jordan_wigner'
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        core = Hamiltonian(transformation=TransformationType.FULL,
                           qubit_mapping=QubitMappingType.JORDAN_WIGNER,
                           two_qubit_reduction=two_qubit_reduction,
                           freeze_core=False,
                           orbital_reduction=[])
        warnings.filterwarnings('always', category=DeprecationWarning)
        qubit_op, _ = core.run(self.molecule)

        num_orbitals = core.molecule_info['num_orbitals']
        num_particles = core.molecule_info['num_particles']

        z2_symmetries = Z2Symmetries.find_Z2_symmetries(qubit_op)
        tapered_op = z2_symmetries.taper(qubit_op)[5]
        eom_ee = QEomEE(num_orbitals=num_orbitals, num_particles=num_particles,
                        qubit_mapping=qubit_mapping, two_qubit_reduction=two_qubit_reduction,
                        z2_symmetries=tapered_op.z2_symmetries, untapered_op=qubit_op)
        result = eom_ee.compute_minimum_eigenvalue(tapered_op)
        np.testing.assert_array_almost_equal(self.reference, result['energies'])


if __name__ == '__main__':
    unittest.main()
