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

""" Test FCIDump Dumping """

import tempfile
import unittest
from abc import ABC, abstractmethod
from test import QiskitNatureTestCase
from pathlib import Path
import numpy as np
from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.operators.tensor_ordering import _phys_to_chem
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
import qiskit_nature.optionals as _optionals


class BaseTestFCIDumpDumper(ABC):
    """FCIDump dumping base test class."""

    def __init__(self):
        self.log = None
        self.dumped = None
        self.core_energy = None
        self.num_molecular_orbitals = None
        self.num_electrons = None
        self.spin_number = None
        self.wf_symmetry = None
        self.orb_symmetries = None
        self.mo_onee = None
        self.mo_eri = None

    @abstractmethod
    def subTest(self, msg, **kwargs):
        # pylint: disable=invalid-name
        """subtest"""
        raise Exception("Abstract method")

    @abstractmethod
    def assertAlmostEqual(self, first, second, places=None, msg=None, delta=None):
        """assert Almost Equal"""
        raise Exception("Abstract method")

    @abstractmethod
    def assertEqual(self, first, second, msg=None):
        """assert equal"""
        raise Exception("Abstract method")

    @abstractmethod
    def assertSequenceEqual(self, seq1, seq2, msg=None, seq_type=None):
        """assert Sequence Equal"""
        raise Exception("Abstract method")

    def test_dumped_inactive_energy(self):
        """dumped inactive energy test"""
        self.log.debug("Dumped inactive energy is %g", self.dumped["ECORE"])
        self.assertAlmostEqual(self.dumped["ECORE"], self.core_energy, places=3)

    def test_dumped_num_molecular_orbitals(self):
        """dumped number of orbitals test"""
        self.log.debug("Dumped number of orbitals is %d", self.dumped["NORB"])
        self.assertEqual(self.dumped["NORB"], self.num_molecular_orbitals)

    def test_dumped_num_electrons(self):
        """dumped number of electrons test"""
        self.log.debug("Dumped number of electrons is %d", self.dumped["NELEC"])
        self.assertEqual(self.dumped["NELEC"], self.num_electrons)

    def test_dumped_spin_number(self):
        """dumped spin number test"""
        self.log.debug("Dumped spin number is %d", self.dumped["MS2"])
        self.assertEqual(self.dumped["MS2"], self.spin_number)

    def test_dumped_wave_function_sym(self):
        """dumped wave function symmetry test"""
        self.log.debug("Dumped wave function symmetry is %d", self.dumped["ISYM"])
        self.assertEqual(self.dumped["ISYM"], self.wf_symmetry)

    def test_dumped_orbital_syms(self):
        """dumped orbital symmetries test"""
        self.log.debug("Dumped orbital symmetries is %s", self.dumped["ORBSYM"])
        self.assertEqual(self.dumped["ORBSYM"], self.orb_symmetries)

    def test_dumped_h1(self):
        """dumped h1 integrals test"""
        self.log.debug("Dumped h1 integrals are %s", self.dumped["H1"])
        np.testing.assert_array_almost_equal(
            np.absolute(self.dumped["H1"]), np.absolute(self.mo_onee), decimal=4
        )

    def test_dumped_h2(self):
        """dumped h2 integrals test"""
        self.log.debug("Dumped h2 integrals are %s", self.dumped["H2"])
        np.testing.assert_array_almost_equal(
            np.absolute(self.dumped["H2"]), np.absolute(self.mo_eri), decimal=4
        )


class TestFCIDumpDumpH2(QiskitNatureTestCase, BaseTestFCIDumpDumper):
    """RHF FCIDump tests."""

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def setUp(self):
        super().setUp()
        self.core_energy = 0.7199
        self.num_molecular_orbitals = 2
        self.num_electrons = 2
        self.spin_number = 0
        self.wf_symmetry = 1
        self.orb_symmetries = [1, 1]
        self.mo_onee = [[1.2563, 0.0], [0.0, 0.4719]]
        self.mo_eri = [0.6757, 0.0, 0.1809, 0.6646, 0.0, 0.6986]
        try:
            driver = PySCFDriver(
                atom="H .0 .0 .0; H .0 .0 0.735",
                unit=DistanceUnit.ANGSTROM,
                charge=0,
                spin=0,
                basis="sto3g",
            )
            problem = driver.run()

            with tempfile.NamedTemporaryFile() as dump:
                self.dump(problem, Path(dump.name))
                # pylint: disable=import-outside-toplevel,import-error
                from pyscf.tools import fcidump as pyscf_fcidump

                self.dumped = pyscf_fcidump.read(dump.name)
        except QiskitNatureError as ex:
            self.skipTest(str(ex))

    @staticmethod
    def dump(problem: ElectronicStructureProblem, outpath: Path) -> None:
        """Convenience method to produce an FCIDump output file.

        Args:
            outpath: Path to the output file.
            problem: The ElectronicStructureProblem to be dumped.
        """

        electronic_energy = problem.hamiltonian
        electronic_integrals = electronic_energy.electronic_integrals
        hijkl = electronic_integrals.alpha.get("++--", None)
        hijkl_ba = electronic_integrals.beta_alpha.get("++--", None)
        hijkl_bb = electronic_integrals.beta.get("++--", None)
        fcidump = FCIDump(
            num_electrons=problem.num_alpha + problem.num_beta,
            hij=electronic_integrals.alpha.get("+-", None),
            hij_b=electronic_integrals.beta.get("+-", None),
            hijkl=_phys_to_chem(hijkl) if hijkl is not None else None,
            hijkl_ba=_phys_to_chem(hijkl_ba) if hijkl_ba is not None else None,
            hijkl_bb=_phys_to_chem(hijkl_bb) if hijkl_bb is not None else None,
            multiplicity=problem.molecule.multiplicity,
            constant_energy=electronic_energy.nuclear_repulsion_energy,
        )
        fcidump.to_file(outpath)


class TestFCIDumpDumpOH(QiskitNatureTestCase):
    """RHF FCIDump tests."""

    def test_dump_unrestricted_spin(self):
        """Tests dumping unrestricted spin data.
        PySCF cannot handle this themselves, so we need to rely on a custom FCIDUMP file, load it,
        dump it again, and make sure it did not change."""
        path = self.get_resource_path("test_fcidump_oh.fcidump", "second_q/formats/fcidump")
        fcidump = FCIDump.from_file(path)
        with tempfile.TemporaryDirectory() as dump_dir:
            dump_file = Path(dump_dir) / "fcidump"
            fcidump.to_file(dump_file)

            with open(path, "r", encoding="utf-8") as reference:
                with open(dump_file, "r", encoding="utf-8") as result:
                    for ref, res in zip(reference.readlines(), result.readlines()):
                        self.assertEqual(ref.strip(), res.strip())


if __name__ == "__main__":
    unittest.main()
