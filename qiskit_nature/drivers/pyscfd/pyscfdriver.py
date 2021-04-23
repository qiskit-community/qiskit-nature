# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" PYSCF Driver """

import importlib
import logging
import os
import tempfile
import warnings
from enum import Enum
from typing import List, Optional, Union

import numpy as np
from qiskit.utils.validation import validate_min

from ...exceptions import QiskitNatureError
from ..fermionic_driver import FermionicDriver, HFMethodType
from ..molecule import Molecule
from ..qmolecule import QMolecule
from ..units_type import UnitsType

logger = logging.getLogger(__name__)

try:
    from pyscf import __version__ as pyscf_version
    from pyscf import ao2mo, gto, scf
    from pyscf.lib import chkfile as lib_chkfile
    from pyscf.lib import logger as pylogger
    from pyscf.lib import param
    from pyscf.tools import dump_mat

    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pyscf")
except ImportError:
    logger.info("PySCF is not installed. See https://sunqm.github.io/pyscf/install.html")


class InitialGuess(Enum):
    """ Initial Guess Enum """

    MINAO = "minao"
    HCORE = "1e"
    ONE_E = "1e"
    ATOM = "atom"


class PySCFDriver(FermionicDriver):
    """
    Qiskit chemistry driver using the PySCF library.

    See https://sunqm.github.io/pyscf/
    """

    def __init__(
        self,
        atom: Union[str, List[str]] = "H 0.0 0.0 0.0; H 0.0 0.0 0.735",
        unit: UnitsType = UnitsType.ANGSTROM,
        charge: int = 0,
        spin: int = 0,
        basis: str = "sto3g",
        hf_method: HFMethodType = HFMethodType.RHF,
        conv_tol: float = 1e-9,
        max_cycle: int = 50,
        init_guess: InitialGuess = InitialGuess.MINAO,
        max_memory: Optional[int] = None,
        chkfile: Optional[str] = None,
        molecule: Optional[Molecule] = None,
    ) -> None:
        """
        Args:
            atom: Atom list or string separated by semicolons or line breaks. Each element in the
                list is an atom followed by position e.g. `H 0.0 0.0 0.5`. The preceding example
                shows the `XYZ` format for position but `Z-Matrix` format is supported too here.
            unit: Angstrom or Bohr
            charge: Charge on the molecule
            spin: Spin (2S), in accordance with how PySCF defines a molecule in pyscf.gto.mole.Mole
            basis: Basis set name as recognized by PySCF, e.g. `sto3g`, `321g` etc.
                See https://sunqm.github.io/pyscf/_modules/pyscf/gto/basis.html for a listing.
                Defaults to the minimal basis 'sto3g'.
            hf_method: Hartree-Fock Method type
            conv_tol: Convergence tolerance see PySCF docs and pyscf/scf/hf.py
            max_cycle: Max convergence cycles see PySCF docs and pyscf/scf/hf.py,
                has a min. value of 1.
            init_guess: See PySCF pyscf/scf/hf.py init_guess_by_minao/1e/atom methods
            max_memory: Maximum memory that PySCF should use
            chkfile: Path to a PySCF chkfile from which to load a previously run calculation.
            molecule: A driver independent Molecule definition instance may be provided. When
                a molecule is supplied the ``atom``, ``unit``, ``charge`` and ``spin`` parameters
                are all ignored as the Molecule instance now defines these instead. The Molecule
                object is read when the driver is run and converted to the driver dependent
                configuration for the computation. This allows, for example, the Molecule geometry
                to be updated to compute different points.

        Raises:
            QiskitNatureError: Invalid Input
        """
        self._check_valid()
        if not isinstance(atom, str) and not isinstance(atom, list):
            raise QiskitNatureError("Invalid atom input for PYSCF Driver '{}'".format(atom))

        if isinstance(atom, list):
            atom = ";".join(atom)
        elif isinstance(atom, str):
            atom = atom.replace("\n", ";")

        validate_min("max_cycle", max_cycle, 1)
        super().__init__(
            molecule=molecule,
            basis=basis,
            hf_method=hf_method.value,
            supports_molecule=True,
        )
        self._mol = None
        self._m_f = None
        self._atom = atom
        self._units = unit.value
        self._charge = charge
        self._spin = spin
        self._conv_tol = conv_tol
        self._max_cycle = max_cycle
        self._init_guess = init_guess.value
        self._max_memory = max_memory
        self._chkfile = chkfile

    @staticmethod
    def _check_valid():
        err_msg = "PySCF is not installed. See https://sunqm.github.io/pyscf/install.html"
        try:
            spec = importlib.util.find_spec("pyscf")
            if spec is not None:
                return
        except Exception as ex:  # pylint: disable=broad-except
            logger.debug("PySCF check error %s", str(ex))
            raise QiskitNatureError(err_msg) from ex

        raise QiskitNatureError(err_msg)

    def run(self) -> QMolecule:
        if self.molecule is not None:
            self._atom = ";".join(
                [name + " " + " ".join(map(str, coord)) for (name, coord) in self.molecule.geometry]
            )
            self._charge = self.molecule.charge
            self._spin = self.molecule.multiplicity - 1
            self._units = self.molecule.units.value

        q_mol = self.compute_integrals()

        q_mol.origin_driver_name = "PYSCF"
        cfg = [
            "atom={}".format(self._atom),
            "unit={}".format(self._units),
            "charge={}".format(self._charge),
            "spin={}".format(self._spin),
            "basis={}".format(self._basis),
            "hf_method={}".format(self._hf_method),
            "conv_tol={}".format(self._conv_tol),
            "max_cycle={}".format(self._max_cycle),
            "init_guess={}".format(self._init_guess),
            "max_memory={}".format(self._max_memory),
            "",
        ]
        q_mol.origin_driver_config = "\n".join(cfg)

        return q_mol

    def compute_integrals(self):
        """ compute integrals """
        # Get config from input parameters
        # molecule is in PySCF atom string format e.g. "H .0 .0 .0; H .0 .0 0.2"
        #          or in Z-Matrix format e.g. "H; O 1 1.08; H 2 1.08 1 107.5"
        # other parameters are as per PySCF got.Mole format

        self._atom = self._check_molecule_format(self._atom)
        self._hf_method = self._hf_method.lower()
        if self._max_memory is None:
            self._max_memory = param.MAX_MEMORY

        try:
            verbose = pylogger.QUIET
            output = None
            if logger.isEnabledFor(logging.DEBUG):
                verbose = pylogger.INFO
                file, output = tempfile.mkstemp(suffix=".log")
                os.close(file)

            self._mol = gto.Mole(
                atom=self._atom,
                unit=self._units,
                basis=self._basis,
                max_memory=self._max_memory,
                verbose=verbose,
                output=output,
            )
            self._mol.symmetry = False
            self._mol.charge = self._charge
            self._mol.spin = self._spin
            self._mol.build(parse_arg=False)
            q_mol = self._calculate_integrals()
            if output is not None:
                self._process_pyscf_log(output)
                try:
                    os.remove(output)
                except Exception:  # pylint: disable=broad-except
                    pass

        except Exception as exc:
            raise QiskitNatureError("Failed electronic structure computation") from exc

        return q_mol

    @staticmethod
    def _check_molecule_format(val):
        """If it seems to be zmatrix rather than xyz format we convert before returning"""
        atoms = [x.strip() for x in val.split(";")]
        if atoms is None or len(atoms) < 1:  # pylint: disable=len-as-condition
            raise QiskitNatureError("Molecule format error: " + val)

        # An xyz format has 4 parts in each atom, if not then do zmatrix convert
        # Allows dummy atoms, using symbol 'X' in zmatrix format for coord computation to xyz
        parts = [x.strip() for x in atoms[0].split(" ")]
        if len(parts) != 4:
            try:
                newval = []
                for entry in gto.mole.from_zmatrix(val):
                    if entry[0].upper() != "X":
                        newval.append(entry)
                return newval
            except Exception as exc:
                raise QiskitNatureError("Failed to convert atom string: " + val) from exc

        return val

    def perform_calculation(self):
        """TODO."""
        hf_method = self._hf_method

        if hf_method == "rhf":
            self._m_f = scf.RHF(self._mol)
        elif hf_method == "rohf":
            self._m_f = scf.ROHF(self._mol)
        elif hf_method == "uhf":
            self._m_f = scf.UHF(self._mol)
        else:
            raise QiskitNatureError("Invalid hf_method type: {}".format(hf_method))

        if self._chkfile is not None and os.path.exists(self._chkfile):
            self._m_f.__dict__.update(lib_chkfile.load(self._chkfile, "scf"))
            # We overwrite the convergence information because the chkfile likely does not contain
            # it. It is better to report no information rather than faulty one.
            self._m_f.converged = None
        else:
            self._m_f.conv_tol = self._conv_tol
            self._m_f.max_cycle = self._max_cycle
            self._m_f.init_guess = self._init_guess
            self._m_f.kernel()

    def _calculate_integrals(self):
        """Function to calculate the one and two electron terms. Perform a Hartree-Fock calculation
        in the given basis.

        Returns:
            QMolecule: QMolecule populated with driver integrals etc
        Raises:
            QiskitNatureError: Invalid hf method type
        """
        enuke = gto.mole.energy_nuc(self._mol)

        self.perform_calculation()

        ehf = self._m_f.e_tot
        logger.info(
            "PySCF kernel() converged: %s, e(hf): %s",
            self._m_f.converged,
            self._m_f.e_tot,
        )

        if isinstance(self._m_f.mo_coeff, tuple):
            mo_coeff = self._m_f.mo_coeff[0]
            mo_coeff_b = self._m_f.mo_coeff[1]
            mo_occ = self._m_f.mo_occ[0]
            mo_occ_b = self._m_f.mo_occ[1]
        else:
            # With PySCF 1.6.2, instead of a tuple of 2 dimensional arrays, its a 3 dimensional
            # array with the first dimension indexing to the coeff arrays for alpha and beta
            if len(self._m_f.mo_coeff.shape) > 2:
                mo_coeff = self._m_f.mo_coeff[0]
                mo_coeff_b = self._m_f.mo_coeff[1]
                mo_occ = self._m_f.mo_occ[0]
                mo_occ_b = self._m_f.mo_occ[1]
            else:
                mo_coeff = self._m_f.mo_coeff
                mo_coeff_b = None
                mo_occ = self._m_f.mo_occ
                mo_occ_b = None
        norbs = mo_coeff.shape[0]

        if isinstance(self._m_f.mo_energy, tuple):
            orbs_energy = self._m_f.mo_energy[0]
            orbs_energy_b = self._m_f.mo_energy[1]
        else:
            # See PYSCF 1.6.2 comment above - this was similarly changed
            if len(self._m_f.mo_energy.shape) > 1:
                orbs_energy = self._m_f.mo_energy[0]
                orbs_energy_b = self._m_f.mo_energy[1]
            else:
                orbs_energy = self._m_f.mo_energy
                orbs_energy_b = None

        if logger.isEnabledFor(logging.DEBUG):
            # Add some more to PySCF output...
            # First analyze() which prints extra information about MO energy and occupation
            self._mol.stdout.write("\n")
            self._m_f.analyze()
            # Now labelled orbitals for contributions to the MOs for s,p,d etc of each atom
            self._mol.stdout.write("\n\n--- Alpha Molecular Orbitals ---\n\n")
            dump_mat.dump_mo(self._mol, mo_coeff, digits=7, start=1)
            if mo_coeff_b is not None:
                self._mol.stdout.write("\n--- Beta Molecular Orbitals ---\n\n")
                dump_mat.dump_mo(self._mol, mo_coeff_b, digits=7, start=1)
            self._mol.stdout.flush()

        hij = self._m_f.get_hcore()
        mohij = np.dot(np.dot(mo_coeff.T, hij), mo_coeff)
        mohij_b = None
        if mo_coeff_b is not None:
            mohij_b = np.dot(np.dot(mo_coeff_b.T, hij), mo_coeff_b)

        eri = self._mol.intor("int2e", aosym=1)
        mo_eri = ao2mo.incore.full(self._m_f._eri, mo_coeff, compact=False)
        mohijkl = mo_eri.reshape(norbs, norbs, norbs, norbs)
        mohijkl_bb = None
        mohijkl_ba = None
        if mo_coeff_b is not None:
            mo_eri_b = ao2mo.incore.full(self._m_f._eri, mo_coeff_b, compact=False)
            mohijkl_bb = mo_eri_b.reshape(norbs, norbs, norbs, norbs)
            mo_eri_ba = ao2mo.incore.general(
                self._m_f._eri,
                (mo_coeff_b, mo_coeff_b, mo_coeff, mo_coeff),
                compact=False,
            )
            mohijkl_ba = mo_eri_ba.reshape(norbs, norbs, norbs, norbs)

        # dipole integrals
        self._mol.set_common_orig((0, 0, 0))
        ao_dip = self._mol.intor_symmetric("int1e_r", comp=3)
        x_dip_ints = ao_dip[0]
        y_dip_ints = ao_dip[1]
        z_dip_ints = ao_dip[2]

        d_m = self._m_f.make_rdm1(self._m_f.mo_coeff, self._m_f.mo_occ)
        if not (isinstance(d_m, np.ndarray) and d_m.ndim == 2):
            d_m = d_m[0] + d_m[1]
        elec_dip = np.negative(np.einsum("xij,ji->x", ao_dip, d_m).real)
        elec_dip = np.round(elec_dip, decimals=8)
        nucl_dip = np.einsum("i,ix->x", self._mol.atom_charges(), self._mol.atom_coords())
        nucl_dip = np.round(nucl_dip, decimals=8)
        logger.info("HF Electronic dipole moment: %s", elec_dip)
        logger.info("Nuclear dipole moment: %s", nucl_dip)
        logger.info("Total dipole moment: %s", nucl_dip + elec_dip)

        # Create driver level molecule object and populate
        _q_ = QMolecule()
        _q_.origin_driver_version = pyscf_version
        # Energies and orbits
        _q_.hf_energy = ehf
        _q_.nuclear_repulsion_energy = enuke
        _q_.num_molecular_orbitals = norbs
        _q_.num_alpha = self._mol.nelec[0]
        _q_.num_beta = self._mol.nelec[1]
        _q_.mo_coeff = mo_coeff
        _q_.mo_coeff_b = mo_coeff_b
        _q_.orbital_energies = orbs_energy
        _q_.orbital_energies_b = orbs_energy_b
        _q_.mo_occ = mo_occ
        _q_.mo_occ_b = mo_occ_b
        # Molecule geometry
        _q_.molecular_charge = self._mol.charge
        _q_.multiplicity = self._mol.spin + 1
        _q_.num_atoms = self._mol.natm
        _q_.atom_symbol = []
        _q_.atom_xyz = np.empty([self._mol.natm, 3])
        _ = self._mol.atom_coords()
        for n_i in range(0, _q_.num_atoms):
            xyz = self._mol.atom_coord(n_i)
            _q_.atom_symbol.append(self._mol.atom_pure_symbol(n_i))
            _q_.atom_xyz[n_i][0] = xyz[0]
            _q_.atom_xyz[n_i][1] = xyz[1]
            _q_.atom_xyz[n_i][2] = xyz[2]
        # 1 and 2 electron integrals AO and MO
        _q_.hcore = hij
        _q_.hcore_b = None
        _q_.kinetic = self._mol.intor_symmetric("int1e_kin")
        _q_.overlap = self._m_f.get_ovlp()
        _q_.eri = eri
        _q_.mo_onee_ints = mohij
        _q_.mo_onee_ints_b = mohij_b
        _q_.mo_eri_ints = mohijkl
        _q_.mo_eri_ints_bb = mohijkl_bb
        _q_.mo_eri_ints_ba = mohijkl_ba
        # dipole integrals AO and MO
        _q_.x_dip_ints = x_dip_ints
        _q_.y_dip_ints = y_dip_ints
        _q_.z_dip_ints = z_dip_ints
        _q_.x_dip_mo_ints = QMolecule.oneeints2mo(x_dip_ints, mo_coeff)
        _q_.x_dip_mo_ints_b = None
        _q_.y_dip_mo_ints = QMolecule.oneeints2mo(y_dip_ints, mo_coeff)
        _q_.y_dip_mo_ints_b = None
        _q_.z_dip_mo_ints = QMolecule.oneeints2mo(z_dip_ints, mo_coeff)
        _q_.z_dip_mo_ints_b = None
        if mo_coeff_b is not None:
            _q_.x_dip_mo_ints_b = QMolecule.oneeints2mo(x_dip_ints, mo_coeff_b)
            _q_.y_dip_mo_ints_b = QMolecule.oneeints2mo(y_dip_ints, mo_coeff_b)
            _q_.z_dip_mo_ints_b = QMolecule.oneeints2mo(z_dip_ints, mo_coeff_b)
        # dipole moment
        _q_.nuclear_dipole_moment = nucl_dip
        _q_.reverse_dipole_sign = True

        return _q_

    def _process_pyscf_log(self, logfile):
        with open(logfile) as file:
            content = file.readlines()

        for i, _ in enumerate(content):
            if content[i].startswith("System:"):
                content = content[i:]
                break

        logger.debug("PySCF processing messages log:\n%s", "".join(content))
