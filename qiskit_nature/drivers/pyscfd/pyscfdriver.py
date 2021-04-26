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

"""The PySCF Driver."""

import importlib
import logging
import os
import tempfile
import warnings
from enum import Enum
from typing import List, Optional, Tuple, Union

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
    from pyscf import ao2mo, dft, gto, scf
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
        xc_functional: str = "lda,vwn",
        xcf_library: str = "libxc",
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
            xc_functional: Exchange-Correlation functional name as recognized by PySCF. See
                https://pyscf.org/user/dft.html#predefined-xc-functionals-and-functional-aliases for
                more details. Defaults to PySCF's default: 'lda,vwn'.
            xcf_library: the Exchange-Correlation functional library to be used. This can be either
                'libxc' (the default) or 'xcfun'. Depending on this value, a different set of values
                for `xc_functional` will be available. Refer to its documentation for mote details.
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
        # First, ensure that PySCF is actually installed
        self._check_installed()

        validate_min("max_cycle", max_cycle, 1)
        super().__init__(
            molecule=molecule,
            basis=basis,
            hf_method=hf_method.value,
            supports_molecule=True,
        )

        # we use the property-setter to deal with conversion
        self.atom = atom  # type: ignore
        self._unit = unit.value
        self._charge = charge
        self._spin = spin
        self._xc_functional = xc_functional
        self.xcf_library = xcf_library  # validate choice in property setter
        self._conv_tol = conv_tol
        self._max_cycle = max_cycle
        self._init_guess = init_guess.value
        self._max_memory = max_memory
        self._chkfile = chkfile

        self._mol: gto.Mole = None
        self._calc: scf.HF = None

    @property
    def atom(self) -> str:
        """Returns the atom."""
        return self._atom

    @atom.setter
    def atom(self, atom: Union[str, List[str]]) -> None:
        """Sets the atom."""
        if isinstance(atom, list):
            atom = ";".join(atom)
        self._atom = atom.replace("\n", ";")

    @property
    def unit(self) -> str:
        """Returns the unit."""
        return self._unit

    @unit.setter
    def unit(self, unit: str) -> None:
        """Sets the unit."""
        self._unit = unit

    @property
    def charge(self) -> int:
        """Returns the charge."""
        return self._charge

    @charge.setter
    def charge(self, charge: int) -> None:
        """Sets the charge."""
        self._charge = charge

    @property
    def spin(self) -> int:
        """Returns the spin."""
        return self._spin

    @spin.setter
    def spin(self, spin: int) -> None:
        """Sets the spin."""
        self._spin = spin

    @property
    def xc_functional(self) -> str:
        """Returns the Exchange-Correlation functional."""
        return self._xc_functional

    @xc_functional.setter
    def xc_functional(self, xc_functional: str) -> None:
        """Sets the Exchange-Correlation functional."""
        self._xc_functional = xc_functional

    @property
    def xcf_library(self) -> str:
        """Returns the Exchange-Correlation functional library."""
        return self._xcf_library

    @xcf_library.setter
    def xcf_library(self, xcf_library: str) -> None:
        """Sets the Exchange-Correlation functional library."""
        if xcf_library not in ("libxc", "xcfun"):
            raise QiskitNatureError(
                "Invalid XCF library. It can be either 'libxc' or 'xcfun', not "
                "'{}'".format(xcf_library)
            )
        self._xcf_library = xcf_library

    @property
    def conv_tol(self) -> float:
        """Returns the conv_tol."""
        return self._conv_tol

    @conv_tol.setter
    def conv_tol(self, conv_tol: float) -> None:
        """Sets the conv_tol."""
        self._conv_tol = conv_tol

    @property
    def max_cycle(self) -> int:
        """Returns the max_cycle."""
        return self._max_cycle

    @max_cycle.setter
    def max_cycle(self, max_cycle: int) -> None:
        """Sets the max_cycle."""
        self._max_cycle = max_cycle

    @property
    def init_guess(self) -> str:
        """Returns the init_guess."""
        return self._init_guess

    @init_guess.setter
    def init_guess(self, init_guess: str) -> None:
        """Sets the init_guess."""
        self._init_guess = init_guess

    @property
    def max_memory(self) -> int:
        """Returns the max_memory."""
        return self._max_memory

    @max_memory.setter
    def max_memory(self, max_memory: int) -> None:
        """Sets the max_memory."""
        self._max_memory = max_memory

    @property
    def chkfile(self) -> str:
        """Returns the chkfile."""
        return self._chkfile

    @chkfile.setter
    def chkfile(self, chkfile: str) -> None:
        """Sets the chkfile."""
        self._chkfile = chkfile

    @staticmethod
    def _check_installed() -> None:
        """Checks the PySCF is actually installed.

        Raises:
            QiskitNatureError: If PySCF is not installed.
        """
        err_msg = "PySCF is not installed. See https://sunqm.github.io/pyscf/install.html"
        try:
            spec = importlib.util.find_spec("pyscf")  # type: ignore
            if spec is not None:
                return
        except Exception as ex:  # pylint: disable=broad-except
            logger.debug("PySCF check error %s", str(ex))
            raise QiskitNatureError(err_msg) from ex

        raise QiskitNatureError(err_msg)

    def run(self) -> QMolecule:
        """Runs the PySCF driver.

        Returns:
            A QMolecule object containing the raw driver results.
        """
        if self.molecule is not None:
            self.atom = [  # type: ignore
                " ".join(map(str, (name, *coord))) for (name, coord) in self.molecule.geometry
            ]
            self._charge = self.molecule.charge
            self._spin = self.molecule.multiplicity - 1
            self._unit = self.molecule.units.value

        self._build_molecule()
        self.run_pyscf()

        q_mol = self._construct_qmolecule()
        return q_mol

    def _build_molecule(self) -> None:
        """Builds the PySCF molecule object.

        Raises:
             QiskitNatureError: If building the PySCF molecule object failed.
        """
        # Get config from input parameters
        # molecule is in PySCF atom string format e.g. "H .0 .0 .0; H .0 .0 0.2"
        #          or in Z-Matrix format e.g. "H; O 1 1.08; H 2 1.08 1 107.5"
        # other parameters are as per PySCF got.Mole format

        atom = self._check_molecule_format(self.atom)
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
                atom=atom,
                unit=self._unit,
                basis=self._basis,
                max_memory=self._max_memory,
                verbose=verbose,
                output=output,
            )
            self._mol.symmetry = False
            self._mol.charge = self._charge
            self._mol.spin = self._spin
            self._mol.build(parse_arg=False)

            if output is not None:
                self._process_pyscf_log(output)
                try:
                    os.remove(output)
                except Exception:  # pylint: disable=broad-except
                    pass

        except Exception as exc:
            raise QiskitNatureError("Failed to build the PySCF Molecule object.") from exc

    @staticmethod
    def _check_molecule_format(val: str) -> Union[str, List[str]]:
        """Ensures the molecule coordinates are in xyz format.

        This utility automatically converts a z-matrix coordinate format into xyz coordinates.

        Args:
            val: the atomic coordinates.

        Raises:
            QiskitNatureError: If the provided coordinate are mis-formatted.

        Returns:
            The coordinates in xyz format.
        """
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

    def run_pyscf(self) -> None:
        """Runs the PySCF calculation.

        This method is part of the public interface to allow the user to easily overwrite it in a
        subclass to further tailor the behavior to some specific use case.

        Raises:
            QiskitNatureError: Invalid hf method type
        """
        try:
            hf_method_name = self._hf_method.upper()
            hf_method_cls = getattr(scf, hf_method_name)
        except AttributeError as exc:
            raise QiskitNatureError("Failed to load {} HF object.".format(hf_method_name)) from exc

        self._calc = hf_method_cls(self._mol)

        if "KS" in hf_method_name:
            self._calc._numint.libxc = getattr(dft, self.xcf_library)
            self._calc.xc = self.xc_functional

        if self._chkfile is not None and os.path.exists(self._chkfile):
            self._calc.__dict__.update(lib_chkfile.load(self._chkfile, "scf"))

            logger.info("PySCF loaded from chkfile e(hf): %s", self._calc.e_tot)
        else:
            self._calc.conv_tol = self._conv_tol
            self._calc.max_cycle = self._max_cycle
            self._calc.init_guess = self._init_guess
            self._calc.kernel()

            logger.info(
                "PySCF kernel() converged: %s, e(hf): %s",
                self._calc.converged,
                self._calc.e_tot,
            )

    def _construct_qmolecule(self) -> QMolecule:
        """Constructs a QMolecule out of the PySCF calculation object.

        Returns:
            QMolecule: QMolecule populated with driver integrals etc
        """
        # Create driver level molecule object and populate
        q_mol = QMolecule()

        self._populate_q_molecule_driver_info(q_mol)
        self._populate_q_molecule_mol_data(q_mol)
        self._populate_q_molecule_orbitals(q_mol)
        self._populate_q_molecule_ao_integrals(q_mol)
        self._populate_q_molecule_mo_integrals(q_mol)
        self._populate_q_molecule_dipole_integrals(q_mol)

        return q_mol

    def _populate_q_molecule_driver_info(self, q_mol: QMolecule) -> None:
        """Populates the driver information fields.

        Args:
            q_mol: the QMolecule to populate.
        """
        q_mol.origin_driver_name = "PYSCF"
        q_mol.origin_driver_version = pyscf_version
        cfg = [
            "atom={}".format(self._atom),
            "unit={}".format(self._unit),
            "charge={}".format(self._charge),
            "spin={}".format(self._spin),
            "basis={}".format(self._basis),
            "hf_method={}".format(self._hf_method),
            "conv_tol={}".format(self._conv_tol),
            "max_cycle={}".format(self._max_cycle),
            "init_guess={}".format(self._init_guess),
            "max_memory={}".format(self._max_memory),
        ]
        if "ks" in self._hf_method.lower():
            cfg.extend(
                [
                    "xc_functional={}".format(self._xc_functional),
                    "xcf_library={}".format(self._xcf_library),
                ]
            )
        q_mol.origin_driver_config = "\n".join(cfg + [""])

    def _populate_q_molecule_mol_data(self, q_mol: QMolecule) -> None:
        """Populates the molecule information fields.

        Args:
            q_mol: the QMolecule to populate.
        """
        # Molecule geometry
        q_mol.molecular_charge = self._mol.charge
        q_mol.multiplicity = self._mol.spin + 1
        q_mol.num_atoms = self._mol.natm
        q_mol.atom_symbol = []
        q_mol.atom_xyz = np.empty([self._mol.natm, 3])
        _ = self._mol.atom_coords()
        for n_i in range(0, q_mol.num_atoms):
            xyz = self._mol.atom_coord(n_i)
            q_mol.atom_symbol.append(self._mol.atom_pure_symbol(n_i))
            q_mol.atom_xyz[n_i][0] = xyz[0]
            q_mol.atom_xyz[n_i][1] = xyz[1]
            q_mol.atom_xyz[n_i][2] = xyz[2]

        q_mol.nuclear_repulsion_energy = gto.mole.energy_nuc(self._mol)

    def _populate_q_molecule_orbitals(self, q_mol: QMolecule) -> None:
        """Populates the orbital information fields.

        Args:
            q_mol: the QMolecule to populate.
        """
        mo_coeff, mo_coeff_b = self._extract_mo_data("mo_coeff", array_dimension=3)
        mo_occ, mo_occ_b = self._extract_mo_data("mo_occ")
        orbs_energy, orbs_energy_b = self._extract_mo_data("mo_energy")

        if logger.isEnabledFor(logging.DEBUG):
            # Add some more to PySCF output...
            # First analyze() which prints extra information about MO energy and occupation
            self._mol.stdout.write("\n")
            self._calc.analyze()
            # Now labelled orbitals for contributions to the MOs for s,p,d etc of each atom
            self._mol.stdout.write("\n\n--- Alpha Molecular Orbitals ---\n\n")
            dump_mat.dump_mo(self._mol, mo_coeff, digits=7, start=1)
            if mo_coeff_b is not None:
                self._mol.stdout.write("\n--- Beta Molecular Orbitals ---\n\n")
                dump_mat.dump_mo(self._mol, mo_coeff_b, digits=7, start=1)
            self._mol.stdout.flush()

        # Energies and orbital data
        q_mol.hf_energy = self._calc.e_tot
        q_mol.num_molecular_orbitals = mo_coeff.shape[0]
        q_mol.num_alpha = self._mol.nelec[0]
        q_mol.num_beta = self._mol.nelec[1]
        q_mol.mo_coeff = mo_coeff
        q_mol.mo_coeff_b = mo_coeff_b
        q_mol.orbital_energies = orbs_energy
        q_mol.orbital_energies_b = orbs_energy_b
        q_mol.mo_occ = mo_occ
        q_mol.mo_occ_b = mo_occ_b

    def _populate_q_molecule_ao_integrals(self, q_mol: QMolecule) -> None:
        """Populates the atomic orbital integral fields.

        Args:
            q_mol: the QMolecule to populate.
        """
        # 1 and 2 electron AO integrals
        q_mol.hcore = self._calc.get_hcore()
        q_mol.hcore_b = None
        q_mol.kinetic = self._mol.intor_symmetric("int1e_kin")
        q_mol.overlap = self._calc.get_ovlp()
        q_mol.eri = self._mol.intor("int2e", aosym=1)

    def _populate_q_molecule_mo_integrals(self, q_mol: QMolecule) -> None:
        """Populates the molecular orbital integral fields.

        Args:
            q_mol: the QMolecule to populate.
        """
        mo_coeff, mo_coeff_b = q_mol.mo_coeff, q_mol.mo_coeff_b
        norbs = mo_coeff.shape[0]

        mohij = np.dot(np.dot(mo_coeff.T, q_mol.hcore), mo_coeff)
        mohij_b = None
        if mo_coeff_b is not None:
            mohij_b = np.dot(np.dot(mo_coeff_b.T, q_mol.hcore), mo_coeff_b)

        mo_eri = ao2mo.incore.full(self._calc._eri, mo_coeff, compact=False)
        mohijkl = mo_eri.reshape(norbs, norbs, norbs, norbs)
        mohijkl_bb = None
        mohijkl_ba = None
        if mo_coeff_b is not None:
            mo_eri_b = ao2mo.incore.full(self._calc._eri, mo_coeff_b, compact=False)
            mohijkl_bb = mo_eri_b.reshape(norbs, norbs, norbs, norbs)
            mo_eri_ba = ao2mo.incore.general(
                self._calc._eri,
                (mo_coeff_b, mo_coeff_b, mo_coeff, mo_coeff),
                compact=False,
            )
            mohijkl_ba = mo_eri_ba.reshape(norbs, norbs, norbs, norbs)

        # 1 and 2 electron MO integrals
        q_mol.mo_onee_ints = mohij
        q_mol.mo_onee_ints_b = mohij_b
        q_mol.mo_eri_ints = mohijkl
        q_mol.mo_eri_ints_bb = mohijkl_bb
        q_mol.mo_eri_ints_ba = mohijkl_ba

    def _populate_q_molecule_dipole_integrals(self, q_mol: QMolecule) -> None:
        """Populates the dipole integral fields.

        Args:
            q_mol: the QMolecule to populate.
        """
        # dipole integrals
        self._mol.set_common_orig((0, 0, 0))
        ao_dip = self._mol.intor_symmetric("int1e_r", comp=3)
        x_dip_ints = ao_dip[0]
        y_dip_ints = ao_dip[1]
        z_dip_ints = ao_dip[2]

        d_m = self._calc.make_rdm1(self._calc.mo_coeff, self._calc.mo_occ)

        if not (isinstance(d_m, np.ndarray) and d_m.ndim == 2):
            d_m = d_m[0] + d_m[1]

        elec_dip = np.negative(np.einsum("xij,ji->x", ao_dip, d_m).real)
        elec_dip = np.round(elec_dip, decimals=8)
        nucl_dip = np.einsum("i,ix->x", self._mol.atom_charges(), self._mol.atom_coords())
        nucl_dip = np.round(nucl_dip, decimals=8)

        logger.info("HF Electronic dipole moment: %s", elec_dip)
        logger.info("Nuclear dipole moment: %s", nucl_dip)
        logger.info("Total dipole moment: %s", nucl_dip + elec_dip)

        # dipole integrals AO and MO
        q_mol.x_dip_ints = x_dip_ints
        q_mol.y_dip_ints = y_dip_ints
        q_mol.z_dip_ints = z_dip_ints
        q_mol.x_dip_mo_ints = QMolecule.oneeints2mo(x_dip_ints, q_mol.mo_coeff)
        q_mol.x_dip_mo_ints_b = None
        q_mol.y_dip_mo_ints = QMolecule.oneeints2mo(y_dip_ints, q_mol.mo_coeff)
        q_mol.y_dip_mo_ints_b = None
        q_mol.z_dip_mo_ints = QMolecule.oneeints2mo(z_dip_ints, q_mol.mo_coeff)
        q_mol.z_dip_mo_ints_b = None
        if q_mol.mo_coeff_b is not None:
            q_mol.x_dip_mo_ints_b = QMolecule.oneeints2mo(x_dip_ints, q_mol.mo_coeff_b)
            q_mol.y_dip_mo_ints_b = QMolecule.oneeints2mo(y_dip_ints, q_mol.mo_coeff_b)
            q_mol.z_dip_mo_ints_b = QMolecule.oneeints2mo(z_dip_ints, q_mol.mo_coeff_b)
        # dipole moment
        q_mol.nuclear_dipole_moment = nucl_dip
        q_mol.reverse_dipole_sign = True

    def _extract_mo_data(
        self, name: str, array_dimension: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract molecular orbital data from a PySCF calculation object.

        Args:
            name: the name of the molecular orbital data field to extract.
            array_dimension: since PySCF 1.6.2, the alpha and beta components are no longer stored
                as a tuple but as a multi-dimensional numpy array. This argument specifies the
                dimension of that array in such a case. Making this configurable permits this
                function to be used to extract both, MO coefficients (3D array) or MO energies (2D
                array).

        Returns:
            The (alpha, beta) tuple of MO data.
        """
        attr = getattr(self._calc, name)
        if isinstance(attr, tuple):
            attr_alpha = attr[0]
            attr_beta = attr[1]
        else:
            # Since PySCF 1.6.2, instead of a tuple it could be a multi-dimensional array with the
            # first dimension indexing the arrays for alpha and beta
            if len(attr.shape) == array_dimension:
                attr_alpha = attr[0]
                attr_beta = attr[1]
            else:
                attr_alpha = attr
                attr_beta = None
        return attr_alpha, attr_beta

    def _process_pyscf_log(self, logfile: str) -> None:
        """Processes a PySCF logfile.

        Args:
            logfile: the path of the PySCF logfile.
        """
        with open(logfile) as file:
            content = file.readlines()

        for i, _ in enumerate(content):
            if content[i].startswith("System:"):
                content = content[i:]
                break

        logger.debug("PySCF processing messages log:\n%s", "".join(content))
