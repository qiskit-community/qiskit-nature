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
from typing import Optional, Union, List

from qiskit.utils.validation import validate_min
from qiskit.exceptions import MissingOptionalLibraryError

from ..qmolecule import QMolecule
from .integrals import compute_integrals
from ..fermionic_driver import FermionicDriver, HFMethodType
from ..molecule import Molecule
from ..units_type import UnitsType
from ...exceptions import QiskitNatureError
from ...deprecation import (
    DeprecatedType,
    warn_deprecated_same_type_name,
    DeprecatedEnum,
    DeprecatedEnumMeta,
)

logger = logging.getLogger(__name__)


class InitialGuess(DeprecatedEnum, metaclass=DeprecatedEnumMeta):
    """Initial Guess Enum"""

    MINAO = "minao"
    HCORE = "1e"
    ONE_E = "1e"
    ATOM = "atom"

    def deprecate(self):
        """show deprecate message"""
        warn_deprecated_same_type_name(
            "0.2.0",
            DeprecatedType.ENUM,
            self.__class__.__name__,
            "from qiskit_nature.drivers.second_quantization as a direct replacement",
            3,
        )


class PySCFDriver(FermionicDriver):
    """**DEPRECATED** Qiskit Nature driver using the PySCF library.

    See https://pyscf.org/install.html
    """

    def __init__(
        self,
        atom: Union[str, List[str]] = "H 0.0 0.0 0.0; H 0.0 0.0 0.735",
        unit: UnitsType = UnitsType.ANGSTROM,
        charge: int = 0,
        spin: int = 0,
        basis: str = "sto3g",
        hf_method: Optional[HFMethodType] = None,
        conv_tol: float = 1e-9,
        max_cycle: int = 50,
        init_guess: Optional[InitialGuess] = None,
        max_memory: Optional[int] = None,
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
                See https://pyscf.org/user/gto.html for a listing.
                Defaults to the minimal basis 'sto3g'.
            hf_method: Hartree-Fock Method type
            conv_tol: Convergence tolerance see PySCF docs and pyscf/scf/hf.py
            max_cycle: Max convergence cycles see PySCF docs and pyscf/scf/hf.py,
                has a min. value of 1.
            init_guess: See PySCF pyscf/scf/hf.py init_guess_by_minao/1e/atom methods
            max_memory: Maximum memory that PySCF should use
            molecule: A driver independent Molecule definition instance may be provided. When
                a molecule is supplied the ``atom``, ``unit``, ``charge`` and ``spin`` parameters
                are all ignored as the Molecule instance now defines these instead. The Molecule
                object is read when the driver is run and converted to the driver dependent
                configuration for the computation. This allows, for example, the Molecule geometry
                to be updated to compute different points.

        Raises:
            QiskitNatureError: Invalid Input
        """
        warn_deprecated_same_type_name(
            "0.2.0",
            DeprecatedType.CLASS,
            "PySCFDriver",
            "from qiskit_nature.drivers.second_quantization.pyscfd",
        )
        self._check_valid()
        if not isinstance(atom, str) and not isinstance(atom, list):
            raise QiskitNatureError(f"Invalid atom input for PYSCF Driver '{atom}'")
        if hf_method is None:
            hf_method = HFMethodType.RHF
        if init_guess is None:
            init_guess = InitialGuess.MINAO

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
        self._atom = atom
        self._units = unit.value
        self._charge = charge
        self._spin = spin
        self._conv_tol = conv_tol
        self._max_cycle = max_cycle
        self._init_guess = init_guess.value
        self._max_memory = max_memory

    @staticmethod
    def _check_valid():
        """Checks that PySCF is actually installed.

        Raises:
            MissingOptionalLibraryError: If PySCF is not installed.
        """
        try:
            spec = importlib.util.find_spec("pyscf")  # type: ignore
            if spec is not None:
                return
        except Exception as ex:  # pylint: disable=broad-except
            logger.debug("PySCF check error %s", str(ex))
            raise MissingOptionalLibraryError(
                libname="PySCF",
                name="PySCFDriver",
                pip_install="pip install 'qiskit-nature[pyscf]'",
                msg="See https://pyscf.org/install.html",
            ) from ex

        raise MissingOptionalLibraryError(
            libname="PySCF",
            name="PySCFDriver",
            pip_install="pip install 'qiskit-nature[pyscf]'",
            msg="See https://pyscf.org/install.html",
        )

    def run(self) -> QMolecule:
        if self.molecule is not None:
            atom = ";".join(
                [name + " " + " ".join(map(str, coord)) for (name, coord) in self.molecule.geometry]
            )
            charge = self.molecule.charge
            spin = self.molecule.multiplicity - 1
            units = self.molecule.units.value
        else:
            atom = self._atom
            charge = self._charge
            spin = self._spin
            units = self._units

        basis = self.basis
        hf_method = self.hf_method

        q_mol = compute_integrals(
            atom=atom,
            unit=units,
            charge=charge,
            spin=spin,
            basis=basis,
            hf_method=hf_method,
            conv_tol=self._conv_tol,
            max_cycle=self._max_cycle,
            init_guess=self._init_guess,
            max_memory=self._max_memory,
        )

        q_mol.origin_driver_name = "PYSCF"
        cfg = [
            f"atom={atom}",
            f"unit={units}",
            f"charge={charge}",
            f"spin={spin}",
            f"basis={basis}",
            f"hf_method={hf_method}",
            f"conv_tol={self._conv_tol}",
            f"max_cycle={self._max_cycle}",
            f"init_guess={self._init_guess}",
            f"max_memory={self._max_memory}",
            "",
        ]
        q_mol.origin_driver_config = "\n".join(cfg)

        return q_mol
