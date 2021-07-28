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

""" Gaussian Forces Driver """

from typing import Union, List, Optional, Dict, Any
from qiskit_nature import QiskitNatureError

from ..watson_hamiltonian import WatsonHamiltonian
from ...units_type import UnitsType
from ..bosonic_driver import BosonicDriver
from ...molecule import Molecule
from .gaussian_utils import check_valid
from .gaussian_log_driver import GaussianLogDriver
from .gaussian_log_result import GaussianLogResult


B3YLP_JCF_DEFAULT = """
#p B3LYP/cc-pVTZ Freq=(Anharm) Int=Ultrafine SCF=VeryTight

CO2 geometry optimization B3LYP/cc-pVTZ

0 1
C        -0.848629    2.067624    0.160992
O         0.098816    2.655801   -0.159738
O        -1.796073    1.479446    0.481721

"""


class GaussianForcesDriver(BosonicDriver):
    """Gaussian™ 16 forces driver."""

    def __init__(
        self,
        jcf: Union[str, List[str]] = B3YLP_JCF_DEFAULT,
        logfile: Optional[str] = None,
        normalize: bool = True,
    ) -> None:
        r"""
        Args:
            jcf: A job control file conforming to Gaussian™ 16 format. This can
                be provided as a single string with '\\n' line separators or as a list of
                strings.
            logfile: Instead of a job control file a log as output from running such a file
                can optionally be given.
            normalize: Whether to normalize the factors used in creation of the WatsonHamiltonian
                 as returned when this driver is run.

        Raises:
            QiskitNatureError: If `jcf` given and Gaussian™ 16 executable
                cannot be located.
        """
        super().__init__()
        self._jcf = jcf
        self._logfile = None
        self._normalize = normalize

        # Logfile has precedence if supplied
        if logfile is not None:
            self._jcf = None
            self._logfile = logfile

        # If running from a jcf we need Gaussian™ 16 so check if we have a
        # valid install.
        if self._logfile is None:
            check_valid()

    @staticmethod
    def from_molecule(
        molecule: Molecule,
        basis: str = "sto-3g",
        driver_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "GaussianForcesDriver":
        """
        Args:
            molecule: If a molecule is supplied then an appropriate job control file will be
                       built from this, and the `basis`, and will be used in precedence of either the
                       `logfile` or the `jcf` params.
            basis: The basis set to be used in the resultant job control file when a
                    molecule is provided.
            driver_kwargs: kwargs to be passed to driver
        Returns:
            driver
        Raises:
            QiskitNatureError: Unknown unit
        """
        # Ignore kwargs parameter for this driver
        del driver_kwargs
        GaussianForcesDriver.check_installed()
        basis = GaussianForcesDriver.to_driver_basis(basis)

        if molecule.units == UnitsType.ANGSTROM:
            units = "Angstrom"
        elif molecule.units == UnitsType.BOHR:
            units = "Bohr"
        else:
            raise QiskitNatureError("Unknown unit '{}'".format(molecule.units.value))
        cfg1 = f"#p B3LYP/{basis} UNITS={units} Freq=(Anharm) Int=Ultrafine SCF=VeryTight\n\n"
        name = "".join([name for (name, _) in molecule.geometry])
        geom = "\n".join(
            [name + " " + " ".join(map(str, coord)) for (name, coord) in molecule.geometry]
        )
        cfg2 = f"{name} geometry optimization\n\n"
        cfg3 = f"{molecule.charge} {molecule.multiplicity}\n{geom}\n\n"

        return GaussianForcesDriver(jcf=cfg1 + cfg2 + cfg3)

    @staticmethod
    def to_driver_basis(basis: str) -> str:
        """
        Converts basis to a driver acceptable basis
        Args:
            basis: The basis set to be used
        Returns:
            driver acceptable basis
        """
        if basis == "sto3g":
            return "sto-3g"
        return basis

    @staticmethod
    def check_installed():
        """check if installed"""
        check_valid()

    def run(self) -> WatsonHamiltonian:
        if self._logfile is not None:
            glr = GaussianLogResult(self._logfile)
        else:
            glr = GaussianLogDriver(jcf=self._jcf).run()

        return glr.get_watson_hamiltonian(self._normalize)
