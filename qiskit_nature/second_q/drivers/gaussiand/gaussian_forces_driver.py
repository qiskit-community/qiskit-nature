# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2020, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Gaussian Forces Driver """

from __future__ import annotations

from typing import Any
from qiskit_nature import QiskitNatureError

from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.formats.watson_translator import watson_to_problem
from qiskit_nature.second_q.problems import VibrationalBasis, VibrationalStructureProblem
import qiskit_nature.optionals as _optionals

from .gaussian_log_driver import GaussianLogDriver
from .gaussian_log_result import GaussianLogResult
from ..vibrational_structure_driver import VibrationalStructureDriver


B3YLP_JCF_DEFAULT = """
#p B3LYP/cc-pVTZ Freq=(Anharm) Int=Ultrafine SCF=VeryTight

CO2 geometry optimization B3LYP/cc-pVTZ

0 1
C        -0.848629    2.067624    0.160992
O         0.098816    2.655801   -0.159738
O        -1.796073    1.479446    0.481721

"""


class GaussianForcesDriver(VibrationalStructureDriver):
    """Gaussian™ 16 forces driver."""

    def __init__(
        self,
        jcf: str | list[str] = B3YLP_JCF_DEFAULT,
        logfile: str | None = None,
        *,
        normalize: bool = True,
    ) -> None:
        r"""
        Args:
            jcf: A job control file conforming to Gaussian™ 16 format. This can
                be provided as a single string with '\\n' line separators or as a list of
                strings.
            logfile: Instead of a job control file a log as output from running such a file
                can optionally be given.
            normalize: Whether to normalize the factors used in creation of the VibrationalEnergy
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
            _optionals.HAS_GAUSSIAN.require_now("GaussianForcesDriver __init__")

    @staticmethod
    @_optionals.HAS_GAUSSIAN.require_in_call
    def from_molecule(
        molecule: MoleculeInfo,
        *,
        basis: str = "sto-3g",
        xcf: str = "B3LYP",
        driver_kwargs: dict[str, Any] | None = None,
    ) -> "GaussianForcesDriver":
        """Creates a driver from a molecule.

        Args:
            molecule: If a molecule is supplied then an appropriate job control file will be built
                from this, and the `basis`, and will be used in precedence of either the `logfile`
                or the `jcf` params.
            basis: The basis set to be used in the resultant job control file when a molecule is
                provided.
            xcf: The exchange-correlation functional to be used in the resultant job control file.
            driver_kwargs: keyword arguments to be passed to driver.

        Returns:
            The constructed driver instance.

        Raises:
            QiskitNatureError: when an unknown unit is encountered.
        """
        # Ignore kwargs parameter for this driver
        del driver_kwargs
        basis = GaussianForcesDriver.to_driver_basis(basis)

        if molecule.units == DistanceUnit.ANGSTROM:
            units = "Angstrom"
        elif molecule.units == DistanceUnit.BOHR:
            units = "Bohr"
        else:
            raise QiskitNatureError(f"Unknown unit '{molecule.units.value}'")
        cfg1 = f"#p {xcf}/{basis} UNITS={units} Freq=(Anharm) Int=Ultrafine SCF=VeryTight\n\n"
        name = "".join(molecule.symbols)
        geom = "\n".join(
            [
                name + " " + " ".join(map(str, coord))
                for (name, coord) in zip(molecule.symbols, molecule.coords)
            ]
        )
        cfg2 = f"{name} geometry optimization\n\n"
        cfg3 = f"{molecule.charge} {molecule.multiplicity}\n{geom}\n\n"

        return GaussianForcesDriver(jcf=cfg1 + cfg2 + cfg3)

    @staticmethod
    def to_driver_basis(basis: str) -> str:
        """Converts basis to a driver acceptable basis.

        Args:
            basis: The basis set to be used.

        Returns:
            A driver acceptable basis.
        """
        if basis == "sto3g":
            return "sto-3g"
        return basis

    def run(self, basis: VibrationalBasis) -> VibrationalStructureProblem:  # type: ignore[override]
        # pylint: disable=arguments-differ
        if self._logfile is not None:
            glr = GaussianLogResult(self._logfile)
        else:
            glr = GaussianLogDriver(jcf=self._jcf).run()

        watson = glr.get_watson_hamiltonian(normalize=self._normalize)

        problem = watson_to_problem(watson, basis=basis)
        return problem
