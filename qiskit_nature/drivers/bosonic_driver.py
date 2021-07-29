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

"""
This module implements the abstract base class for bosonic driver modules.
"""

from abc import abstractmethod
from typing import Optional

from qiskit_nature.drivers.watson_hamiltonian import WatsonHamiltonian
from .base_driver import BaseDriver
from .molecule import Molecule
from ..deprecation import DeprecatedType, warn_deprecated


class BosonicDriver(BaseDriver):
    """
    Base class for Qiskit Nature's bosonic drivers.
    """

    @abstractmethod
    def __init__(
        self,
        molecule: Optional[Molecule] = None,
        basis: str = "sto3g",
        hf_method: str = "rhf",
        supports_molecule: bool = False,
    ) -> None:
        """
        Args:
            molecule: molecule
            basis: basis set
            hf_method: Hartree-Fock Method type
            supports_molecule: Indicates if driver supports molecule
        """
        super().__init__(
            molecule, basis=basis, hf_method=hf_method, supports_molecule=supports_molecule
        )
        warn_deprecated(
            "0.2.0",
            old_type=DeprecatedType.CLASS,
            old_name="BosonicDriver",
            new_name="VibrationalStructureDriver",
            additional_msg="from qiskit_nature.drivers.second_quantization",
        )

    @abstractmethod
    def run(self) -> WatsonHamiltonian:
        """
        Runs driver to produce a WatsonHamiltonian output.

        Returns:
            A WatsonHamiltonian comprising the bosonic data.
        """
        pass
