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

"""
This module implements the abstract base class for driver modules.
"""

from typing import Optional
from abc import ABC, abstractmethod

from enum import Enum, EnumMeta
import warnings

from .molecule import Molecule
from ..exceptions import QiskitNatureError


class DeprecatedEnum(Enum):
    """
    Shows deprecate message whenever member is accessed
    """

    def __new__(cls, value, *args):
        member = object.__new__(cls)
        member._value_ = value
        member._args = args
        member._show_deprecate = member._deprecate
        return member

    def _deprecate(self):
        warnings.warn(
            f"This {self.__class__.__name__} is deprecated as of 0.2.0, "
            "and will be removed no earlier than 3 months after the release. "
            f"You should use the qiskit_nature.drivers.second_quantization "
            f"{self.__class__.__name__} as a direct replacement instead.",
            DeprecationWarning,
            stacklevel=2,
        )


class DeprecatedEnumMeta(EnumMeta):
    """
    Shows deprecate message whenever member is accessed
    """

    def __getattribute__(cls, name):
        obj = super().__getattribute__(name)
        if isinstance(obj, DeprecatedEnum) and obj._show_deprecate:
            obj._show_deprecate()
        return obj

    def __getitem__(cls, name):
        member = super().__getitem__(name)
        if member._show_deprecate:
            member._show_deprecate()
        return member

    # pylint: disable=redefined-builtin
    def __call__(cls, value, names=None, *, module=None, qualname=None, type=None, start=1):
        obj = super().__call__(
            value, names, module=module, qualname=qualname, type=type, start=start
        )
        if isinstance(obj, DeprecatedEnum) and obj._show_deprecate:
            obj._show_deprecate()
        return obj


class BaseDriver(ABC):
    """**DEPRECATED** Base class for Qiskit Nature drivers."""

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

        Raises:
            QiskitNatureError: Molecule passed but driver doesn't support it.
        """
        warnings.warn(
            "This BaseDriver is deprecated as of 0.2.0, "
            "and will be removed no earlier than 3 months after the release. "
            "You should use the qiskit_nature.drivers.second_quantization "
            "BaseDriver as a direct replacement instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if molecule is not None and not supports_molecule:
            raise QiskitNatureError("Driver doesn't support molecule.")

        self._molecule = molecule
        self._basis = basis
        self._hf_method = hf_method
        self._supports_molecule = supports_molecule

    @abstractmethod
    def run(self):
        """
        Runs a driver to produce an output data structure.
        """
        raise NotImplementedError()

    @property
    def supports_molecule(self) -> bool:
        """
        True for derived classes that support Molecule.

        Returns:
            True if Molecule is supported.
        """
        return self._supports_molecule

    @property
    def molecule(self) -> Optional[Molecule]:
        """return molecule"""
        return self._molecule

    @molecule.setter
    def molecule(self, value: Molecule) -> None:
        """set molecule"""
        if not self.supports_molecule:
            raise QiskitNatureError("Driver doesn't support molecule.")
        self._molecule = value

    @property
    def basis(self) -> str:
        """return basis"""
        return self._basis

    @basis.setter
    def basis(self, value: str) -> None:
        """set basis"""
        self._basis = value

    @property
    def hf_method(self) -> str:
        """return Hartree-Fock method"""
        return self._hf_method

    @hf_method.setter
    def hf_method(self, value: str) -> None:
        """set Hartree-Fock method"""
        self._hf_method = value
