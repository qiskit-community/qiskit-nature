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

"""A dataclass storing Molecule information."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from qiskit_nature.units import DistanceUnit


@dataclass
class MoleculeInfo:
    """A dataclass storing molecule information."""

    symbols: Sequence[str]
    """The ordered sequence of atoms which make up this molecule."""

    coords: Sequence[tuple[float, float, float]]
    """The XYZ coordinates of the atoms."""

    multiplicity: int = 1
    """The multiplicity of the molecule (`= 2 * spin + 1`)."""

    charge: int = 0
    """The total charge of the molecule."""

    units: DistanceUnit = DistanceUnit.ANGSTROM
    """The distance unit in which the XYZ coordinates are stored."""

    masses: Sequence[float] | None = None
    """The sequence of masses for all atoms part of the molecule."""

    def __str__(self) -> str:
        string = ["Molecule:"]
        string += [f"\tMultiplicity: {self.multiplicity}"]
        string += [f"\tCharge: {self.charge}"]
        string += [f"\tUnit: {self.units.value}"]
        string += ["\tGeometry:"]
        for atom, xyz in zip(self.symbols, self.coords):
            string += [f"\t\t{atom}\t{xyz}"]
        if self.masses is not None:
            string += ["\tMasses:"]
            for mass, atom in zip(self.masses, self.symbols):
                string += [f"\t\t{atom}\t{mass}"]
        return "\n".join(string)
