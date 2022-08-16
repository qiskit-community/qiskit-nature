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

import numpy as np

from qiskit_nature.units import DistanceUnit


@dataclass
class MoleculeInfo:
    """TODO."""

    symbols: Sequence[str]
    """TODO."""

    coords: np.ndarray
    """TODO."""

    multiplicity: int = 1
    """TODO."""

    charge: int = 0
    """TODO."""

    units: DistanceUnit = DistanceUnit.BOHR
    """TODO."""

    masses: Sequence[float] | None = None
    """TODO."""

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
