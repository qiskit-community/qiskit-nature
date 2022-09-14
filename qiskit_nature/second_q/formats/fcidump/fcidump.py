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

"""FCIDump"""

from __future__ import annotations

from typing import List
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from qiskit_nature import QiskitNatureError


from .dumper import _dump_1e_ints, _dump_2e_ints, _write_to_outfile
from .parser import _parse


@dataclass
class FCIDump:
    """
    Qiskit Nature dataclass for representing the FCIDump format.

    The FCIDump format is partially defined in Knowles1989.

    References:
        Knowles1989: Peter J. Knowles, Nicholas C. Handy,
            A determinant based full configuration interaction program,
            Computer Physics Communications, Volume 54, Issue 1, 1989, Pages 75-83,
            ISSN 0010-4655, https://doi.org/10.1016/0010-4655(89)90033-7.
    """

    hij: np.ndarray
    """The alpha 1-electron integrals."""
    hijkl: np.ndarray
    """The alpha/alpha 2-electron integrals."""
    hij_b: np.ndarray | None
    """The beta 1-electron integrals."""
    hijkl_ba: np.ndarray | None
    """The beta/alpha 2-electron integrals."""
    hijkl_bb: np.ndarray | None
    """The beta/beta 2-electron integrals."""
    multiplicity: int
    """The multiplicity."""
    num_electrons: int
    """The number of electrons."""
    num_orbitals: int
    """The number of orbitals."""
    constant_energy: float | None
    """The constant energy comprising (for example) the nuclear repulsion energy and inactive energies."""
    orbsym: List[str] | None
    """A list of spatial symmetries of the orbitals."""
    isym: int
    """The spatial symmetry of the wave function."""

    @classmethod
    def from_file(cls, fcidump: str | Path) -> FCIDump:
        """Constructs an FCIDump object from a file."""
        data = _parse(fcidump if isinstance(fcidump, Path) else Path(fcidump))
        return cls(
            hij=data.get("hij"),
            hijkl=data.get("hijkl"),
            hij_b=data.get("hij_b", None),
            hijkl_ba=data.get("hijkl_ba", None),
            hijkl_bb=data.get("hijkl_bb", None),
            multiplicity=data.get("MS2", 0) + 1,
            num_electrons=data.get("NELEC"),
            num_orbitals=data.get("NORB"),
            constant_energy=data.get("ecore", None),
            orbsym=data.get("ORBSYM", None),
            isym=data.get("ISYM"),
        )

    def to_file(self, fcidump: str | Path) -> None:
        """Dumps an FCIDump object to a file.

        Args:
            fcidump: Path to the output file.
        Raises:
            QiskitNatureError: invalid number of orbitals.
            QiskitNatureError: not all beta-spin related matrices are either None or not None.
        """
        outpath = fcidump if isinstance(fcidump, Path) else Path(fcidump)
        # either all beta variables are None or all of them are not
        if not all(h is None for h in [self.hij_b, self.hijkl_ba, self.hijkl_bb]) and not all(
            h is not None for h in [self.hij_b, self.hijkl_ba, self.hijkl_bb]
        ):
            raise QiskitNatureError("Invalid beta variables.")
        norb = self.num_orbitals
        nelec = self.num_electrons
        einact = self.constant_energy
        ms2 = self.multiplicity - 1
        if norb != self.hij.shape[0] or norb != self.hijkl.shape[0]:
            raise QiskitNatureError(
                f"Invalid number of orbitals {norb} {self.hij.shape[0]} {self.hijkl.shape[0]}"
            )

        mos = range(norb)
        with outpath.open("w", encoding="utf8") as outfile:
            # print header
            outfile.write(f"&FCI NORB={norb:4d},NELEC={nelec:4d},MS2={ms2:4d}\n")
            if self.orbsym is None:
                outfile.write(" ORBSYM=" + "1," * norb + "\n")
            else:
                if len(self.orbsym) != norb:
                    raise QiskitNatureError(f"Invalid number of orbitals {norb} {len(self.orbsym)}")
                outfile.write(" ORBSYM=" + ",".join(self.orbsym) + "\n")
            outfile.write(f" ISYM={self.isym:d},\n&END\n")
            # append 2e integrals
            _dump_2e_ints(self.hijkl, mos, outfile)
            if self.hijkl_ba is not None:
                _dump_2e_ints(self.hijkl_ba.transpose(), mos, outfile)
            if self.hijkl_bb is not None:
                _dump_2e_ints(self.hijkl_bb, mos, outfile)
            # append 1e integrals
            _dump_1e_ints(self.hij, mos, outfile)
            if self.hij_b is not None:
                _dump_1e_ints(self.hij_b, mos, outfile)
            # TODO append MO energies (last three indices are 0)
            # append inactive energy
            _write_to_outfile(outfile, einact, (0, 0, 0, 0))
