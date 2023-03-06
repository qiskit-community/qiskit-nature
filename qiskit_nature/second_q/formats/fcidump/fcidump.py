# This code is part of Qiskit.
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

"""FCIDump"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.operators.symmetric_two_body import SymmetricTwoBodyIntegrals
from qiskit_nature.second_q.operators.tensor_ordering import (
    IndexType,
    find_index_order,
    to_chemist_ordering,
)

from .dumper import _dump_1e_ints, _dump_2e_ints, _write_to_outfile


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

    num_electrons: int
    """The number of electrons."""
    hij: np.ndarray
    """The alpha 1-electron integrals."""
    hijkl: np.ndarray | SymmetricTwoBodyIntegrals
    """The alpha/alpha 2-electron integrals ordered in chemist' notation."""
    hij_b: np.ndarray | None = None
    """The beta 1-electron integrals."""
    hijkl_bb: np.ndarray | SymmetricTwoBodyIntegrals | None = None
    """The beta/beta 2-electron integrals ordered in chemist' notation."""
    hijkl_ba: np.ndarray | SymmetricTwoBodyIntegrals | None = None
    """The beta/alpha 2-electron integrals ordered in chemist' notation."""
    constant_energy: float | None = None
    """The constant energy comprising (for example) the nuclear repulsion energy and inactive
    energies."""
    multiplicity: int = 1
    """The multiplicity."""
    orbsym: Sequence[int] | None = None
    """A list of spatial symmetries of the orbitals."""
    isym: int = 1
    """The spatial symmetry of the wave function."""

    @property
    def num_orbitals(self) -> int:
        """The number of orbitals."""
        return self.hij.shape[0]

    @property
    def _hijkl(self) -> np.ndarray | SymmetricTwoBodyIntegrals:
        if not isinstance(self._chemist_hijkl, SymmetricTwoBodyIntegrals):
            warnings.warn(
                DeprecationWarning(
                    "The `np.ndarray` return type of the `FCIDump.hijkl` property is deprecated as "
                    "of version 0.6.0 and support for it will be removed no sooner than 3 months "
                    "after the release. Instead, the new return type will be a "
                    "`qiskit_nature.second_q.operators.symmetric_two_body.SymmetricTwoBodyIntegrals`"
                    " (which can be used in place of `np.ndarray` objects). You may switch to the "
                    "new return type immediately by setting "
                    "`qiskit_nature.settings.use_symmetry_reduced_integrals` to `True`."
                ),
                stacklevel=3,
            )
        return self._chemist_hijkl

    @_hijkl.setter
    def _hijkl(self, hijkl: np.ndarray | SymmetricTwoBodyIntegrals) -> None:
        self._original_index_order = find_index_order(hijkl)
        if self._original_index_order != IndexType.CHEMIST:
            warnings.warn(
                DeprecationWarning(
                    "Setting `FCIDump.hijkl` to non-chemistry ordered integrals is deprecated as of "
                    "version 0.6.0 and support for it will be removed no sooner than 3 months after "
                    "the release. Instead, ensure that your integrals are already ordered in "
                    "chemist' notation for example by using the "
                    "`qiskit_nature.second_q.operators.tensor_ordering.to_chemist_ordering` utility."
                ),
                stacklevel=3,
            )
        if isinstance(hijkl, SymmetricTwoBodyIntegrals):
            self._chemist_hijkl = hijkl
        else:
            self._chemist_hijkl = to_chemist_ordering(hijkl)  # type: ignore[assignment]

    @property
    def _hijkl_bb(self) -> np.ndarray | SymmetricTwoBodyIntegrals | None:
        if self._chemist_hijkl_bb is not None and not isinstance(
            self._chemist_hijkl_bb, SymmetricTwoBodyIntegrals
        ):
            warnings.warn(
                DeprecationWarning(
                    "The `np.ndarray` return type of the `FCIDump.hijkl_bb` property is deprecated as "
                    "of version 0.6.0 and support for it will be removed no sooner than 3 months "
                    "after the release. Instead, the new return type will be a "
                    "`qiskit_nature.second_q.operators.symmetric_two_body.SymmetricTwoBodyIntegrals`"
                    " (which can be used in place of `np.ndarray` objects). You may switch to the "
                    "new return type immediately by setting "
                    "`qiskit_nature.settings.use_symmetry_reduced_integrals` to `True`."
                ),
                stacklevel=3,
            )
        return self._chemist_hijkl_bb

    @_hijkl_bb.setter
    def _hijkl_bb(self, hijkl_bb: np.ndarray | SymmetricTwoBodyIntegrals | None) -> None:
        if hijkl_bb is None:
            self._chemist_hijkl_bb = None
            return
        if isinstance(hijkl_bb, SymmetricTwoBodyIntegrals):
            self._chemist_hijkl_bb = hijkl_bb
        else:
            if find_index_order(hijkl_bb) != IndexType.CHEMIST:
                warnings.warn(
                    DeprecationWarning(
                        "Setting `FCIDump.hijkl_bb` to non-chemistry ordered integrals is deprecated "
                        "as of version 0.6.0 and support for it will be removed no sooner than 3 "
                        "months after the release. Instead, ensure that your integrals are already "
                        "ordered in chemist' notation for example by using the "
                        "`qiskit_nature.second_q.operators.tensor_ordering.to_chemist_ordering` "
                        "utility."
                    ),
                    stacklevel=3,
                )
            self._chemist_hijkl_bb = to_chemist_ordering(hijkl_bb)  # type: ignore[assignment]

    @property
    def _hijkl_ba(self) -> np.ndarray | SymmetricTwoBodyIntegrals | None:
        if self._chemist_hijkl_ba is not None and not isinstance(
            self._chemist_hijkl_ba, SymmetricTwoBodyIntegrals
        ):
            warnings.warn(
                DeprecationWarning(
                    "The `np.ndarray` return type of the `FCIDump.hijkl_ba` property is deprecated as "
                    "of version 0.6.0 and support for it will be removed no sooner than 3 months "
                    "after the release. Instead, the new return type will be a "
                    "`qiskit_nature.second_q.operators.symmetric_two_body.SymmetricTwoBodyIntegrals`"
                    " (which can be used in place of `np.ndarray` objects). You may switch to the "
                    "new return type immediately by setting "
                    "`qiskit_nature.settings.use_symmetry_reduced_integrals` to `True`."
                ),
                stacklevel=3,
            )
        return self._chemist_hijkl_ba

    @_hijkl_ba.setter
    def _hijkl_ba(self, hijkl_ba: np.ndarray | SymmetricTwoBodyIntegrals | None) -> None:
        if hijkl_ba is None:
            self._chemist_hijkl_ba = None
            return
        if isinstance(hijkl_ba, SymmetricTwoBodyIntegrals):
            self._chemist_hijkl_ba = hijkl_ba
        else:
            if self._original_index_order != IndexType.CHEMIST:
                warnings.warn(
                    DeprecationWarning(
                        "Setting `FCIDump.hijkl_ba` to non-chemistry ordered integrals is deprecated "
                        "as of version 0.6.0 and support for it will be removed no sooner than 3 "
                        "months after the release. Instead, ensure that your integrals are already "
                        "ordered in chemist' notation for example by using the "
                        "`qiskit_nature.second_q.operators.tensor_ordering.to_chemist_ordering` "
                        "utility."
                    ),
                    stacklevel=3,
                )
            self._chemist_hijkl_ba = to_chemist_ordering(  # type: ignore[assignment]
                hijkl_ba, index_order=self._original_index_order
            )

    @classmethod
    def from_file(cls, fcidump: str | Path) -> FCIDump:
        """Constructs an FCIDump object from a file.

        Args:
            fcidump: Path to the input file.

        Returns:
            A :class:`.FCIDump` instance.
        """
        # pylint: disable=cyclic-import
        from .parser import _parse

        return _parse(fcidump if isinstance(fcidump, Path) else Path(fcidump))

    def to_file(self, fcidump: str | Path) -> None:
        """Dumps an FCIDump object to a file.

        Args:
            fcidump: Path to the output file.
        Raises:
            QiskitNatureError: not all beta-spin related matrices are either None or not None.
            QiskitNatureError: if the dimensions of the provided integral matrices do not match.
        """
        outpath = fcidump if isinstance(fcidump, Path) else Path(fcidump)
        # either all beta variables are None or all of them are not
        if not all(h is None for h in [self.hij_b, self.hijkl_ba, self.hijkl_bb]) and not all(
            h is not None for h in [self.hij_b, self.hijkl_ba, self.hijkl_bb]
        ):
            raise QiskitNatureError("Invalid beta variables.")
        if set(self.hij.shape) != set(self.hijkl.shape):
            raise QiskitNatureError(
                "The number of orbitals of the 1- and 2-body matrices do not match: "
                f"{set(self.hij.shape)} vs. {set(self.hijkl.shape)}"
            )
        norb = self.hij.shape[0]
        nelec = self.num_electrons
        einact = self.constant_energy
        ms2 = self.multiplicity - 1

        mos = range(norb)
        with outpath.open("w", encoding="utf8") as outfile:
            # print header
            outfile.write(f"&FCI NORB={norb:4d},NELEC={nelec:4d},MS2={ms2:4d},\n")
            if self.orbsym is None:
                outfile.write(" ORBSYM=" + "1," * norb + ",\n")
            else:
                if len(self.orbsym) != norb:
                    raise QiskitNatureError(f"Invalid number of orbitals {norb} {len(self.orbsym)}")
                outfile.write(" ORBSYM=" + ",".join(str(o) for o in self.orbsym) + ",\n")
            outfile.write(f" ISYM={self.isym:d},\n&END\n")
            # append 2e integrals
            _dump_2e_ints(self.hijkl, mos, outfile)
            if self.hijkl_ba is not None:
                _dump_2e_ints(np.transpose(self.hijkl_ba), mos, outfile, beta=1)
            if self.hijkl_bb is not None:
                _dump_2e_ints(self.hijkl_bb, mos, outfile, beta=2)
            # append 1e integrals
            _dump_1e_ints(self.hij, mos, outfile)
            if self.hij_b is not None:
                _dump_1e_ints(self.hij_b, mos, outfile, beta=True)
            # TODO append MO energies (last three indices are 0)
            # append inactive energy
            if einact is not None:
                _write_to_outfile(outfile, einact, (0, 0, 0, 0))


# inject the property getter/setter methods for the dataclass attributes
# See also: https://stackoverflow.com/a/61480946
FCIDump.hijkl = FCIDump._hijkl  # type: ignore[assignment]
FCIDump.hijkl_bb = FCIDump._hijkl_bb  # type: ignore[assignment]
FCIDump.hijkl_ba = FCIDump._hijkl_ba  # type: ignore[assignment]
