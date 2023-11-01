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

"""FCIDump parser."""

from __future__ import annotations

from typing import Any
import re
from pathlib import Path
import numpy as np

from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.operators.symmetric_two_body import S4Integrals, S8Integrals
from .fcidump import FCIDump


def _parse(fcidump: Path) -> FCIDump:
    """Parses a FCIDump output.

    Args:
        fcidump: Path to the FCIDump file.
    Raises:
        QiskitNatureError: If the input file cannot be found, if a required field in the FCIDump
            file is missing, if wrong integral indices are encountered, or if the alpha/beta or
            beta/alpha 2-electron integrals are mixed.
    Returns:
        A dictionary storing the parsed data.
    """
    try:
        with fcidump.open("r", encoding="utf8") as file:
            fcidump_str = file.read()
    except OSError as ex:
        raise QiskitNatureError(f"Input file '{fcidump}' cannot be read!") from ex

    output: dict[str, Any] = {}

    # FCIDump starts with a Fortran namelist of meta data
    namelist_end = re.search("(/|&END)", fcidump_str)
    metadata = fcidump_str[: namelist_end.start(0)]
    metadata = " ".join(metadata.split())  # replace duplicate whitespace and newlines
    # we know what elements to look for so we don't get too fancy with the parsing
    # pattern explanation:
    #  .*?      any text
    #  (*|*),   match either part of this group followed by a comma
    #  [-+]?    match up to a single - or +
    #  \d*.\d+  number format
    pattern = r".*?([-+]?\d*\.\d+|[-+]?\d+),"
    # we parse the values in the order in which they are listed in Knowles1989
    _norb = re.search("NORB" + pattern, metadata)
    if _norb is None:
        raise QiskitNatureError("The required NORB entry of the FCIDump format is missing!")
    norb = int(_norb.groups()[0])
    output["NORB"] = norb
    _nelec = re.search("NELEC" + pattern, metadata)
    if _nelec is None:
        raise QiskitNatureError("The required NELEC entry of the FCIDump format is missing!")
    output["NELEC"] = int(_nelec.groups()[0])
    # the rest of these values may occur and are set to their defaults otherwise
    _ms2 = re.search("MS2" + pattern, metadata)
    output["MS2"] = int(_ms2.groups()[0]) if _ms2 else 0
    _isym = re.search("ISYM" + pattern, metadata)
    output["ISYM"] = int(_isym.groups()[0]) if _isym else 1
    # ORBSYM holds a list, thus it requires a little different treatment
    _orbsym = re.search(r"ORBSYM.*?" + r"(\d+)," * norb, metadata)
    output["ORBSYM"] = [int(s) for s in _orbsym.groups()] if _orbsym else [1] * norb
    _iprtim = re.search("IPRTIM" + pattern, metadata)
    output["IPRTIM"] = int(_iprtim.groups()[0]) if _iprtim else -1
    _int = re.search("INT" + pattern, metadata)
    output["INT"] = int(_int.groups()[0]) if _int else 5
    _memory = re.search("MEMORY" + pattern, metadata)
    output["MEMORY"] = int(_memory.groups()[0]) if _memory else 10000
    _core = re.search("CORE" + pattern, metadata)
    output["CORE"] = float(_core.groups()[0]) if _core else 0.0
    _maxit = re.search("MAXIT" + pattern, metadata)
    output["MAXIT"] = int(_maxit.groups()[0]) if _maxit else 25
    _thr = re.search("THR" + pattern, metadata)
    output["THR"] = float(_thr.groups()[0]) if _thr else 1e-5
    _thrres = re.search("THRRES" + pattern, metadata)
    output["THRRES"] = float(_thrres.groups()[0]) if _thrres else 0.1
    _nroot = re.search("NROOT" + pattern, metadata)
    output["NROOT"] = int(_nroot.groups()[0]) if _nroot else 1

    # If the FCIDump file resulted from an unrestricted spin calculation the indices will label spin
    # rather than molecular orbitals. This means, that a line must exist which encodes the
    # coefficient for the spin orbital with index (norb*2, norb*2). By checking for such a line we
    # can distinguish between unrestricted and restricted FCIDump files.
    _uhf = bool(
        re.search(
            rf".*(\s+{norb * 2}\s+{norb * 2}\s+0\s+0)",
            fcidump_str[namelist_end.start(0) :],
        )
    )

    # the rest of the FCIDump will hold lines of the form x i a j b
    # a few cases have to be treated differently:
    # i, a, j and b are all zero: x is the core energy
    # TODO: a, j and b are all zero: x is the energy of the i-th MO  (often not supported)
    # j and b are both zero: x is the 1e-integral between i and a (x = <i|h|a>)
    # otherwise: x is the Coulomb integral ( x = (ia|jb) )
    hij = np.zeros((norb, norb))
    s8_hijkl = S8Integrals.zero(norb)
    hij_b = s4_hijkl_ba = s8_hijkl_bb = None
    if _uhf:
        hij_b = np.zeros((norb, norb))
        s4_hijkl_ba = S4Integrals.zero(norb)
        s8_hijkl_bb = S8Integrals.zero(norb)

    orbital_data = fcidump_str[namelist_end.end(0) :].split("\n")
    for orbital in orbital_data:
        if not orbital:
            continue
        x = float(orbital.split()[0])
        # Note: differing naming than ijkl due to E741 and this iajb is inline with this:
        # https://hande.readthedocs.io/en/latest/manual/integrals.html#fcidump-format
        i, a, j, b = [int(i) for i in orbital.split()[1:]]
        if i == a == j == b == 0:
            output["ecore"] = x
        elif a == j == b == 0:
            # TODO: x is the energy of the i-th MO
            continue
        elif j == b == 0:
            if i > a:
                # ensure that we set the upper triangular values
                i, a = a, i
            try:
                hij[i - 1][a - 1] = x
            except IndexError as ex:
                if _uhf:
                    hij_b[i - 1 - norb][a - 1 - norb] = x
                else:
                    raise QiskitNatureError(
                        "Unknown 1-electron integral indices encountered in " f"'{(i, a)}'"
                    ) from ex
        else:
            try:
                s8_hijkl[i - 1, a - 1, j - 1, b - 1] = x  # type: ignore[assignment]
            except IndexError as ex:
                if _uhf:
                    try:
                        # NOTE: we exploit the 4-fold symmetry here and greedily use j and b to
                        # index the beta-spin
                        s4_hijkl_ba[
                            j - 1 - norb, b - 1 - norb, i - 1, a - 1
                        ] = x  # type: ignore[assignment]
                    except IndexError:
                        try:
                            s4_hijkl_ba[
                                i - 1 - norb, a - 1 - norb, j - 1, b - 1
                            ] = x  # type: ignore[assignment]
                        except IndexError:
                            s8_hijkl_bb[
                                i - 1 - norb, a - 1 - norb, j - 1 - norb, b - 1 - norb
                            ] = x  # type: ignore[assignment]
                else:
                    raise QiskitNatureError(
                        "Unknown 2-electron integral indices encountered in " f"'{(i, a, j, b)}'"
                    ) from ex

    # complement the 1-body matrices by placing the upper-triangular values into the lower-triangle
    tril_indices = np.tril_indices_from(hij)
    hij[tril_indices] = hij.T[tril_indices]

    if _uhf:
        hij_b[tril_indices] = hij_b.T[tril_indices]

    return FCIDump(
        num_electrons=output.get("NELEC"),
        hij=hij,
        hijkl=s8_hijkl,
        hij_b=hij_b,
        hijkl_ba=s4_hijkl_ba,
        hijkl_bb=s8_hijkl_bb,
        multiplicity=output.get("MS2", 0) + 1,
        constant_energy=output.get("ecore", None),
        orbsym=output.get("ORBSYM", None),
        isym=output.get("ISYM"),
    )
