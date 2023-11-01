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

"""FCIDump dumper."""

from typing import List, Union, TextIO, Tuple, Iterator, Any
import itertools
import numpy as np


def _dump_1e_ints(
    hij: np.ndarray,
    mos: Union[range, List[int]],
    outfile: TextIO,
    beta: bool = False,
) -> None:
    idx_offset = 1 if not beta else 1 + len(mos)
    hij_elements = set()
    for i, j in itertools.product(mos, repeat=2):
        if i == j:
            _write_to_outfile(outfile, hij[i][j], (i + idx_offset, j + idx_offset, 0, 0))
            continue
        if (j, i) in hij_elements and np.isclose(hij[i][j], hij[j][i]):
            continue
        _write_to_outfile(outfile, hij[i][j], (i + idx_offset, j + idx_offset, 0, 0))
        hij_elements.add((i, j))


def _dump_2e_ints(
    hijkl: np.ndarray, mos: Union[range, List[int]], outfile: TextIO, beta: int = 0
) -> None:
    idx_offsets = [1, 1]
    for b in range(beta):
        idx_offsets[1 - b] += len(mos)

    # TODO: refactor to leverage symmetry-reduced integral containers
    hijkl_elements = set()
    for elem in itertools.product(mos, repeat=4):
        if np.isclose(hijkl[elem], 0.0, atol=1e-14):
            continue
        if len(set(elem)) == 1:
            _write_to_outfile(
                outfile,
                hijkl[elem],
                (
                    *[e + idx_offsets[0] for e in elem[:2]],
                    *[e + idx_offsets[1] for e in elem[2:]],
                ),
            )
            continue
        if (
            beta != 1
            and elem[::-1] in hijkl_elements
            and np.isclose(hijkl[elem], hijkl[elem[::-1]])
        ):
            continue
        bra_perms = set(itertools.permutations(elem[:2]))
        ket_perms = set(itertools.permutations(elem[2:]))
        permutations: Iterator[Any]
        if beta == 1:
            permutations = itertools.product(bra_perms, ket_perms)
        else:
            permutations = itertools.chain(
                itertools.product(bra_perms, ket_perms),
                itertools.product(ket_perms, bra_perms),
            )
        for perm in {e1 + e2 for e1, e2 in permutations}:
            if perm in hijkl_elements and np.isclose(hijkl[elem], hijkl[perm]):
                break
        else:
            _write_to_outfile(
                outfile,
                hijkl[elem],
                (
                    *[e + idx_offsets[0] for e in elem[:2]],
                    *[e + idx_offsets[1] for e in elem[2:]],
                ),
            )
            hijkl_elements.add(elem)


def _write_to_outfile(outfile: TextIO, value: float, indices: Tuple):
    outfile.write(f"{value:23.16E}{indices[0]:4d}{indices[1]:4d}{indices[2]:4d}{indices[3]:4d}\n")
