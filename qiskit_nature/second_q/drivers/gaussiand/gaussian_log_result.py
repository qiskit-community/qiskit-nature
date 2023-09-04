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

""" Gaussian Log File Result """

from __future__ import annotations

import math
from typing import List, Sequence, Tuple, Union, cast
import copy
import logging
import re

import numpy as np

from qiskit_nature.second_q.formats.watson import WatsonHamiltonian
import qiskit_nature.optionals as _optionals

if _optionals.HAS_SPARSE:
    # pylint: disable=import-error
    from sparse import as_coo
else:

    def as_coo(*args):
        """Empty as_coo function
        Replacement if sparse.as_coo is not present.
        """
        del args
        return 0


logger = logging.getLogger(__name__)


class GaussianLogResult:
    """Result for Gaussian™ 16 log driver.

    This result allows access to selected data from the log file that is not available
    via the use Gaussian 16 interfacing code when using the MatrixElement file.
    Since this parses the text output it is subject to the format of the log file.
    """

    def __init__(self, log: Union[str, list[str]]) -> None:
        """
        Args:
            log: The log contents conforming to Gaussian™ 16 format either as a single string
                 containing new line characters, or as a list of strings. If the single string
                 has no new line characters it is treated a file name and the file contents
                 will be read (a valid log file would be multiple lines).
        Raises:
            ValueError: Invalid Input
        """

        self._log = None

        if isinstance(log, str):
            lines = log.split("\n")

            if len(lines) == 1:
                with open(lines[0], "r", encoding="utf8") as file:
                    self._log = file.read().split("\n")
            else:
                self._log = lines

        elif isinstance(log, list):
            self._log = log

        else:
            raise ValueError(f"Invalid input for Gaussian Log Parser '{log}'")

    @property
    def log(self) -> list[str]:
        """The complete Gaussian log in the form of a list of strings."""
        return copy.copy(self._log)

    def __str__(self):
        return "\n".join(self._log)

    # Sections of interest in the log file
    _SECTION_QUADRATIC = r":\s+QUADRATIC\sFORCE\sCONSTANTS\sIN\sNORMAL\sMODES"
    _SECTION_CUBIC = r":\s+CUBIC\sFORCE\sCONSTANTS\sIN\sNORMAL\sMODES"
    _SECTION_QUARTIC = r":\s+QUARTIC\sFORCE\sCONSTANTS\sIN\sNORMAL\sMODES"

    @property
    def quadratic_force_constants(self) -> list[tuple[str, str, float, float, float]]:
        """Quadratic force constants. (2 indices, 3 values)

        Returns:
            A list of tuples each with 2 index values and 3 constant values.
            An empty list is returned if no such data is present in the log.
        """
        qfc = self._force_constants(self._SECTION_QUADRATIC, 2)
        return cast(List[Tuple[str, str, float, float, float]], qfc)

    @property
    def cubic_force_constants(self) -> list[tuple[str, str, str, float, float, float]]:
        """Cubic force constants. (3 indices, 3 values)

        Returns:
            A list of tuples each with 3 index values and 3 constant values.
            An empty list is returned if no such data is present in the log.
        """
        cfc = self._force_constants(self._SECTION_CUBIC, 3)
        return cast(List[Tuple[str, str, str, float, float, float]], cfc)

    @property
    def quartic_force_constants(
        self,
    ) -> list[tuple[str, str, str, str, float, float, float]]:
        """Quartic force constants. (4 indices, 3 values)

        Returns:
            A list of tuples each with 4 index values and 3 constant values.
            An empty list is returned if no such data is present in the log.
        """
        qfc = self._force_constants(self._SECTION_QUARTIC, 4)
        return cast(List[Tuple[str, str, str, str, float, float, float]], qfc)

    def _force_constants(self, section_name: str, indices: int) -> list[tuple]:
        constants = []
        pattern_constants = ""
        for i in range(indices):
            pattern_constants += rf"\s+(?P<index{i + 1}>\w+)"
        for i in range(3):
            pattern_constants += rf"\s+(?P<const{i + 1}>[+-]?\d+\.\d+)"

        # Find the section of interest
        i = 0
        found_section = False
        for i, line in enumerate(self._log):
            if re.search(section_name, line) is not None:
                found_section = True
                break

        # Now if section found look from this line onwards to get the corresponding constant data
        # lines which are from when we start to get a match against the constants pattern until we
        # do not again.
        const_found = False
        if found_section:
            for line in self._log[i:]:
                if not const_found:
                    # If we have not found the first line that matches we keep looking
                    # until we get a match (non-None) and then drop through into found
                    # section which we use thereafter
                    const = re.match(pattern_constants, line)
                    const_found = const is not None

                if const_found:
                    # If we found the match then for each line we want the contents until
                    # such point as it does not match anymore then we break out
                    const = re.match(pattern_constants, line)
                    if const is not None:
                        clist: list[Union[str, float]] = []
                        for i in range(indices):
                            clist.append(const.group(f"index{i + 1}"))
                        for i in range(3):
                            clist.append(float(const.group(f"const{i + 1}")))
                        constants.append(tuple(clist))
                    else:
                        break  # End of matching lines

        return constants

    @property
    def a_to_h_numbering(self) -> dict[str, int]:
        """A to H numbering mapping.

        Returns:
            Dictionary mapping string A numbering such as '1', '3a' etc from forces modes
            to H integer numbering
        """
        a2h: dict[str, int] = {}

        found_section = False
        found_h = False
        found_a = False
        h_nums = []
        a_nums = []
        for line in self._log:
            if not found_section:
                if re.search(r"Input/Output\sinformation", line) is not None:
                    logger.debug(line)
                    found_section = True
            else:
                if re.search(r"\s+\(H\)\s+\|", line) is not None:
                    logger.debug(line)
                    found_h = True
                    h_nums += [x.strip() for x in line.split("|") if x and "(H)" not in x]
                elif re.search(r"\s+\(A\)\s+\|", line) is not None:
                    logger.debug(line)
                    found_a = True
                    a_nums += [x.strip() for x in line.split("|") if x and "(A)" not in x]

                if found_h and found_a and re.search(r"NOTE:", line) is not None:
                    for i, a_num in enumerate(a_nums):
                        a2h[a_num] = int(h_nums[i])
                    break

        return a2h

    # ----------------------------------------------------------------------------------------
    # The following is to process the constants and produce an n-body array for input
    # to the Bosonic Operator. It maybe these methods all should be in some other module
    # but for now they are here

    @staticmethod
    def _multinomial(indices: list[int]) -> float:
        # For a given list of integers, computes the associated multinomial
        tmp = set(indices)  # Set of unique indices
        multinomial = 1
        for val in tmp:
            count = indices.count(val)
            multinomial *= math.factorial(count)
        return multinomial

    def _process_entry_indices(self, entry: list[Union[str, float]]) -> list[int]:
        # a2h gives us say '3a' -> 1, '3b' -> 2 etc. The H values can be 1 through 4
        # but we want them numbered in reverse order so the 'a2h_vals + 1 - a2h[x]'
        # takes care of this
        a2h = self.a_to_h_numbering
        a2h_vals = max(list(a2h.values()))

        # There are 3 float entries in the list at the end, the other entries up
        # front are the indices (string type).
        num_indices = len(entry) - 3
        return [a2h_vals + 1 - a2h[cast(str, x)] for x in entry[0:num_indices]]

    def _force_constants_array(
        self,
        force_constants: Sequence[tuple],
        factor: float,
        *,
        normalize: bool = True,
    ):
        sparse_data = {}
        max_index = -1

        for entry in force_constants:
            indices = self._process_entry_indices(list(entry))
            if indices:
                max_index = max(max_index, *set(indices))
                fac = factor
                fac *= self._multinomial(indices) if normalize else 1.0
                coeff = entry[-3] / fac
                sparse_data[tuple(i - 1 for i in indices)] = coeff

        return sparse_data, max_index

    def get_watson_hamiltonian(self, *, normalize: bool = True) -> WatsonHamiltonian:
        """Extracts a Watson Hamiltonian from the Gaussian log.

        Args:
            normalize: whether or not to normalize the force constants.

        Returns:
            The constructed ``WatsonHamiltonian``.
        """
        quadratic_data, quadratic_max_index = self._force_constants_array(
            self.quadratic_force_constants, factor=2.0, normalize=normalize
        )
        cubic_data, cubic_max_index = self._force_constants_array(
            self.cubic_force_constants, factor=2.0 * math.sqrt(2.0), normalize=normalize
        )
        quartic_data, quartic_max_index = self._force_constants_array(
            self.quartic_force_constants, factor=4.0, normalize=normalize
        )

        max_index = max(quadratic_max_index, cubic_max_index, quartic_max_index)

        if _optionals.HAS_SPARSE:
            watson = WatsonHamiltonian(
                as_coo(quadratic_data, shape=(max_index,) * 2),
                as_coo(cubic_data, shape=(max_index,) * 3),
                as_coo(quartic_data, shape=(max_index,) * 4),
                -as_coo(quadratic_data, shape=(max_index,) * 2),
            )
        else:
            quadratic_numpy = np.zeros((max_index,) * 2)
            for coord, value in quadratic_data.items():
                quadratic_numpy[coord] = value
            cubic_numpy = np.zeros((max_index,) * 3)
            for coord, value in cubic_data.items():
                cubic_numpy[coord] = value
            quartic_numpy = np.zeros((max_index,) * 4)
            for coord, value in quartic_data.items():
                quartic_numpy[coord] = value
            watson = WatsonHamiltonian(
                quadratic_numpy,
                cubic_numpy,
                quartic_numpy,
                -quadratic_numpy,
            )

        return watson
