# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Freeze-Core Reduction interface."""

from typing import List, Optional, Sequence

from qiskit_nature import QiskitNatureError
from qiskit_nature.constants import PERIODIC_TABLE
from qiskit_nature.second_q.problems import ElectronicStructureProblem

from .active_space_transformer import ActiveSpaceTransformer


class FreezeCoreTransformer(ActiveSpaceTransformer):
    """The Freeze-Core reduction."""

    def __init__(
        self,
        freeze_core: bool = True,
        remove_orbitals: Optional[List[int]] = None,
    ):
        """Initializes a transformer which can reduce an `ElectronicStructureProblem` to a
        configured active space.

        The orbitals to be removed are specified in two ways:

            #. When ``freeze_core`` is enabled (the default), the "core" orbitals will be determined
               automatically according to ``count_core_orbitals``. These will then be made inactive
               and removed in the same fashion as in the :class:`ActiveSpaceTransformer`.
            #. Additionally, unoccupied spatial orbitals can be removed via a list of indices
               passed to ``remove_orbitals``. It is the user's responsibility to ensure that these are
               indeed unoccupied orbitals, as no checks are performed.

        If you want to remove additional occupied orbitals, please use the
        :class:`ActiveSpaceTransformer` instead.

        Args:
            freeze_core: A boolean indicating whether to remove the "core" orbitals.
            remove_orbitals: A list of indices specifying spatial orbitals which are removed.
                             No checks are performed on the nature of these orbitals, so the user
                             must make sure that these are _unoccupied_ orbitals, which can be
                             removed without taking any energy shifts into account.
        """
        self._freeze_core = freeze_core
        self._remove_orbitals = remove_orbitals

        super().__init__(-1, -1)

    def _check_configuration(self):
        pass

    def _determine_active_space(self, problem: ElectronicStructureProblem) -> List[int]:
        """Determines the active and inactive orbital indices.

        Args:
            problem: the `ElectronicStructureProblem` to be transformed.

        Returns:
            The list of active and inactive orbital indices.

        Raises:
            QiskitNatureError: if a BaseProblem is provided which is not also an
                               ElectronicStructureProblem.
        """
        if not isinstance(problem, ElectronicStructureProblem):
            raise QiskitNatureError(
                "The FreezeCoreTransformer requires an `ElectronicStructureProblem`, not a "
                f"problem of type {type(problem)}."
            )

        molecule = problem.molecule

        inactive_orbs_idxs: List[int] = []
        if self._freeze_core:
            inactive_orbs_idxs.extend(range(self.count_core_orbitals(molecule.symbols)))
        if self._remove_orbitals is not None:
            inactive_orbs_idxs.extend(self._remove_orbitals)
        active_orbs_idxs = [
            o for o, _ in enumerate(problem.orbital_occupations) if o not in inactive_orbs_idxs
        ]
        self._active_orbitals = active_orbs_idxs
        self._num_spatial_orbitals = len(active_orbs_idxs)

        return active_orbs_idxs

    def count_core_orbitals(self, atoms: Sequence[str]) -> int:
        """Counts the number of core orbitals in a list of atoms.

        Args:
            atoms: the list of atoms.

        Returns:
            The number of core orbitals.
        """
        count = 0
        for atom in atoms:
            z = self.Z(atom)
            if z > 2:
                count += 1
            if z > 10:
                count += 4
            if z > 18:
                count += 4
            if z > 36:
                count += 9
            if z > 54:
                count += 9
            if z > 86:
                count += 16
        return count

    def Z(self, atom: str) -> int:  # pylint: disable=invalid-name
        """Atomic Number (Z) of an atom.

        Args:
            atom: the atom kind (symbol) whose atomic number to return.

        Returns:
            The atomic number of the queried atom kind.
        """
        return PERIODIC_TABLE.index(atom.lower().capitalize())
