# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Freeze-Core Reduction interface."""

from typing import List, Optional, Tuple

from qiskit_nature import QiskitNatureError
from qiskit_nature.properties.second_quantization.electronic import ElectronicStructureDriverResult
from qiskit_nature.properties.second_quantization.electronic.types import GroupedElectronicProperty

from .active_space_transformer import ActiveSpaceTransformer


class FreezeCoreTransformer(ActiveSpaceTransformer):
    """The Freeze-Core reduction."""

    def __init__(
        self,
        freeze_core: bool = True,
        remove_orbitals: Optional[List[int]] = None,
    ):
        """Initializes a transformer which can reduce an `ElectronicStructureDriverResult` to a
        configured active space.

        The orbitals to be removed are specified in two ways:
            1. When `freeze_core` is enabled (the default), the "core" orbitals will be determined
               automatically according to `count_core_orbitals`. These will then be made inactive
               and removed in the same fashion as in the :class:`ActiveSpaceTransformer`.
            2. Additionally, unoccupied molecular orbitals can be removed via a list of indices
               passed to `remove_orbitals`. It is the user's responsibility to ensure that these are
               indeed unoccupied orbitals, as no checks are performed.

        If you want to remove additional occupied orbitals, please use the
        :class:`ActiveSpaceTransformer` instead.

        Args:
            freeze_core: A boolean indicating whether to remove the "core" orbitals.
            remove_orbitals: A list of indices specifying molecular orbitals which are removed.
                             No checks are performed on the nature of these orbitals, so the user
                             must make sure that these are _unoccupied_ orbitals, which can be
                             removed without taking any energy shifts into account.
        """
        self._freeze_core = freeze_core
        self._remove_orbitals = remove_orbitals

        super().__init__()

    def _check_configuration(self):
        pass

    def _determine_active_space(
        self, grouped_property: GroupedElectronicProperty
    ) -> Tuple[List[int], List[int]]:
        """Determines the active and inactive orbital indices.

        Args:
            grouped_property: the `ElectronicStructureDriverResult` to be transformed.

        Returns:
            The list of active and inactive orbital indices.

        Raises:
            QiskitNatureError: if a GroupedElectronicProperty is provided which is not also an
                               ElectronicElectronicStructureDriverResult.
        """
        if not isinstance(grouped_property, ElectronicStructureDriverResult):
            raise QiskitNatureError(
                "The FreezeCoreTransformer requires an `ElectronicStructureDriverResult`, not a "
                f"property of type {type(grouped_property)}."
            )

        molecule = grouped_property.molecule
        particle_number = grouped_property.get_property("ParticleNumber")

        inactive_orbs_idxs = list(range(self.count_core_orbitals(molecule.atoms)))
        if self._remove_orbitals is not None:
            inactive_orbs_idxs.extend(self._remove_orbitals)
        active_orbs_idxs = [
            o for o, _ in enumerate(particle_number.occupation_alpha) if o not in inactive_orbs_idxs
        ]
        self._active_orbitals = active_orbs_idxs
        self._num_molecular_orbitals = len(active_orbs_idxs)

        return (active_orbs_idxs, inactive_orbs_idxs)

    def count_core_orbitals(self, atoms: List[str]) -> int:
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
        return self.periodic_table.index(atom.lower().capitalize())

    periodic_table = [
        # pylint: disable=line-too-long
        # fmt: off
        "_",
         "H", "He",
        "Li", "Be",                                                              "B",  "C",  "N",  "O",  "F", "Ne",
        "Na", "Mg",                                                             "Al", "Si",  "P",  "S", "Cl", "Ar",
         "K", "Ca", "Sc", "Ti",  "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
        "Rb", "Sr",  "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te",  "I", "Xe",
        "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
                          "Hf", "Ta",  "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
        "Fr", "Ra", "Ac", "Th", "Pa",  "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
                          "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
        # fmt: on
    ]
