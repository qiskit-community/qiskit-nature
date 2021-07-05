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

from qiskit_nature.properties.second_quantization.electronic import ElectronicDriverResult

from .active_space_transformer import ActiveSpaceTransformer


class FreezeCoreTransformer(ActiveSpaceTransformer):
    """The Freeze-Core reduction."""

    def __init__(
        self,
        freeze_core: bool = True,
        remove_orbitals: Optional[List[int]] = None,
    ):
        """Initializes a transformer which reduces a `QMolecule` by removing some molecular
        orbitals.

        The orbitals to be removed are specified in two ways:
            1. When `freeze_core` is enabled (the default), the `core_orbitals` listed in the
               `QMolecule` are made inactive and removed in the same fashion as in the
               :class:`ActiveSpaceTransformer`.
            2. Additionally, unoccupied molecular orbitals can be removed via a list of indices
               passed to `remove_orbitals`. It is the user's responsibility to ensure that these are
               indeed unoccupied orbitals, as no checks are performed.

        If you want to remove additional occupied orbitals, please use the
        :class:`ActiveSpaceTransformer` instead.

        Args:
            freeze_core: A boolean indicating whether to remove the molecular orbitals specified by
                        `QMolecule.core_orbitals`.
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
        self, molecule_data: ElectronicDriverResult
    ) -> Tuple[List[int], List[int]]:
        """Determines the active and inactive orbital indices.

        Args:
            molecule_data: the ElectronicDriverResult.

        Returns:
            The list of active and inactive orbital indices.
        """
        molecule = molecule_data.molecule
        particle_number = molecule_data.get_property("ParticleNumber")

        inactive_orbs_idxs = molecule.core_orbitals
        if self._remove_orbitals is not None:
            inactive_orbs_idxs.extend(self._remove_orbitals)
        active_orbs_idxs = [
            o for o, _ in enumerate(particle_number.occupation_alpha) if o not in inactive_orbs_idxs
        ]
        self._active_orbitals = active_orbs_idxs
        self._num_molecular_orbitals = len(active_orbs_idxs)

        return (active_orbs_idxs, inactive_orbs_idxs)
