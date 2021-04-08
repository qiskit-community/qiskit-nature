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

from typing import List, Optional
import logging

from qiskit_nature.drivers import QMolecule

from .active_space_transformer import ActiveSpaceTransformer

logger = logging.getLogger(__name__)


class FreezeCoreTransformer(ActiveSpaceTransformer):
    """The Freeze-Core reduction."""

    def __init__(self, freeze_core: bool = True,
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

    def transform(self, molecule_data: QMolecule) -> QMolecule:
        """Reduces the given `QMolecule` by removing the core and optionally defined unoccupied
        molecular orbitals.

        Args:
            molecule_data: the `QMolecule` to be transformed.

        Returns:
            A new `QMolecule` instance.

        Raises:
            QiskitNatureError: If more electrons or orbitals are requested than are available, if an
                               uneven number of inactive electrons remains, or if the number of
                               selected active orbital indices does not match
                               `num_molecular_orbitals`.
        """
        molecule_data_new = super().transform(molecule_data)

        def rename_dict_key(energy_shift_dict):
            try:
                energy_shift_dict['FreezeCoreTransformer'] = \
                    energy_shift_dict.pop('ActiveSpaceTransformer')
            except KeyError:
                pass

        rename_dict_key(molecule_data_new.energy_shift)
        rename_dict_key(molecule_data_new.x_dip_energy_shift)
        rename_dict_key(molecule_data_new.y_dip_energy_shift)
        rename_dict_key(molecule_data_new.z_dip_energy_shift)

        return molecule_data_new

    def _check_configuration(self):
        pass

    def _determine_active_space(self, molecule_data: QMolecule):
        nelec_total = molecule_data.num_alpha + molecule_data.num_beta

        inactive_orbs_idxs = molecule_data.core_orbitals
        if self._remove_orbitals is not None:
            inactive_orbs_idxs.extend(self._remove_orbitals)
        active_orbs_idxs = [o for o in range(molecule_data.num_molecular_orbitals)
                            if o not in inactive_orbs_idxs]
        self._active_orbitals = active_orbs_idxs
        self._num_molecular_orbitals = len(active_orbs_idxs)

        # compute number of active electrons
        nelec_inactive = int(sum([self._mo_occ_total[o] for o in inactive_orbs_idxs]))
        nelec_active = nelec_total - nelec_inactive

        num_alpha = (nelec_active - (molecule_data.multiplicity - 1)) // 2
        num_beta = nelec_active - num_alpha

        self._num_particles = (num_alpha, num_beta)

        return (active_orbs_idxs, inactive_orbs_idxs)
