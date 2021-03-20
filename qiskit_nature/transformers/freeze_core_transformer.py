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

from .active_space_transformer import ActiveSpaceTransformer
from ..drivers import QMolecule

logger = logging.getLogger(__name__)


class FreezeCoreTransformer(ActiveSpaceTransformer):
    """The Freeze-Core reduction.

    """

    def __init__(self, freeze_core: bool = True,
                 remove_orbitals: Optional[List[int]] = None,
                 ):
        """Initializes a transformer which can reduce a `QMolecule` to a configured active space.

        Args:
            freeze_core: A convenience argument to quickly enable the inactivity of the
                         `QMolecule.core_orbitals`. This keyword overwrites the use of all other
                         keywords (except `remove_orbitals`) and, thus, cannot be used in
                         combination with them.
            remove_orbitals: A list of indices specifying molecular orbitals which are removed in
                             combination with the `freeze_core` option. No checks are performed on
                             the nature of these orbitals, so the user must make sure that these are
                             _unoccupied_ orbitals, which can be removed without taking any energy
                             shifts into account.
        """
        self._freeze_core = freeze_core
        self._remove_orbitals = remove_orbitals

        super().__init__()

    def transform(self, q_molecule: QMolecule) -> QMolecule:
        """Reduces the given `QMolecule` to a given active space.

        Args:
            q_molecule: the `QMolecule` to be transformed.

        Returns:
            A new `QMolecule` instance.

        Raises:
            QiskitNatureError: If more electrons or orbitals are requested than are available, if an
                               uneven number of inactive electrons remains, or if the number of
                               selected active orbital indices does not match
                               `num_molecular_orbitals`.
        """
        q_molecule_new = super().transform(q_molecule)

        def rename_dict_key(energy_shift_dict):
            try:
                energy_shift_dict['FreezeCoreTransformer'] = \
                    energy_shift_dict.pop('ActiveSpaceTransformer')
            except KeyError:
                pass

        rename_dict_key(q_molecule_new.energy_shift)
        rename_dict_key(q_molecule_new.x_dip_energy_shift)
        rename_dict_key(q_molecule_new.y_dip_energy_shift)
        rename_dict_key(q_molecule_new.z_dip_energy_shift)

        return q_molecule_new

    def _check_configuration(self):
        pass

    def _determine_active_space(self, q_molecule: QMolecule):
        nelec_total = q_molecule.num_alpha + q_molecule.num_beta

        inactive_orbs_idxs = q_molecule.core_orbitals
        if self._remove_orbitals is not None:
            inactive_orbs_idxs.extend(self._remove_orbitals)
        active_orbs_idxs = [o for o in range(q_molecule.num_orbitals)
                            if o not in inactive_orbs_idxs]

        # compute number of active electrons
        nelec_inactive = sum([self._mo_occ_total[o] for o in inactive_orbs_idxs])
        nelec_active = nelec_total - nelec_inactive

        num_alpha = (nelec_active - (q_molecule.multiplicity - 1)) // 2
        num_beta = nelec_active - num_alpha

        self._num_particles = (num_alpha, num_beta)

        return (active_orbs_idxs, inactive_orbs_idxs)
