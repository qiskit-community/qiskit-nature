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

"""The Active-Space Reduction interface."""

from typing import List, Optional
import logging
import numpy as np

from qiskit_nature import QMolecule, QiskitNatureError
from .base_transformer import BaseTransformer

logger = logging.getLogger(__name__)


class ActiveSpaceTransformer(BaseTransformer):
    """The Active-Space reduction."""

    def __init__(self, num_electrons: int, num_orbitals: int, num_alpha: Optional[int] = None,
                 active_orbitals: Optional[List[int]] = None):
        """Reduces the 1- and 2-electron integrals of a `QMolecule` into an active space.

        Args:
            num_electrons: the number of active electrons.
            num_orbitals: the number of active orbitals.
            num_alpha: the optional number of active alpha-spin electrons.
            active_orbitals: A list of indices specifying the molecular orbitals of the active
                             space. This argument must match with the remaining arguments and should
                             only be used to enforce an active space that is not chosen purely
                             around the Fermi level.
        """
        self.num_electrons = num_electrons
        self.num_orbitals = num_orbitals
        self.num_alpha = num_alpha
        self.active_orbitals = active_orbitals

    def transform(self, q_molecule: QMolecule) -> QMolecule:
        """Reduces the given `QMolecule` to a given active space.

        Args:
            q_molecule: the `QMolecule` to be transformed.

        Returns:
            A new `QMolecule` instance.

        Raises:
            QiskitNatureError: If more electrons or orbitals are requested than are available, if an
                               uneven number of inactive electrons remains, or if the number of
                               selected active orbital indices does not match `num_orbitals`.
        """
        # get molecular orbital coefficients
        mo_coeff_full = q_molecule.mo_coeff
        mo_coeff_full_b = q_molecule.mo_coeff_b
        # get molecular orbital occupation numbers
        mo_occ_full = q_molecule.mo_occ
        mo_occ_full_b = q_molecule.mo_occ_b
        beta = mo_coeff_full_b is not None
        mo_occ_total = mo_occ_full + mo_occ_full_b if beta else mo_occ_full
        # compute number of inactive electrons
        nelec_total = q_molecule.num_alpha + q_molecule.num_beta
        nelec_inactive = nelec_total - self.num_electrons
        if self.num_alpha is not None:
            if not beta:
                warning = 'The provided instance of QMolecule does not provide any beta ' \
                          + 'coefficients but you tried to specify a separate number of alpha ' \
                          + 'electrons. Continuing as if it does not matter.'
                logger.warning(warning)
            num_alpha = self.num_alpha
            num_beta = self.num_electrons - self.num_alpha
        else:
            num_beta = (self.num_electrons - (q_molecule.multiplicity - 1)) // 2
            num_alpha = self.num_electrons - num_beta
        if nelec_inactive < 0:
            raise QiskitNatureError("More electrons requested than available.")
        if nelec_inactive % 2 != 0:
            raise QiskitNatureError("The number of inactive electrons must be even.")

        # determine active and inactive orbital indices
        if self.active_orbitals is None:
            norbs_inactive = nelec_inactive // 2
            if norbs_inactive + self.num_orbitals > q_molecule.num_orbitals:
                raise QiskitNatureError("More orbitals requested than available.")
            inactive_orbs_idxs = list(range(norbs_inactive))
            active_orbs_idxs = list(range(norbs_inactive, norbs_inactive+self.num_orbitals))
        else:
            if self.num_orbitals != len(self.active_orbitals):
                raise QiskitNatureError("The number of selected active orbital indices does not "
                                        "match the specified number of active orbitals.")
            if max(self.active_orbitals) >= q_molecule.num_orbitals:
                raise QiskitNatureError("More orbitals requested than available.")
            active_orbs_idxs = self.active_orbitals
            inactive_orbs_idxs = [o for o in range(nelec_total // 2) if o not in
                                  self.active_orbitals and mo_occ_total[o] > 0]

        # split molecular orbitals coefficients into active and inactive parts
        mo_coeff_inactive = mo_coeff_full[:, inactive_orbs_idxs]
        mo_coeff_active = mo_coeff_full[:, active_orbs_idxs]
        mo_coeff_active_b = mo_coeff_inactive_b = None
        if beta:
            mo_coeff_inactive_b = mo_coeff_full_b[:, inactive_orbs_idxs]
            mo_coeff_active_b = mo_coeff_full_b[:, active_orbs_idxs]

        # compute inactive density matrix
        mo_occ_inactive = mo_occ_full[inactive_orbs_idxs]
        density_inactive = np.dot(mo_coeff_inactive*mo_occ_inactive,
                                  np.transpose(mo_coeff_inactive))
        density_inactive_b = None
        if beta:
            mo_occ_inactive_b = mo_occ_full_b[inactive_orbs_idxs]
            density_inactive_b = np.dot(mo_coeff_inactive_b*mo_occ_inactive_b,
                                        np.transpose(mo_coeff_inactive_b))

        # compute inactive Fock matrix
        hcore = q_molecule.hcore
        eri = q_molecule.eri
        coulomb_inactive = np.einsum('ijkl,ji->kl', eri, density_inactive)
        exchange_inactive = np.einsum('ijkl,jk->il', eri, density_inactive)
        fock_inactive = hcore + coulomb_inactive - 0.5 * exchange_inactive
        fock_inactive_b = hcore_b = coulomb_inactive_b = exchange_inactive_b = None
        if beta:
            hcore_b = q_molecule.hcore_b or hcore
            coulomb_inactive_b = np.einsum('ijkl,ji->kl', eri, density_inactive_b)
            exchange_inactive_b = np.einsum('ijkl,jk->il', eri, density_inactive_b)
            fock_inactive = hcore + coulomb_inactive + coulomb_inactive_b - exchange_inactive
            fock_inactive_b = hcore_b + coulomb_inactive + coulomb_inactive_b - exchange_inactive_b

        # compute inactive energy
        e_inactive = 0.0
        if not beta and mo_coeff_inactive.size > 0:
            e_inactive += 0.5 * np.einsum('ij,ji', density_inactive, hcore+fock_inactive)
        elif beta and mo_coeff_inactive_b.size > 0:
            e_inactive += 0.5 * np.einsum('ij,ji', density_inactive, hcore+fock_inactive)
            e_inactive += 0.5 * np.einsum('ij,ji', density_inactive_b, hcore_b+fock_inactive_b)

        # compute new 1- and 2-electron integrals
        hij = np.dot(np.dot(np.transpose(mo_coeff_active), fock_inactive), mo_coeff_active)
        hijkl = np.einsum('pqrs,pi,qj,rk,sl->ijkl', eri, mo_coeff_active, mo_coeff_active,
                          mo_coeff_active, mo_coeff_active, optimize=True)
        hij_b = hijkl_bb = hijkl_ba = None
        if beta:
            hij_b = np.dot(np.dot(np.transpose(mo_coeff_active_b), fock_inactive_b),
                           mo_coeff_active_b)
            hijkl_bb = np.einsum('pqrs,pi,qj,rk,sl->ijkl', eri, mo_coeff_active_b,
                                 mo_coeff_active_b, mo_coeff_active_b, mo_coeff_active_b,
                                 optimize=True)
            hijkl_ba = np.einsum('pqrs,pi,qj,rk,sl->ijkl', eri, mo_coeff_active_b,
                                 mo_coeff_active_b, mo_coeff_active, mo_coeff_active,
                                 optimize=True)

        # TODO: maybe deep-copying and overwriding fields is less prone to information loss
        # construct new QMolecule
        q_molecule_reduced = QMolecule()
        # Driver origin from which this QMolecule was created
        q_molecule_reduced.origin_driver_name = q_molecule.origin_driver_name
        q_molecule_reduced.origin_driver_version = q_molecule.origin_driver_version
        q_molecule_reduced.origin_driver_config = q_molecule.origin_driver_config
        # Energies and orbits
        q_molecule_reduced.hf_energy = q_molecule.hf_energy
        q_molecule_reduced.nuclear_repulsion_energy = q_molecule.nuclear_repulsion_energy
        q_molecule_reduced.energy_shift = q_molecule.energy_shift + e_inactive
        q_molecule_reduced.num_orbitals = self.num_orbitals
        q_molecule_reduced.num_alpha = num_alpha
        q_molecule_reduced.num_beta = num_beta
        q_molecule_reduced.mo_coeff = mo_coeff_active
        q_molecule_reduced.mo_coeff_b = mo_coeff_active_b
        q_molecule_reduced.orbital_energies = q_molecule.orbital_energies[active_orbs_idxs]
        if beta:
            q_molecule_reduced.orbital_energies_b = q_molecule.orbital_energies_b[active_orbs_idxs]
        # Molecule geometry. xyz coords are in Bohr
        q_molecule_reduced.molecular_charge = q_molecule.molecular_charge
        q_molecule_reduced.multiplicity = q_molecule.multiplicity
        q_molecule_reduced.num_atoms = q_molecule.num_atoms
        q_molecule_reduced.atom_symbol = q_molecule.atom_symbol
        q_molecule_reduced.atom_xyz = q_molecule.atom_xyz
        # TODO 1 and 2 electron ints in AO basis
        # 1 and 2 electron integrals in MO basis
        q_molecule_reduced.mo_onee_ints = hij
        q_molecule_reduced.mo_onee_ints_b = hij_b
        q_molecule_reduced.mo_eri_ints = hijkl
        q_molecule_reduced.mo_eri_ints_bb = hijkl_bb
        q_molecule_reduced.mo_eri_ints_ba = hijkl_ba
        # TODO dipole moment integrals in AO and MO basis

        return q_molecule_reduced
