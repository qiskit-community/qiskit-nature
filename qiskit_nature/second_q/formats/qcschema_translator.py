# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Translator methods for the QCSchema."""

from __future__ import annotations

import numpy as np

from qiskit_nature import QiskitNatureError
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.properties import (
    AngularMomentum,
    Magnetization,
    ParticleNumber,
)
from qiskit_nature.second_q.properties.bases import ElectronicBasis, ElectronicBasisTransform
from qiskit_nature.second_q.properties.integrals import (
    ElectronicIntegrals,
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
)

from .molecule_info import MoleculeInfo
from .qcschema import QCSchema


# TODO: make use of basis argument, pending further ElectronicStructureProblem refactoring
# pylint: disable=unused-argument
def qcschema_to_problem(
    qcschema: QCSchema, *, basis: ElectronicBasis = ElectronicBasis.MO
) -> ElectronicStructureProblem:
    """Builds out an :class:`.ElectronicStructureProblem` from a :class:`.QCSchema` instance.

    This method centralizes the construction of an :class:`.ElectronicStructureProblem` from a
    :class:`.QCSchema`.

    Args:
        qcschema: the :class:`.QCSchema` object from which to build the problem.
        basis: the :class:`.ElectronicBasis` of the generated problem.

    Raises:
        QiskitNatureError: if either of the required 1- or 2-body electronic integrals are missing.

    Returns:
        An :class:`.ElectronicStructureProblem` instance.
    """
    nao = qcschema.properties.calcinfo_nmo
    nmo = qcschema.properties.calcinfo_nmo

    ints: list[ElectronicIntegrals] = []

    # AO data
    hij: np.ndarray | None = None
    hij_b: np.ndarray | None = None
    eri: np.ndarray | None = None
    one_body_ao: OneBodyElectronicIntegrals
    two_body_ao: TwoBodyElectronicIntegrals

    coeff_a: np.ndarray | None = None
    coeff_b: np.ndarray | None = None
    basis_transform: ElectronicBasisTransform | None = None

    def reshape_2(arr, dim, dim_2=None):
        return arr.reshape((dim, dim_2 if dim_2 is not None else dim))

    def reshape_4(arr, dim):
        return arr.reshape((dim,) * 4)

    if qcschema.wavefunction.scf_fock_a is not None:
        # TODO: deal with this properly
        hij = np.asarray(qcschema.wavefunction.scf_fock_a)
        nao = int(np.sqrt(len(hij)))
        hij = reshape_2(hij, nao)

    if qcschema.wavefunction.scf_fock_b is not None:
        hij_b = reshape_2(np.asarray(qcschema.wavefunction.scf_fock_b), nao)

    if hij is not None:
        one_body_ao = OneBodyElectronicIntegrals(ElectronicBasis.AO, (hij, hij_b))
        ints.append(one_body_ao)

    if qcschema.wavefunction.scf_eri is not None:
        eri = reshape_4(np.asarray(qcschema.wavefunction.scf_eri), nao)
        two_body_ao = TwoBodyElectronicIntegrals(ElectronicBasis.AO, (eri, None, None, None))
        ints.append(two_body_ao)

    if qcschema.wavefunction.scf_orbitals_a is not None:
        coeff_a = reshape_2(np.asarray(qcschema.wavefunction.scf_orbitals_a), nao, nmo)
    if qcschema.wavefunction.scf_orbitals_b is not None:
        coeff_b = reshape_2(np.asarray(qcschema.wavefunction.scf_orbitals_b), nao, nmo)

    if coeff_a is not None:
        basis_transform = ElectronicBasisTransform(
            ElectronicBasis.AO, ElectronicBasis.MO, coeff_a, coeff_b
        )

    # MO data
    hij_mo: np.ndarray | None = None
    hij_mo_b: np.ndarray | None = None
    eri_mo: np.ndarray | None = None
    eri_mo_ba: np.ndarray | None = None
    eri_mo_bb: np.ndarray | None = None
    eri_mo_ab: np.ndarray | None = None
    one_body_mo: OneBodyElectronicIntegrals
    two_body_mo: TwoBodyElectronicIntegrals

    if qcschema.wavefunction.scf_fock_mo_a is not None:
        hij_mo = reshape_2(np.asarray(qcschema.wavefunction.scf_fock_mo_a), nmo)

    if qcschema.wavefunction.scf_fock_mo_b is not None:
        hij_mo_b = reshape_2(np.asarray(qcschema.wavefunction.scf_fock_mo_b), nmo)

    if hij_mo is not None:
        one_body_mo = OneBodyElectronicIntegrals(ElectronicBasis.MO, (hij_mo, hij_mo_b))
    elif hij is not None:
        one_body_mo = one_body_ao.transform_basis(basis_transform)
    else:
        raise QiskitNatureError(
            "The provided QCSchema object is missing the required 1-body electronic integrals."
        )

    if one_body_mo is not None:
        ints.append(one_body_mo)

    if qcschema.wavefunction.scf_eri_mo_aa is not None:
        eri_mo = reshape_4(np.asarray(qcschema.wavefunction.scf_eri_mo_aa), nmo)

    if qcschema.wavefunction.scf_eri_mo_ba is not None:
        eri_mo_ba = reshape_4(np.asarray(qcschema.wavefunction.scf_eri_mo_ba), nmo)

    if qcschema.wavefunction.scf_eri_mo_bb is not None:
        eri_mo_bb = reshape_4(np.asarray(qcschema.wavefunction.scf_eri_mo_bb), nmo)

    if qcschema.wavefunction.scf_eri_mo_ab is not None:
        eri_mo_ab = reshape_4(np.asarray(qcschema.wavefunction.scf_eri_mo_ab), nmo)

    if eri_mo is not None:
        two_body_mo = TwoBodyElectronicIntegrals(
            ElectronicBasis.MO, (eri_mo, eri_mo_ba, eri_mo_bb, eri_mo_ab)
        )
    elif eri is not None:
        two_body_mo = two_body_ao.transform_basis(basis_transform)
    else:
        raise QiskitNatureError(
            "The provided QCSchema object is missing the required 2-body electronic integrals."
        )

    if two_body_mo is not None:
        ints.append(two_body_mo)

    e_nuc = qcschema.properties.nuclear_repulsion_energy
    e_ref = qcschema.properties.return_energy
    hamiltonian = ElectronicEnergy(
        ints,
        nuclear_repulsion_energy=e_nuc,
        reference_energy=e_ref,
    )
    if qcschema.wavefunction.scf_eigenvalues_a is not None:
        hamiltonian.orbital_energies = np.asarray(qcschema.wavefunction.scf_eigenvalues_a)

    natm = len(qcschema.molecule.symbols)
    geo = qcschema.molecule.geometry
    molecule = MoleculeInfo(
        symbols=qcschema.molecule.symbols,
        # the following format makes mypy happy:
        coords=[(geo[3 * i], geo[3 * i + 1], geo[3 * i + 2]) for i in range(natm)],
        multiplicity=qcschema.molecule.molecular_multiplicity or 1,
        charge=qcschema.molecule.molecular_charge or 0,
        units=DistanceUnit.BOHR,
        masses=qcschema.molecule.masses,
    )

    num_spin_orbitals = 2 * nmo
    num_particles = (qcschema.properties.calcinfo_nalpha, qcschema.properties.calcinfo_nbeta)

    problem = ElectronicStructureProblem(hamiltonian)
    problem.molecule = molecule
    problem.basis_transform = basis_transform
    problem.properties.angular_momentum = AngularMomentum(num_spin_orbitals)
    problem.properties.magnetization = Magnetization(num_spin_orbitals)
    problem.properties.particle_number = ParticleNumber(num_spin_orbitals, num_particles)

    return problem
