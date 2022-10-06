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

from typing import cast

import numpy as np

from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_nature.second_q.properties import (
    AngularMomentum,
    Magnetization,
    ParticleNumber,
)
from qiskit_nature.second_q.properties.bases import ElectronicBasis
from qiskit_nature.second_q.transformers import BasisTransformer

from .molecule_info import MoleculeInfo
from .qcschema import QCSchema


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
        NotImplementedError: if an unsupported :class:`.ElectronicBasis` is requested.

    Returns:
        An :class:`.ElectronicStructureProblem` instance.
    """
    norb = qcschema.properties.calcinfo_nmo

    hamiltonian: ElectronicEnergy = None
    if basis == ElectronicBasis.AO:
        hamiltonian = _get_ao_hamiltonian(qcschema)
    elif basis == ElectronicBasis.MO:
        hamiltonian = _get_mo_hamiltonian(qcschema)
    else:
        raise NotImplementedError(f"The basis {basis} is not supported by the translation method.")

    hamiltonian.nuclear_repulsion_energy = qcschema.properties.nuclear_repulsion_energy

    natm = len(qcschema.molecule.symbols)
    geo = qcschema.molecule.geometry
    molecule = MoleculeInfo(
        symbols=qcschema.molecule.symbols,
        coords=[(geo[3 * i], geo[3 * i + 1], geo[3 * i + 2]) for i in range(natm)],
        multiplicity=qcschema.molecule.molecular_multiplicity or 1,
        charge=qcschema.molecule.molecular_charge or 0,
        units=DistanceUnit.BOHR,
        masses=qcschema.molecule.masses,
    )

    num_spin_orbitals = 2 * norb
    num_particles = (qcschema.properties.calcinfo_nalpha, qcschema.properties.calcinfo_nbeta)

    problem = ElectronicStructureProblem(hamiltonian)
    problem.basis = basis
    problem.molecule = molecule
    problem.reference_energy = qcschema.properties.return_energy
    problem.properties.angular_momentum = AngularMomentum(num_spin_orbitals)
    problem.properties.magnetization = Magnetization(num_spin_orbitals)
    problem.properties.particle_number = ParticleNumber(num_spin_orbitals, num_particles)

    if qcschema.wavefunction.scf_eigenvalues_a is not None:
        problem.orbital_energies = np.asarray(qcschema.wavefunction.scf_eigenvalues_a)
    if qcschema.wavefunction.scf_eigenvalues_b is not None:
        problem.orbital_energies_b = np.asarray(qcschema.wavefunction.scf_eigenvalues_b)

    return problem


def _reshape_2(arr, dim, dim_2=None):
    return np.asarray(arr).reshape((dim, dim_2 if dim_2 is not None else dim))


def _reshape_4(arr, dim):
    return np.asarray(arr).reshape((dim,) * 4)


def _get_ao_hamiltonian(qcschema) -> ElectronicEnergy:
    nao = int(np.sqrt(len(qcschema.wavefunction.scf_fock_a)))
    hcore = _reshape_2(qcschema.wavefunction.scf_fock_a, nao)
    hcore_b = None
    if qcschema.wavefunction.scf_fock_b is not None:
        hcore_b = _reshape_2(qcschema.wavefunction.scf_fock_b, nao)
    eri = _reshape_4(qcschema.wavefunction.scf_eri, nao)

    hamiltonian = ElectronicEnergy.from_raw_integrals(hcore, eri, hcore_b)

    return hamiltonian


def _get_mo_hamiltonian(qcschema) -> ElectronicEnergy:
    if qcschema.wavefunction.scf_fock_mo_a is not None:
        return _get_mo_hamiltonian_direct(qcschema)

    hamiltonian = _get_ao_hamiltonian(qcschema)
    transformer = get_ao_to_mo_from_qcschema(qcschema)

    return cast(ElectronicEnergy, transformer.transform_hamiltonian(hamiltonian))


def _get_mo_hamiltonian_direct(qcschema) -> ElectronicEnergy:
    norb = qcschema.properties.calcinfo_nmo
    hij = _reshape_2(qcschema.wavefunction.scf_fock_mo_a, norb)
    hijkl = _reshape_4(qcschema.wavefunction.scf_eri_mo_aa, norb)
    hij_b = None
    hijkl_bb = None
    hijkl_ba = None
    if qcschema.wavefunction.scf_fock_mo_b is not None:
        hij_b = _reshape_2(qcschema.wavefunction.scf_fock_mo_b, norb)
    if qcschema.wavefunction.scf_eri_mo_bb is not None:
        hijkl_bb = _reshape_4(qcschema.wavefunction.scf_eri_mo_bb, norb)
    if qcschema.wavefunction.scf_eri_mo_ba is not None:
        hijkl_ba = _reshape_4(qcschema.wavefunction.scf_eri_mo_ba, norb)

    hamiltonian = ElectronicEnergy.from_raw_integrals(hij, hijkl, hij_b, hijkl_bb, hijkl_ba)

    return hamiltonian


def get_ao_to_mo_from_qcschema(qcschema: QCSchema) -> BasisTransformer:
    """Builds out a :class:`.BasisTransformer` from a :class:`.QCSchema` instance.

    This utility extracts the AO-to-MO conversion coefficients, contained in a :class:`.QCSchema`
    object.

    Args:
        qcschema: the :class:`.QCSchema` object from which to build the problem.

    Returns:
        A :class:`.BasisTransformer` instance.
    """
    nmo = qcschema.properties.calcinfo_nmo
    nao = len(qcschema.wavefunction.scf_orbitals_a) // nmo
    coeff_a = _reshape_2(qcschema.wavefunction.scf_orbitals_a, nao, nmo)
    coeff_b = None
    if qcschema.wavefunction.scf_orbitals_b is not None:
        coeff_b = _reshape_2(qcschema.wavefunction.scf_orbitals_b, nao, nmo)

    transformer = BasisTransformer(
        ElectronicBasis.AO,
        ElectronicBasis.MO,
        ElectronicIntegrals.from_raw_integrals(coeff_a, h1_b=coeff_b),
    )

    return transformer
