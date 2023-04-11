# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
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
from qiskit_nature.second_q.problems import ElectronicBasis, ElectronicStructureProblem
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_nature.second_q.operators.symmetric_two_body import (
    S1Integrals,
    S4Integrals,
    S8Integrals,
)
from qiskit_nature.second_q.properties import (
    AngularMomentum,
    ElectronicDipoleMoment,
    Magnetization,
    ParticleNumber,
)
from qiskit_nature.second_q.transformers import BasisTransformer
from qiskit_nature.settings import settings

from .molecule_info import MoleculeInfo
from .qcschema import QCSchema


def qcschema_to_problem(
    qcschema: QCSchema,
    *,
    basis: ElectronicBasis = ElectronicBasis.MO,
    include_dipole: bool = True,
) -> ElectronicStructureProblem:
    """Builds out an :class:`.ElectronicStructureProblem` from a :class:`.QCSchema` instance.

    This method centralizes the construction of an :class:`.ElectronicStructureProblem` from a
    :class:`.QCSchema`.

    Args:
        qcschema: the :class:`.QCSchema` object from which to build the problem.
        basis: the :class:`.ElectronicBasis` of the generated problem.
        include_dipole: whether or not to include an :class:`.ElectronicDipoleMoment` property
            in the generated problem (if the data is available).

    Raises:
        QiskitNatureError: if either of the required 1- or 2-body electronic integrals are missing.
        NotImplementedError: if an unsupported :class:`.ElectronicBasis` is requested.

    Returns:
        An :class:`.ElectronicStructureProblem` instance.
    """
    norb = qcschema.properties.calcinfo_nmo

    hamiltonian: ElectronicEnergy = None
    dipole_x: ElectronicIntegrals | None = None
    dipole_y: ElectronicIntegrals | None = None
    dipole_z: ElectronicIntegrals | None = None

    if basis == ElectronicBasis.AO:
        hamiltonian = _get_ao_hamiltonian(qcschema)

        if include_dipole:
            dipole_x = _get_ao_dipole(qcschema, "x")
            dipole_y = _get_ao_dipole(qcschema, "y")
            dipole_z = _get_ao_dipole(qcschema, "z")

    elif basis == ElectronicBasis.MO:
        hamiltonian = _get_mo_hamiltonian(qcschema)

        if include_dipole:
            dipole_x = _get_mo_dipole(qcschema, "x")
            dipole_y = _get_mo_dipole(qcschema, "y")
            dipole_z = _get_mo_dipole(qcschema, "z")

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

    num_particles = (qcschema.properties.calcinfo_nalpha, qcschema.properties.calcinfo_nbeta)

    problem = ElectronicStructureProblem(hamiltonian)
    problem.basis = basis
    problem.molecule = molecule
    problem.num_particles = num_particles
    problem.num_spatial_orbitals = norb
    problem.reference_energy = qcschema.properties.return_energy
    problem.properties.angular_momentum = AngularMomentum(norb)
    problem.properties.magnetization = Magnetization(norb)
    problem.properties.particle_number = ParticleNumber(norb)

    if qcschema.wavefunction.scf_eigenvalues_a is not None:
        problem.orbital_energies = np.asarray(qcschema.wavefunction.scf_eigenvalues_a)
    if qcschema.wavefunction.scf_eigenvalues_b is not None:
        problem.orbital_energies_b = np.asarray(qcschema.wavefunction.scf_eigenvalues_b)

    if include_dipole and dipole_x is not None and dipole_y is not None and dipole_z is not None:
        dipole = ElectronicDipoleMoment(dipole_x, dipole_y, dipole_z)
        if qcschema.properties.nuclear_dipole_moment is not None:
            dipole.nuclear_dipole_moment = qcschema.properties.nuclear_dipole_moment

        problem.properties.electronic_dipole_moment = dipole

    return problem


def _reshape_2(arr, dim, dim_2=None):
    return np.asarray(arr).reshape((dim, dim_2 if dim_2 is not None else dim))


def _reshape_4(arr, dim):
    npair = dim * (dim + 1) // 2

    if len(arr) == npair * (npair + 1) // 2:
        return S8Integrals(np.asarray(arr))

    if len(arr) == npair**2:
        return S4Integrals(np.asarray(arr).reshape((npair,) * 2))

    if len(arr) == dim**4:
        if not settings.use_symmetry_reduced_integrals:
            return np.asarray(arr).reshape((dim,) * 4)
        return S1Integrals(np.asarray(arr).reshape((dim,) * 4))

    return arr


def _get_ao_hamiltonian(qcschema: QCSchema) -> ElectronicEnergy:
    nao = int(np.sqrt(len(qcschema.wavefunction.scf_fock_a)))
    hcore = _reshape_2(qcschema.wavefunction.scf_fock_a, nao)
    hcore_b = None
    if qcschema.wavefunction.scf_fock_b is not None:
        hcore_b = _reshape_2(qcschema.wavefunction.scf_fock_b, nao)
    eri = _reshape_4(qcschema.wavefunction.scf_eri, nao)

    hamiltonian = ElectronicEnergy.from_raw_integrals(hcore, eri, hcore_b)

    return hamiltonian


def _get_mo_hamiltonian(qcschema: QCSchema) -> ElectronicEnergy:
    if qcschema.wavefunction.scf_fock_mo_a is not None:
        return _get_mo_hamiltonian_direct(qcschema)

    hamiltonian = _get_ao_hamiltonian(qcschema)
    transformer = get_ao_to_mo_from_qcschema(qcschema)

    return cast(ElectronicEnergy, transformer.transform_hamiltonian(hamiltonian))


def _get_mo_hamiltonian_direct(qcschema: QCSchema) -> ElectronicEnergy:
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
    if qcschema.wavefunction.scf_eri_mo_ab is not None and hijkl_ba is None:
        hijkl_ba = np.transpose(_reshape_4(qcschema.wavefunction.scf_eri_mo_ab, norb))

    hamiltonian = ElectronicEnergy.from_raw_integrals(hij, hijkl, hij_b, hijkl_bb, hijkl_ba)

    return hamiltonian


def _get_ao_dipole(qcschema, axis) -> ElectronicIntegrals | None:
    name = f"scf_dipole_{axis}"
    integrals = getattr(qcschema.wavefunction, name, None)
    if integrals is None:
        return None
    nao = int(np.sqrt(len(integrals)))
    ints = _reshape_2(integrals, nao)
    return ElectronicIntegrals.from_raw_integrals(ints)


def _get_mo_dipole(qcschema, axis) -> ElectronicIntegrals | None:
    name_a = f"scf_dipole_mo_{axis}_a"
    if getattr(qcschema.wavefunction, name_a, None) is not None:
        return _get_mo_dipole_direct(qcschema, axis)

    dipole = _get_ao_dipole(qcschema, axis)
    transformer = get_ao_to_mo_from_qcschema(qcschema)

    return transformer.transform_electronic_integrals(dipole)


def _get_mo_dipole_direct(qcschema, axis) -> ElectronicIntegrals | None:
    norb = qcschema.properties.calcinfo_nmo
    name_a = f"scf_dipole_mo_{axis}_a"
    integrals_a = getattr(qcschema.wavefunction, name_a, None)
    if integrals_a is None:
        return None
    ints_a = _reshape_2(integrals_a, norb)

    name_b = f"scf_dipole_mo_{axis}_b"
    integrals_b = getattr(qcschema.wavefunction, name_b, None)
    ints_b = None
    if integrals_b is not None:
        ints_b = _reshape_2(integrals_b, norb)

    return ElectronicIntegrals.from_raw_integrals(ints_a, h1_b=ints_b)


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
