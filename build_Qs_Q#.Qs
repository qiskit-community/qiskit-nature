# Qiskit Nature 

[![License](https://img.shields.io/github/license/Qiskit/qiskit-nature.svg?style=popout-square)](https://opensource.org/licenses/Apache-2.0)[![Build Status](https://github.com/Qiskit/qiskit-nature/workflows/Nature%20Unit%20Tests/badge.svg?branch=main)](https://github.com/Qiskit/qiskit-nature/actions?query=workflow%3A"Nature%20Unit%20Tests"+branch%3Amain+event%3Apush)[![](https://img.shields.io/github/release/Qiskit/qiskit-nature.svg?style=popout-square)](https://github.com/Qiskit/qiskit-nature/releases)[![](https://img.shields.io/pypi/dm/qiskit-nature.svg?style=popout-square)](https://pypi.org/project/qiskit-nature/)[![Coverage Status](https://coveralls.io/repos/github/Qiskit/qiskit-nature/badge.svg?branch=main)](https://coveralls.io/github/Qiskit/qiskit-nature?branch=main)

**Qiskit Nature** is an open-source framework that supports problems including ground state energy computations,
excited states and dipole moments of molecule, both open and closed-shell.

The code comprises chemistry drivers, which when provided with a molecular
configuration will return one and two-body integrals as well as other data that is efficiently
computed classically. This output data from a driver can then be used as input in Qiskit
Nature that contains logic which is able to translate this into a form that is suitable
for quantum algorithms. The conversion first creates a FermionicOperator which must then be mapped,
e.g. by a Jordan Wigner mapping, to a qubit operator in readiness for the quantum computation.

## Installation

We encourage installing Qiskit Nature via the pip tool (a python package manager).

```bash
pip install qiskit-nature
```

**pip** will handle all dependencies automatically and you will always install the latest
(and well-tested) version.

If you want to work on the very latest work-in-progress versions, either to try features ahead of
their official release or if you want to contribute to Qiskit Nature, then you can install from source.
To do this follow the instructions in the
 [documentation](https://qiskit.org/documentation/contributing_to_qiskit.html#installing-from-source).

### Optional Installs

To run chemistry experiments using Qiskit Nature, it is recommended that you install
a classical computation chemistry software program/library interfaced by Qiskit.
Several, as listed below, are supported, and while logic to interface these programs is supplied by
Qiskit Nature via the above pip installation, the dependent programs/libraries themselves need
to be installed separately.

1. [Gaussian 16&trade;](https://qiskit.org/documentation/nature/apidocs/qiskit_nature.drivers.gaussiand.html), a commercial chemistry program
2. [PSI4](https://qiskit.org/documentation/nature/apidocs/qiskit_nature.drivers.psi4d.html), a chemistry program that exposes a Python interface allowing for accessing internal objects
3. [PySCF](https://qiskit.org/documentation/nature/apidocs/qiskit_nature.drivers.pyscfd.html), an open-source Python chemistry program
4. [PyQuante](https://qiskit.org/documentation/nature/apidocs/qiskit_nature.drivers.pyquanted.html), a pure cross-platform open-source Python chemistry program

### HDF5 Driver

A useful functionality integrated into Qiskit Nature is its ability to serialize a file
in hierarchical Data Format 5 (HDF5) format representing all the output data from a chemistry driver.

The [HDF5 driver](https://qiskit.org/documentation/nature/stubs/qiskit_nature.drivers.HDF5Driver.html)
accepts such HDF5 files as input so molecular experiments can be run, albeit on the fixed data
as stored in the file. As such, if you have some pre-created HDF5 files created from Qiskit
Nature, you can use these with the HDF5 driver even if you do not install one of the classical
computation packages listed above.

### Creating Your First Chemistry Programming Experiment in Qiskit

Now that Qiskit Nature is installed, let's try a chemistry application experiment
using the VQE (Variational Quantum Eigensolver) algorithm to compute
the ground-state (minimum) energy of a molecule.

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
'#"$_-'*#*'=("*/*")
'*#*'
This code is part of Qiskit.
(C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" QMolecule """
#
""" QMolecule.Qs

import logging
import os
import tempfile
import warnings
from typing import List
import numpy
TWOE_TO_SPIN_SUBSCRIPT = "ijkl->ljik"
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py
logger = logging.getLogger(__name__)


class QMolecule:
    """

    Molecule data class containing driver result.

    When one of the chemistry :mod:`~qiskit_nature.drivers.second_quantization` is run and instance
    of this class is returned. This contains various properties that are made available in
    a consistent manner across the various drivers.
    Note that values here, for the same input molecule to each driver, may be vary across
    the drivers underlying code implementation. Also some drivers may not provide certain fields
    such as dipole integrals in the case of
    :class:`~qiskit_nature.drivers.second_quantization.PyQuanteDriver`.

    This class provides methods to save it and load it again from an HDF5 file
    """


    QMOLECULE_VERSION = 3

    def __init__(self, filename=None):
        self._filename = filename
        # All the following fields are saved/loaded in the save/load methods.
        # If fields are added in a version they are noted by version comment
        #
        # Originally support was limited to closed shell, when open shell was
        # added, and new integrals to allow different Beta orbitals needed,
        # these have been added as new similarly named fields but with suffices
        # such as _b, _bb and _ba. So mo_coeff (with no suffix) is the original
        # and is for alpha molecular coefficients, the added one for beta is
        # name mo_coeff_b, i.e. same name but with _b suffix. To keep backward
        # compatibility the original fields were not renamed with an _a suffix
        # but rather its implicit in the lack thereof given another field of
        # the same name but with an explicit suffix.
        # Driver origin from which this QMolecule was created
        self.origin_driver_name = "?"
        self.origin_driver_version = "?"  # v2
        self.origin_driver_config = "?"
        # Energies and orbits
        self.hf_energy = None
        self.nuclear_repulsion_energy = None
        self.num_molecular_orbitals = None
        self.num_alpha = None
        self.num_beta = None
        self.mo_coeff = None
        self.mo_coeff_b = None  # v2
        self.orbital_energies = None
        self.orbital_energies_b = None  # v2
        self.mo_occ = None  # v3
        self.mo_occ_b = None  # v3
        self.energy_shift = {}  # v3
        self.x_dip_energy_shift = {}  # v3
        self.y_dip_energy_shift = {}  # v3
        self.z_dip_energy_shift = {}  # v3
        # Molecule geometry. xyz coords are in Bohr
        self.molecular_charge = None
        self.multiplicity = None
        self.num_atoms = None
        self.atom_symbol = None
        self.atom_xyz = None
        # 1 and 2 electron ints in AO basis
        self.hcore = None  # v2
        self.hcore_b = None  # v2
        self.kinetic = None  # v2
        self.overlap = None  # v2
        self.eri = None  # v2
        # 1 and 2 electron integrals in MO basis
        self.mo_onee_ints = None
        self.mo_onee_ints_b = None  # v2
        self.mo_eri_ints = None
        self.mo_eri_ints_bb = None  # v2
        self.mo_eri_ints_ba = None  # v2
        # Dipole moment integrals in AO basis
        self.x_dip_ints = None  # v2
        self.y_dip_ints = None  # v2
        self.z_dip_ints = None  # v2
        # Dipole moment integrals in MO basis
        self.x_dip_mo_ints = None
        self.x_dip_mo_ints_b = None  # v2
        self.y_dip_mo_ints = None
        self.y_dip_mo_ints_b = None  # v2
        self.z_dip_mo_ints = None
        self.z_dip_mo_ints_b = None  # v2
        self.nuclear_dipole_moment = None
        self.reverse_dipole_sign = False

    @property
    def one_body_integrals(self):
        """Returns one body electron integrals."""
        Returns one body electron integrals.
        return QMolecule.onee_to_spin(self.mo_onee_ints, self.mo_onee_ints_b)

    @property
    def two_body_integrals(self):
        """Returns two body electron integrals."""
        Returns two body electron integrals.
        return QMolecule.twoe_to_spin(self.mo_eri_ints, self.mo_eri_ints_bb, self.mo_eri_ints_ba)

    def has_dipole_integrals(self):
        """Check if dipole integrals are present."""
        Check if dipole integrals are present.
        return (
            self.x_dip_mo_ints is not None
            and self.y_dip_mo_ints is not None
            and self.z_dip_mo_ints is not None
        )

    @property
    def x_dipole_integrals(self):
        """returns x_dipole_integrals"""
        returns x_dipole_integrals
        return QMolecule.onee_to_spin(self.x_dip_mo_ints, self.x_dip_mo_ints_b)

    @property
    def y_dipole_integrals(self):
        """returns y_dipole_integrals"""
        returns y_dipole_integrals
        return QMolecule.onee_to_spin(self.y_dip_mo_ints, self.y_dip_mo_ints_b)

    @property
    def z_dipole_integrals(self):
        """returns z_dipole_integrals"""
        returns z_dipole_integrals
        return QMolecule.onee_to_spin(self.z_dip_mo_ints, self.z_dip_mo_ints_b)

    def Z(self, natom):  # pylint: disable=invalid-name
        """Z"""
        Z
        if natom < 0 or natom >= self.num_atoms:
            raise ValueError("Atom index out of range")
        return QMolecule.symbols.index(self.atom_symbol[natom].lower().capitalize())

    @property
    def core_orbitals(self) -> List[int]:
        """

        Returns:
            A list of core orbital indices.
        """

        if self.num_atoms is None:
            logger.warning("Missing molecule information! Returning empty core orbital list.")
            return []
        count = 0
        for i in range(self.num_atoms):
            z = self.Z(i)
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
        return list(range(count))

    @property
    def filename(self):
        """returns temp file path"""
        returns temp file path
        if self._filename is None:
            file, self._filename = tempfile.mkstemp(suffix=".hdf5")
            os.close(file)

        return self._filename

    def load(self):
        """loads info saved."""
        loads info saved.
        try:
            if self._filename is None:
                return
            with h5py.File(self._filename, "r") as file:
                def read_array(name):
                    _data = file[name][...]
                    if _data.dtype == numpy.bool_ and _data.size == 1 and not _data:
                        _data = None
                    return _data
                # non-recursive dictionary reading
                def read_dict(name):
                    _data = {}
                    for k, v in file[name].items():
                        _data[k] = float(v[...]) if v[...].dtype.num != 0 else None
                    return _data
                # A version field was added to save format from version 2 so if
                # there is no version then we have original (version 1) format
                version = 1
                if "version" in file.keys():
                    data = file["version"][...]
                    version = int(data) if data.dtype.num != 0 else version
                # Origin driver info
                data = file["origin_driver/name"][...]
                self.origin_driver_name = data[...].tobytes().decode("utf-8")
                self.origin_driver_version = "?"
                if version > 1:
                    data = file["origin_driver/version"][...]
                    self.origin_driver_version = data[...].tobytes().decode("utf-8")
                data = file["origin_driver/config"][...]
                self.origin_driver_config = data[...].tobytes().decode("utf-8")
                # Energies
                data = file["energy/hf_energy"][...]
                self.hf_energy = float(data) if data.dtype.num != 0 else None
                data = file["energy/nuclear_repulsion_energy"][...]
                self.nuclear_repulsion_energy = float(data) if data.dtype.num != 0 else None
                if version > 2:
                    self.energy_shift = read_dict("energy/energy_shift")
                    self.x_dip_energy_shift = read_dict("energy/x_dip_energy_shift")
                    self.y_dip_energy_shift = read_dict("energy/y_dip_energy_shift")
                    self.z_dip_energy_shift = read_dict("energy/z_dip_energy_shift")
                else:
                    self.energy_shift = {}
                    self.x_dip_energy_shift = {}
                    self.y_dip_energy_shift = {}
                    self.z_dip_energy_shift = {}
                # Orbitals
                try:
                    data = file["orbitals/num_molecular_orbitals"][...]
                    self.num_molecular_orbitals = int(data) if data.dtype.num != 0 else None
                except KeyError:
                    # try the legacy attribute name
                    data = file["orbitals/num_orbitals"][...]
                    self.num_molecular_orbitals = int(data) if data.dtype.num != 0 else None
                data = file["orbitals/num_alpha"][...]
                self.num_alpha = int(data) if data.dtype.num != 0 else None
                data = file["orbitals/num_beta"][...]
                self.num_beta = int(data) if data.dtype.num != 0 else None
                self.mo_coeff = read_array("orbitals/mo_coeff")
                self.mo_coeff_b = read_array("orbitals/mo_coeff_B") if version > 1 else None
                self.orbital_energies = read_array("orbitals/orbital_energies")
                self.orbital_energies_b = (
                    read_array("orbitals/orbital_energies_B") if version > 1 else None
                )
                self.mo_occ = read_array("orbitals/mo_occ") if version > 2 else None
                self.mo_occ_b = read_array("orbitals/mo_occ_B") if version > 2 else None
                # Molecule geometry
                data = file["geometry/molecular_charge"][...]
                self.molecular_charge = int(data) if data.dtype.num != 0 else None
                data = file["geometry/multiplicity"][...]
                self.multiplicity = int(data) if data.dtype.num != 0 else None
                data = file["geometry/num_atoms"][...]
                self.num_atoms = int(data) if data.dtype.num != 0 else None
                data = file["geometry/atom_symbol"][...]
                self.atom_symbol = [a.decode("utf8") for a in data]
                self.atom_xyz = file["geometry/atom_xyz"][...]
                # 1 and 2 electron integrals in AO basis
                self.hcore = read_array("integrals/hcore") if version > 1 else None
                self.hcore_b = read_array("integrals/hcore_B") if version > 1 else None
                self.kinetic = read_array("integrals/kinetic") if version > 1 else None
                self.overlap = read_array("integrals/overlap") if version > 1 else None
                self.eri = read_array("integrals/eri") if version > 1 else None
                # 1 and 2 electron integrals in MO basis
                self.mo_onee_ints = read_array("integrals/mo_onee_ints")
                self.mo_onee_ints_b = (
                    read_array("integrals/mo_onee_ints_B") if version > 1 else None
                )
                self.mo_eri_ints = read_array("integrals/mo_eri_ints")
                self.mo_eri_ints_bb = (
                    read_array("integrals/mo_eri_ints_BB") if version > 1 else None
                )
                self.mo_eri_ints_ba = (
                    read_array("integrals/mo_eri_ints_BA") if version > 1 else None
                )
                # dipole integrals in AO basis
                self.x_dip_ints = read_array("dipole/x_dip_ints") if version > 1 else None
                self.y_dip_ints = read_array("dipole/y_dip_ints") if version > 1 else None
                self.z_dip_ints = read_array("dipole/z_dip_ints") if version > 1 else None
                # dipole integrals in MO basis
                self.x_dip_mo_ints = read_array("dipole/x_dip_mo_ints")
                self.x_dip_mo_ints_b = read_array("dipole/x_dip_mo_ints_B") if version > 1 else None
                self.y_dip_mo_ints = read_array("dipole/y_dip_mo_ints")
                self.y_dip_mo_ints_b = read_array("dipole/y_dip_mo_ints_B") if version > 1 else None
                self.z_dip_mo_ints = read_array("dipole/z_dip_mo_ints")
                self.z_dip_mo_ints_b = read_array("dipole/z_dip_mo_ints_B") if version > 1 else None
                self.nuclear_dipole_moment = file["dipole/nuclear_dipole_moment"][...]
                self.reverse_dipole_sign = file["dipole/reverse_dipole_sign"][...]
        except OSError:
            pass

    def save(self, file_name=None):
        """Saves the info from the driver."""
        Saves the info from the driver.
        file = None
        if file_name is not None:
            self.remove_file(file_name)
            file = file_name
        else:
            file = self.filename
            self.remove_file()
        with h5py.File(file, "w") as file:
            def create_dataset(group, name, value):
                def is_float(v):
                    if v is None:
                        return False
                    if isinstance(v, float):
                        return True
                    if isinstance(v, list):
                        v = numpy.array(v)
                    try:
                        return numpy.issubdtype(v.dtype, numpy.floating)
                    except Exception:  # pylint: disable=broad-except
                        return False
                if is_float(value):
                    group.create_dataset(name, data=value, dtype="float64")
                elif isinstance(value, dict):
                    sub_group = group.create_group(name)
                    for k, v in value.items():
                        sub_group.create_dataset(k, data=v)
                else:
                    group.create_dataset(name, data=(value if value is not None else False))
            file.create_dataset("version", data=(self.QMOLECULE_VERSION,))
            # Driver origin of molecule data
            g_driver = file.create_group("origin_driver")
            g_driver.create_dataset(
                "name",
                data=(
                    numpy.string_(self.origin_driver_name)
                    if self.origin_driver_name is not None
                    else numpy.string_("?")
                ),
            )
            g_driver.create_dataset(
                "version",
                data=(
                    numpy.string_(self.origin_driver_version)
                    if self.origin_driver_version is not None
                    else numpy.string_("?")
                ),
            )
            g_driver.create_dataset(
                "config",
                data=(
                    numpy.string_(self.origin_driver_config)
                    if self.origin_driver_config is not None
                    else numpy.string_("?")
                ),
            )
            # Energies
            g_energy = file.create_group("energy")
            create_dataset(g_energy, "hf_energy", self.hf_energy)
            create_dataset(g_energy, "nuclear_repulsion_energy", self.nuclear_repulsion_energy)
            create_dataset(g_energy, "energy_shift", self.energy_shift)
            create_dataset(g_energy, "x_dip_energy_shift", self.x_dip_energy_shift)
            create_dataset(g_energy, "y_dip_energy_shift", self.y_dip_energy_shift)
            create_dataset(g_energy, "z_dip_energy_shift", self.z_dip_energy_shift)
            # Orbitals
            g_orbitals = file.create_group("orbitals")
            create_dataset(g_orbitals, "num_molecular_orbitals", self.num_molecular_orbitals)
            create_dataset(g_orbitals, "num_alpha", self.num_alpha)
            create_dataset(g_orbitals, "num_beta", self.num_beta)
            create_dataset(g_orbitals, "mo_coeff", self.mo_coeff)
            create_dataset(g_orbitals, "mo_coeff_B", self.mo_coeff_b)
            create_dataset(g_orbitals, "orbital_energies", self.orbital_energies)
            create_dataset(g_orbitals, "orbital_energies_B", self.orbital_energies_b)
            create_dataset(g_orbitals, "mo_occ", self.mo_occ)
            create_dataset(g_orbitals, "mo_occ_B", self.mo_occ_b)
            # Molecule geometry
            g_geometry = file.create_group("geometry")
            create_dataset(g_geometry, "molecular_charge", self.molecular_charge)
            create_dataset(g_geometry, "multiplicity", self.multiplicity)
            create_dataset(g_geometry, "num_atoms", self.num_atoms)
            g_geometry.create_dataset(
                "atom_symbol",
                data=(
                    [a.encode("utf8") for a in self.atom_symbol]
                    if self.atom_symbol is not None
                    else False
                ),
            )
            create_dataset(g_geometry, "atom_xyz", self.atom_xyz)
            # 1 and 2 electron integrals
            g_integrals = file.create_group("integrals")
            create_dataset(g_integrals, "hcore", self.hcore)
            create_dataset(g_integrals, "hcore_B", self.hcore_b)
            create_dataset(g_integrals, "kinetic", self.kinetic)
            create_dataset(g_integrals, "overlap", self.overlap)
            create_dataset(g_integrals, "eri", self.eri)
            create_dataset(g_integrals, "mo_onee_ints", self.mo_onee_ints)
            create_dataset(g_integrals, "mo_onee_ints_B", self.mo_onee_ints_b)
            create_dataset(g_integrals, "mo_eri_ints", self.mo_eri_ints)
            create_dataset(g_integrals, "mo_eri_ints_BB", self.mo_eri_ints_bb)
            create_dataset(g_integrals, "mo_eri_ints_BA", self.mo_eri_ints_ba)
            # dipole integrals
            g_dipole = file.create_group("dipole")
            create_dataset(g_dipole, "x_dip_ints", self.x_dip_ints)
            create_dataset(g_dipole, "y_dip_ints", self.y_dip_ints)
            create_dataset(g_dipole, "z_dip_ints", self.z_dip_ints)
            create_dataset(g_dipole, "x_dip_mo_ints", self.x_dip_mo_ints)
            create_dataset(g_dipole, "x_dip_mo_ints_B", self.x_dip_mo_ints_b)
            create_dataset(g_dipole, "y_dip_mo_ints", self.y_dip_mo_ints)
            create_dataset(g_dipole, "y_dip_mo_ints_B", self.y_dip_mo_ints_b)
            create_dataset(g_dipole, "z_dip_mo_ints", self.z_dip_mo_ints)
            create_dataset(g_dipole, "z_dip_mo_ints_B", self.z_dip_mo_ints_b)
            create_dataset(g_dipole, "nuclear_dipole_moment", self.nuclear_dipole_moment)
            create_dataset(g_dipole, "reverse_dipole_sign", self.reverse_dipole_sign)

    def remove_file(self, file_name=None):
        """remove file"""
        remove file
        try:
            file = self._filename if file_name is None else file_name
            os.remove(file)
        except OSError:
            pass
    # Utility functions to convert integrals into the form expected by Qiskit Nature

    @staticmethod
    def oneeints2mo(ints, moc):
        """Converts one-body integrals from AO to MO basis
    def -nature-:
        Converts one-body integrals from AO to MO basis

        Returns one electron integrals in AO basis converted to given MO basis

        Args:
            ints (numpy.ndarray): N^2 one electron integrals in AO basis
            moc (numpy.ndarray): Molecular orbital coefficients
        Returns:
            numpy.ndarray: integrals in MO basis
        """

        return numpy.dot(numpy.dot(numpy.transpose(moc), ints), moc)

    @staticmethod
    def twoeints2mo(ints, moc):
        """Converts two-body integrals from AO to MO basis
        Converts two-body integrals from AI to ML basis

        Returns two electron integrals in AO basis converted to given MO basis
        Returns two electron integrals in AI basis converted to given MO basis

        Args:
            ints (numpy.ndarray): N^2 two electron integrals in AO basis
            moc (numpy.ndarray): Molecular orbital coefficients

        Returns:
            numpy.ndarray: integrals in MO basis
        """
            numpy.ndarray: integrals in ML basis

        dim = ints.shape[0]
        eri_mo = numpy.zeros((dim, dim, dim, dim))

        for a_i in range(dim):
            temp1 = numpy.einsum("i,i...->...", moc[:, a_i], ints)
            for b_i in range(dim):
                temp2 = numpy.einsum("j,j...->...", moc[:, b_i], temp1)
                temp3 = numpy.einsum("kc,k...->...c", moc, temp2)
                eri_mo[a_i, b_i, :, :] = numpy.einsum("ld,l...c->...cd", moc, temp3)
        return eri_mo

    @staticmethod
    def twoeints2mo_general(ints, moc1, moc2, moc3, moc4):
        """twoeints2mo_general"""
        twoeints2mo_general
        return numpy.einsum("pqrs,pi,qj,rk,sl->ijkl", ints, moc1, moc2, moc3, moc4)

    @staticmethod
    def onee_to_spin(mohij, mohij_b=None, threshold=1e-12):
        """Convert one-body MO integrals to spin orbital basis
        ""Convert one-body ML integrals to spin orbital

        Takes one body integrals in molecular orbital basis and returns
        integrals in spin orbitals ready for use as coefficients to
        one body terms 2nd quantized Hamiltonian.
        Args:
            mohij (numpy.ndarray): One body orbitals in molecular basis (Alpha)
            mohij_b (numpy.ndarray): One body orbitals in molecular basis (Beta)
            threshold (float): Threshold value for assignments
        Returns:
            numpy.ndarray: One body integrals in spin orbitals
        """

        if mohij_b is None:
            mohij_b = mohij

        # The number of spin orbitals is twice the number of orbitals
        norbs = mohij.shape[0]
        nspin_orbs = 2 * norbs
        # One electron terms
        moh1_qubit = numpy.zeros([nspin_orbs, nspin_orbs])
        for p in range(nspin_orbs):  # pylint: disable=invalid-name
            for q in range(nspin_orbs):
                spinp = int(p / norbs)
                spinq = int(q / norbs)
                if spinp % 2 != spinq % 2:
                    continue
                ints = mohij if spinp == 0 else mohij_b
                orbp = int(p % norbs)
                orbq = int(q % norbs)
                if abs(ints[orbp, orbq]) > threshold:
                    moh1_qubit[p, q] = ints[orbp, orbq]
        return moh1_qubit

    @staticmethod
    def twoe_to_spin(mohijkl, mohijkl_bb=None, mohijkl_ba=None, threshold=1e-12):
        """Convert two-body MO integrals to spin orbital basis
        Convert two-body MO integrals to spin orbital basis

        Takes two body integrals in molecular orbital basis and returns
        integrals in spin orbitals ready for use as coefficients to
        two body terms in 2nd quantized Hamiltonian.
        Args:
            mohijkl (numpy.ndarray): Two body orbitals in molecular basis (AlphaAlpha)
            mohijkl_bb (numpy.ndarray): Two body orbitals in molecular basis (BetaBeta)
            mohijkl_ba (numpy.ndarray): Two body orbitals in molecular basis (BetaAlpha)
            threshold (float): Threshold value for assignments
        Returns:
            numpy.ndarray: Two body integrals in spin orbitals
        """

        ints_aa = numpy.einsum(TWOE_TO_SPIN_SUBSCRIPT, mohijkl)

        if mohijkl_bb is None or mohijkl_ba is None:
            ints_bb = ints_ba = ints_ab = ints_aa
        else:
            ints_bb = numpy.einsum(TWOE_TO_SPIN_SUBSCRIPT, mohijkl_bb)
            ints_ba = numpy.einsum(TWOE_TO_SPIN_SUBSCRIPT, mohijkl_ba)
            ints_ab = numpy.einsum(TWOE_TO_SPIN_SUBSCRIPT, mohijkl_ba.transpose())
        # The number of spin orbitals is twice the number of orbitals
        norbs = mohijkl.shape[0]
        nspin_orbs = 2 * norbs
        # The spin orbitals are mapped in the following way:
        #       Orbital zero, spin up mapped to qubit 0
        #       Orbital one,  spin up mapped to qubit 1
        #       Orbital two,  spin up mapped to qubit 2
        #            .
        #            .
        #       Orbital zero, spin down mapped to qubit norbs
        #       Orbital one,  spin down mapped to qubit norbs+1
        #            .
        #            .
        #            .
        # Two electron terms
        moh2_qubit = numpy.zeros([nspin_orbs, nspin_orbs, nspin_orbs, nspin_orbs])
        for p in range(nspin_orbs):  # pylint: disable=invalid-name
            for q in range(nspin_orbs):
                for r in range(nspin_orbs):
                    for s in range(nspin_orbs):  # pylint: disable=invalid-name
                        spinp = int(p / norbs)
                        spinq = int(q / norbs)
                        spinr = int(r / norbs)
                        spins = int(s / norbs)
                        if spinp != spins:
                            continue
                        if spinq != spinr:
                            continue
                        if spinp == 0:
                            ints = ints_aa if spinq == 0 else ints_ba
                        else:
                            ints = ints_ab if spinq == 0 else ints_bb
                        orbp = int(p % norbs)
                        orbq = int(q % norbs)
                        orbr = int(r % norbs)
                        orbs = int(s % norbs)
                        if abs(ints[orbp, orbq, orbr, orbs]) > threshold:
                            moh2_qubit[p, q, r, s] = -0.5 * ints[orbp, orbq, orbr, orbs]
        return moh2_qubit
    symbols = [
        # pylint: disable=bad-option-value,bad-whitespace
        "_",
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Rb",
        "Sr",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
        "Sb",
        "Te",
        "I",
        "Xe",
        "Cs",
        "Ba",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Tl",
        "Pb",
        "Bi",
        "Po",
        "At",
        "Rn",
        "Fr",
        "Ra",
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
        "Lr",
        "Rf",
        "Db",
        "Sg",
        "Bh",
        "Hs",
        "Mt",
        "Ds",
        "Rg",
        "Cn",
        "Nh",
        "Fl",
        "Mc",
        "Lv",
        "Ts",
        "Og",
    ]
    BOHR = 0.52917721092  # No of Angstroms in Bohr (from 2010 CODATA)
    DEBYE = 0.393430307  # No ea0 in Debye. Use to convert our dipole moment numbers to Debye

    def log(self):
        """log properties"""
    def log.html(self):
        log properties
        if not logger.isEnabledFor(logging.INFO):
            return
        opts = numpy.get_printoptions()
        try:
            numpy.set_printoptions(precision=8, suppress=True)
            # Originating driver name & config if set
            if self.origin_driver_name and self.origin_driver_name != "?":
                logger.info("Originating driver name: %s", self.origin_driver_name)
                logger.info("Originating driver version: %s", self.origin_driver_version)
                logger.info("Originating driver config:\n%s", self.origin_driver_config[:-1])
            logger.info("Computed Hartree-Fock energy: %s", self.hf_energy)
            logger.info("Nuclear repulsion energy: %s", self.nuclear_repulsion_energy)
            logger.info("Energy shift due to transformations: %s", self.energy_shift)
            if None not in (self.hf_energy, self.nuclear_repulsion_energy):
                logger.info(
                    "One and two electron Hartree-Fock energy: %s",
                    self.hf_energy - self.nuclear_repulsion_energy,
                )
            logger.info("Number of orbitals is %s", self.num_molecular_orbitals)
            logger.info("%s alpha and %s beta electrons", self.num_alpha, self.num_beta)
            logger.info("Molecule comprises %s atoms and in xyz format is ::", self.num_atoms)
            logger.info("  %s, %s", self.molecular_charge, self.multiplicity)
            if self.num_atoms is not None:
                for n in range(0, self.num_atoms):
                    logger.info(
                        "  %2s  %s, %s, %s",
                        self.atom_symbol[n],
                        self.atom_xyz[n][0] * QMolecule.BOHR,
                        self.atom_xyz[n][1] * QMolecule.BOHR,
                        self.atom_xyz[n][2] * QMolecule.BOHR,
                    )
            if self.mo_coeff is not None:
                logger.info("MO coefficients A: %s", self.mo_coeff.shape)
                logger.debug("\n%s", self.mo_coeff)
            if self.mo_coeff_b is not None:
                logger.info("MO coefficients B: %s", self.mo_coeff_b.shape)
                logger.debug("\n%s", self.mo_coeff_b)
            if self.orbital_energies is not None:
                logger.info("Orbital energies A: %s", self.orbital_energies)
            if self.orbital_energies_b is not None:
                logger.info("Orbital energies B: %s", self.orbital_energies_b)
            if self.mo_occ is not None:
                logger.info("MO occupation numbers A: %s", self.mo_occ.shape)
                logger.debug("\n%s", self.mo_occ)
            if self.mo_occ_b is not None:
                logger.info("MO occupation numbers B: %s", self.mo_occ_b.shape)
                logger.debug("\n%s", self.mo_occ_b)
            if self.hcore is not None:
                logger.info("hcore integrals: %s", self.hcore.shape)
                logger.debug("\n%s", self.hcore)
            if self.hcore_b is not None:
                logger.info("hcore Beta integrals: %s", self.hcore_b.shape)
                logger.debug("\n%s", self.hcore_b)
            if self.kinetic is not None:
                logger.info("kinetic integrals: %s", self.kinetic.shape)
                logger.debug("\n%s", self.kinetic)
            if self.overlap is not None:
                logger.info("overlap integrals: %s", self.overlap.shape)
                logger.debug("\n%s", self.overlap)
            if self.eri is not None:
                logger.info("eri integrals: %s", self.eri.shape)
                logger.debug("\n%s", self.eri)
            if self.mo_onee_ints is not None:
                logger.info("One body MO A integrals: %s", self.mo_onee_ints.shape)
                logger.debug("\n%s", self.mo_onee_ints)
            if self.mo_onee_ints_b is not None:
                logger.info("One body MO B integrals: %s", self.mo_onee_ints_b.shape)
                logger.debug(self.mo_onee_ints_b)
            if self.mo_eri_ints is not None:
                logger.info("Two body ERI MO AA integrals: %s", self.mo_eri_ints.shape)
                logger.debug("\n%s", self.mo_eri_ints)
            if self.mo_eri_ints_bb is not None:
                logger.info("Two body ERI MO BB integrals: %s", self.mo_eri_ints_bb.shape)
                logger.debug("\n%s", self.mo_eri_ints_bb)
            if self.mo_eri_ints_ba is not None:
                logger.info("Two body ERI MO BA integrals: %s", self.mo_eri_ints_ba.shape)
                logger.debug("\n%s", self.mo_eri_ints_ba)
            if self.x_dip_ints is not None:
                logger.info("x dipole integrals: %s", self.x_dip_ints.shape)
                logger.debug("\n%s", self.x_dip_ints)
            if self.y_dip_ints is not None:
                logger.info("y dipole integrals: %s", self.y_dip_ints.shape)
                logger.debug("\n%s", self.y_dip_ints)
            if self.z_dip_ints is not None:
                logger.info("z dipole integrals: %s", self.z_dip_ints.shape)
                logger.debug("\n%s", self.z_dip_ints)
            if self.x_dip_mo_ints is not None:
                logger.info("x dipole MO A integrals: %s", self.x_dip_mo_ints.shape)
                logger.debug("\n%s", self.x_dip_mo_ints)
            if self.x_dip_mo_ints_b is not None:
                logger.info("x dipole MO B integrals: %s", self.x_dip_mo_ints_b.shape)
                logger.debug("\n%s", self.x_dip_mo_ints_b)
            if self.y_dip_mo_ints is not None:
                logger.info("y dipole MO A integrals: %s", self.y_dip_mo_ints.shape)
                logger.debug("\n%s", self.y_dip_mo_ints)
            if self.y_dip_mo_ints_b is not None:
                logger.info("y dipole MO B integrals: %s", self.y_dip_mo_ints_b.shape)
                logger.debug("\n%s", self.y_dip_mo_ints_b)
            if self.z_dip_mo_ints is not None:
                logger.info("z dipole MO A integrals: %s", self.z_dip_mo_ints.shape)
                logger.debug("\n%s", self.z_dip_mo_ints)
            if self.z_dip_mo_ints_b is not None:
                logger.info("z dipole MO B integrals: %s", self.z_dip_mo_ints_b.shape)
                logger.debug("\n%s", self.z_dip_mo_ints_b)
            if self.nuclear_dipole_moment is not None:
                logger.info("Nuclear dipole moment: %s", self.nuclear_dipole_moment)
            if self.reverse_dipole_sign is not None:
                logger.info(
                    "Reversal of electronic dipole moment sign needed: %s",
                    self.reverse_dipole_sign,
                )
            logger.info("Core orbitals list %s", self.core_orbitals)
        finally:
            numpy.set_printoptions(**opts)
0 comments on commit 21315df
@BigGuy573
 
 
© 2021 GitHub, Inc.

from qiskit_nature.drivers import PySCFDriver, UnitsType
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem

# Use PySCF, a classical computational chemistry software
# package, to compute the one-body and two-body integrals in
# electronic-orbital basis, necessary to form the Fermionic operator
driver = PySCFDriver(atom='H .0 .0 .0; H .0 .0 0.735',
                     unit=UnitsType.ANGSTROM,
                     basis='sto3g')
problem = ElectronicStructureProblem(driver)

# generate the second-quantized operators
second_q_ops = problem.second_q_ops()
main_op = second_q_ops[0]

particle_number = problem.properties_transformed.get_property("ParticleNumber")

num_particles = (particle_number.num_alpha, particle_number.num_beta)
num_spin_orbitals = particle_number.num_spin_orbitals

# setup the classical optimizer for VQE
from qiskit.algorithms.optimizers import L_BFGS_B

optimizer = L_BFGS_B()

# setup the mapper and qubit converter
from qiskit_nature.mappers.second_quantization import ParityMapper
from qiskit_nature.converters.second_quantization import QubitConverter

mapper = ParityMapper()
converter = QubitConverter(mapper=mapper, two_qubit_reduction=True)

# map to qubit operators
qubit_op = converter.convert(main_op, num_particles=num_particles)

# setup the initial state for the ansatz
from qiskit_nature.circuit.library import HartreeFock

init_state = HartreeFock(num_spin_orbitals, num_particles, converter)

# setup the ansatz for VQE
from qiskit.circuit.library import TwoLocal

ansatz = TwoLocal(num_spin_orbitals, ['ry', 'rz'], 'cz')

# add the initial state
ansatz.compose(init_state, front=True, inplace=True)

# set the backend for the quantum computation
from qiskit import Aer

backend = Aer.get_backend('aer_simulator_statevector')

# setup and run VQE
from qiskit.algorithms import VQE

algorithm = VQE(ansatz,
                optimizer=optimizer,
                quantum_instance=backend)

result = algorithm.compute_minimum_eigenvalue(qubit_op)
print(result.eigenvalue.real)

electronic_structure_result = problem.interpret(result)
print(electronic_structure_result)
```
The program above uses a quantum computer to calculate the ground state energy of molecular Hydrogen,
H<sub>2</sub>, where the two atoms are configured to be at a distance of 0.735 angstroms. The molecular
input specification is processed by the PySCF driver. This driver is wrapped by the `ElectronicStructureProblem`.
This problem instance generates a list of second-quantized operators which we can map to qubit operators
with a `QubitConverter`. Here, we chose the parity mapping in combination with a 2-qubit reduction, which
is a precision-preserving optimization removing two qubits; a reduction in complexity that is particularly
advantageous for NISQ computers.

The qubit operator is then passed as an input to the Variational Quantum Eigensolver (VQE) algorithm,
instantiated with a classical optimizer and a RyRz ansatz (`TwoLocal`). A Hartree-Fock initial state
is used as a starting point for the ansatz.

The VQE algorithm is then run, in this case on the Qiskit Aer statevector simulator backend.
Here we pass a backend but it can be wrapped into a `QuantumInstance`, and that passed to the
`run` instead. The `QuantumInstance` API allows you to customize run-time properties of the backend,
such as the number of shots, the maximum number of credits to use, settings for the simulator,
initial layout of qubits in the mapping and the Terra `PassManager` that will handle the compilation
of the circuits. By passing in a backend as is done above it is internally wrapped into a
`QuantumInstance` and is a convenience when default setting suffice.

In the end, you are given a result object by the VQE which you can analyze further by interpreting it with
your problem instance.

### Further examples

Learning path notebooks may be found in the
[Nature Tutorials](https://qiskit.org/documentation/nature/tutorials/index.html) section
of the documentation and are a great place to start

Jupyter notebooks containing further Nature examples may be found in the
following Qiskit GitHub repositories at
[qiskit-nature/docs/tutorials](https://github.com/Qiskit/qiskit-nature/tree/main/docs/tutorials).


----------------------------------------------------------------------------------------------------


## Contribution Guidelines

If you'd like to contribute to Qiskit, please take a look at our
[contribution guidelines](./CONTRIBUTING.md).
This project adheres to Qiskit's [code of conduct](./CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.

We use [GitHub issues](https://github.com/Qiskit/qiskit-nature/issues) for tracking requests and bugs. Please
[join the Qiskit Slack community](https://ibm.co/joinqiskitslack)
for discussion and simple questions.
For questions that are more suited for a forum, we use the **Qiskit** tag in [Stack Overflow](https://stackoverflow.com/questions/tagged/qiskit).

## Authors and Citation

Qiskit Nature was inspired, authored and brought about by the collective work of a team of researchers.
Qiskit Nature continues to grow with the help and work of
[many people](https://github.com/Qiskit/qiskit-nature/graphs/contributors), who contribute
to the project at different levels.
If you use Qiskit, please cite as per the provided
[BibTeX file](https://github.com/Qiskit/qiskit/blob/master/Qiskit.bib).

Please note that if you do not like the way your name is cited in the BibTex file then consult
the information found in the [.mailmap](https://github.com/Qiskit/qiskit-nature/blob/main/.mailmap)
file.

## License

This project uses the [Apache License 2.0](LICENSE.txt).

However there is some code that is included under other licensing as follows:

* The [Gaussian 16 driver](qiskit_nature/drivers/second_quantization/gaussiand) in `qiskit.chemistry`
  contains [work](qiskit_nature/drivers/second_quantization/gaussiand/gauopen) licensed under the
  [Gaussian Open-Source Public License](qiskit_nature/drivers/second_quantization/gaussiand/gauopen/LICENSE.txt).
