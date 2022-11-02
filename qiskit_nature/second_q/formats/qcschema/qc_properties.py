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

"""The QCSchema properties dataclass."""
# pylint: disable=invalid-name

from __future__ import annotations

from dataclasses import dataclass

from .qc_base import _QCBase


@dataclass
class QCProperties(_QCBase):
    """A dataclass to store the computed properties of the original calculation.

    For more information refer to
    [here](https://molssi-qc-schema.readthedocs.io/en/latest/auto_props.html#properties-schema).
    """

    calcinfo_nbasis: int | None = None
    """The number of basis functions in the computation."""
    calcinfo_nmo: int | None = None
    """The number of molecular orbitals in the computation."""
    calcinfo_nalpha: int | None = None
    """The number of alpha-spin electrons in the computation."""
    calcinfo_nbeta: int | None = None
    """The number of beta-spin electrons in the computation."""
    calcinfo_natom: int | None = None
    """The number of atoms in the computation."""
    return_energy: float | None = None
    """The returned energy of the computation. When :attr:`QCSchemaInput.driver` is `energy`, this
    value is identical to :attr:`QCSchema.return_result`."""

    scf_one_electron_energy: float | None = None
    """The one-electron energy contribution to the total SCF energy."""
    scf_two_electron_energy: float | None = None
    """The two-electron energy contribution to the total SCF energy."""
    nuclear_repulsion_energy: float | None = None
    """The nuclear repulsion energy contribution to the total SCF energy."""
    nuclear_dipole_moment: tuple[float, float, float] | None = None
    """The nuclear X, Y, and Z dipole components."""
    scf_vv10_energy: float | None = None
    """The VV10 functional energy contribution to the total SCF energy."""
    scf_xc_energy: float | None = None
    """The XC functional energy contribution to the total SCF energy."""
    scf_dispersion_correction_energy: float | None = None
    """The dispersion correction appended to the underlying functional in a DFT-D method."""
    scf_dipole_moment: tuple[float, float, float] | None = None
    """The total SCF X, Y, and Z dipole components."""
    scf_total_energy: float | None = None
    """The total SCF energy."""
    scf_iterations: int | None = None
    """The number of SCF iterations taken during the computation."""

    mp2_same_spin_correlation_energy: float | None = None
    """The MP2 doubles correlation energy contribution from same-spin (e.g. triplet) correlations,
    without any user scaling."""
    mp2_opposite_spin_correlation_energy: float | None = None
    """The MP2 doubles correlation energy contribution from opposite-spin (e.g. singlet)
    correlations, without any user scaling."""
    mp2_singles_energy: float | None = None
    """The MP2 singles correlation energy. This value is `0.0` except in ROHF."""
    mp2_doubles_energy: float | None = None
    """The total MP2 doubles correlation energy."""
    mp2_correlation_energy: float | None = None
    """The total MP2 correlation energy."""
    mp2_total_energy: float | None = None
    """The total MP2 energy (i.e. the sum of the SCF energy and MP2 correlation energy)."""
    mp2_dipole_moment: tuple[float, float, float] | None = None
    """The total MP2 X, Y, and Z dipole components."""

    ccsd_same_spin_correlation_energy: float | None = None
    """The CCSD doubles correlation energy contribution from same-spin (e.g. triplet) correlations,
    without any user scaling."""
    ccsd_opposite_spin_correlation_energy: float | None = None
    """The CCSD doubles correlation energy contribution from opposite-spin (e.g. singlet)
    correlations, without any user scaling."""
    ccsd_singles_energy: float | None = None
    """The CCSD singles correlation energy. This value is `0.0` except in ROHF."""
    ccsd_doubles_energy: float | None = None
    """The total CCSD doubles correlation energy."""
    ccsd_correlation_energy: float | None = None
    """The total CCSD correlation energy."""
    ccsd_total_energy: float | None = None
    """The total CCSD energy (i.e. the sum of the SCF energy and CCSD correlation energy)."""
    ccsd_prt_pr_correlation_energy: float | None = None
    """The total CCSD(T) correlation energy."""
    ccsd_prt_pr_total_energy: float | None = None
    """The total CCSD(T) energy (i.e. the sum of the SCF energy and CCSD(T) correlation energy)."""
    ccsdt_correlation_energy: float | None = None
    """The total CCSDT correlation energy."""
    ccsdt_total_energy: float | None = None
    """The total CCSDT energy (i.e. the sum of the SCF energy and CCSDT correlation energy)."""
    ccsdtq_correlation_energy: float | None = None
    """The total CCSDTQ correlation energy."""
    ccsdtq_total_energy: float | None = None
    """The total CCSDTQ energy (i.e. the sum of the SCF energy and CCSDTQ correlation energy)."""
    ccsd_dipole_moment: tuple[float, float, float] | None = None
    """The total CCSD X, Y, and Z dipole components."""
    ccsd_prt_pr_dipole_moment: tuple[float, float, float] | None = None
    """The total CCSD(T) X, Y, and Z dipole components."""
    ccsdt_dipole_moment: tuple[float, float, float] | None = None
    """The total CCSDT X, Y, and Z dipole components."""
    ccsdtq_dipole_moment: tuple[float, float, float] | None = None
    """The total CCSDTQ X, Y, and Z dipole components."""
    ccsd_iterations: int | None = None
    """The number of CCSD iterations taken during the computation."""
    ccsdt_iterations: int | None = None
    """The number of CCSDT iterations taken during the computation."""
    ccsdtq_iterations: int | None = None
    """The number of CCSDTQ iterations taken during the computation."""
