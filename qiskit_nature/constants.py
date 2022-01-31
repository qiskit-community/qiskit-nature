# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Defines some constants used in chemical calculations.
"""

# multiplicative conversions
N_A = 6.02214129e23  # particles per mol

KCAL_PER_MOL_TO_J_PER_MOLECULE = 6.947695e-21
HARTREE_TO_KCAL_PER_MOL = 627.509474
HARTREE_TO_J_PER_MOL = 2625499.63922
HARTREE_TO_KJ_PER_MOL = 2625.49963922
HARTREE_TO_PER_CM = 219474.63
J_PER_MOL_TO_PER_CM = 0.08359347178
CAL_TO_J = 4.184
HARTREE_TO_J = 4.3597443380807824e-18  # HARTREE_TO_J_PER_MOL / N_A
J_TO_HARTREE = 2.293712480489655e17  # 1.0 / HARTREE_TO_J
M_TO_ANGSTROM = 1e10
ANGSTROM_TO_M = 1e-10


# physical constants
C_CM_PER_S = 2.9979245800e10
C_M_PER_S = 2.9979245800e8
HBAR_J_S = 1.054571800e-34  # note this is h/2Pi
H_J_S = 6.62607015e-34
KB_J_PER_K = 1.3806488e-23
BOHR = 0.52917721092  # No of Angstroms in Bohr (from 2010 CODATA)
DEBYE = 0.393430307  # No ea0 in Debye. Use to convert our dipole moment numbers to Debye

PERIODIC_TABLE = [
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
