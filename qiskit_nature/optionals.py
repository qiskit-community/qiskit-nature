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

"""
Additional optional constants.
"""

from typing import Optional, Callable
from shutil import which
from qiskit.utils import LazyDependencyManager, LazyImportTester, LazySubprocessTester


class NatureLazyCommandTester(LazyDependencyManager):
    """
    A lazy checker that a command-line tool is available. This just checks if the command is
    not empty.
    """

    def __init__(
        self,
        command: str,
        *,
        name: Optional[str] = None,
        callback: Optional[Callable[[bool], None]] = None,
        install: Optional[str] = None,
        msg: Optional[str] = None,
    ):
        """
        Args:
            command: the string that make up the command.  For example,``"g16"``.
        """
        self._command = "" if command is None else command
        super().__init__(name=name or self._command, callback=callback, install=install, msg=msg)

    def _is_available(self):
        return bool(self._command)


HAS_PYQUANTE2 = LazyImportTester(
    {
        "pyquante2": ("molecule", "rhf", "uhf", "rohf", "basisset", "onee_integrals"),
        "pyquante2.geo.zmatrix": ("z2xyz",),
        "pyquante2.ints.integrals": ("twoe_integrals",),
    },
    name="pyquante2",
    msg="See https://github.com/rpmuller/pyquante2",
)

HAS_PYSCF = LazyImportTester(
    {
        "pyscf": ("__version__", "gto", "scf"),
        "pyscf.lib": ("chkfile", "logger", "param"),
        "pyscf.tools": ("dump_mat",),
    },
    name="pyscf",
    msg="See https://pyscf.org/install.html",
)

GAUSSIAN_16 = "g16"
GAUSSIAN_16_DESC = "Gaussian 16"
G16PROG = which(GAUSSIAN_16)

HAS_GAUSSIAN = NatureLazyCommandTester(
    G16PROG,
    name=GAUSSIAN_16_DESC,
    msg="Please check that it is installed correctly",
)

PSI4 = "psi4"
PSI4_DESC = "PSI4"
PSI4PROG = which(PSI4)
if PSI4PROG is None:
    PSI4PROG = PSI4

HAS_PSI4 = LazySubprocessTester(
    (PSI4PROG, "--version"),
    name=PSI4_DESC,
    msg="See https://psicode.org",
)
