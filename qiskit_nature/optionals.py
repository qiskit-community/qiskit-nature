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

from typing import Optional, Callable, Iterable, Union
import shutil
from qiskit.utils import LazyImportTester, LazySubprocessTester


class NatureLazySubprocessTester(LazySubprocessTester):
    """A lazy checker that a command-line tool is available.
    First it checks with `shutil.which`.
    Then it will run the command based on flag passed.
    """

    def __init__(
        self,
        command: Union[str, Iterable[str]],
        run: Optional[bool] = True,
        *,
        name: Optional[str] = None,
        callback: Optional[Callable[[bool], None]] = None,
        install: Optional[str] = None,
        msg: Optional[str] = None,
    ):
        """
        Args:
            command: the strings that make up the command to be run.  For example,
                ``["pdflatex", "-version"]``.
            run: flag to indicate if the command should be run

        Raises:
            ValueError: if an empty command is given.
        """
        self._run = run
        super().__init__(command=command, name=name, callback=callback, install=install, msg=msg)

    def _is_available(self):
        if shutil.which(self._command[0]) is None:
            return False
        return super()._is_available() if self._run else True


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
HAS_GAUSSIAN = NatureLazySubprocessTester(
    GAUSSIAN_16,
    False,
    name=GAUSSIAN_16,
    msg="Please check that it is installed correctly",
)

PSI4 = "psi4"
PSI4_DESC = "PSI4"
HAS_PSI4 = NatureLazySubprocessTester(
    (PSI4, "--version"),
    name=PSI4_DESC,
    msg="See https://psicode.org",
)
