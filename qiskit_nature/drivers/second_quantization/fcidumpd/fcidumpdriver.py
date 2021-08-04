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

"""FCIDump Driver."""

from typing import List, Optional, cast

from qiskit_nature import QiskitNatureError
from qiskit_nature.properties.second_quantization.electronic import (
    ElectronicStructureDriverResult,
    ElectronicEnergy,
    ParticleNumber,
)
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis
from qiskit_nature.properties.second_quantization.electronic.integrals import (
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
)

from ...qmolecule import QMolecule
from .dumper import dump
from .parser import parse  # pylint: disable=deprecated-module
from ..electronic_structure_driver import ElectronicStructureDriver


class FCIDumpDriver(ElectronicStructureDriver):
    """
    Qiskit Nature driver for reading an FCIDump file.

    The FCIDump format is partially defined in Knowles1989.

    References:
        Knowles1989: Peter J. Knowles, Nicholas C. Handy,
            A determinant based full configuration interaction program,
            Computer Physics Communications, Volume 54, Issue 1, 1989, Pages 75-83,
            ISSN 0010-4655, https://doi.org/10.1016/0010-4655(89)90033-7.
    """

    def __init__(self, fcidump_input: str, atoms: Optional[List[str]] = None) -> None:
        """
        Args:
            fcidump_input: Path to the FCIDump file.
            atoms: Allows to specify the atom list of the molecule. If it is provided, the created
                QMolecule instance will permit frozen core Hamiltonians. This list must consist of
                valid atom symbols.

        Raises:
            QiskitNatureError: If ``fcidump_input`` is not a string or if ``atoms`` is not a list
                of valid atomic symbols as specified in ``QMolecule``.
        """
        super().__init__()

        if not isinstance(fcidump_input, str):
            raise QiskitNatureError("The fcidump_input must be str, not '{}'".format(fcidump_input))
        self._fcidump_input = fcidump_input

        if (
            atoms
            and not isinstance(atoms, list)
            and not all(sym in QMolecule.symbols for sym in atoms)
        ):
            raise QiskitNatureError(
                "The atoms must be a list of valid atomic symbols, not '{}'".format(atoms)
            )
        self.atoms = atoms

    def run(self) -> ElectronicStructureDriverResult:
        """Constructs a QMolecule instance out of a FCIDump file.

        Returns:
            A QMolecule instance populated with a minimal set of required data.
        """
        fcidump_data = parse(self._fcidump_input)

        hij = fcidump_data.get("hij", None)
        hij_b = fcidump_data.get("hij_b", None)
        hijkl = fcidump_data.get("hijkl", None)
        hijkl_ba = fcidump_data.get("hijkl_ba", None)
        hijkl_bb = fcidump_data.get("hijkl_bb", None)

        multiplicity = fcidump_data.get("MS2", 0) + 1
        num_beta = (fcidump_data.get("NELEC") - (multiplicity - 1)) // 2
        num_alpha = fcidump_data.get("NELEC") - num_beta

        particle_number = ParticleNumber(
            num_spin_orbitals=fcidump_data.get("NORB") * 2,
            num_particles=(num_alpha, num_beta),
        )

        electronic_energy = ElectronicEnergy(
            [
                OneBodyElectronicIntegrals(ElectronicBasis.MO, (hij, hij_b)),
                TwoBodyElectronicIntegrals(ElectronicBasis.MO, (hijkl, hijkl_ba, hijkl_bb, None)),
            ],
            nuclear_repulsion_energy=fcidump_data.get("ecore", None),
        )

        driver_result = ElectronicStructureDriverResult()
        driver_result.add_property(electronic_energy)
        driver_result.add_property(particle_number)

        return driver_result

    @staticmethod
    def dump(
        driver_result: ElectronicStructureDriverResult,
        outpath: str,
        orbsym: Optional[List[str]] = None,
        isym: int = 1,
    ) -> None:
        """Convenience method to produce an FCIDump output file.

        Args:
            outpath: Path to the output file.
            driver_result: The ElectronicStructureDriverResult to be dumped. It is assumed that the
                nuclear_repulsion_energy in its ElectronicEnergy property contains the inactive core
                energy.
            orbsym: A list of spatial symmetries of the orbitals.
            isym: The spatial symmetry of the wave function.
        """
        particle_number = cast(ParticleNumber, driver_result.get_property(ParticleNumber))
        electronic_energy = cast(ElectronicEnergy, driver_result.get_property(ElectronicEnergy))
        one_body_integrals = electronic_energy.get_electronic_integral(ElectronicBasis.MO, 1)
        two_body_integrals = electronic_energy.get_electronic_integral(ElectronicBasis.MO, 2)
        dump(
            outpath,
            particle_number.num_spin_orbitals // 2,
            particle_number.num_alpha + particle_number.num_beta,
            one_body_integrals._matrices,  # type: ignore
            two_body_integrals._matrices[0:3],  # type: ignore
            electronic_energy.nuclear_repulsion_energy,
            ms2=driver_result.molecule.multiplicity - 1,
            orbsym=orbsym,
            isym=isym,
        )
