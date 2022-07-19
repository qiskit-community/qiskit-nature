# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The electronic structure result."""

from functools import reduce
from typing import Dict, List, Optional, Tuple, cast

import numpy as np

from qiskit_nature.constants import DEBYE

from .eigenstate_result import EigenstateResult

# A dipole moment, when present as X, Y and Z components will normally have float values for all
# the components. However when using Z2Symmetries, if the dipole component operator does not
# commute with the symmetry then no evaluation is done and None will be used as the 'value'
# indicating no measurement of the observable took place
DipoleTuple = Tuple[Optional[float], Optional[float], Optional[float]]


class ElectronicStructureResult(EigenstateResult):
    """The electronic structure result."""

    def __init__(self) -> None:
        super().__init__()
        self._hartree_fock_energy: float = 0.0
        self._nuclear_repulsion_energy: Optional[float] = None
        self._nuclear_dipole_moment: Optional[DipoleTuple] = None
        self._computed_energies: Optional[np.ndarray] = None
        self._computed_dipole_moment: Optional[List[DipoleTuple]] = None
        self._extracted_transformer_energies: Dict[str, float] = {}
        self._extracted_transformer_dipoles: Optional[List[Dict[str, DipoleTuple]]] = None
        self._reverse_dipole_sign: bool = False
        self._num_particles: Optional[List[float]] = None
        self._magnetization: Optional[List[float]] = None
        self._total_angular_momentum: Optional[List[float]] = None

    @property
    def hartree_fock_energy(self) -> float:
        """Returns Hartree-Fock energy"""
        return self._hartree_fock_energy

    @hartree_fock_energy.setter
    def hartree_fock_energy(self, value: float) -> None:
        """Sets Hartree-Fock energy"""
        self._hartree_fock_energy = value

    @property
    def nuclear_repulsion_energy(self) -> Optional[float]:
        """Returns nuclear repulsion energy when available from driver"""
        return self._nuclear_repulsion_energy

    @nuclear_repulsion_energy.setter
    def nuclear_repulsion_energy(self, value: float) -> None:
        """Sets nuclear repulsion energy"""
        self._nuclear_repulsion_energy = value

    @property
    def nuclear_dipole_moment(self) -> Optional[DipoleTuple]:
        """Returns nuclear dipole moment X,Y,Z components in A.U when available from driver"""
        return self._nuclear_dipole_moment

    @nuclear_dipole_moment.setter
    def nuclear_dipole_moment(self, value: DipoleTuple) -> None:
        """Sets nuclear dipole moment in A.U"""
        self._nuclear_dipole_moment = value

    # TODO we need to be able to extract the statevector or the optimal parameters that can
    # construct the circuit of the GS from here (if the algorithm supports this)

    @property
    def total_energies(self) -> Optional[np.ndarray]:
        """Returns ground state energy if nuclear_repulsion_energy is available from driver"""
        if self.electronic_energies is None:
            return None
        nre = self.nuclear_repulsion_energy if self.nuclear_repulsion_energy is not None else 0
        # Adding float to np.ndarray adds it to each entry
        return self.electronic_energies + nre

    @property
    def electronic_energies(self) -> Optional[np.ndarray]:
        """Returns electronic part of ground state energy"""
        # TODO the fact that this property is computed on the fly breaks the `.combine()`
        # functionality
        # Adding float to np.ndarray adds it to each entry
        if self.computed_energies is None:
            return None
        return self.computed_energies + self.extracted_transformer_energy

    @property
    def computed_energies(self) -> Optional[np.ndarray]:
        """Returns computed electronic part of ground state energy"""
        return self._computed_energies

    @computed_energies.setter
    def computed_energies(self, value: np.ndarray) -> None:
        """Sets computed electronic part of ground state energy"""
        self._computed_energies = value

    @property
    def extracted_transformer_energies(self) -> Dict[str, float]:
        """Returns the energies extracted by any applied transformers."""
        return self._extracted_transformer_energies

    @extracted_transformer_energies.setter
    def extracted_transformer_energies(self, value: Dict[str, float]) -> None:
        """Sets the energies extracted by any applied transformers."""
        self._extracted_transformer_energies = value

    @property
    def extracted_transformer_energy(self) -> float:
        """Returns the sum of all extracted energies."""
        return sum(self.extracted_transformer_energies.values())

    # Dipole moment results. Note dipole moments of tuples of X, Y and Z components. Chemistry
    # drivers either support dipole integrals or not. Note that when using Z2 symmetries of

    def has_dipole(self) -> bool:
        """Returns whether dipole moment is present in result or not"""
        return self.nuclear_dipole_moment is not None and self.electronic_dipole_moment is not None

    @property
    def reverse_dipole_sign(self) -> bool:
        """Returns if electronic dipole moment sign should be reversed when adding to nuclear"""
        return self._reverse_dipole_sign

    @reverse_dipole_sign.setter
    def reverse_dipole_sign(self, value: bool) -> None:
        """Sets if electronic dipole moment sign should be reversed when adding to nuclear"""
        self._reverse_dipole_sign = value

    @property
    def total_dipole_moment(self) -> Optional[List[float]]:
        """Returns total dipole of moment"""
        if self.dipole_moment is None:
            return None  # No dipole at all
        tdm: List[float] = []
        for dip in self.dipole_moment:
            if np.any(np.equal(list(dip), None)):
                tdm.append(None)  # One or more components in the dipole is None
            else:
                tdm.append(np.sqrt(np.sum(np.power(list(dip), 2))))
        return tdm

    @property
    def total_dipole_moment_in_debye(self) -> Optional[List[float]]:
        """Returns total dipole of moment in Debye"""
        tdm = self.total_dipole_moment
        return [dip / DEBYE if dip is not None else None for dip in tdm]

    @property
    def dipole_moment(self) -> Optional[List[DipoleTuple]]:
        """Returns dipole moment"""
        edm = self.electronic_dipole_moment
        if edm is None:
            return None
        nrd = self.nuclear_dipole_moment if self.nuclear_dipole_moment is not None else (0, 0, 0)
        if self.reverse_dipole_sign:
            edm = [
                cast(DipoleTuple, tuple(-1 * x if x is not None else None for x in dip))
                for dip in edm
            ]
        return [_dipole_tuple_add(dip, nrd) for dip in edm]

    @property
    def dipole_moment_in_debye(self) -> Optional[List[DipoleTuple]]:
        """Returns dipole moment in Debye"""
        dipm = self.dipole_moment
        if dipm is None:
            return None
        dipmd = []
        for dip in dipm:
            dipmd0 = dip[0] / DEBYE if dip[0] is not None else None
            dipmd1 = dip[1] / DEBYE if dip[1] is not None else None
            dipmd2 = dip[2] / DEBYE if dip[2] is not None else None
            dipmd += [(dipmd0, dipmd1, dipmd2)]
        return dipmd

    @property
    def electronic_dipole_moment(self) -> Optional[List[DipoleTuple]]:
        """Returns electronic dipole moment"""
        if self.computed_dipole_moment is None or self.extracted_transformer_dipoles is None:
            return None
        return [
            _dipole_tuple_add(comp_dip, extr_dip)
            for comp_dip, extr_dip in zip(
                self.computed_dipole_moment, self.extracted_transformer_dipole
            )
        ]

    @property
    def computed_dipole_moment(self) -> Optional[List[DipoleTuple]]:
        """Returns computed electronic part of dipole moment"""
        return self._computed_dipole_moment

    @computed_dipole_moment.setter
    def computed_dipole_moment(self, value: List[DipoleTuple]) -> None:
        """Sets computed electronic part of dipole moment"""
        self._computed_dipole_moment = value

    @property
    def extracted_transformer_dipoles(self) -> Optional[List[Dict[str, DipoleTuple]]]:
        """Returns the dipole moments extracted by any applied transformers."""
        return self._extracted_transformer_dipoles

    @extracted_transformer_dipoles.setter
    def extracted_transformer_dipoles(self, value: List[Dict[str, DipoleTuple]]) -> None:
        """Sets the dipole moments extracted by any applied transformers."""
        self._extracted_transformer_dipoles = value

    @property
    def extracted_transformer_dipole(self) -> List[DipoleTuple]:
        """Returns the sum of all extracted dipole moments."""
        extracted_dips = self.extracted_transformer_dipoles
        if extracted_dips is None:
            return []
        extracted_dipms = []
        for dipm in self.extracted_transformer_dipoles:
            if not dipm:
                extracted_dipms.append(cast(DipoleTuple, (0, 0, 0)))
            else:
                extracted_dipms.append(reduce(_dipole_tuple_add, dipm.values()))
        return extracted_dipms

    # Other measured operators. If these are not evaluated then None will be returned
    # instead of any measured value.

    def has_observables(self):
        """Returns whether result has aux op observables such as spin, num particles"""
        return (
            self.total_angular_momentum is not None
            or self.num_particles is not None
            or self.magnetization is not None
        )

    @property
    def total_angular_momentum(self) -> Optional[List[float]]:
        """Returns total angular momentum (S^2)"""
        return self._total_angular_momentum

    @total_angular_momentum.setter
    def total_angular_momentum(self, value: List[float]) -> None:
        """Sets total angular momentum"""
        self._total_angular_momentum = value

    @property
    def spin(self) -> Optional[List[float]]:
        """Returns computed spin"""
        if self.total_angular_momentum is None:
            return None
        spin = []
        for total_angular_momentum in self.total_angular_momentum:
            spin.append((-1.0 + np.sqrt(1 + 4 * total_angular_momentum)) / 2)
        return spin

    @property
    def num_particles(self) -> Optional[List[float]]:
        """Returns measured number of particles"""
        return self._num_particles

    @num_particles.setter
    def num_particles(self, value: List[float]) -> None:
        """Sets measured number of particles"""
        self._num_particles = value

    @property
    def magnetization(self) -> Optional[List[float]]:
        """Returns measured magnetization"""
        return self._magnetization

    @magnetization.setter
    def magnetization(self, value: List[float]) -> None:
        """Sets measured magnetization"""
        self._magnetization = value

    def __str__(self) -> str:
        """Printable formatted result"""
        return "\n".join(self.formatted())

    def formatted(self) -> List[str]:
        """Formatted result as a list of strings"""
        lines = []
        lines.append("=== GROUND STATE ENERGY ===")
        lines.append(" ")
        if self.electronic_energies is not None:
            lines.append(
                "* Electronic ground state energy (Hartree): "
                f"{_complex_to_string(self.electronic_energies[0], 12)}"
            )
            lines.append(
                f"  - computed part:      {_complex_to_string(self.computed_energies[0], 12)}"
            )
            for name, value in self.extracted_transformer_energies.items():
                lines.append(f"  - {name} extracted energy part: {_complex_to_string(value, 12)}")
        if self.nuclear_repulsion_energy is not None:
            lines.append(
                "~ Nuclear repulsion energy (Hartree): "
                f"{_complex_to_string(self.nuclear_repulsion_energy, 12)}"
            )
            lines.append(
                "> Total ground state energy (Hartree): "
                f"{_complex_to_string(self.total_energies[0], 12)}"
            )

        if self.computed_energies is not None and len(self.computed_energies) > 1:
            lines.append(" ")
            lines.append("=== EXCITED STATE ENERGIES ===")
            lines.append(" ")
            for idx, (elec_energy, total_energy) in enumerate(
                zip(self.electronic_energies[1:], self.total_energies[1:])
            ):
                lines.append(f"{(idx + 1): 3d}: ")
                lines.append(
                    f"* Electronic excited state energy (Hartree): {_complex_to_string(elec_energy, 12)}"
                )
                lines.append(
                    f"> Total excited state energy (Hartree): {_complex_to_string(total_energy, 12)}"
                )

        if self.has_observables():
            lines.append(" ")
            lines.append("=== MEASURED OBSERVABLES ===")
            lines.append(" ")
            for idx, (num_particles, spin, total_angular_momentum, magnetization,) in enumerate(
                zip(
                    self.num_particles,
                    self.spin,
                    self.total_angular_momentum,
                    self.magnetization,
                )
            ):
                line = f"{idx: 3d}: "
                if num_particles is not None:
                    line += f" # Particles: {num_particles:.3f}"
                if spin is not None:
                    line += f" S: {spin:.3f}"
                if total_angular_momentum is not None:
                    line += f" S^2: {total_angular_momentum:.3f}"
                if magnetization is not None:
                    line += f" M: {magnetization:.3f}"
                lines.append(line)

        if self.has_dipole():
            lines.append(" ")
            lines.append("=== DIPOLE MOMENTS ===")
            lines.append(" ")
            if self.nuclear_dipole_moment is not None:
                lines.append(
                    f"~ Nuclear dipole moment (a.u.): {_dipole_to_string(self.nuclear_dipole_moment)}"
                )
                lines.append(" ")
            for idx, (elec_dip, comp_dip, extr_dip, dip, tot_dip, dip_db, tot_dip_db,) in enumerate(
                zip(
                    self.electronic_dipole_moment,
                    self.computed_dipole_moment,
                    self.extracted_transformer_dipoles,
                    self.dipole_moment,
                    self.total_dipole_moment,
                    self.dipole_moment_in_debye,
                    self.total_dipole_moment_in_debye,
                )
            ):
                lines.append(f"{idx: 3d}: ")
                lines.append(f"  * Electronic dipole moment (a.u.): {_dipole_to_string(elec_dip)}")
                lines.append(f"    - computed part:      {_dipole_to_string(comp_dip)}")
                for name, ex_dip in extr_dip.items():
                    lines.append(f"    - {name} extracted energy part: {_dipole_to_string(ex_dip)}")
                if self.nuclear_dipole_moment is not None:
                    lines.append(
                        f"  > Dipole moment (a.u.): { _dipole_to_string(dip)}  "
                        f"Total: {_complex_to_string(tot_dip)}"
                    )
                    lines.append(
                        f"                 (debye): {_dipole_to_string(dip_db)}  "
                        f"Total: {_complex_to_string(tot_dip_db)}"
                    )
                lines.append(" ")

        return lines


def _dipole_tuple_add(x: Optional[DipoleTuple], y: Optional[DipoleTuple]) -> Optional[DipoleTuple]:
    """Utility to add two dipole tuples element-wise for dipole additions"""
    if x is None or y is None:
        return None
    return _element_add(x[0], y[0]), _element_add(x[1], y[1]), _element_add(x[2], y[2])


def _element_add(x: Optional[float], y: Optional[float]):
    """Add dipole elements where a value may be None then None is returned"""
    return x + y if x is not None and y is not None else None


def _dipole_to_string(dipole: DipoleTuple):
    value = "["
    for i, dip in enumerate(dipole):
        value += _complex_to_string(dip, 8) if dip is not None else "None"
        value += "  " if i < len(dipole) - 1 else "]"
    return value


def _complex_to_string(value: Optional[complex], precision: int = 8) -> str:
    if value is None:
        return "None"
    else:
        real = (
            "0.0"
            if round(value.real, precision) == 0
            else ("{:." + str(precision) + "f}").format(value.real).rstrip("0")
        )
        imag = (
            ""
            if round(value.imag, precision) == 0
            else ("{:." + str(precision) + "f}").format(value.imag).rstrip("0") + "j"
        )
        string = real
        if imag != "" and value.imag > 0:
            string += "+" + imag
        elif imag != "" and value.imag < 0:
            string += imag
        return string
