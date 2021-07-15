# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The ParticleNumber property."""

from typing import List, Optional, Tuple, Union, cast

import numpy as np

from qiskit_nature.drivers.second_quantization import QMolecule
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.results import EigenstateResult

from ..second_quantized_property import (
    LegacyDriverResult,
    LegacyElectronicDriverResult,
    SecondQuantizedProperty,
)


class ParticleNumber(SecondQuantizedProperty):
    """The ParticleNumber property."""

    def __init__(
        self,
        num_spin_orbitals: int,
        num_particles: Union[int, Tuple[int, int]],
        occupation: Optional[List[int]] = None,
        occupation_beta: Optional[List[int]] = None,
    ):
        """
        Args:
            num_spin_orbitals: the number of spin orbitals in the system.
            num_particles: the number of particles in the system. If this is a pair of integers, the
                first is the alpha and the second is the beta spin. If it is an int, the number is
                halved, with the remainder being added onto the alpha spin number.
            occupation: the occupation numbers. If ``occupation_beta`` is ``None``, these are the
                total occupation numbers, otherwise these are treated as the alpha-spin occupation.
            occupation_beta: the beta-spin occupation numbers.
        """
        super().__init__(self.__class__.__name__)
        self._num_spin_orbitals = num_spin_orbitals
        if isinstance(num_particles, int):
            self._num_alpha = num_particles // 2 + num_particles % 2
            self._num_beta = num_particles // 2
        else:
            self._num_alpha, self._num_beta = num_particles

        if occupation is None:
            self._occupation_alpha = [1 for _ in range(self._num_alpha)]
            self._occupation_alpha += [0] * (num_spin_orbitals // 2 - len(self._occupation_alpha))
            self._occupation_beta = [1 for _ in range(self._num_beta)]
            self._occupation_beta += [0] * (num_spin_orbitals // 2 - len(self._occupation_beta))
        elif occupation_beta is None:
            self._occupation_alpha = [np.ceil(o / 2) for o in occupation]
            self._occupation_beta = [np.floor(o / 2) for o in occupation]
        else:
            self._occupation_alpha = occupation
            self._occupation_beta = occupation_beta

    @property
    def num_spin_orbitals(self) -> int:
        """Returns the num_spin_orbitals."""
        return self._num_spin_orbitals

    @property
    def num_alpha(self) -> int:
        """Returns the number of alpha electrons."""
        return self._num_alpha

    @property
    def num_beta(self) -> int:
        """Returns the number of beta electrons."""
        return self._num_beta

    @property
    def occupation_alpha(self) -> np.ndarray:
        """Returns the occupation_alpha."""
        return np.asarray(self._occupation_alpha, dtype=int)

    @property
    def occupation_beta(self) -> np.ndarray:
        """Returns the occupation_beta."""
        return np.asarray(self._occupation_beta, dtype=int)

    def __str__(self) -> str:
        string = [super().__str__() + ":"]
        string += [f"\t{self._num_spin_orbitals} SOs"]
        string += [f"\t{self._num_alpha} alpha electrons: {self.occupation_alpha}"]
        string += [f"\t{self._num_beta} beta electrons: {self.occupation_beta}"]
        return "\n".join(string)

    @classmethod
    def from_legacy_driver_result(cls, result: LegacyDriverResult) -> "ParticleNumber":
        """Construct a ParticleNumber instance from a QMolecule.

        Args:
            result: the driver result from which to extract the raw data. For this property, a
                QMolecule is required!

        Returns:
            An instance of this property.

        Raises:
            QiskitNatureError: if a WatsonHamiltonian is provided.
        """
        cls._validate_input_type(result, LegacyElectronicDriverResult)

        qmol = cast(QMolecule, result)

        return cls(
            qmol.num_molecular_orbitals * 2,
            (qmol.num_alpha, qmol.num_beta),
            qmol.mo_occ,
            qmol.mo_occ_b,
        )

    def reduce_system_size(self, active_orbital_indices: List[int]) -> "ParticleNumber":
        """Reduces the system size to a subset of active orbitals.

        Args:
            active_orbital_indices: the list of active orbital indices.

        Returns:
            A new ParticleNumber property instance of the reduced size.
        """
        active_occ_alpha = self.occupation_alpha[active_orbital_indices]
        active_occ_beta = self.occupation_beta[active_orbital_indices]
        num_alpha = sum(active_occ_alpha)
        num_beta = sum(active_occ_beta)
        return ParticleNumber(
            len(active_orbital_indices) * 2,
            (num_alpha, num_beta),
            active_occ_alpha,
            active_occ_beta,
        )

    def second_q_ops(self) -> List[FermionicOp]:
        """Returns a list containing the particle number operator."""
        op = FermionicOp(
            [(f"N_{o}", 1.0) for o in range(self._num_spin_orbitals)],
            register_length=self._num_spin_orbitals,
        )
        return [op]

    def interpret(self, result: EigenstateResult) -> None:
        """Interprets an :class:~qiskit_nature.result.EigenstateResult in this property's context.

        Args:
            result: the result to add meaning to.
        """
        result.num_particles = []

        if not isinstance(result.aux_operator_eigenvalues, list):
            aux_operator_eigenvalues = [result.aux_operator_eigenvalues]
        else:
            aux_operator_eigenvalues = result.aux_operator_eigenvalues  # type: ignore
        for aux_op_eigenvalues in aux_operator_eigenvalues:
            if aux_op_eigenvalues is None:
                continue

            if aux_op_eigenvalues[0] is not None:
                result.num_particles.append(aux_op_eigenvalues[0][0].real)  # type: ignore
            else:
                result.num_particles.append(None)
