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

import logging
from typing import List, Optional, Tuple, Union, cast

import numpy as np

from qiskit_nature.drivers import QMolecule
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.results import EigenstateResult

from ..second_quantized_property import LegacyDriverResult
from .types import ElectronicProperty

LOGGER = logging.getLogger(__file__)


class ParticleNumber(ElectronicProperty):
    """The ParticleNumber property.

    Note that this Property serves a two purposes:
        1. it stores the expected number of electrons (`self.num_particles`)
        2. it is used to evaluate the measured number of electrons via auxiliary operators.
           If this measured number does not match the expected number a warning will be logged.
    """

    def __init__(
        self,
        num_spin_orbitals: int,
        num_particles: Union[int, Tuple[int, int]],
        occupation: Optional[Union[np.ndarray, List[float]]] = None,
        occupation_beta: Optional[Union[np.ndarray, List[float]]] = None,
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
            self._occupation_alpha = [1.0 for _ in range(self._num_alpha)]
            self._occupation_alpha += [0.0] * (num_spin_orbitals // 2 - len(self._occupation_alpha))
            self._occupation_beta = [1.0 for _ in range(self._num_beta)]
            self._occupation_beta += [0.0] * (num_spin_orbitals // 2 - len(self._occupation_beta))
        elif occupation_beta is None:
            self._occupation_alpha = [o / 2.0 for o in occupation]
            self._occupation_beta = [o / 2.0 for o in occupation]
        else:
            self._occupation_alpha = occupation  # type: ignore
            self._occupation_beta = occupation_beta  # type: ignore

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
    def num_particles(self) -> Tuple[int, int]:
        """Returns the number of electrons."""
        return (self.num_alpha, self.num_beta)

    @property
    def occupation_alpha(self) -> np.ndarray:
        """Returns the occupation numbers of the alpha-spin orbitals.

        The occupation numbers may be float because in non-Hartree Fock methods you may encounter
        superpositions of determinants.
        """
        return np.asarray(self._occupation_alpha)

    @property
    def occupation_beta(self) -> np.ndarray:
        """Returns the occupation numbers of the beta-spin orbitals.

        The occupation numbers may be float because in non-Hartree Fock methods you may encounter
        superpositions of determinants.
        """
        return np.asarray(self._occupation_beta)

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
        cls._validate_input_type(result, QMolecule)

        qmol = cast(QMolecule, result)

        return cls(
            qmol.num_molecular_orbitals * 2,
            (qmol.num_alpha, qmol.num_beta),
            qmol.mo_occ,
            qmol.mo_occ_b,
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
        expected = self.num_alpha + self.num_beta
        result.num_particles = []

        if not isinstance(result.aux_operator_eigenvalues, list):
            aux_operator_eigenvalues = [result.aux_operator_eigenvalues]
        else:
            aux_operator_eigenvalues = result.aux_operator_eigenvalues  # type: ignore
        for aux_op_eigenvalues in aux_operator_eigenvalues:
            if aux_op_eigenvalues is None:
                continue

            if aux_op_eigenvalues[0] is not None:
                n_particles = aux_op_eigenvalues[0][0].real  # type: ignore
                result.num_particles.append(n_particles)

                if not np.isclose(n_particles, expected):
                    LOGGER.warning(
                        "The measured number of particles %s does NOT match the expected number of "
                        "particles %s!",
                        n_particles,
                        expected,
                    )
            else:
                result.num_particles.append(None)
