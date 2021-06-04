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

from typing import cast, List, Optional, Tuple, Union

from qiskit_nature import QiskitNatureError
from qiskit_nature.drivers import QMolecule, WatsonHamiltonian
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.results import EigenstateResult

from ..property import Property


class ParticleNumber(Property):
    """The ParticleNumber property."""

    def __init__(
        self,
        num_spin_orbitals: int,
        num_particles: Union[int, Tuple[int, int]],
        occupation: Optional[List[float]] = None,
        occupation_beta: Optional[List[float]] = None,
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
            self._occupation_alpha += [0] * (num_spin_orbitals // 2 - len(self._occupation_alpha))
            self._occupation_beta = [1.0 for _ in range(self._num_beta)]
            self._occupation_beta += [0] * (num_spin_orbitals // 2 - len(self._occupation_beta))
        elif occupation_beta is None:
            self._occupation_alpha = [o / 2.0 for o in occupation]
            self._occupation_beta = [o / 2.0 for o in occupation]
        else:
            self._occupation_alpha = occupation
            self._occupation_beta = occupation_beta

    @classmethod
    def from_driver_result(cls, result: Union[QMolecule, WatsonHamiltonian]) -> "ParticleNumber":
        """Construct a ParticleNumber instance from a QMolecule.

        Args:
            result: the driver result from which to extract the raw data. For this property, a
                QMolecule is required!

        Returns:
            An instance of this property.

        Raises:
            QiskitNatureError: if a WatsonHamiltonian is provided.
        """
        if isinstance(result, WatsonHamiltonian):
            raise QiskitNatureError(
                "You cannot construct an AngularMomentum from a WatsonHamiltonian. Please provide "
                "a QMolecule object instead."
            )

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
        """Interprets an `qiskit_nature.result.EigenstateResult` in the context of this Property.

        This is currently a method stub which may be used in the future.

        Args:
            result: the result to add meaning to.
        """
        raise NotImplementedError()
