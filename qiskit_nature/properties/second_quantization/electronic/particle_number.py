# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The ParticleNumber property."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union, cast

import h5py
import numpy as np

from qiskit_nature import ListOrDictType, settings
from qiskit_nature.drivers import QMolecule
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.results import EigenstateResult

from ..second_quantized_property import LegacyDriverResult
from .types import ElectronicProperty

LOGGER = logging.getLogger(__name__)


class ParticleNumber(ElectronicProperty):
    """The ParticleNumber property.

    Note that this Property serves a two purposes:
        1. it stores the expected number of electrons (``self.num_particles``)
        2. it is used to evaluate the measured number of electrons via auxiliary operators.
           If this measured number does not match the expected number, it will be logged on the INFO
           level.
    """

    ABSOLUTE_TOLERANCE = 1e-05
    RELATIVE_TOLERANCE = 1e-02

    def __init__(
        self,
        num_spin_orbitals: int,
        num_particles: Union[int, Tuple[int, int]],
        occupation: Optional[Union[np.ndarray, List[float]]] = None,
        occupation_beta: Optional[Union[np.ndarray, List[float]]] = None,
        absolute_tolerance: float = ABSOLUTE_TOLERANCE,
        relative_tolerance: float = RELATIVE_TOLERANCE,
    ) -> None:
        """
        Args:
            num_spin_orbitals: the number of spin orbitals in the system.
            num_particles: the number of particles in the system. If this is a pair of integers, the
                first is the alpha and the second is the beta spin. If it is an int, the number is
                halved, with the remainder being added onto the alpha spin number.
            occupation: the occupation numbers. If ``occupation_beta`` is ``None``, these are the
                total occupation numbers, otherwise these are treated as the alpha-spin occupation.
            occupation_beta: the beta-spin occupation numbers.
            absolute_tolerance: the absolute tolerance used for checking whether the measured
                particle number matches the expected one.
            relative_tolerance: the relative tolerance used for checking whether the measured
                particle number matches the expected one.
        """
        super().__init__(self.__class__.__name__)
        self._num_spin_orbitals = num_spin_orbitals
        if isinstance(num_particles, int):
            self._num_alpha = num_particles // 2 + num_particles % 2
            self._num_beta = num_particles // 2
        else:
            self._num_alpha, self._num_beta = num_particles

        self._occupation_alpha: Union[np.ndarray, list[float]]
        self._occupation_beta: Union[np.ndarray, list[float]]
        if occupation is None:
            self._occupation_alpha = [1.0 for _ in range(self._num_alpha)]
            self._occupation_alpha += [0.0] * (num_spin_orbitals // 2 - len(self._occupation_alpha))
            self._occupation_beta = [1.0 for _ in range(self._num_beta)]
            self._occupation_beta += [0.0] * (num_spin_orbitals // 2 - len(self._occupation_beta))
        elif occupation_beta is None:
            self._occupation_alpha = [np.ceil(o / 2) for o in occupation]
            self._occupation_beta = [np.floor(o / 2) for o in occupation]
        else:
            self._occupation_alpha = occupation
            self._occupation_beta = occupation_beta

        self._absolute_tolerance = absolute_tolerance
        self._relative_tolerance = relative_tolerance

    @property
    def num_spin_orbitals(self) -> int:
        """Returns the num_spin_orbitals."""
        return self._num_spin_orbitals

    @num_spin_orbitals.setter
    def num_spin_orbitals(self, num_spin_orbitals: int) -> None:
        """Sets the number of spin orbitals."""
        self._num_spin_orbitals = num_spin_orbitals

    @property
    def num_alpha(self) -> int:
        """Returns the number of alpha electrons."""
        return self._num_alpha

    @num_alpha.setter
    def num_alpha(self, num_alpha: int) -> None:
        """Sets the number of alpha electrons."""
        self._num_alpha = num_alpha

    @property
    def num_beta(self) -> int:
        """Returns the number of beta electrons."""
        return self._num_beta

    @num_beta.setter
    def num_beta(self, num_beta: int) -> None:
        """Sets the number of beta electrons."""
        self._num_beta = num_beta

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

    @occupation_alpha.setter
    def occupation_alpha(self, occ_alpha: Union[np.ndarray, List[float]]) -> None:
        """Sets the occupation numbers of the alpha-spin orbitals."""
        self._occupation_alpha = occ_alpha

    @property
    def occupation_beta(self) -> np.ndarray:
        """Returns the occupation numbers of the beta-spin orbitals.

        The occupation numbers may be float because in non-Hartree Fock methods you may encounter
        superpositions of determinants.
        """
        return np.asarray(self._occupation_beta)

    @occupation_beta.setter
    def occupation_beta(self, occ_beta: Union[np.ndarray, List[float]]) -> None:
        """Sets the occupation numbers of the beta-spin orbitals."""
        self._occupation_beta = occ_beta

    @property
    def absolute_tolerance(self) -> float:
        """Returns the absolute tolerance."""
        return self._absolute_tolerance

    @absolute_tolerance.setter
    def absolute_tolerance(self, absolute_tolerance: float) -> None:
        """Sets the absolute tolerance."""
        self._absolute_tolerance = absolute_tolerance

    @property
    def relative_tolerance(self) -> float:
        """Returns the relative tolerance."""
        return self._relative_tolerance

    @relative_tolerance.setter
    def relative_tolerance(self, relative_tolerance: float) -> None:
        """Sets the relative tolerance."""
        self._relative_tolerance = relative_tolerance

    def __str__(self) -> str:
        string = [super().__str__() + ":"]
        string += [f"\t{self._num_spin_orbitals} SOs"]
        string += [f"\t{self._num_alpha} alpha electrons"]
        string += [f"\t\torbital occupation: {self.occupation_alpha}"]
        string += [f"\t{self._num_beta} beta electrons"]
        string += [f"\t\torbital occupation: {self.occupation_beta}"]
        return "\n".join(string)

    def to_hdf5(self, parent: h5py.Group) -> None:
        """Stores this instance in an HDF5 group inside of the provided parent group.

        See also :func:`~qiskit_nature.hdf5.HDF5Storable.to_hdf5` for more details.

        Args:
            parent: the parent HDF5 group.
        """
        super().to_hdf5(parent)
        group = parent.require_group(self.name)

        group.attrs["num_spin_orbitals"] = self._num_spin_orbitals
        group.attrs["num_alpha"] = self._num_alpha
        group.attrs["num_beta"] = self._num_beta
        group.attrs["absolute_tolerance"] = self._absolute_tolerance
        group.attrs["relative_tolerance"] = self._relative_tolerance

        group.create_dataset("occupation_alpha", data=self.occupation_alpha)
        group.create_dataset("occupation_beta", data=self.occupation_beta)

    @staticmethod
    def from_hdf5(h5py_group: h5py.Group) -> ParticleNumber:
        """Constructs a new instance from the data stored in the provided HDF5 group.

        See also :func:`~qiskit_nature.hdf5.HDF5Storable.from_hdf5` for more details.

        Args:
            h5py_group: the HDF5 group from which to load the data.

        Returns:
            A new instance of this class.
        """
        return ParticleNumber(
            h5py_group.attrs["num_spin_orbitals"],
            (h5py_group.attrs["num_alpha"], h5py_group.attrs["num_beta"]),
            h5py_group["occupation_alpha"][...],
            h5py_group["occupation_beta"][...],
            h5py_group.attrs["absolute_tolerance"],
            h5py_group.attrs["relative_tolerance"],
        )

    @classmethod
    def from_legacy_driver_result(cls, result: LegacyDriverResult) -> ParticleNumber:
        """Construct a ParticleNumber instance from a :class:`~qiskit_nature.drivers.QMolecule`.

        Args:
            result: the driver result from which to extract the raw data. For this property, a
                :class:`~qiskit_nature.drivers.QMolecule` is required!

        Returns:
            An instance of this property.

        Raises:
            QiskitNatureError: if a :class:`~qiskit_nature.drivers.WatsonHamiltonian` is provided.
        """
        cls._validate_input_type(result, QMolecule)

        qmol = cast(QMolecule, result)

        return cls(
            qmol.num_molecular_orbitals * 2,
            (qmol.num_alpha, qmol.num_beta),
            qmol.mo_occ,
            qmol.mo_occ_b,
        )

    def second_q_ops(self) -> ListOrDictType[FermionicOp]:
        """Returns the second quantized particle number operator.

        The actual return-type is determined by `qiskit_nature.settings.dict_aux_operators`.

        Returns:
            A `list` or `dict` of `FermionicOp` objects.
        """
        op = FermionicOp(
            [(f"N_{o}", 1.0) for o in range(self._num_spin_orbitals)],
            register_length=self._num_spin_orbitals,
            display_format="sparse",
        )

        if not settings.dict_aux_operators:
            return [op]

        return {self.name: op}

    def interpret(self, result: EigenstateResult) -> None:
        """Interprets an :class:`~qiskit_nature.results.EigenstateResult` in this property's context.

        Args:
            result: the result to add meaning to.
        """
        expected = self.num_alpha + self.num_beta
        result.num_particles = []

        if not isinstance(result.aux_operator_eigenvalues, list):
            aux_operator_eigenvalues = [result.aux_operator_eigenvalues]
        else:
            aux_operator_eigenvalues = result.aux_operator_eigenvalues
        for aux_op_eigenvalues in aux_operator_eigenvalues:
            if aux_op_eigenvalues is None:
                continue

            _key = self.name if isinstance(aux_op_eigenvalues, dict) else 0

            if aux_op_eigenvalues[_key] is not None:
                n_particles = aux_op_eigenvalues[_key][0].real
                result.num_particles.append(n_particles)

                if not np.isclose(
                    n_particles,
                    expected,
                    rtol=self._relative_tolerance,
                    atol=self._absolute_tolerance,
                ):
                    LOGGER.info(
                        "The measured number of particles %s does NOT match the expected number of "
                        "particles %s!",
                        n_particles,
                        expected,
                    )
            else:
                result.num_particles.append(None)
