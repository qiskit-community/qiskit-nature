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

"""The Magnetization property."""

from __future__ import annotations

from typing import cast, TYPE_CHECKING

import h5py

from qiskit_nature import ListOrDictType, settings
from qiskit_nature.deprecation import deprecate_method
from qiskit_nature.second_q._qmolecule import QMolecule
from qiskit_nature.second_q.operators import FermionicOp


from .second_quantized_property import LegacyDriverResult
from .electronic_types import ElectronicProperty

if TYPE_CHECKING:
    from qiskit_nature.second_q.problems import EigenstateResult


class Magnetization(ElectronicProperty):
    """The Magnetization property."""

    def __init__(self, num_spin_orbitals: int) -> None:
        """
        Args:

            num_spin_orbitals: the number of spin orbitals in the system.
        """
        super().__init__(self.__class__.__name__)
        self._num_spin_orbitals = num_spin_orbitals

    @property
    def num_spin_orbitals(self) -> int:
        """Returns the number of spin orbitals."""
        return self._num_spin_orbitals

    @num_spin_orbitals.setter
    def num_spin_orbitals(self, num_spin_orbitals: int) -> None:
        """Sets the number of spin orbitals."""
        self._num_spin_orbitals = num_spin_orbitals

    def __str__(self) -> str:
        string = [super().__str__() + ":"]
        string += [f"\t{self._num_spin_orbitals} SOs"]
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

    @staticmethod
    def from_hdf5(h5py_group: h5py.Group) -> Magnetization:
        """Constructs a new instance from the data stored in the provided HDF5 group.

        See also :func:`~qiskit_nature.hdf5.HDF5Storable.from_hdf5` for more details.

        Args:
            h5py_group: the HDF5 group from which to load the data.

        Returns:
            A new instance of this class.
        """
        return Magnetization(int(h5py_group.attrs["num_spin_orbitals"]))

    @classmethod
    @deprecate_method("0.4.0")
    def from_legacy_driver_result(cls, result: LegacyDriverResult) -> Magnetization:
        """Construct a Magnetization instance from a :class:`~qiskit_nature.second_q.drivers.QMolecule`.

        Args:
            result: the driver result from which to extract the raw data. For this property, a
                :class:`~qiskit_nature.second_q.drivers.QMolecule` is required!

        Returns:
            An instance of this property.

        Raises:
            QiskitNatureError: if a :class:`~qiskit_nature.second_q.drivers.WatsonHamiltonian`
            is provided.
        """
        cls._validate_input_type(result, QMolecule)

        qmol = cast(QMolecule, result)

        return cls(
            qmol.num_molecular_orbitals * 2,
        )

    def second_q_ops(self) -> ListOrDictType[FermionicOp]:
        """Returns the second quantized magnetization operator.

        The actual return-type is determined by `qiskit_nature.settings.dict_aux_operators`.

        Returns:
            A `list` or `dict` of `SecondQuantizedOp` objects.
        """
        op = FermionicOp(
            [
                (f"N_{o}", 0.5 if o < self._num_spin_orbitals // 2 else -0.5)
                for o in range(self._num_spin_orbitals)
            ],
            register_length=self._num_spin_orbitals,
            display_format="sparse",
        )

        if not settings.dict_aux_operators:
            return [op]

        return {self.name: op}

    def interpret(self, result: "EigenstateResult") -> None:
        """Interprets an :class:`~qiskit_nature.second_q.problems.EigenstateResult` in this property's context.

        Args:
            result: the result to add meaning to.
        """
        result.magnetization = []

        if not isinstance(result.aux_operator_eigenvalues, list):
            aux_operator_eigenvalues = [result.aux_operator_eigenvalues]
        else:
            aux_operator_eigenvalues = result.aux_operator_eigenvalues
        for aux_op_eigenvalues in aux_operator_eigenvalues:
            if aux_op_eigenvalues is None:
                continue

            _key = self.name if isinstance(aux_op_eigenvalues, dict) else 2

            if aux_op_eigenvalues[_key] is not None:
                result.magnetization.append(aux_op_eigenvalues[_key][0].real)
            else:
                result.magnetization.append(None)
