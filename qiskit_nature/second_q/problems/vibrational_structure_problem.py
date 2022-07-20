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
"""The Vibrational Structure Problem class."""

from __future__ import annotations

from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from qiskit.algorithms import EigensolverResult, MinimumEigensolverResult
from qiskit.opflow import PauliSumOp

from qiskit_nature.second_q.operators import SecondQuantizedOp
from qiskit_nature.second_q.mappers import QubitConverter
from qiskit_nature.second_q.properties import (
    OccupiedModals,
    VibrationalEnergy,
    VibrationalStructureDriverResult,
)
from qiskit_nature.second_q.properties.property import Interpretable
from qiskit_nature.second_q.properties.bases import VibrationalBasis, HarmonicBasis

from .builders.vibrational_hopping_ops_builder import _build_qeom_hopping_ops
from .base_problem import BaseProblem

from .vibrational_structure_result import VibrationalStructureResult
from .eigenstate_result import EigenstateResult


class VibrationalStructureProblem(BaseProblem):
    """Vibrational Structure Problem"""

    def __init__(
        self,
        hamiltonian: VibrationalEnergy,
        num_modes: int,
        num_modals: Optional[Union[int, List[int]]] = None,
        truncation_order: Optional[int] = None,
    ):
        """
        Args:
            bosonic_driver: a bosonic driver encoding the molecule information.
            num_modals: the number of modals per mode.
            truncation_order: order at which an n-body expansion is truncated
            transformers: a list of transformations to be applied to the driver result.
        """
        super().__init__(hamiltonian)
        self.num_modes = num_modes
        self.num_modals = num_modals
        self.truncation_order = truncation_order
        self.basis: Optional[VibrationalBasis] = None

    def second_q_ops(self) -> Tuple[SecondQuantizedOp, Optional[Dict[str, SecondQuantizedOp]]]:
        """Returns the second quantized operators created based on the driver and transformations.

        If the arguments are returned as a `list`, the operators are in the following order: the
        Vibrational Hamiltonian operator, occupied modal operators for each mode.

        The actual return-type is determined by `qiskit_nature.settings.dict_aux_operators`.

        Returns:
            A `list` or `dict` of `SecondQuantizedOp` objects.
        """
        num_modes = self.num_modes
        if isinstance(self.num_modals, int):
            num_modals = [self.num_modals] * num_modes
        else:
            num_modals = self.num_modals

        # TODO: expose this as an argument in __init__
        basis = HarmonicBasis(num_modals)
        self.basis = basis
        self.hamiltonian.basis = basis  # type: ignore[union-attr]
        self.hamiltonian.truncation_order = self.truncation_order  # type: ignore[union-attr]

        aux_ops: dict[str, SecondQuantizedOp] = {}
        for prop in self.properties.values():
            if hasattr(prop, "truncation_order"):
                prop.truncation_order = self.truncation_order  # type: ignore[attr-defined]
            prop.basis = basis  # type: ignore[attr-defined]
            aux_ops.update(prop.second_q_ops())

        # TODO: refactor once Hamiltonian base-class exposes single second_q_op() generator
        return list(self.hamiltonian.second_q_ops().values())[0], aux_ops

    def hopping_qeom_ops(
        self,
        qubit_converter: QubitConverter,
        excitations: Union[
            str,
            int,
            List[int],
            Callable[[int, Tuple[int, int]], List[Tuple[Tuple[int, ...], Tuple[int, ...]]]],
        ] = "sd",
    ) -> Tuple[
        Dict[str, PauliSumOp],
        Dict[str, List[bool]],
        Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]],
    ]:
        """Generates the hopping operators and their commutativity information for the specified set
        of excitations.

        Args:
            qubit_converter: the `QubitConverter` to use for mapping and symmetry reduction. The
                             Z2 symmetries stored in this instance are the basis for the
                             commutativity information returned by this method.
            excitations: the types of excitations to consider. The simple cases for this input are

                :`str`: containing any of the following characters: `s`, `d`, `t` or `q`.
                :`int`: a single, positive integer denoting the excitation type (1 == `s`, etc.).
                :`List[int]`: a list of positive integers.
                :`Callable`: a function which is used to generate the excitations.
                    For more details on how to write such a function refer to the default method,
                    :meth:`generate_vibrational_excitations`.

        Returns:
            A tuple containing the hopping operators, the types of commutativities and the
            excitation indices.
        """

        if isinstance(self.num_modals, int):
            num_modals = [self.num_modals] * self.num_modes
        else:
            num_modals = self.num_modals

        return _build_qeom_hopping_ops(num_modals, qubit_converter, excitations)

    def interpret(
        self,
        raw_result: Union[EigenstateResult, EigensolverResult, MinimumEigensolverResult],
    ) -> VibrationalStructureResult:
        """Interprets an EigenstateResult in the context of this transformation.
        Args:
            raw_result: an eigenstate result object.
        Returns:
            An vibrational structure result.
        """
        eigenstate_result = None
        if isinstance(raw_result, EigenstateResult):
            eigenstate_result = raw_result
        elif isinstance(raw_result, EigensolverResult):
            eigenstate_result = EigenstateResult()
            eigenstate_result.raw_result = raw_result
            eigenstate_result.eigenenergies = raw_result.eigenvalues
            eigenstate_result.eigenstates = raw_result.eigenstates
            eigenstate_result.aux_operator_eigenvalues = raw_result.aux_operator_eigenvalues
        elif isinstance(raw_result, MinimumEigensolverResult):
            eigenstate_result = EigenstateResult()
            eigenstate_result.raw_result = raw_result
            eigenstate_result.eigenenergies = np.asarray([raw_result.eigenvalue])
            eigenstate_result.eigenstates = [raw_result.eigenstate]
            eigenstate_result.aux_operator_eigenvalues = [raw_result.aux_operator_eigenvalues]
        result = VibrationalStructureResult()
        result.combine(eigenstate_result)
        self.hamiltonian.interpret(result)  # type: ignore[union-attr]
        for prop in self.properties.values():
            if isinstance(prop, Interpretable):
                prop.interpret(result)
        result.computed_vibrational_energies = eigenstate_result.eigenenergies
        return result

    def get_default_filter_criterion(
        self,
    ) -> Optional[Callable[[Union[List, np.ndarray], float, Optional[List[float]]], bool]]:
        """Returns a default filter criterion method to filter the eigenvalues computed by the
        eigen solver. For more information see also
        aqua.algorithms.eigen_solvers.NumPyEigensolver.filter_criterion.
        In the fermionic case the default filter ensures that the number of particles is being
        preserved.
        """

        # pylint: disable=unused-argument
        def filter_criterion(self, eigenstate, eigenvalue, aux_values):
            # the first num_modes aux_value is the evaluated number of particles for the given mode
            for mode in range(self.num_modes):
                _key = str(mode) if isinstance(aux_values, dict) else mode
                if aux_values is None or not np.isclose(aux_values[_key][0], 1):
                    return False
            return True

        return partial(filter_criterion, self)

    @classmethod
    def from_legacy_driver_result(
        cls, result: VibrationalStructureDriverResult
    ) -> VibrationalStructureProblem:
        """Converts a :class:`~qiskit_nature.second_q.drivers.WatsonHamiltonian` into an
        ``VibrationalStructureDriverResult``.

        Args:
            result: the :class:`~qiskit_nature.second_q.drivers.WatsonHamiltonian` to convert.

        Returns:
            An instance of this property.

        Raises:
            QiskitNatureError: if a :class:`~qiskit_nature.second_q.drivers.QMolecule` is provided.
        """
        vib_energy = result.get_property(VibrationalEnergy)

        ret = cls(vib_energy, result.num_modes)

        ret.properties["OccupiedModals"] = OccupiedModals()

        return ret
