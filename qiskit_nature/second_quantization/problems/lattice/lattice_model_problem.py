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
"""The Lattice Model Problem class."""

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from qiskit.algorithms import EigensolverResult, MinimumEigensolverResult
from qiskit.opflow import PauliSumOp
from qiskit_nature import ListOrDictType, settings
from qiskit_nature.second_quantization.operators import QubitConverter
from qiskit_nature.second_quantization.operators import SecondQuantizedOp
from qiskit_nature.second_quantization.results import EigenstateResult, LatticeModelResult

from ..base_problem import BaseProblem
from .models.lattice_model import LatticeModel


class LatticeModelProblem(BaseProblem):
    """Lattice Model Problem class to create second quantized operators from a lattice model."""

    def __init__(self, lattice_model: LatticeModel) -> None:
        """
        Args:
            lattice_model: A lattice model class to create second quantized operators.
        """
        super().__init__(main_property_name="LatticeEnergy")
        self._lattice_model = lattice_model

    def second_q_ops(self) -> ListOrDictType[SecondQuantizedOp]:
        """Returns the second quantized operators created based on the lattice models.

        Returns:
            A ``list`` or ``dict`` of
            :class:`~qiskit_nature.second_quantization.operators.SecondQuantizedOp`
        """
        second_q_ops: ListOrDictType[SecondQuantizedOp] = self._lattice_model.second_q_ops()
        if settings.dict_aux_operators:
            second_q_ops = {self._main_property_name: second_q_ops}
        else:
            second_q_ops = [second_q_ops]

        return second_q_ops

    def interpret(
        self,
        raw_result: Union[EigenstateResult, EigensolverResult, MinimumEigensolverResult],
    ) -> LatticeModelResult:
        """Interprets a raw result in the context of this transformation.

        Args:
            raw_result: a raw result to be interpreted

        Returns:
            A lattice model result.
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
        result = LatticeModelResult()
        result.combine(eigenstate_result)
        result.computed_lattice_energies = eigenstate_result.eigenenergies
        return result

    def hopping_qeom_ops(
        self,
        qubit_converter: QubitConverter,
        excitations: Union[
            str,
            int,
            List[int],
            Callable[[int, Tuple[int, int]], List[Tuple[Tuple[int, ...], Tuple[int, ...]]]],
        ] = "sd",
    ) -> Optional[
        Tuple[
            Dict[str, PauliSumOp],
            Dict[str, List[bool]],
            Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]],
        ]
    ]:
        """Generates the hopping operators and their commutativity information
        for the specified set of excitations. Raises `NotImplementedError` for the
        `LatticeProblemModel` class, currently.

        Args:
            qubit_converter: the `QubitConverter` to use for mapping and symmetry reduction. The
                             Z2 symmetries stored in this instance are the basis for the
                             commutativity information returned by this method.
            excitations: the types of excitations to consider. The simple cases for this input are

                :`str`: containing any of the following characters: `s`, `d`, `t` or `q`.
                :`int`: a single, positive integer denoting the excitation type (1 == `s`, etc.).
                :`List[int]`: a list of positive integers.
                :`Callable`: a function which is used to generate the excitations.
                    For more details on how to write such a function refer to one of the default
                    methods, :meth:`generate_fermionic_excitations` or
                    :meth:`generate_vibrational_excitations`.

        Raises:
            Currently, this function is not implemented in the `LatticeProblemModel` class and
            always raises `NotImplementedError`.
        """
        raise NotImplementedError(
            "Currently, it's not implemented in the `LatticeProblemModel` class."
        )

    def get_default_filter_criterion(
        self,
    ) -> Optional[Callable[[Union[List, np.ndarray], float, Optional[List[float]]], bool]]:
        return None
