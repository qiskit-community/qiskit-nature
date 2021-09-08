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

"""The Qiskit Nature VQE Quantum Program."""


from typing import List, Callable, Optional, Any, Dict, Union
import warnings
import numpy as np

from qiskit import QuantumCircuit
from qiskit.providers import Provider
from qiskit.providers.backend import Backend
from qiskit.algorithms import MinimumEigensolverResult
from qiskit.algorithms.optimizers import Optimizer
from qiskit.opflow import OperatorBase

from ..deprecation import warn_deprecated, DeprecatedType

from .vqe_runtime_client import VQEClient, VQERuntimeResult


class VQEProgram(VQEClient):
    """DEPRECATED. This class has been renamed to ``qiskit_nature.runtime.VQEClient``.

    This renaming reflects that this class is a client for a program executed in the cloud.
    """

    def __init__(
        self,
        ansatz: QuantumCircuit,
        optimizer: Optional[Union[Optimizer, Dict[str, Any]]] = None,
        initial_point: Optional[np.ndarray] = None,
        provider: Optional[Provider] = None,
        backend: Optional[Backend] = None,
        shots: int = 1024,
        measurement_error_mitigation: bool = False,
        callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None,
        store_intermediate: bool = False,
    ) -> None:
        warn_deprecated(
            version="0.2.1",
            old_type=DeprecatedType.CLASS,
            old_name="VQEProgram",
            new_name="VQEClient",
            additional_msg="from qiskit_nature.runtime",
        )
        super().__init__(
            ansatz,
            optimizer,
            initial_point,
            provider,
            backend,
            shots,
            measurement_error_mitigation,
            callback,
            store_intermediate,
        )

    def compute_minimum_eigenvalue(
        self, operator: OperatorBase, aux_operators: Optional[List[Optional[OperatorBase]]] = None
    ) -> MinimumEigensolverResult:
        result = super().compute_minimum_eigenvalue(operator, aux_operators)

        # convert to previous result type
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            vqe_result = VQEProgramResult()

        vqe_result.combine(result)
        return vqe_result


class VQEProgramResult(VQERuntimeResult):
    """DEPRECATED. The ``VQEProgramResult`` result object has been renamed to ``VQERuntimeResult``.

    This result objects contains the same as the VQEResult and additionally the history
    of the optimizer, containing information such as the function and parameter values per step.
    """

    def __init__(self) -> None:
        super().__init__()
        warn_deprecated(
            version="0.2.1",
            old_type=DeprecatedType.CLASS,
            old_name="VQEProgramResult",
            new_name="VQERuntimeResult",
            additional_msg="from qiskit_nature.runtime",
        )
