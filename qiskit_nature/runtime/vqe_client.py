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

"""The Qiskit Nature VQE Runtime Client."""


from typing import Callable, Optional, Any, Dict, Union
import numpy as np

from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.providers import Provider
from qiskit.providers.backend import Backend
from qiskit.algorithms import (
    MinimumEigensolver,
    MinimumEigensolverResult,
    VQEResult,
    VariationalAlgorithm,
)
from qiskit.algorithms.optimizers import Optimizer, SPSA
from qiskit.opflow import OperatorBase, PauliSumOp
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature import ListOrDictType

from ..converters.second_quantization.utils import ListOrDict


class VQEClient(VariationalAlgorithm, MinimumEigensolver):
    """The Qiskit Nature VQE Runtime Client.

    This class is a client to call the VQE program in Qiskit Runtime."""

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
        """
        Args:
            ansatz: A parameterized circuit used as Ansatz for the wave function.
            optimizer: An optimizer or dictionary specifying a classical optimizer.
                If a dictionary, only SPSA and QN-SPSA are supported. The dictionary must contain a
                key ``name`` for the name of the optimizer and may contain additional keys for the
                settings. E.g. ``{'name': 'SPSA', 'maxiter': 100}``.
                Per default, SPSA is used.
            backend: The backend to run the circuits on.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` a random vector is used.
            provider: Provider that supports the runtime feature.
            shots: The number of shots to be used
            measurement_error_mitigation: Whether or not to use measurement error mitigation.
            callback: a callback that can access the intermediate data during the optimization.
                Four parameter values are passed to the callback as follows during each evaluation
                by the optimizer for its current set of parameters as it works towards the minimum.
                These are: the evaluation count, the optimizer parameters for the
                ansatz, the evaluated mean and the evaluated standard deviation.
            store_intermediate: Whether or not to store intermediate values of the optimization
                steps. Per default False.
        """
        if optimizer is None:
            optimizer = SPSA(maxiter=300)

        # define program name
        self._program_id = "vqe"

        # store settings
        self._provider = None
        self._ansatz = ansatz
        self._optimizer = None
        self._backend = backend
        self._initial_point = initial_point
        self._shots = shots
        self._measurement_error_mitigation = measurement_error_mitigation
        self._callback = callback
        self._store_intermediate = store_intermediate

        # use setter to check for valid inputs
        if provider is not None:
            self.provider = provider

        self.optimizer = optimizer

    @property
    def provider(self) -> Optional[Provider]:
        """Return the provider."""
        return self._provider

    @provider.setter
    def provider(self, provider: Provider) -> None:
        """Set the provider. Must be a provider that supports the runtime feature."""
        try:
            _ = hasattr(provider, "runtime")
        except QiskitError:
            # pylint: disable=raise-missing-from
            raise ValueError(f"The provider {provider} does not provide a runtime environment.")

        self._provider = provider

    @property
    def program_id(self) -> str:
        """Return the program ID."""
        return self._program_id

    @classmethod
    def supports_aux_operators(cls) -> bool:
        return True

    @property
    def ansatz(self) -> QuantumCircuit:
        """Return the ansatz."""
        return self._ansatz

    @ansatz.setter
    def ansatz(self, ansatz: QuantumCircuit) -> None:
        """Set the ansatz."""
        self._ansatz = ansatz

    @property
    def optimizer(self) -> Union[Optimizer, Dict[str, Any]]:
        """Return the dictionary describing the optimizer."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Union[Optimizer, Dict[str, Any]]) -> None:
        """Set the optimizer."""
        if isinstance(optimizer, Optimizer):
            self._optimizer = optimizer
        else:
            if "name" not in optimizer.keys():
                raise ValueError(
                    "The optimizer dictionary must contain a ``name`` key specifying the type "
                    "of the optimizer."
                )

            _validate_optimizer_settings(optimizer)

            self._optimizer = optimizer

    @property
    def backend(self) -> Optional[Backend]:
        """Returns the backend."""
        return self._backend

    @backend.setter
    def backend(self, backend) -> None:
        """Sets the backend."""
        self._backend = backend

    @property
    def initial_point(self) -> Optional[np.ndarray]:
        """Returns the initial point."""
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: Optional[np.ndarray]) -> None:
        """Sets the initial point."""
        self._initial_point = initial_point

    @property
    def shots(self) -> int:
        """Return the number of shots."""
        return self._shots

    @shots.setter
    def shots(self, shots: int) -> None:
        """Set the number of shots."""
        self._shots = shots

    @property
    def measurement_error_mitigation(self) -> bool:
        """Returns whether or not to use measurement error mitigation.

        Readout error mitigation is done using a complete measurement fitter with the
        ``self.shots`` number of shots and re-calibrations every 30 minutes.
        """
        return self._measurement_error_mitigation

    @measurement_error_mitigation.setter
    def measurement_error_mitigation(self, measurement_error_mitigation: bool) -> None:
        """Whether or not to use readout error mitigation."""
        self._measurement_error_mitigation = measurement_error_mitigation

    @property
    def store_intermediate(self) -> bool:
        """Returns whether or not to store intermediate information of the optimization."""
        return self._store_intermediate

    @store_intermediate.setter
    def store_intermediate(self, store: bool) -> None:
        """Whether or not to store intermediate information of the optimization."""
        self._store_intermediate = store

    @property
    def callback(self) -> Callable:
        """Returns the callback."""
        return self._callback

    @callback.setter
    def callback(self, callback: Callable) -> None:
        """Set the callback."""
        self._callback = callback

    def _wrap_vqe_callback(self) -> Optional[Callable]:
        """Wraps and returns the given callback to match the signature of the runtime callback."""

        def wrapped_callback(*args) -> None:
            _, data = args  # first element is the job id
            if isinstance(data, dict):
                return  # not expected params. skip
            iteration_count = data[0]
            params = data[1]
            mean = data[2]
            sigma = data[3]
            self._callback(iteration_count, params, mean, sigma)
            return

        # if callback is set, return wrapped callback, else return None
        if self._callback:
            return wrapped_callback
        else:
            return None

    def compute_minimum_eigenvalue(
        self, operator: OperatorBase, aux_operators: Optional[ListOrDictType[OperatorBase]] = None
    ) -> MinimumEigensolverResult:
        """Calls the VQE Runtime to approximate the ground state of the given operator.

        Args:
            operator: Qubit operator of the observable
            aux_operators: Optional list of auxiliary operators or dictionary with
                auxiliary operators as values and their names as keys to be evaluated with the
                (approximate) eigenstate of the minimum eigenvalue main result and their expectation
                values returned. For instance in chemistry these can be dipole operators, total
                particle count operators so we can get values for these at the ground state.

        Returns:
            MinimumEigensolverResult

        Raises:
            ValueError: If the backend has not yet been set.
            ValueError: If the provider has not yet been set.
            RuntimeError: If the job execution failed.

        """
        if self.backend is None:
            raise ValueError("The backend has not been set.")

        if self.provider is None:
            raise ValueError("The provider has not been set.")

        # try to convert the operators to a PauliSumOp, if it isn't already one
        operator = _convert_to_paulisumop(operator)
        wrapped_aux_operators = {
            str(aux_op_name_or_idx): _convert_to_paulisumop(aux_op)
            for aux_op_name_or_idx, aux_op in ListOrDict(aux_operators).items()
        }

        # combine the settings with the given operator to runtime inputs
        inputs = {
            "operator": operator,
            "aux_operators": wrapped_aux_operators,
            "ansatz": self.ansatz,
            "optimizer": self.optimizer,
            "initial_point": self.initial_point,
            "shots": self.shots,
            "measurement_error_mitigation": self.measurement_error_mitigation,
            "store_intermediate": self.store_intermediate,
        }

        # define runtime options
        options = {"backend_name": self.backend.name()}

        # send job to runtime and return result
        job = self.provider.runtime.run(
            program_id=self.program_id,
            inputs=inputs,
            options=options,
            callback=self._wrap_vqe_callback(),
        )
        # print job ID if something goes wrong
        try:
            result = job.result()
        except Exception as exc:
            raise RuntimeError(f"The job {job.job_id()} failed unexpectedly.") from exc

        # re-build result from serialized return value
        vqe_result = VQERuntimeResult()
        vqe_result.job_id = job.job_id()
        vqe_result.cost_function_evals = result.get("cost_function_evals", None)
        vqe_result.eigenstate = result.get("eigenstate", None)
        vqe_result.eigenvalue = result.get("eigenvalue", None)
        aux_op_eigenvalues = result.get("aux_operator_eigenvalues", None)
        if isinstance(aux_operators, dict) and aux_op_eigenvalues is not None:
            aux_op_eigenvalues = dict(
                zip(wrapped_aux_operators.keys(), aux_op_eigenvalues.values())
            )
            if not aux_op_eigenvalues:  # For consistency set to None for empty dict
                aux_op_eigenvalues = None
        vqe_result.aux_operator_eigenvalues = aux_op_eigenvalues
        vqe_result.optimal_parameters = result.get("optimal_parameters", None)
        vqe_result.optimal_point = result.get("optimal_point", None)
        vqe_result.optimal_value = result.get("optimal_value", None)
        vqe_result.optimizer_evals = result.get("optimizer_evals", None)
        vqe_result.optimizer_time = result.get("optimizer_time", None)
        vqe_result.optimizer_history = result.get("optimizer_history", None)

        return vqe_result


class VQERuntimeResult(VQEResult):
    """The VQEClient result object.

    This result objects contains the same as the VQEResult and additionally the history
    of the optimizer, containing information such as the function and parameter values per step.
    """

    def __init__(self) -> None:
        super().__init__()
        self._job_id = None  # type: str
        self._optimizer_history = None  # type: Dict[str, Any]

    @property
    def job_id(self) -> str:
        """The job ID associated with the VQE runtime job."""
        return self._job_id

    @job_id.setter
    def job_id(self, job_id: str) -> None:
        """Set the job ID associated with the VQE runtime job."""
        self._job_id = job_id

    @property
    def optimizer_history(self) -> Optional[Dict[str, Any]]:
        """The optimizer history."""
        return self._optimizer_history

    @optimizer_history.setter
    def optimizer_history(self, history: Dict[str, Any]) -> None:
        """Set the optimizer history."""
        self._optimizer_history = history


def _validate_optimizer_settings(settings):
    name = settings.get("name", None)
    if name not in ["SPSA", "QN-SPSA"]:
        raise NotImplementedError("Only SPSA and QN-SPSA are currently supported.")

    allowed_settings = [
        "name",
        "maxiter",
        "blocking",
        "allowed_increase",
        "trust_region",
        "learning_rate",
        "perturbation",
        "resamplings",
        "last_avg",
        "second_order",
        "hessian_delay",
        "regularization",
        "initial_hessian",
    ]

    if name == "QN-SPSA":
        allowed_settings.remove("trust_region")
        allowed_settings.remove("second_order")

    unsupported_args = set(settings.keys()) - set(allowed_settings)

    if len(unsupported_args) > 0:
        raise ValueError(
            f"The following settings are unsupported for the {name} optimizer: "
            f"{unsupported_args}"
        )


def _convert_to_paulisumop(operator):
    """Attempt to convert the operator to a PauliSumOp."""
    if isinstance(operator, PauliSumOp):
        return operator

    try:
        primitive = SparsePauliOp(operator.primitive)
        return PauliSumOp(primitive, operator.coeff)
    except Exception as exc:
        raise ValueError(
            f"Invalid type of the operator {type(operator)} "
            "must be PauliSumOp, or castable to one."
        ) from exc
