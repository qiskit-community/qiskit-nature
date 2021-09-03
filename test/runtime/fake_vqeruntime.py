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

"""Fake runtime provider and VQE runtime."""

from typing import Dict, Any
import numpy as np
from qiskit.algorithms.optimizers import Optimizer
from qiskit.circuit import QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.providers import Provider
from qiskit_nature.runtime import VQERuntimeResult


class FakeVQEJob:
    """A fake job for unit tests."""

    def result(self) -> Dict[str, Any]:
        """Return a VQE result."""
        result = VQERuntimeResult()
        serialized_result = {
            "optimizer_evals": result.optimizer_evals,
            "optimizer_time": result.optimizer_time,
            "optimal_value": result.optimal_value,
            "optimal_point": result.optimal_point,
            "optimal_parameters": result.optimal_parameters,
            "cost_function_evals": result.cost_function_evals,
            "eigenstate": result.eigenstate,
            "eigenvalue": result.eigenvalue,
            "aux_operator_eigenvalues": result.aux_operator_eigenvalues,
            "optimizer_history": result.optimizer_history,
        }
        return serialized_result

    def job_id(self) -> str:
        """Return a fake job ID."""
        return "c2985khdm6upobbnmll0"


class FakeVQERuntime:
    """A fake VQE runtime for unit tests."""

    def run(self, program_id, inputs, options, callback=None):
        """Run the fake program. Checks the input types."""

        if program_id != "vqe":
            raise ValueError("program_id is not vqe.")

        allowed_inputs = {
            "operator": PauliSumOp,
            "aux_operators": (list, type(None)),
            "ansatz": QuantumCircuit,
            "initial_point": (np.ndarray, str),
            "optimizer": (Optimizer, dict),
            "shots": int,
            "measurement_error_mitigation": bool,
            "store_intermediate": bool,
        }
        for arg, value in inputs.items():
            if not isinstance(value, allowed_inputs[arg]):
                raise ValueError(f"{arg} does not have the right type: {allowed_inputs[arg]}")

        allowed_options = {"backend_name": str}
        for arg, value in options.items():
            if not isinstance(value, allowed_options[arg]):
                raise ValueError(f"{arg} does not have the right type: {allowed_inputs[arg]}")

        if callback is not None:
            try:
                fake_job_id = "c2985khdm6upobbnmll0"
                fake_data = [3, np.arange(10), 1.3]
                _ = callback(fake_job_id, fake_data)
            except Exception as exc:
                raise ValueError("Callback failed") from exc

        return FakeVQEJob()


class FakeRuntimeProvider(Provider):
    """A fake runtime provider for unit tests."""

    def has_service(self, service):
        """Check if a service is available."""
        if service == "runtime":
            return True
        return False

    @property
    def runtime(self) -> FakeVQERuntime:
        """Return the runtime."""
        return FakeVQERuntime()
