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

"""Two-qubit YX - XY interaction gate."""

from typing import Optional

import numpy as np
from qiskit.circuit.gate import Gate
from qiskit.circuit.library import CXGate, HGate, RYGate, SdgGate, SGate
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister


class YXMinusXYInteractionGate(Gate):
    r"""A parametric 2-qubit :math:`Y \otimes X - X \otimes Y` interaction.

    These gates are used in the Givens rotation strategy for preparing
    Slater determinants and fermionic Gaussian states.

    **Circuit Symbol:**

    .. parsed-literal::

             ┌───────────┐
        q_0: ┤0          ├
             │  Ryxxy(θ) │
        q_1: ┤1          ├
             └───────────┘

    **Matrix Representation:**

    .. math::

        R_{YXXY}(\theta)\ q_0, q_1 = \exp(-i \frac{\theta}{2} (X{\otimes}Y - Y{\otimes}X)) =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & \cos(\theta) & \sin(\theta) & 0 \\
                0 & -\sin(\theta) & \cos(\theta) & 0 \\
                0 & 0 & 0 & 1
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In the above example we apply the gate
        on (q_0, q_1) which results in the :math:`X \otimes Y - Y \otimes X` tensor order.
        Instead, if we apply it on (q_1, q_0), the matrix will
        be :math:`Y \otimes X - X \otimes Y`:

        .. parsed-literal::

                 ┌───────────┐
            q_0: ┤1          ├
                 │  Ryxxy(θ) │
            q_1: ┤0          ├
                 └───────────┘

        .. math::

            R_{YXXY}(\theta)\ q_0, q_1 = \exp(-i \frac{\theta}{2} (Y{\otimes}X - X{\otimes}Y)) =
                \begin{pmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & \cos(\theta) & -\sin(\theta) & 0 \\
                    0 & \sin(\theta) & \cos(\theta) & 0 \\
                    0 & 0 & 0 & 1
                \end{pmatrix}

    **Examples:**

        .. math::

            R_{YXXY}(\theta = 0) = I

        .. math::

            R_{YXXY}(\theta = \frac{\pi}{4}) =
                \begin{pmatrix}
                    1  & 0 & 0 & 0 \\
                    0  & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}  & 0 \\
                    0 & -\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}  & 0 \\
                    0  & 0 & 0  & 1
                \end{pmatrix}

        .. math::

            R_{YXXY}(\theta = \frac{\pi}{2}) =
                \begin{pmatrix}
                    1  & 0 & 0 & 0 \\
                    0  & 0 & 1  & 0 \\
                    0 & -1 & 0  & 0 \\
                    0  & 0 & 0  & 1
                \end{pmatrix}
    """

    def __init__(self, theta: ParameterValueType, label: Optional[str] = None):
        """Create new YXMinusXYInteractionGate gate."""
        super().__init__("ryxxy", 2, [theta], label=label)

    def __array__(self, dtype=None):
        """Return a numpy array for the YXMinusXYInteractionGate gate."""
        (theta,) = self.params
        cos = np.cos(theta)
        sin = np.sin(theta)
        return np.array(
            [[1, 0, 0, 0], [0, cos, sin, 0], [0, -sin, cos, 0], [0, 0, 0, 1]],
            dtype=dtype,
        )

    def _define(self):
        """Decomposition of the gate."""
        (theta,) = self.params
        register = QuantumRegister(2, "q")
        circuit = QuantumCircuit(register, name=self.name)
        a, b = register
        rules = [
            (SGate(), [a], []),
            (SGate(), [b], []),
            (HGate(), [b], []),
            (CXGate(), [b, a], []),
            (RYGate(theta), [a], []),
            (RYGate(theta), [b], []),
            (CXGate(), [b, a], []),
            (HGate(), [b], []),
            (SdgGate(), [b], []),
            (SdgGate(), [a], []),
        ]
        for instr, qargs, cargs in rules:
            circuit.append(instr, qargs, cargs)

        self.definition = circuit

    def inverse(self):
        """Return inverse YXMinusXYInteractionGate gate."""
        (theta,) = self.params
        return YXMinusXYInteractionGate(-theta)
