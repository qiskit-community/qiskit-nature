# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Particle/Hole Transformation interface."""

from qiskit_nature.operators.second_quantization import SecondQuantizedSumOp

from .second_quantized_transformation import SecondQuantizedTransformation


class ParticleHoleTransformation(SecondQuantizedTransformation):
    """The Particle/Hole transformation."""

    def transform(self, second_q_op: SecondQuantizedSumOp) -> SecondQuantizedSumOp:
        """Transforms the given `SecondQuantizedSumOp` into the particle/hole view.

        Args:
            second_q_op: the `SecondQuantizedSumOp` to be transformed.

        Returns:
            A new `SecondQuantizedSumOp` instance.
        """
        # TODO
        raise NotImplementedError()
