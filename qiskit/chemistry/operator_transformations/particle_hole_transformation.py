# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Particle/Hole Transformation interface."""

from qiskit.chemistry.operators import SecondQuantizedOperator

from .second_quantized_transformation import SecondQuantizedTransformation


class ParticleHoleTransformation(SecondQuantizedTransformation):
    """The Particle/Hole transformation."""

    def transform(self, second_q_op: SecondQuantizedOperator) -> SecondQuantizedOperator:
        """Transforms the given `SecondQuantizedOperator` into the particle/hole view.

        Args:
            second_q_op: the `SecondQuantizedOperator` to be transformed.

        Returns:
            A new `SecondQuantizedOperator` instance.
        """
        # TODO
        raise NotImplementedError()
