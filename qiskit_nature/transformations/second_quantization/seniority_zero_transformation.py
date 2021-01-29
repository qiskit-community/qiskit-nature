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

"""The Seniority-Zero Transformation interface."""

from qiskit_nature.operators.second_quantization.particle_op import ParticleOp

from .second_quantized_transformation import SecondQuantizedTransformation


class SeniorityZeroTransformation(SecondQuantizedTransformation):
    """The Seniority-Zero transformation."""

    def transform(self, second_q_op: ParticleOp) -> ParticleOp:
        """Transforms the given `ParticleOp` into a seniority-zero (i.e.
        restricted-spin) variant.

        Args:
            second_q_op: the `ParticleOp` to be transformed.

        Returns:
            A new `ParticleOp` instance.
        """
        # TODO
        raise NotImplementedError()
