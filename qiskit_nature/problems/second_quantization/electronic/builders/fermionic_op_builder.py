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

""" Fermionic operator builder. """

from typing import List

import numpy as np

from qiskit_nature.deprecation import DeprecatedType, deprecate_function
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis
from qiskit_nature.properties.second_quantization.electronic.integrals import (
    ElectronicIntegrals,
    IntegralProperty,
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
)


@deprecate_function(
    "0.2.0",
    DeprecatedType.CLASS,
    "IntegralProperty",
    "from qiskit_nature.properties.second_quantization.electronic.integrals in combination with the"
    " more versatile ElectronicIntegrals containers",
)
def build_ferm_op_from_ints(
    one_body_integrals: np.ndarray, two_body_integrals: np.ndarray = None
) -> FermionicOp:
    """**DEPRECATED!**
    Builds a fermionic operator based on 1- and/or 2-body integrals. Integral values are used for
    the coefficients of the second-quantized Hamiltonian that is built. If integrals are stored
    in the '*chemist*' notation
             h2(i,j,k,l) --> adag_i adag_k a_l a_j
    they are required to be in block spin format and also have indices reordered as follows
    'ijkl->ljik'.
    There is another popular notation, the '*physicist*' notation
             h2(i,j,k,l) --> adag_i adag_j a_k a_l
    If you are using the '*physicist*' notation, you need to convert it to
    the '*chemist*' notation. E.g. h2=numpy.einsum('ikmj->ijkm', h2)
    The :class:`~qiskit_nature.drivers.QMolecule` class has
    :attr:`~qiskit_nature.drivers.QMolecule.one_body_integrals` and
    :attr:`~qiskit_nature.drivers.QMolecule.two_body_integrals` properties that
    can be directly supplied to the `h1` and `h2` parameters here respectively.

    Args:
        one_body_integrals (numpy.ndarray): One-body integrals stored in the chemist notation.
        two_body_integrals (numpy.ndarray): Two-body integrals stored in the chemist notation.

    Returns:
        FermionicOp: FermionicOp built from 1- and/or 2-body integrals.
    """
    integrals: List[ElectronicIntegrals] = []
    integrals.append(OneBodyElectronicIntegrals(ElectronicBasis.SO, one_body_integrals))
    if two_body_integrals is not None:
        integrals.append(TwoBodyElectronicIntegrals(ElectronicBasis.SO, two_body_integrals))

    prop = IntegralProperty("", integrals)

    fermionic_op = prop.second_q_ops()[0]

    return fermionic_op
