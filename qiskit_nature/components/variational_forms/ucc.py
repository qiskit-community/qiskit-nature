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
"""
TODO.
"""

from typing import Callable, List, Optional, Tuple, Union

import logging

from .evolved_operator_ansatz import EvolvedOperatorAnsatz

logger = logging.getLogger(__name__)


class UCC(EvolvedOperatorAnsatz):
    """The Unitary Coupled-Cluster Ansatz."""

    def __init__(self, qubit_op_converter: "QubitOpConverter",
                 num_particles: Optional[Tuple[int, int]] = None,
                 num_spin_orbitals: Optional[int] = None,
                 excitations: Optional[Union[str, int, List[int], List[Callable]]] = None):
        """

        Args:
            qubit_op_converter: the QubitOpConverter instance which takes care of mapping a
            :code:`~.SecondQuantizedOp` to a :code:`~.PauliSumOp` as well as performing all
            configured symmetry reductions on it.
            num_particles: the tuple of the number of alpha- and beta-spin particles.
            num_spin_orbitals: the number of spin orbitals.
            excitations: this can be any of the following:
                - a `str` which contains the types of excitations. Allowed characters are:
                    - `s` for singles
                    - `d` for doubles
                    - `t` for triples
                    - `q` for quadruples
                - a single, positive `int` which denotes the number of excitations (1 == `s`, etc.)
                - a list of positive integers
                - a list of `callable` objects which are used to generate the excitations. Each
                  `callable` must take the following arguments:
                      - `excitation_type`: the type of excitation to generate
                      - `num_particles`: the same as above
                      - `num_spin_orbitals`: the same as above
        """
        self._qubit_op_converter = qubit_op_converter
        self._num_particles = num_particles
        self._num_spin_orbitals = num_spin_orbitals
        self._excitations = excitations

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        pass

    def _build(self) -> None:
        pass
