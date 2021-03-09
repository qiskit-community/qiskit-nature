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

from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import logging

from qiskit_nature import QiskitNatureError
from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from .evolved_operator_ansatz import EvolvedOperatorAnsatz
from .excitation_builder import ExcitationBuilder

logger = logging.getLogger(__name__)


# TODO: Added to pass mypy, need change to real class
class QubitOpConverter:
    """ The QubitOpConverter """

    def convert(self, op):
        """TODO."""
        pass


class UCC(EvolvedOperatorAnsatz):
    """The Unitary Coupled-Cluster Ansatz."""

    EXCITATION_TYPE = {
        's': 1,
        'd': 2,
        't': 3,
        'q': 4,
    }

    def __init__(self, qubit_op_converter: Optional[QubitOpConverter] = None,
                 num_particles: Optional[Tuple[int, int]] = None,
                 num_spin_orbitals: Optional[int] = None,
                 excitations: Optional[Union[str, int, List[int], Callable[
                     [Tuple[int, int], int], List[SecondQuantizedOp]]]] = None,
                 reps: int = 1):
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
                - a `callable` object which is used to generate the excitations. The `callable` must
                  take the following arguments:
                      - `num_particles`: the same as above
                      - `num_spin_orbitals`: the same as above
                  and must return a `List[SecondQuantizedOp]`.
            reps: The number of times to repeat the evolved operators.
        """
        self._qubit_op_converter = qubit_op_converter
        self._num_particles = num_particles
        self._num_spin_orbitals = num_spin_orbitals
        self._excitations = excitations
        # TODO: Added to pass lint, need change
        super().__init__([], reps=reps, evolution=None)

    @property
    def qubit_op_converter(self) -> QubitOpConverter:
        """The qubit operator converter."""
        return self._qubit_op_converter

    @qubit_op_converter.setter
    def qubit_op_converter(self, conv: QubitOpConverter) -> None:
        """Sets the qubit operator converter."""
        self._invalidate()
        self._qubit_op_converter = conv

    @property
    def num_spin_orbitals(self) -> int:
        """The number of spin orbitals."""
        return self._num_spin_orbitals

    @num_spin_orbitals.setter
    def num_spin_orbitals(self, n: int) -> None:
        """Sets the number of spin orbitals."""
        self._invalidate()
        self._num_spin_orbitals = n

    @property
    def num_particles(self) -> Tuple[int, int]:
        """The number of particles."""
        return self._num_particles

    @num_particles.setter
    def num_particles(self, n: Tuple[int, int]) -> None:
        """Sets the number of particles."""
        self._invalidate()
        self._num_particles = n

    @property
    def excitations(self) -> Union[str, int, List[int], List[Callable]]:
        """The excitations."""
        return self._excitations

    @excitations.setter
    def excitations(self, exc: Union[str, int, List[int], List[Callable]]) -> None:
        """Sets the excitations."""
        self._invalidate()
        self._excitations = exc

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        if self.num_spin_orbitals < 0:
            if raise_on_failure:
                raise ValueError('The number of spin orbitals cannot be smaller than 0.')
            return False

        if any(n < 0 for n in self.num_particles):
            if raise_on_failure:
                raise ValueError('The number of particles cannot be smaller than 0.')
            return False

        if self.excitations is None:
            if raise_on_failure:
                raise ValueError('The excitations cannot be `None`.')
            return False

        return True

    def _build(self) -> None:
        if self._data is not None:
            return

        excitation_ops = self.excitation_ops()

        converted_ops = []
        for op in excitation_ops:
            converted_ops.append(self.qubit_op_converter.convert(op))

        # we don't append to this property directly in order to only perform the checks done during
        # its setter once
        self.operators = converted_ops

        super()._build()

    def excitation_ops(self) -> List[SecondQuantizedOp]:
        """Parses the excitations and generates the list of operators.

        Raises:
            QiskitNatureError: if invalid excitations are specified.

        Returns:
            The list of generated excitation operators.
        """
        generators = []

        if isinstance(self.excitations, str):
            for exc in self.excitations:
                generators.append(partial(
                    ExcitationBuilder.build_excitation_ops,
                    num_excitation=self.EXCITATION_TYPE[exc]
                ))
        elif isinstance(self.excitations, int):
            generators.append(partial(
                ExcitationBuilder.build_excitation_ops,
                num_excitation=self.excitations
            ))
        elif isinstance(self.excitations, list):
            for exc in self.excitations:
                generators.append(partial(
                    ExcitationBuilder.build_excitation_ops,
                    num_excitation=exc
                ))
        elif isinstance(self.excitations, callable):
            generators.append(self.excitations)
        else:
            raise QiskitNatureError("Invalid excitation configuration: {}".format(self.excitations))

        excitation_ops = []
        for gen in generators:
            excitation_ops.extend(gen(
                num_spin_orbitals=self.num_spin_orbitals,
                num_particles=self.num_particles
            ))

        return excitation_ops
