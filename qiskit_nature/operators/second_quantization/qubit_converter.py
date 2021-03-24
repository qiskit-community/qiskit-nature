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

"""A converter from Second-Quantized to Qubit Operators."""
import copy
import logging
from typing import cast, List, Optional, Tuple, Union

import numpy as np

from qiskit.opflow import PauliSumOp
from qiskit.opflow.converters import TwoQubitReduction
from qiskit.opflow.primitive_ops import Z2Symmetries

from qiskit_nature import QiskitNatureError
from qiskit_nature.mappers.second_quantization import QubitMapper

from . import SecondQuantizedOp

logger = logging.getLogger(__name__)


class QubitConverter:
    """A converter from Second-Quantized to Qubit Operators."""

    def __init__(self,
                 mapper: QubitMapper,
                 two_qubit_reduction: bool = False,
                 z2symmetry_reduction: Optional[List[int]] = None):

        self._mapper = mapper
        self._two_qubit_reduction = two_qubit_reduction
        self._z2symmetry_reduction = None
        self.z2symmetry_reduction = z2symmetry_reduction  # Setter does validation

        self._conversion_done = False
        self._did_two_qubit_reduction = False
        self._num_particles = None
        self._z2symmetries = self._no_symmetries

    def _invalidate(self):
        self._conversion_done = False
        self._did_two_qubit_reduction = False
        self._num_particles = None
        self._z2symmetries = self._no_symmetries

    def _set_valid(self, did_two_qubit_reduction: bool,
                   num_particles: Optional[Tuple[int, int]],
                   z2symmetries: Z2Symmetries):
        self._did_two_qubit_reduction = did_two_qubit_reduction
        self._num_particles = num_particles
        self._z2symmetries = z2symmetries
        self._conversion_done = True

    def _check_valid(self, converting: bool = False):
        if not self._conversion_done:
            if not converting:
                raise QiskitNatureError("Properties are valid only after a convert() has "
                                        "been called successfully. "
                                        "Conversion state is reset if any settings are updated.")
            else:
                raise QiskitNatureError("convert_more() can only be used after a convert() has "
                                        "been called successfully as long as no settings "
                                        "were updated since that call.")

    @property
    def _no_symmetries(self) -> Z2Symmetries:
        return Z2Symmetries([], [], [], None)

    @property
    def mapper(self) -> QubitMapper:
        """Get mapper"""
        return self._mapper

    @mapper.setter
    def mapper(self, value: QubitMapper) -> None:
        """Set mapper"""
        self._mapper = value
        self._invalidate()

    @property
    def two_qubit_reduction(self) -> bool:
        """Get two_qubit_reduction"""
        return self._two_qubit_reduction

    @two_qubit_reduction.setter
    def two_qubit_reduction(self, value: bool) -> None:
        """Set two_qubit_reduction"""
        self._two_qubit_reduction = value
        self._invalidate()

    @property
    def z2symmetry_reduction(self) -> Optional[List[int]]:
        """Get z2symmetry_reduction"""
        return self._z2symmetry_reduction

    @z2symmetry_reduction.setter
    def z2symmetry_reduction(self, z2symmetry_reduction: Optional[List[int]]) -> None:
        """Set z2symmetry_reduction"""
        if z2symmetry_reduction is not None:
            if not np.all(np.isin(z2symmetry_reduction, [-1, 1])):
                raise ValueError('z2symmetry_reduction tapering values list must '
                                 'contain -1\'s and/or 1\'s only but was {}'.
                                 format(z2symmetry_reduction))

        self._z2symmetry_reduction = z2symmetry_reduction
        self._invalidate()

    @property
    def did_two_qubit_reduction(self) -> bool:
        """Get two qubit reduction as used from conversion

        Raises:
            QiskitNatureError: If property is accessed before a successful conversion
        """
        self._check_valid()
        return self._did_two_qubit_reduction

    @property
    def num_particles(self) -> Optional[Tuple[int, int]]:
        """Get the number of particles as supplied to conversion

        Raises:
            QiskitNatureError: If property is accessed before a successful conversion
        """
        self._check_valid()
        return self._num_particles

    @property
    def z2_symmetries(self) -> Z2Symmetries:
        """Get z2_symmetries as detected from conversion. This may indicate no symmetries
        if none were detected.

        Raises:
            QiskitNatureError: If property is accessed before a successful conversion
        """
        self._check_valid()
        return copy.deepcopy(self._z2symmetries)

    def convert(self, second_q_ops: Union[SecondQuantizedOp, List[SecondQuantizedOp]],
                num_particles: Optional[Tuple[int, int]] = None,
                z2symmetry_reduction: Optional[List[int]] = None,
                ) -> Union[PauliSumOp, List[Optional[PauliSumOp]]]:
        """
        Maps the given list of second quantized operators to qubit operators. Also it will
        carry out z2 symmetry reduction on the qubit operators if z2symmetry_reduction has
        been specified. For convenience a single operator may be passed, without wrapping
        it in a List, and in which case a single qubit operator will likewise be returned.

        Args:
            second_q_ops: A second quantized operator, or list thereof to be converted.
            num_particles: Needed for two qubit reduction to determine correct sector. If
                not supplied, even if two_qubit_reduction is possible, it will not be done.
            z2symmetry_reduction: Optional z2symmetry reduction, the sector of the symmetry

        Returns:
            A qubit operator or list of thereof the same length as the second_q_ops list. The first
            operator in the second_q_ops list is treated as the main operator and others must
            commute with its symmetry, when symmetry reduction is being done. If it does not
            then the position in the output list will be set to `None` to preserve the order.
        """
        self._invalidate()  # Invalidate state before conversion is attempted

        # To allow a single operator to be converted, but use the same logic that does the
        # actual conversions, we make a single entry list of it here and unwrap to return.
        wrapped = False
        if isinstance(second_q_ops, SecondQuantizedOp):
            second_q_ops = [second_q_ops]
            wrapped = True

        if z2symmetry_reduction is not None:
            self.z2symmetry_reduction = z2symmetry_reduction

        qubit_ops = self._map_to_qubits(second_q_ops)
        qubit_ops_reduced, did_2_q = self._two_qubit_reduce(qubit_ops, num_particles)
        qubit_ops_tapered, z2symmetries = self._find_symmetry_reduce(qubit_ops_reduced)

        # Set the state (i.e. whether we did two qubit reductions and the Z2Symmetries)
        # now that a conversion was successful
        self._set_valid(did_2_q, num_particles, z2symmetries)

        if wrapped:
            qubit_ops_tapered = qubit_ops_tapered[0]

        return qubit_ops_tapered

    def convert_more(self, second_q_ops: Union[SecondQuantizedOp, List[SecondQuantizedOp]]
                ) -> Union[PauliSumOp, List[Optional[PauliSumOp]]]:
        """
        Maps the given second quantized operators to qubit operators using the same mapping,
        reductions as was done on a preceding successful call to :meth:`convert`. This allows
        other modules, that may build operators, to conform them to the mapping and reduction
        that would have been  done to the problem operators.
        but will do so only after

        Args:
            second_q_ops: A second quantized operator, or list thereof to be converted.

        Returns:
            A qubit operator or list of thereof the same length as the second_q_ops list. All
            operators in the second_q_ops list must commute with the symmetry detected when
            :meth:`convert` was called. If it does not then the position in the output list
            will be set to `None` to preserve the order; or None may be directly returned
            in the case when a single operator is provided

        Raises:
            QiskitNatureError: If property is accessed before a successful conversion
        """
        self._check_valid(True)

        # To allow a single operator to be converted, but use the same logic that does the
        # actual conversions, we make a single entry list of it here and unwrap to return.
        wrapped = False
        if isinstance(second_q_ops, SecondQuantizedOp):
            second_q_ops = [second_q_ops]
            wrapped = True

        qubit_ops = self._map_to_qubits(second_q_ops)
        qubit_ops_reduced, _ = self._two_qubit_reduce(qubit_ops, self.num_particles)
        qubit_ops_tapered = self._symmetry_reduce(qubit_ops_reduced)

        if wrapped:
            qubit_ops_tapered = qubit_ops_tapered[0]

        return qubit_ops_tapered

    def _map_to_qubits(self, second_q_ops: List[SecondQuantizedOp]) -> List[PauliSumOp]:

        qubit_ops: List[PauliSumOp] = []
        for op in second_q_ops:
            qubit_ops.append(self._mapper.map(op))

        return qubit_ops

    def _two_qubit_reduce(self, qubit_ops: List[PauliSumOp],
                          num_particles: Optional[Tuple[int, int]])\
            -> Tuple[List[Optional[PauliSumOp]], bool]:
        if self._two_qubit_reduction and not self._mapper.allows_two_qubit_reduction:
            logger.warning("Ignoring requested two qubit reduction as mapping does not support it")

        did_2_q = False
        if (self._two_qubit_reduction and
                self._mapper.allows_two_qubit_reduction and
                num_particles is not None):
            two_q_reducer = TwoQubitReduction(num_particles)
            reduced_qubit_ops = [cast(PauliSumOp, two_q_reducer.convert(op)) for op in qubit_ops]
            did_2_q = True
        else:
            reduced_qubit_ops = qubit_ops

        return reduced_qubit_ops, did_2_q

    def _find_symmetry_reduce(self, qubit_ops: List[PauliSumOp])\
            -> Tuple[List[Optional[PauliSumOp]], Z2Symmetries]:

        if self.z2symmetry_reduction is None:
            tapered_qubit_ops = qubit_ops
            z2_symmetries = self._no_symmetries
        else:
            z2_symmetries = Z2Symmetries.find_Z2_symmetries(qubit_ops[0])
            if z2_symmetries.is_empty():
                logger.debug('No Z2 symmetries found')
                tapered_qubit_ops = qubit_ops
            else:
                # check sector definition fits to symmetries found
                if len(self._z2symmetry_reduction) != len(z2_symmetries.symmetries):
                    raise QiskitNatureError('z2symmetry_reduction tapering values list has '
                                            'invalid length {} should be {}'.
                                            format(len(self._z2symmetry_reduction),
                                                   len(z2_symmetries.symmetries)))
                # Check all operators commute with main operator's symmetry
                logger.debug('Checking operators commute with symmetry:')
                symmetry_ops = []
                for symmetry in z2_symmetries.symmetries:
                    symmetry_ops.append(PauliSumOp.from_list([(symmetry.to_label(), 1.0)]))
                commuted = []
                for i, qubit_op in enumerate(qubit_ops):
                    commutes = QubitConverter._check_commutes(symmetry_ops, qubit_op)
                    if i == 0 and not commutes:
                        raise QiskitNatureError('Z2 symmetry failure. Main operator must commute '
                                                'with symmetries found from it!')
                    commuted.append(commutes)
                    logger.debug("Qubit operators commuted with symmetry %s", commuted)

                # set tapering for the given sector and taper operators
                z2_symmetries.tapering_values = self._z2symmetry_reduction
                tapered_qubit_ops = []
                for i, commutes in enumerate(commuted):
                    op = z2_symmetries.taper(qubit_ops[i]) if commutes else None
                    tapered_qubit_ops.append(op)

        return tapered_qubit_ops, z2_symmetries

    def _symmetry_reduce(self, qubit_ops: List[PauliSumOp])\
            -> List[Optional[PauliSumOp]]:

        if self.z2symmetry_reduction is None:
            tapered_qubit_ops = qubit_ops
        else:
            z2_symmetries = self.z2_symmetries
            if z2_symmetries.is_empty():
                tapered_qubit_ops = qubit_ops
            else:
                logger.debug('Checking operators commute with symmetry:')
                symmetry_ops = []
                for symmetry in z2_symmetries.symmetries:
                    symmetry_ops.append(PauliSumOp.from_list([(symmetry.to_label(), 1.0)]))
                commuted = []
                for i, qubit_op in enumerate(qubit_ops):
                    commutes = QubitConverter._check_commutes(symmetry_ops, qubit_op)
                    commuted.append(commutes)
                    logger.debug("Qubit operators commuted with symmetry %s", commuted)

                # Tapering values were set from prior convert so we go ahead and taper operators
                tapered_qubit_ops = []
                for i, commutes in enumerate(commuted):
                    op = z2_symmetries.taper(qubit_ops[i]) if commutes else None
                    tapered_qubit_ops.append(op)

        return tapered_qubit_ops

    @staticmethod
    def _check_commutes(cliffords: List[PauliSumOp], qubit_op: PauliSumOp) -> bool:
        commutes = []
        for clifford in cliffords:
            commutes.append(qubit_op.primitive.table.commutes_with_all(
                clifford.primitive.table))
        does_commute = np.all(commutes) is True
        logger.debug('  \'%s\' commutes: %s, %s', id(qubit_op), does_commute, commutes)

        return does_commute
