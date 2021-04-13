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
from typing import cast, Callable, List, Optional, Tuple, Union

import numpy as np

from qiskit.opflow import PauliSumOp
from qiskit.opflow.converters import TwoQubitReduction
from qiskit.opflow.primitive_ops import Z2Symmetries

from qiskit_nature import QiskitNatureError
from qiskit_nature.mappers.second_quantization import QubitMapper

from . import SecondQuantizedOp

logger = logging.getLogger(__name__)


class QubitConverter:
    """A converter from Second-Quantized to Qubit Operators.

    This converter can be configured with a mapper instance which will later be used
    when 2nd quantized operators are requested to be converted (mapped) to qubit operators.

    For a Fermionic system, when its a electronic problem, there are certain mappers, such as
    the :class:`~qiskit_nature.mappers.second_quantization.ParityMapper` that introduces known
    symmetries, by virtue of the mapping, that can be exploited to reduce the size of the
    problem, i.e the qubit operator, by two qubits. The two qubit reduction setting indicates
    to do this where possible - i.e. mapper supports it and the number of particles in the
    Fermionic system is provided for the conversion. (The number of particles is used to
    determine the symmetry)

    Also this converter supports Z2 Symmetry reduction to reduce a problem (operator) size
    based on mathematical symmetries that can be detected in the operator. A knowledgeable user
    can define which sector the problem solution lies in. This sector information can also
    be passed in on :meth:`convert` which will override this value should both be given.
    """

    def __init__(self,
                 mapper: QubitMapper,
                 two_qubit_reduction: bool = False,
                 z2symmetry_reduction: Optional[Union[str, List[int]]] = None):
        """

        Args:
            mapper: A mapper instance used to convert second quantized to qubit operators
            two_qubit_reduction: Whether to carry out two qubit reduction when possible
            z2symmetry_reduction: Indicates whether a z2 symmetry reduction should be applied to
                resulting qubit operators that are computed. For each symmetry detected the operator
                will be split into two where each requires one qubit less for computation. So for
                example 3 symmetries will split the original operator into 8 new operators each
                requiring 3 less qubits. Now only one of these operators will have the ground state
                and be the correct symmetry sector needed for the ground state. Setting 'auto' will
                use an automatic computation of the correct sector. If the sector is known
                from other experiments with the z2symmetry logic, then the tapering values of that
                sector can be provided (a list of int of values -1, and 1). The default is None
                meaning no symmetry reduction is done.
        """

        self._mapper: QubitMapper = mapper
        self._two_qubit_reduction: bool = two_qubit_reduction
        self._z2symmetry_reduction: Optional[Union[str, List[int]]] = None
        self.z2symmetry_reduction = z2symmetry_reduction  # Setter does validation

        self._did_two_qubit_reduction: bool = False
        self._num_particles: Optional[Tuple[int, int]] = None
        self._z2symmetries: Z2Symmetries = self._no_symmetries

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
        self._z2symmetries = None  # Reset as symmetries my change due to mapper change

    @property
    def two_qubit_reduction(self) -> bool:
        """Get two_qubit_reduction"""
        return self._two_qubit_reduction

    @two_qubit_reduction.setter
    def two_qubit_reduction(self, value: bool) -> None:
        """Set two_qubit_reduction"""
        self._two_qubit_reduction = value
        self._z2symmetries = None  # Reset as symmetries my change due to this reduction

    @property
    def z2symmetry_reduction(self) -> Optional[Union[str, List[int]]]:
        """Get z2symmetry_reduction"""
        return self._z2symmetry_reduction

    @z2symmetry_reduction.setter
    def z2symmetry_reduction(self, z2symmetry_reduction: Optional[Union[str, List[int]]]) -> None:
        """Set z2symmetry_reduction"""
        if z2symmetry_reduction is not None:
            if isinstance(z2symmetry_reduction, str):
                if z2symmetry_reduction != 'auto':
                    raise ValueError("The only string-like option for z2symmetry_reduction is "
                                     "'auto', not {}".format(z2symmetry_reduction))
            elif not np.all(np.isin(z2symmetry_reduction, [-1, 1])):
                raise ValueError('z2symmetry_reduction tapering values list must '
                                 'contain -1\'s and/or 1\'s only but was {}'.
                                 format(z2symmetry_reduction))

        self._z2symmetry_reduction = z2symmetry_reduction

    @property
    def num_particles(self) -> Optional[Tuple[int, int]]:
        """Get the number of particles as supplied to :meth:`convert`.

        This can also be set, for advanced usage, using :meth:`force_match`
        """
        return self._num_particles

    @property
    def z2symmetries(self) -> Z2Symmetries:
        """Get Z2Symmetries as detected from conversion via :meth:`convert`.

        This may indicate no symmetries, i.e. be empty, if none were detected.

        This can also be set, for advanced usage, using :meth:`force_match`
        """
        return copy.deepcopy(self._z2symmetries)

    def convert(self, second_q_op: SecondQuantizedOp,
                num_particles: Optional[Tuple[int, int]] = None,
                sector_locator: Optional[Callable[[Z2Symmetries], Optional[List[int]]]] = None
                ) -> PauliSumOp:
        """
        Map the given second quantized operator to a qubit operators. Also it will
        carry out z2 symmetry reduction on the qubit operators if z2symmetry_reduction has
        been specified whether via the constructor or indirectly via the sector locator which
        is passed the detected symmetries to inform the determination.

        Args:
            second_q_op: A second quantized operator.
            num_particles: Needed for two qubit reduction to determine correct sector. If
                not supplied, even if two_qubit_reduction is possible, it will not be done.
            sector_locator: An optional callback, that given the detected Z2Symmetries can
                determine the correct sector of the tapered operators so the correct one
                can be returned, which contains the problem solution, out of the set that are
                the result of tapering.

        Returns:
            PauliSumOp qubit operator
        """
        qubit_op = self._map(second_q_op)
        reduced_op = self._two_qubit_reduce(qubit_op, num_particles)
        tapered_op, z2symmetries = self._find_taper_op(reduced_op, sector_locator)

        self._num_particles = num_particles
        self._z2symmetries = z2symmetries

        return tapered_op

    def force_match(self, num_particles: Optional[Tuple[int, int]] = None,
                    z2symmetries: Optional[Z2Symmetries] = None) -> None:
        """ This is for advanced use where :meth:`convert` may not have been called or the
            converter should be used to taper to some external characteristics to be matched
            when using :meth:`convert_match`. Parameters passed here, when not None,
            will override any values stored from :meth:`convert`.

            Args:
                num_particles: The number or particles pertaining to two qubit reduction
                z2symmetries: Z2Symmetry information for tapering

            Raises:
                ValueError: If format of Z2Symmetry tapering values is invalid
        """
        if num_particles is not None:
            self._num_particles = num_particles

        if z2symmetries is not None:
            if not z2symmetries.is_empty():
                if len(z2symmetries.tapering_values) != len(z2symmetries.sq_list):
                    raise ValueError(f'Z2Symmetries tapering value length was '
                                     f'{len(z2symmetries.tapering_values)} but expected '
                                     f'{len(z2symmetries.sq_list)}.')
                if not np.all(np.isin(z2symmetries.tapering_values, [-1, 1])):
                    raise ValueError(f'Z2Symmetries values list must contain only '
                                     f'-1\'s and/or 1\'s but was {z2symmetries.tapering_values}.')

            self._z2symmetries = z2symmetries

    def convert_match(self, second_q_ops: Union[SecondQuantizedOp, List[SecondQuantizedOp]],
                      suppress_none: bool = False
                      ) -> Union[PauliSumOp, List[Optional[PauliSumOp]]]:
        """ Convert further operators to match that done in :meth:`convert`, or as set by
            :meth:`force_match`.

        Args:
            second_q_ops: A second quantized operator or list thereof to be converted
            suppress_none: If None should be placed in the output list where an operator
               did not commute with symmetry, to maintain order, or whether that should
               be suppressed where the output list length may then be smaller than the input

        Returns:
            A qubit operator or list thereof of the same length as the second_q_ops list. All
            operators in the second_q_ops list must commute with the symmetry detected when
            :meth:`convert` was called. If it does not then the position in the output list
            will be set to `None` to preserve the order, unless suppress_none is set; or None may
            be directly returned in the case when a single operator is provided (that cannot be
            suppressed as it's a single value)
        """
        # To allow a single operator to be converted, but use the same logic that does the
        # actual conversions, we make a single entry list of it here and unwrap to return.
        wrapped = False
        if isinstance(second_q_ops, SecondQuantizedOp):
            second_q_ops = [second_q_ops]
            wrapped = True
            suppress_none = False  # When only a single op we will return None back

        qubit_ops = [self._map(second_q_op) for second_q_op in second_q_ops]
        reduced_ops = [self._two_qubit_reduce(qubit_op, self._num_particles)
                       for qubit_op in qubit_ops]
        tapered_ops = self._symmetry_reduce(reduced_ops, suppress_none)

        if wrapped:
            tapered_ops = tapered_ops[0]

        return tapered_ops

    def map(self, second_q_ops: Union[SecondQuantizedOp, List[SecondQuantizedOp]]) \
            -> Union[PauliSumOp, List[Optional[PauliSumOp]]]:
        """ A convenience method to map second quantized operators based on current mapper.

        Args:
            second_q_ops: A second quantized operator, or list thereof

        Returns:
            A qubit operator in the form of a PauliSumOp, or list thereof if a list of
            second quantized operators was supplied
        """
        if isinstance(second_q_ops, SecondQuantizedOp):
            qubit_ops = self._map(second_q_ops)
        else:
            qubit_ops = [self._map(second_q_op) for second_q_op in second_q_ops]

        return qubit_ops

    def _map(self, second_q_op: SecondQuantizedOp) -> PauliSumOp:
        return self._mapper.map(second_q_op)

    def _two_qubit_reduce(self, qubit_op: PauliSumOp,
                          num_particles: Optional[Tuple[int, int]]) -> PauliSumOp:
        reduced_op = qubit_op

        if num_particles is not None:
            if self._two_qubit_reduction and self._mapper.allows_two_qubit_reduction:
                two_q_reducer = TwoQubitReduction(num_particles)
                reduced_op = cast(PauliSumOp, two_q_reducer.convert(qubit_op))

        return reduced_op

    def _find_taper_op(self, qubit_op: PauliSumOp,
                       sector_locator: Optional[Callable[[Z2Symmetries],
                                                         Optional[List[int]]]] = None
                       ) -> Tuple[PauliSumOp, Z2Symmetries]:
        # Return operator unchanged and empty symmetries if we do not taper
        tapered_qubit_op = qubit_op
        z2_symmetries = self._no_symmetries

        # If we were given a sector, or one might be located, we first need to find any symmetries
        if self.z2symmetry_reduction is not None:
            z2_symmetries = Z2Symmetries.find_Z2_symmetries(qubit_op)
            if z2_symmetries.is_empty():
                logger.debug('No Z2 symmetries found')
            else:
                # As we have symmetries, if we have a sector locator, if that provides one back
                # it will override any value defined on constructor
                if sector_locator is not None and self.z2symmetry_reduction == 'auto':
                    z2symmetry_reduction = sector_locator(z2_symmetries)
                    if z2symmetry_reduction is not None:
                        self.z2symmetry_reduction = z2symmetry_reduction  # Overrides any value

                    # We may end up that neither were we given a sector nor that the locator
                    # returned one. Since though we may have found valid symmetries above we should
                    # simply just forget about them so as not to return something we are not using.
                    if self.z2symmetry_reduction is None:
                        z2_symmetries = self._no_symmetries

        # So now if we have a sector and have symmetries we found we can attempt to taper
        if self.z2symmetry_reduction is not None and self.z2symmetry_reduction != 'auto' \
                and not z2_symmetries.is_empty():
            # check sector definition fits to symmetries found
            if len(self._z2symmetry_reduction) != len(z2_symmetries.symmetries):
                raise QiskitNatureError('z2symmetry_reduction tapering values list has '
                                        'invalid length {} should be {}'.
                                        format(len(self._z2symmetry_reduction),
                                               len(z2_symmetries.symmetries)))
            # Check all operators commute with main operator's symmetry
            logger.debug('Sanity check that operator commutes with the symmetry')
            symmetry_ops = []
            for symmetry in z2_symmetries.symmetries:
                symmetry_ops.append(PauliSumOp.from_list([(symmetry.to_label(), 1.0)]))
            commutes = QubitConverter._check_commutes(symmetry_ops, qubit_op)
            if not commutes:
                raise QiskitNatureError('Z2 symmetry failure. The operator must commute '
                                        'with symmetries found from it!')

            z2_symmetries.tapering_values = self._z2symmetry_reduction
            tapered_qubit_op = z2_symmetries.taper(qubit_op) if commutes else None

        return tapered_qubit_op, z2_symmetries

    def _symmetry_reduce(self, qubit_ops: List[PauliSumOp],
                         suppress_none: bool) -> List[Optional[PauliSumOp]]:

        if self._z2symmetries is None or self._z2symmetries.is_empty():
            tapered_qubit_ops = qubit_ops
        else:
            logger.debug('Checking operators commute with symmetry:')
            symmetry_ops = []
            for symmetry in self._z2symmetries.symmetries:
                symmetry_ops.append(PauliSumOp.from_list([(symmetry.to_label(), 1.0)]))
            commuted = []
            for i, qubit_op in enumerate(qubit_ops):
                commutes = QubitConverter._check_commutes(symmetry_ops, qubit_op)
                commuted.append(commutes)
                logger.debug("Qubit operators commuted with symmetry %s", commuted)

            # Tapering values were set from prior convert so we go ahead and taper operators
            tapered_qubit_ops = []
            for i, commutes in enumerate(commuted):
                if commutes:
                    tapered_qubit_ops.append(self._z2symmetries.taper(qubit_ops[i]))
                elif not suppress_none:
                    tapered_qubit_ops.append(None)

        return tapered_qubit_ops

    @staticmethod
    def _check_commutes(cliffords: List[PauliSumOp], qubit_op: PauliSumOp) -> bool:
        commutes = []
        for clifford in cliffords:
            commuting_rows = qubit_op.primitive.table.commutes_with_all(clifford.primitive.table)
            commutes.append(len(commuting_rows) == qubit_op.primitive.size)
        does_commute = bool(np.all(commutes))
        logger.debug('  \'%s\' commutes: %s, %s', id(qubit_op), does_commute, commutes)

        return does_commute
