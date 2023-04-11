# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A converter from Second-Quantized to Qubit Operators."""

from __future__ import annotations

import copy
import logging
from typing import Callable

import numpy as np

from qiskit.algorithms.list_or_dict import ListOrDict as ListOrDictType
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp
from qiskit.opflow.primitive_ops import Z2Symmetries

from qiskit_nature import QiskitNatureError, settings
from qiskit_nature.deprecation import warn_deprecated, DeprecatedType
from qiskit_nature.second_q.operators import SparseLabelOp

from .qubit_mapper import QubitMapper, _ListOrDict
from .parity_mapper import ParityMapper
from .tapered_qubit_mapper import TaperedQubitMapper

logger = logging.getLogger(__name__)


class QubitConverter:
    """DEPRECATED A converter from Second-Quantized to Qubit Operators.

    This converter can be configured with a mapper instance which will later be used
    when 2nd quantized operators are requested to be converted (mapped) to qubit operators.

    For a Fermionic system, when its a electronic problem, there are certain mappers, such as
    the :class:`~qiskit_nature.second_q.mappers.ParityMapper` that introduces known
    symmetries, by virtue of the mapping, that can be exploited to reduce the size of the
    problem, i.e the qubit operator, by two qubits. The two qubit reduction setting indicates
    to do this where possible - i.e. mapper supports it and the number of particles in the
    Fermionic system is provided for the conversion. (The number of particles is used to
    determine the symmetry.)

    Also this converter supports Z2 Symmetry reduction to reduce a problem (operator) size
    based on mathematical symmetries that can be detected in the operator. A knowledgeable user
    can define which sector the problem solution lies in. This sector information can also
    be passed in on :meth:`convert` which will override this value should both be given.
    """

    def __init__(
        self,
        mapper: QubitMapper,
        *,
        two_qubit_reduction: bool = False,
        z2symmetry_reduction: str | list[int] | None = None,
        sort_operators: bool = False,
    ):
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
            sort_operators: Whether or not the second-quantized operators should be sorted before
                mapping them to the qubit space. Enable this if you encounter non-reproducible
                results which can occur when operator terms are not consistently ordered.
                This is disabled by default, because in practice the Pauli-terms will be grouped
                later on anyways.

        Raises:
            ValueError: If the mapper is a ``TaperedQubitMapper``.
        """
        warn_deprecated(
            "0.6.0",
            DeprecatedType.CLASS,
            "QubitConverter",
            additional_msg=(
                ". Instead you should directly use the QubitMapper instance which you used to pass "
                "into the QubitConverter as the first argument. Refer to the documentation of the "
                "qiskit_nature.second_q.mappers module for more information"
            ),
        )
        if isinstance(mapper, TaperedQubitMapper):
            raise ValueError(
                "The TaperedQubitMapper is not supported by the QubitConverter. "
                "If you want to use tapering please either use the tapering built "
                "directly into the QubitConverter (see its documentation) "
                "or use the TaperedQubitMapper standalone (recommended)."
            )
        self._mapper: QubitMapper = mapper

        self._two_qubit_reduction: bool = two_qubit_reduction
        self._z2symmetry_reduction: str | list[int] | None = None
        # We use the setter for the additional validation
        self.z2symmetry_reduction = z2symmetry_reduction
        self._z2symmetries: Z2Symmetries = self._no_symmetries

        self._sort_operators: bool = sort_operators

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
    def z2symmetry_reduction(self) -> str | list[int] | None:
        """Get z2symmetry_reduction"""
        return self._z2symmetry_reduction

    @z2symmetry_reduction.setter
    def z2symmetry_reduction(self, z2symmetry_reduction: str | list[int] | None) -> None:
        """Set z2symmetry_reduction"""
        if z2symmetry_reduction is not None:
            if isinstance(z2symmetry_reduction, str):
                if z2symmetry_reduction != "auto":
                    raise ValueError(
                        "The only string-like option for z2symmetry_reduction is "
                        f"'auto', not {z2symmetry_reduction}"
                    )
            elif not np.all(np.isin(z2symmetry_reduction, [-1, 1])):
                raise ValueError(
                    "z2symmetry_reduction tapering values list must "
                    f"contain -1's and/or 1's only but was {z2symmetry_reduction}"
                )

        self._z2symmetry_reduction = z2symmetry_reduction

    @property
    def num_particles(self) -> tuple[int, int] | None:
        """Get the number of particles as supplied to :meth:`convert`.

        This can also be set, for advanced usage, using :meth:`force_match`
        """
        if isinstance(self._mapper, ParityMapper):
            return self._mapper.num_particles
        else:
            return None

    @property
    def z2symmetries(self) -> Z2Symmetries:
        """Get Z2Symmetries as detected from conversion via :meth:`convert`.

        This may indicate no symmetries, i.e. be empty, if none were detected.

        This can also be set, for advanced usage, using :meth:`force_match`
        """
        return copy.deepcopy(self._z2symmetries)

    def _check_reset_mapper(self) -> None:
        """Resets the ``ParityMapper`` if the attribute :attr:`two_qubit_reduction` is set to False. This
        makes the behavior of the QubitConverter compatible with the new ParityMapper class which only
        has one attribute :attr:`num_particles`. This must be called right before any mapping method of
        the mappers.
        """
        if not self.two_qubit_reduction and isinstance(self._mapper, ParityMapper):
            self._mapper.num_particles = None

    def convert(
        self,
        second_q_op: SparseLabelOp,
        num_particles: tuple[int, int] | None = None,
        sector_locator: Callable[[Z2Symmetries, "QubitConverter"], list[int] | None] | None = None,
    ) -> SparsePauliOp | PauliSumOp:
        """
        Map the given second quantized operator to a qubit operators. Also it will
        carry out z2 symmetry reduction on the qubit operators if z2symmetry_reduction has
        been specified whether via the constructor or indirectly via the sector locator which
        is passed the detected symmetries to inform the determination.

        Args:
            second_q_op: A second quantized operator.
            num_particles: Needed for two qubit reduction to determine correct sector. If
                not supplied, even if two_qubit_reduction is possible, it will not be done.
            sector_locator: An optional callback, that given the detected Z2Symmetries, and also
                the instance of the converter, can determine the correct sector of the tapered
                operators so the correct one can be returned, which contains the problem solution,
                out of the set that are the result of tapering.

        Returns:
            A qubit operator.
        """
        if isinstance(self._mapper, ParityMapper):
            self._mapper.num_particles = num_particles

        self._check_reset_mapper()

        reduced_op = self._mapper.map(second_q_op)
        tapered_op, z2symmetries = self.find_taper_op(reduced_op, sector_locator)

        self._z2symmetries = z2symmetries

        return tapered_op

    def convert_only(
        self,
        second_q_op: SparseLabelOp,
        num_particles: tuple[int, int] | None = None,
    ) -> SparsePauliOp | PauliSumOp:
        """
        Map the given second quantized operator to a qubit operators using the mapper
        and possibly two qubit reduction. No tapering is done, nor is any conversion state saved,
        as is done in :meth:`convert` where a later :meth:`convert_match` will convert
        further operators in an identical manner.

        Args:
            second_q_op: A second quantized operator.
            num_particles: Needed for two qubit reduction to determine correct sector. If
                not supplied, even if two_qubit_reduction is possible, it will not be done.

        Returns:
            A qubit operator.
        """
        if num_particles is not None and isinstance(self._mapper, ParityMapper):
            self._mapper.num_particles = num_particles

        self._check_reset_mapper()

        reduced_op = self._mapper.map(second_q_op)

        return reduced_op

    def force_match(
        self,
        *,
        num_particles: tuple[int, int] | None = None,
        z2symmetries: Z2Symmetries | None = None,
    ) -> None:
        """This is for advanced use where :meth:`convert` may not have been called or the
        converter should be used to taper to some external characteristics to be matched
        when using :meth:`convert_match`. Parameters passed here, when not None,
        will override any values stored from :meth:`convert`.

        Args:
            num_particles: The number or particles pertaining to two qubit reduction
            z2symmetries: Z2Symmetry information for tapering

        Raises:
            ValueError: If format of Z2Symmetry tapering values is invalid
        """
        if num_particles is not None and isinstance(self._mapper, ParityMapper):
            self._mapper.num_particles = num_particles

        if z2symmetries is not None:
            if not z2symmetries.is_empty():
                if len(z2symmetries.tapering_values) != len(z2symmetries.sq_list):
                    raise ValueError(
                        f"Z2Symmetries tapering value length was "
                        f"{len(z2symmetries.tapering_values)} but expected "
                        f"{len(z2symmetries.sq_list)}."
                    )
                if not np.all(np.isin(z2symmetries.tapering_values, [-1, 1])):
                    raise ValueError(
                        f"Z2Symmetries values list must contain only "
                        f"-1's and/or 1's but was {z2symmetries.tapering_values}."
                    )

            self._z2symmetries = z2symmetries

    def convert_match(
        self,
        second_q_ops: SparseLabelOp | ListOrDictType[SparseLabelOp],
        *,
        suppress_none: bool = False,
        check_commutes: bool = True,
    ) -> SparsePauliOp | PauliSumOp | ListOrDictType[SparsePauliOp | PauliSumOp]:
        """Convert further operators to match that done in :meth:`convert`, or as set by
            :meth:`force_match`.

        Args:
            second_q_ops: A second quantized operator or list thereof to be converted
            suppress_none: If None should be placed in the output list where an operator
                did not commute with symmetry, to maintain order, or whether that should
                be suppressed where the output list length may then be smaller than the input
            check_commutes: If True (default) an operator must commute with the
                symmetry to be tapered otherwise None is returned for that operator. When
                False the operator is tapered with no check so due consideration needs to
                be given in this case to how such operator(s) are eventually used.

        Returns:
            A qubit operator or list thereof of the same length as the second_q_ops list. All
            operators in the second_q_ops list must commute with the symmetry detected when
            :meth:`convert` was called. If it does not then the position in the output list
            will be set to `None` to preserve the order, unless suppress_none is set; or None may
            be directly returned in the case when a single operator is provided (that cannot be
            suppressed as it's a single value)
        """

        self._check_reset_mapper()

        # To allow a single operator to be converted, but use the same logic that does the
        # actual conversions, we make a single entry list of it here and unwrap to return.
        wrapped_type = type(second_q_ops)

        if isinstance(second_q_ops, SparseLabelOp):
            second_q_ops = [second_q_ops]
            suppress_none = False  # When only a single op we will return None back

        wrapped_second_q_ops: _ListOrDict[SparseLabelOp] = _ListOrDict(second_q_ops)

        reduced_ops: _ListOrDict[PauliSumOp] = _ListOrDict()
        for name, second_q_op in iter(wrapped_second_q_ops):
            reduced_op = self._mapper.map(second_q_op)
            # NOTE: we ensure all operators are PauliSumOp here, because this is required by the
            # opflow-based Z2Symmetries class still used by the QubitConverter
            reduced_ops[name] = (
                PauliSumOp(reduced_op) if not isinstance(reduced_op, PauliSumOp) else reduced_op
            )

        tapered_ops: _ListOrDict[PauliSumOp] = self._symmetry_reduce(reduced_ops, check_commutes)

        # NOTE: _ListOrDict.unwrap takes care of the conversion to/from PauliSumOp based on
        # settings.use_pauli_sum_op
        returned_ops: PauliSumOp | ListOrDictType[PauliSumOp] = tapered_ops.unwrap(
            wrapped_type, suppress_none=suppress_none
        )

        return returned_ops

    def find_taper_op(
        self,
        qubit_op: SparsePauliOp | PauliSumOp,
        sector_locator: Callable[[Z2Symmetries, "QubitConverter"], list[int] | None] | None = None,
    ) -> tuple[SparsePauliOp | PauliSumOp, Z2Symmetries]:
        r"""
        Find the $Z_2$-symmetries associated with the qubit operator and taper it accordingly.

        Args:
            qubit_op: Qubit main operator - often the hamiltonian - from which symmetries
                will be identified.
            sector_locator: Method associated to the problem of interest which identifies the
                symmetry sector of the solution. Defaults to None.

        Raises:
            QiskitNatureError: The user-specified or identified symmetry sector is not compatible with
                the symmetries found for this problem.
            QiskitNatureError: The main operator does not commute with its expected symmetries.

        Returns:
            Tuple of the form (tapered qubit operator, identified $Z_2$-symmetry object)
        """
        if not isinstance(qubit_op, PauliSumOp):
            # NOTE: we ensure the operator is a PauliSumOp here, because this is required by the
            # opflow-based Z2Symmetries class still used by the QubitConverter
            qubit_op = PauliSumOp(qubit_op)

        # Return operator unchanged and empty symmetries if we do not taper
        tapered_qubit_op = qubit_op
        z2_symmetries = self._no_symmetries

        # If we were given a sector, or one might be located, we first need to find any symmetries
        if self.z2symmetry_reduction is not None:
            z2_symmetries = Z2Symmetries.find_Z2_symmetries(qubit_op)
            if z2_symmetries.is_empty():
                logger.debug("No Z2 symmetries found")
            else:
                # As we have symmetries, if we have a sector locator, if that provides one back
                # it will override any value defined on constructor
                if sector_locator is not None and self.z2symmetry_reduction == "auto":
                    z2symmetry_reduction = sector_locator(z2_symmetries, self)
                    if z2symmetry_reduction is not None:
                        self.z2symmetry_reduction = z2symmetry_reduction  # Overrides any value

                    # We may end up that neither were we given a sector nor that the locator
                    # returned one. Since though we may have found valid symmetries above we should
                    # simply just forget about them so as not to return something we are not using.
                    if self.z2symmetry_reduction is None:
                        z2_symmetries = self._no_symmetries

        # So now if we have a sector and have symmetries we found we can attempt to taper
        if (
            self.z2symmetry_reduction is not None
            and self.z2symmetry_reduction != "auto"
            and not z2_symmetries.is_empty()
        ):
            # check sector definition fits to symmetries found
            if len(self._z2symmetry_reduction) != len(z2_symmetries.symmetries):
                raise QiskitNatureError(
                    "z2symmetry_reduction tapering values list has "
                    f"invalid length {len(self._z2symmetry_reduction)} "
                    f"should be {len(z2_symmetries.symmetries)}"
                )
            # Check all operators commute with main operator's symmetry
            logger.debug("Sanity check that operator commutes with the symmetry")
            symmetry_ops = []
            for symmetry in z2_symmetries.symmetries:
                symmetry_ops.append(PauliSumOp.from_list([(symmetry.to_label(), 1.0)]))
            commutes = QubitConverter._check_commutes(symmetry_ops, qubit_op)
            if not commutes:
                raise QiskitNatureError(
                    "Z2 symmetry failure. The operator must commute "
                    "with symmetries found from it!"
                )

            z2_symmetries.tapering_values = self._z2symmetry_reduction
            tapered_qubit_op = z2_symmetries.taper(qubit_op) if commutes else None

        if (
            tapered_qubit_op is not None
            and not isinstance(tapered_qubit_op, SparsePauliOp)
            and not settings.use_pauli_sum_op
        ):
            tapered_qubit_op = tapered_qubit_op.primitive

        return tapered_qubit_op, z2_symmetries

    def _symmetry_reduce(
        self,
        qubit_ops: _ListOrDict[PauliSumOp],
        check_commutes: bool,
    ) -> _ListOrDict[PauliSumOp]:

        if self._z2symmetries is None or self._z2symmetries.is_empty():
            tapered_qubit_ops = qubit_ops
        else:
            if check_commutes:
                logger.debug("Checking operators commute with symmetry:")
                symmetry_ops = []
                for symmetry in self._z2symmetries.symmetries:
                    symmetry_ops.append(PauliSumOp.from_list([(symmetry.to_label(), 1.0)]))
                commuted = {}
                for name, qubit_op in iter(qubit_ops):
                    commutes = QubitConverter._check_commutes(symmetry_ops, qubit_op)
                    commuted[name] = commutes
                    logger.debug("Qubit operator '%s' commuted with symmetry: %s", name, commutes)

                # Tapering values were set from prior convert so we go ahead and taper operators
                tapered_qubit_ops = _ListOrDict()
                for name, commutes in commuted.items():
                    if commutes:
                        tapered_qubit_ops[name] = self._z2symmetries.taper(qubit_ops[name])
            else:
                logger.debug("Tapering operators whether they commute with symmetry or not:")
                tapered_qubit_ops = _ListOrDict()
                for name, qubit_op in iter(qubit_ops):
                    tapered_qubit_ops[name] = self._z2symmetries.taper(qubit_ops[name])

        return tapered_qubit_ops

    def symmetry_reduce_clifford(
        self,
        converted_ops: ListOrDictType[SparsePauliOp | PauliSumOp],
        *,
        check_commutes: bool = True,
    ) -> ListOrDictType[SparsePauliOp | PauliSumOp]:
        """
        Applies the tapering to a list of operators previously converted with the Clifford
        transformation from the current symmetry.

        Args:
            converted_ops: Operators to taper.
            check_commutes: If True (default) an operator must commute with the
                symmetry to be tapered otherwise None is returned for that operator. When
                False the operator is tapered with no check so due consideration needs to
                be given in this case to how such operator(s) are eventually used.

        Returns:
            Tapered operators.
        """
        if converted_ops is None or self._z2symmetries is None or self._z2symmetries.is_empty():
            return_ops = converted_ops
        else:
            wrapped_converted_ops, wrapped_type = _ListOrDict.wrap(converted_ops)

            pauli_sum_ops: ListOrDictType[PauliSumOp] = _ListOrDict()
            for name, qubit_op in iter(wrapped_converted_ops):
                if not isinstance(qubit_op, PauliSumOp):
                    qubit_op = PauliSumOp(qubit_op)
                pauli_sum_ops[name] = qubit_op

            if check_commutes:
                logger.debug("Checking operators commute with symmetry:")
                symmetry_ops = []
                for sq_pauli in self._z2symmetries._sq_paulis:
                    symmetry_ops.append(PauliSumOp.from_list([(sq_pauli.to_label(), 1.0)]))
                commuted = {}
                for name, qubit_op in iter(pauli_sum_ops):
                    commutes = QubitConverter._check_commutes(symmetry_ops, qubit_op)
                    commuted[name] = commutes
                    logger.debug("Qubit operator '%s' commuted with symmetry: %s", name, commutes)

                # Tapering values were set from prior convert, so we go ahead and taper operators
                tapered_qubit_ops: _ListOrDict[PauliSumOp] = _ListOrDict()
                for name, commutes in commuted.items():
                    if commutes:
                        tapered_qubit_ops[name] = self._z2symmetries.taper_clifford(
                            pauli_sum_ops[name]
                        )
            else:
                logger.debug("Tapering operators whether they commute with symmetry or not:")
                tapered_qubit_ops = _ListOrDict()
                for name, qubit_op in iter(pauli_sum_ops):
                    tapered_qubit_ops[name] = self._z2symmetries.taper_clifford(pauli_sum_ops[name])

            # NOTE: _ListOrDict.unwrap takes care of the conversion to/from PauliSumOp based on
            # settings.use_pauli_sum_op
            return_ops = tapered_qubit_ops.unwrap(wrapped_type)

        return return_ops

    def convert_clifford(
        self,
        qubit_ops: SparsePauliOp | PauliSumOp | ListOrDictType[SparsePauliOp | PauliSumOp],
    ) -> SparsePauliOp | PauliSumOp | ListOrDictType[SparsePauliOp | PauliSumOp]:
        """
        Applies the Clifford transformation from the current symmetry to all operators.

        Args:
            qubit_ops: Operators to convert.

        Returns:
            Converted operators
        """
        if qubit_ops is None or self._z2symmetries is None or self._z2symmetries.is_empty():
            converted_ops = qubit_ops
        else:
            wrapped_second_q_ops, wrapped_type = _ListOrDict.wrap(qubit_ops)

            pauli_sum_ops: ListOrDictType[PauliSumOp] = _ListOrDict()
            for name, qubit_op in iter(wrapped_second_q_ops):
                if not isinstance(qubit_op, PauliSumOp):
                    qubit_op = PauliSumOp(qubit_op)
                pauli_sum_ops[name] = qubit_op

            converted_ops = _ListOrDict()
            for name, second_q_op in iter(pauli_sum_ops):
                converted_ops[name] = self._z2symmetries.convert_clifford(second_q_op)

            # NOTE: _ListOrDict.unwrap takes care of the conversion to/from PauliSumOp based on
            # settings.use_pauli_sum_op
            converted_ops = converted_ops.unwrap(wrapped_type)

        return converted_ops

    @staticmethod
    def _check_commutes(cliffords: list[PauliSumOp], qubit_op: PauliSumOp) -> bool:
        commutes = []
        for clifford in cliffords:
            commuting_rows = qubit_op.primitive.paulis.commutes_with_all(clifford.primitive.paulis)
            commutes.append(len(commuting_rows) == qubit_op.primitive.size)
        does_commute = bool(np.all(commutes))
        logger.debug("  '%s' commutes: %s, %s", id(qubit_op), does_commute, commutes)

        return does_commute
