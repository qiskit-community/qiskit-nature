# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Electronic Structure Problem class."""

from __future__ import annotations

from functools import partial
from typing import cast, Callable, List, Optional, Tuple, Union

import numpy as np

from qiskit.algorithms import EigensolverResult, MinimumEigensolverResult
from qiskit.opflow.primitive_ops import Z2Symmetries

from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.circuit.library.initial_states.hartree_fock import (
    hartree_fock_bitstring_mapped,
)
from qiskit_nature.second_q.drivers import ElectronicStructureDriver
from qiskit_nature.second_q.operators import SecondQuantizedOp
from qiskit_nature.second_q.mappers import QubitConverter
from qiskit_nature.second_q.properties import ParticleNumber
from qiskit_nature.second_q.transformers.base_transformer import BaseTransformer

from .electronic_structure_result import ElectronicStructureResult
from .eigenstate_result import EigenstateResult

from .base_problem import BaseProblem


class ElectronicStructureProblem(BaseProblem):
    """The Electronic Structure Problem.

    The attributes `num_particles` and `num_spin_orbitals` are only available _after_ the
    `second_q_ops()` method has been called! Note, that if you do so, the method will be executed
    again when the problem is being solved.

    In the fermionic case the default filter ensures that the number of particles is being
    preserved.

    .. note::

        The default filter_criterion assumes a singlet spin configuration. This means, that the
        number of alpha-spin electrons is equal to the number of beta-spin electrons.
        If the :class:`~qiskit_nature.second_q.properties.AngularMomentum`
        property is available, one can correctly filter a non-singlet spin configuration with a
        custom `filter_criterion` similar to the following:

    .. code-block:: python

        import numpy as np
        from qiskit_nature.second_q.algorithms import NumPyEigensolverFactory

        expected_spin = 2
        expected_num_electrons = 6

        def filter_criterion_custom(eigenstate, eigenvalue, aux_values):
            num_particles_aux = aux_values["ParticleNumber"][0]
            total_angular_momentum_aux = aux_values["AngularMomentum"][0]

            return (
                np.isclose(expected_spin, total_angular_momentum_aux) and
                np.isclose(expected_num_electrons, num_particles_aux)
            )

        solver = NumPyEigensolverFactory(filter_criterion=filter_criterion_spin)

    """

    def __init__(
        self,
        driver: ElectronicStructureDriver,
        transformers: Optional[List[BaseTransformer]] = None,
    ):
        """

        Args:
            driver: A fermionic driver encoding the molecule information.
            transformers: A list of transformations to be applied to the driver result.
        """
        super().__init__(driver, transformers, "ElectronicEnergy")

    @property
    def num_particles(self) -> Tuple[int, int]:
        if self._grouped_property_transformed is None:
            raise QiskitNatureError(
                "`num_particles` is only available _after_ `second_q_ops()` has been called! "
                "Note, that if you run this manually, the method will run again during solving."
            )
        return self._grouped_property_transformed.get_property("ParticleNumber").num_particles

    @property
    def num_spin_orbitals(self) -> int:
        """Returns the number of spin orbitals."""
        if self._grouped_property_transformed is None:
            raise QiskitNatureError(
                "`num_spin_orbitals` is only available _after_ `second_q_ops()` has been called! "
                "Note, that if you run this manually, the method will run again during solving."
            )
        return self._grouped_property_transformed.get_property("ParticleNumber").num_spin_orbitals

    def second_q_ops(self) -> tuple[SecondQuantizedOp, dict[str, SecondQuantizedOp]]:
        """Returns the second quantized operators associated with this Property.

        Returns:
            A tuple, with the first object being the main operator and the second being a dictionary
            of auxiliary operators.
        """
        driver_result = self.driver.run()

        self._grouped_property = driver_result
        self._grouped_property_transformed = self._transform(self._grouped_property)

        second_quantized_ops = self._grouped_property_transformed.second_q_ops()
        main_op = second_quantized_ops.pop(self._main_property_name)

        return main_op, second_quantized_ops

    def interpret(
        self,
        raw_result: Union[EigenstateResult, EigensolverResult, MinimumEigensolverResult],
    ) -> ElectronicStructureResult:
        """Interprets an EigenstateResult in the context of this transformation.

        Args:
            raw_result: an eigenstate result object.

        Returns:
            An electronic structure result.
        """
        eigenstate_result = None
        if isinstance(raw_result, EigenstateResult):
            eigenstate_result = raw_result
        elif isinstance(raw_result, EigensolverResult):
            eigenstate_result = EigenstateResult()
            eigenstate_result.raw_result = raw_result
            eigenstate_result.eigenenergies = raw_result.eigenvalues
            eigenstate_result.eigenstates = raw_result.eigenstates
            eigenstate_result.aux_operator_eigenvalues = raw_result.aux_operator_eigenvalues
        elif isinstance(raw_result, MinimumEigensolverResult):
            eigenstate_result = EigenstateResult()
            eigenstate_result.raw_result = raw_result
            eigenstate_result.eigenenergies = np.asarray([raw_result.eigenvalue])
            eigenstate_result.eigenstates = [raw_result.eigenstate]
            eigenstate_result.aux_operator_eigenvalues = [raw_result.aux_operator_eigenvalues]
        result = ElectronicStructureResult()
        result.combine(eigenstate_result)
        self._grouped_property_transformed.interpret(result)
        result.computed_energies = np.asarray([e.real for e in eigenstate_result.eigenenergies])
        return result

    def get_default_filter_criterion(
        self,
    ) -> Optional[Callable[[Union[List, np.ndarray], float, Optional[List[float]]], bool]]:
        """Returns a default filter criterion method to filter the eigenvalues computed by the
        eigen solver. For more information see also
        qiskit.algorithms.eigen_solvers.NumPyEigensolver.filter_criterion.

        In the fermionic case the default filter ensures that the number of particles is being
        preserved.
        """

        # pylint: disable=unused-argument
        def filter_criterion(self, eigenstate, eigenvalue, aux_values):
            num_particles_aux = aux_values["ParticleNumber"][0]
            total_angular_momentum_aux = aux_values["AngularMomentum"][0]
            particle_number = cast(
                ParticleNumber, self.grouped_property_transformed.get_property(ParticleNumber)
            )
            return np.isclose(
                particle_number.num_alpha + particle_number.num_beta,
                num_particles_aux,
            ) and np.isclose(0.0, total_angular_momentum_aux)

        return partial(filter_criterion, self)

    def symmetry_sector_locator(
        self,
        z2_symmetries: Z2Symmetries,
        converter: QubitConverter,
    ) -> Optional[List[int]]:
        """Given the detected Z2Symmetries this determines the correct sector of the tapered
        operator that contains the ground state we need and returns that information.

        Args:
            z2_symmetries: the z2 symmetries object.
            converter: the qubit converter instance used for the operator conversion that
                symmetries are to be determined for.

        Returns:
            The sector of the tapered operators with the problem solution.
        """
        # We need the HF bitstring mapped to the qubit space but without any tapering done
        # by the converter (just qubit mapping and any two qubit reduction) since we are
        # going to determine the tapering sector
        hf_bitstr = hartree_fock_bitstring_mapped(
            num_spin_orbitals=self.num_spin_orbitals,
            num_particles=self.num_particles,
            qubit_converter=converter,
            match_convert=False,
        )
        sector = ElectronicStructureProblem._pick_sector(z2_symmetries, hf_bitstr)

        return sector

    @staticmethod
    def _pick_sector(z2_symmetries: Z2Symmetries, hf_str: List[bool]) -> List[int]:
        # Finding all the symmetries using the find_Z2_symmetries:
        taper_coeff: List[int] = []
        for sym in z2_symmetries.symmetries:
            coeff = -1 if np.logical_xor.reduce(np.logical_and(sym.z, hf_str)) else 1
            taper_coeff.append(coeff)

        return taper_coeff
