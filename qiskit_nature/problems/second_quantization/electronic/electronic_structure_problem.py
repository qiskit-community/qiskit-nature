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

"""The Electronic Structure Problem class."""
from functools import partial
from typing import cast, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from qiskit.algorithms import EigensolverResult, MinimumEigensolverResult
from qiskit.opflow import PauliSumOp
from qiskit.opflow.primitive_ops import Z2Symmetries

from qiskit_nature import ListOrDictType
from qiskit_nature.circuit.library.initial_states.hartree_fock import hartree_fock_bitstring_mapped
from qiskit_nature.drivers import QMolecule
from qiskit_nature.drivers.second_quantization import ElectronicStructureDriver
from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.properties.second_quantization.electronic import (
    ElectronicStructureDriverResult,
    ParticleNumber,
)
from qiskit_nature.results import EigenstateResult, ElectronicStructureResult
from qiskit_nature.transformers import BaseTransformer as LegacyBaseTransformer
from qiskit_nature.transformers.second_quantization import BaseTransformer

from .builders.hopping_ops_builder import _build_qeom_hopping_ops
from ..base_problem import BaseProblem


class ElectronicStructureProblem(BaseProblem):
    """Electronic Structure Problem"""

    def __init__(
        self,
        driver: ElectronicStructureDriver,
        transformers: Optional[List[Union[LegacyBaseTransformer, BaseTransformer]]] = None,
    ):
        """

        Args:
            driver: A fermionic driver encoding the molecule information.
            transformers: A list of transformations to be applied to the driver result.
        """
        super().__init__(driver, transformers)
        self._main_property_name = "ElectronicEnergy"

    @property
    def num_particles(self) -> Tuple[int, int]:
        return self._grouped_property_transformed.get_property("ParticleNumber").num_particles

    @property
    def num_spin_orbitals(self) -> int:
        """Returns the number of spin orbitals."""
        return self._grouped_property_transformed.get_property("ParticleNumber").num_spin_orbitals

    def second_q_ops(self) -> ListOrDictType[SecondQuantizedOp]:
        """Returns the second quantized operators associated with this Property.

        If the arguments are returned as a `list`, the operators are in the following order: the
        Hamiltonian operator, total particle number operator, total angular momentum operator, total
        magnetization operator, and (if available) x, y, z dipole operators.

        The actual return-type is determined by `qiskit_nature.settings.dict_aux_operators`.

        Returns:
            A `list` or `dict` of `SecondQuantizedOp` objects.
        """
        driver_result = self.driver.run()

        if self._legacy_driver:
            self._molecule_data = cast(QMolecule, driver_result)
            self._grouped_property = ElectronicStructureDriverResult.from_legacy_driver_result(
                self._molecule_data
            )

            if self._legacy_transform:
                self._molecule_data_transformed = self._transform(self._molecule_data)
                self._grouped_property_transformed = (
                    ElectronicStructureDriverResult.from_legacy_driver_result(
                        self._molecule_data_transformed
                    )
                )

            else:
                if not self.transformers:
                    # if no transformers are supplied, we can still provide
                    # `molecule_data_transformed` as a copy of `molecule_data`
                    self._molecule_data_transformed = self._molecule_data
                self._grouped_property_transformed = self._transform(self._grouped_property)

        else:
            self._grouped_property = driver_result
            self._grouped_property_transformed = self._transform(self._grouped_property)

        second_quantized_ops = self._grouped_property_transformed.second_q_ops()

        return second_quantized_ops

    def hopping_qeom_ops(
        self,
        qubit_converter: QubitConverter,
        excitations: Union[
            str,
            int,
            List[int],
            Callable[[int, Tuple[int, int]], List[Tuple[Tuple[int, ...], Tuple[int, ...]]]],
        ] = "sd",
    ) -> Tuple[
        Dict[str, PauliSumOp],
        Dict[str, List[bool]],
        Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]],
    ]:
        """Generates the hopping operators and their commutativity information for the specified set
        of excitations.

        Args:
            qubit_converter: the `QubitConverter` to use for mapping and symmetry reduction. The
                             Z2 symmetries stored in this instance are the basis for the
                             commutativity information returned by this method.
            excitations: the types of excitations to consider. The simple cases for this input are:

                :`str`: containing any of the following characters: `s`, `d`, `t` or `q`.
                :`int`: a single, positive integer denoting the excitation type (1 == `s`, etc.).
                :`List[int]`: a list of positive integers.
                :`Callable`: a function which is used to generate the excitations.
                    For more details on how to write such a function refer to the default method,
                    :meth:`generate_fermionic_excitations`.

        Returns:
            A tuple containing the hopping operators, the types of commutativities and the
            excitation indices.
        """
        particle_number = self.grouped_property_transformed.get_property("ParticleNumber")
        return _build_qeom_hopping_ops(particle_number, qubit_converter, excitations)

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
            # the first aux_value is the evaluated number of particles
            try:
                num_particles_aux = aux_values["ParticleNumber"][0]
            except TypeError:
                num_particles_aux = aux_values[0][0]
            # the second aux_value is the total angular momentum which (for singlets) should be zero
            try:
                total_angular_momentum_aux = aux_values["AngularMomentum"][0]
            except TypeError:
                total_angular_momentum_aux = aux_values[1][0]
            particle_number = cast(
                ParticleNumber, self.grouped_property_transformed.get_property(ParticleNumber)
            )
            return (
                np.isclose(
                    particle_number.num_alpha + particle_number.num_beta,
                    num_particles_aux,
                )
                and np.isclose(0.0, total_angular_momentum_aux)
            )

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
