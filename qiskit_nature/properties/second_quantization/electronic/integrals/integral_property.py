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

"""The ElectronicEnergy property."""

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np

from qiskit_nature.operators.second_quantization import FermionicOp

from ...second_quantized_property import DriverResult, SecondQuantizedProperty
from ..bases import ElectronicBasis, ElectronicBasisTransform
from . import ElectronicIntegrals


class IntegralProperty(SecondQuantizedProperty):
    """A common Property object based on `ElectronicIntegrals` as its raw data.

    This is a common base class, extracted to be used by (at the time of writing) the
    `ElectronicEnergy` and the `DipoleMoment` properties. More subclasses may be added in the
    future.
    """

    def __init__(
        self,
        name: str,
        electronic_integrals: Dict[ElectronicBasis, List[ElectronicIntegrals]],
        shift: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            basis: the basis which the integrals in ``electronic_integrals`` are stored in.
            electronic_integrals: a dictionary mapping the ``# body terms`` to the corresponding
                ``ElectronicIntegrals``.
            shift: an optional dictionary of value shifts.
        """
        super().__init__(name)
        self._electronic_integrals = electronic_integrals
        self._shift = shift or {}

    def reduce_system_size(
        self, electronic_density: Tuple[np.ndarray, np.ndarray], transform: ElectronicBasisTransform
    ) -> "IntegralProperty":
        """TODO."""
        ao_integrals = self._electronic_integrals[ElectronicBasis.AO]

        beta_spin = electronic_density[1] is not None

        hcore = ao_integrals[0]._matrices
        if beta_spin and hcore[1] is None:
            hcore = (hcore[0], hcore[0])

        fock_operator = deepcopy(hcore)

        if len(ao_integrals) > 1:
            eri = ao_integrals[1]._matrices[0]
            coulomb_inactive = np.einsum("ijkl,ji->kl", eri, electronic_density[0])
            exchange_inactive = np.einsum("ijkl,jk->il", eri, electronic_density[0])

            if not beta_spin:
                fock_operator = (
                    fock_operator[0] + coulomb_inactive - 0.5 * exchange_inactive,
                    None,
                )
            else:
                coulomb_inactive_b = np.einsum("ijkl,ji->kl", eri, electronic_density[1])
                exchange_inactive_b = np.einsum("ijkl,jk->il", eri, electronic_density[1])
                fock_operator = (
                    fock_operator[0] + coulomb_inactive + coulomb_inactive_b - exchange_inactive,
                    fock_operator[1] + coulomb_inactive + coulomb_inactive_b - exchange_inactive_b,
                )

        e_inactive = 0.0
        if not beta_spin and electronic_density[0][0].size > 0:
            e_inactive += 0.5 * np.einsum(
                "ij,ji", electronic_density[0], hcore[0] + fock_operator[0]
            )
        elif beta_spin and electronic_density[1][0].size > 0:
            e_inactive += 0.5 * (
                np.einsum("ij,ji", electronic_density[0], hcore[0] + fock_operator[0])
                + np.einsum("ij,ji", electronic_density[1], hcore[1] + fock_operator[1])
            )

        ret = deepcopy(self)
        ret._electronic_integrals[ElectronicBasis.AO][0]._matrices = fock_operator
        ret._electronic_integrals[ElectronicBasis.MO] = [
            ints.transform_basis(transform)
            for ints in ret._electronic_integrals[ElectronicBasis.AO]
        ]
        ret._shift["Inactive Energy"] = e_inactive

        return ret

    def second_q_ops(self) -> List[FermionicOp]:
        """Returns a list containing the Hamiltonian constructed by the stored electronic integrals."""
        ints = None
        if ElectronicBasis.SO in self._electronic_integrals.keys():
            ints = self._electronic_integrals[ElectronicBasis.SO]
        elif ElectronicBasis.MO in self._electronic_integrals.keys():
            ints = self._electronic_integrals[ElectronicBasis.MO]
        return [sum(int.to_second_q_op() for int in ints).reduce()]  # type: ignore

    @classmethod
    def from_driver_result(cls, result: DriverResult) -> "IntegralProperty":
        """This property does not support construction from a driver result (yet).

        Args:
            result: ignored.

        Raises:
            NotImplemented
        """
        raise NotImplementedError()
