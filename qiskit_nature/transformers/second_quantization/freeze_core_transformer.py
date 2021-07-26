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

"""The Freeze-Core Reduction interface."""

from typing import List, Optional

from qiskit_nature.deprecation import DeprecatedType, warn_deprecated_same_type_name
from qiskit_nature.properties.second_quantization.electronic import ElectronicDriverResult

from .active_space_transformer import ActiveSpaceTransformer
from .electronic import FreezeCoreTransformer as NewFreezeCoreTransformer


class FreezeCoreTransformer(ActiveSpaceTransformer):
    """**DEPRECATED!**"""

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        freeze_core: bool = True,
        remove_orbitals: Optional[List[int]] = None,
    ) -> None:
        """Initializes a transformer which reduces a `QMolecule` by removing some molecular
        orbitals.

        The orbitals to be removed are specified in two ways:
            1. When `freeze_core` is enabled (the default), the `core_orbitals` listed in the
               `QMolecule` are made inactive and removed in the same fashion as in the
               :class:`ActiveSpaceTransformer`.
            2. Additionally, unoccupied molecular orbitals can be removed via a list of indices
               passed to `remove_orbitals`. It is the user's responsibility to ensure that these are
               indeed unoccupied orbitals, as no checks are performed.

        If you want to remove additional occupied orbitals, please use the
        :class:`ActiveSpaceTransformer` instead.

        Args:
            freeze_core: A boolean indicating whether to remove the molecular orbitals specified by
                        `QMolecule.core_orbitals`.
            remove_orbitals: A list of indices specifying molecular orbitals which are removed.
                             No checks are performed on the nature of these orbitals, so the user
                             must make sure that these are _unoccupied_ orbitals, which can be
                             removed without taking any energy shifts into account.
        """
        warn_deprecated_same_type_name(
            "0.2.0",
            DeprecatedType.CLASS,
            "FreezeCoreTransformer",
            "from qiskit_nature.transformers.second_quantization.electronic as a direct replacement",
        )

        self.inner = NewFreezeCoreTransformer(freeze_core, remove_orbitals)

    def transform(self, molecule_data):
        if not isinstance(molecule_data, ElectronicDriverResult):
            molecule_data = ElectronicDriverResult.from_legacy_driver_result(molecule_data)
        return self.inner.transform(molecule_data)
