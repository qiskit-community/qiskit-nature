# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Reference Transition amplitudes for testing."""

from typing import Any, Dict, Tuple

reference_trans_amps: Dict[Tuple[int, int], Dict[str, Any]] = {
    (0, 1): {
        "AngularMomentum": (0.0, {}),
        "Magnetization": (0.0, {}),
        "ParticleNumber": (0.0, {}),
        "XDipole": (0.0, {}),
        "YDipole": (0.0, {}),
        "ZDipole": (0.0, {}),
        "hamiltonian_derivative": (0.0, {}),
    },
    (0, 2): {
        "AngularMomentum": (0.0, {}),
        "Magnetization": (0.0, {}),
        "ParticleNumber": (0.0, {}),
        "XDipole": (0.0, {}),
        "YDipole": (0.0, {}),
        "ZDipole": (0.0, {}),
        "hamiltonian_derivative": (0.0, {}),
    },
    (0, 3): {
        "AngularMomentum": (0.0, {}),
        "Magnetization": (0.0, {}),
        "ParticleNumber": (0.0, {}),
        "XDipole": (0.0, {}),
        "YDipole": (0.0, {}),
        "ZDipole": (0.0, {}),
        "hamiltonian_derivative": (-77.14309256038261, {}),
    },
    (1, 2): {
        "AngularMomentum": (0.0, {}),
        "Magnetization": (0.0, {}),
        "ParticleNumber": (0.0, {}),
        "XDipole": (0.0, {}),
        "YDipole": (0.0, {}),
        "ZDipole": (0.0, {}),
        "hamiltonian_derivative": (0.0, {}),
    },
    (1, 3): {
        "AngularMomentum": (0.0, {}),
        "Magnetization": (0.0, {}),
        "ParticleNumber": (0.0, {}),
        "XDipole": (0.0, {}),
        "YDipole": (0.0, {}),
        "ZDipole": (0.0, {}),
        "hamiltonian_derivative": (0.0, {}),
    },
    (2, 3): {
        "AngularMomentum": (0.0, {}),
        "Magnetization": (0.0, {}),
        "ParticleNumber": (0.0, {}),
        "XDipole": (0.0, {}),
        "YDipole": (0.0, {}),
        "ZDipole": (0.0, {}),
        "hamiltonian_derivative": (0.0, {}),
    },
}
