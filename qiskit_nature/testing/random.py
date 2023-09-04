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

"""Methods to sample random objects."""

from __future__ import annotations

from typing import Any

import numpy as np
from qiskit.quantum_info import random_hermitian

from qiskit_nature.second_q.hamiltonians import QuadraticHamiltonian


def random_antisymmetric_matrix(dim: int, seed: Any = None) -> np.ndarray:
    """Return a random antisymmetric matrix.

    Args:
        dim: The width and height of the matrix.
        seed: The pseudorandom number generator or seed. Should be a valid input to
            :func:`numpy.random.default_rng`.

    Returns:
        The sampled antisymmetric matrix.
    """
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    return mat - mat.T


def random_quadratic_hamiltonian(
    n_orbitals: int, num_conserving: bool = False, seed: Any = None
) -> QuadraticHamiltonian:
    """Generate a random instance of QuadraticHamiltonian.

    Args:
        n_orbitals: The number of orbitals.
        num_conserving: Whether the Hamiltonian should conserve particle number.
        seed: The pseudorandom number generator or seed. Should be a valid input to
            :func:`numpy.random.default_rng`.

    Returns:
        The sampled QuadraticHamiltonian.
    """
    rng = np.random.default_rng(seed)
    hermitian_part = np.array(random_hermitian(n_orbitals, seed=rng))
    antisymmetric_part = (
        None if num_conserving else random_antisymmetric_matrix(n_orbitals, seed=rng)
    )
    constant = rng.standard_normal()
    return QuadraticHamiltonian(
        hermitian_part=hermitian_part, antisymmetric_part=antisymmetric_part, constant=constant
    )


def random_two_body_tensor_real(size: int, rank: int | None = None, seed: Any = None) -> np.ndarray:
    """Sample a random two-body tensor with real-valued orbitals.

    Args:
        size: The length of one dimension of the tensor. The shape of the returned
            tensor will be ``(size, size, size, size)``.
        rank: Rank of the sampled tensor. The default behavior is to use
            the maximum rank, which is ``size * (size + 1) // 2``.
        seed: The pseudorandom number generator or seed. Should be a valid input to
            :func:`numpy.random.default_rng`.

    Returns:
        The sampled two-body tensor.
    """
    rng = np.random.default_rng(seed)
    if rank is None:
        rank = size * (size + 1) // 2
    cholesky_vecs = rng.standard_normal((rank, size, size))
    cholesky_vecs += cholesky_vecs.transpose((0, 2, 1))
    return np.einsum("ipr,iqs->prqs", cholesky_vecs, cholesky_vecs)
