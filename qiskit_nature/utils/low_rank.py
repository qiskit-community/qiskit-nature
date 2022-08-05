# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Low rank decomposition utilities."""

from __future__ import annotations

import dataclasses
import functools
import itertools
from test.random import parse_random_seed
from typing import Any, Optional

import numpy as np
import scipy.linalg
import scipy.optimize

from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.properties.electronic_energy import ElectronicBasis, ElectronicEnergy
from qiskit_nature.utils.linalg import modified_cholesky


@dataclasses.dataclass
class DoubleFactorizedHamiltonian:
    """A Hamiltonian in the double-factorized form of the low rank decomposition.
    See :func:`~.low_rank_decomposition` for a description of the data
    stored in this class.

    Attributes:
        one_body_tensor: The one-body tensor.
        leaf_tensors: The leaf tensors.
        core_tensors: The core tensors.
        constant: The constant.
        z_representation: Whether the Hamiltonian is in the "Z" representation.
    """

    one_body_tensor: np.ndarray
    leaf_tensors: np.ndarray
    core_tensors: np.ndarray
    constant: float
    z_representation: bool = False

    @property
    def n_orbitals(self):
        """The number of spatial orbitals."""
        return self.one_body_tensor.shape[0]

    @functools.cached_property
    def two_body_tensor(self):
        """The two-body tensor."""
        return np.einsum(
            "tpk,tqk,tkl,trl,tsl->pqrs",
            self.leaf_tensors,
            self.leaf_tensors,
            self.core_tensors,
            self.leaf_tensors,
            self.leaf_tensors,
        )

    def to_fermionic_op(self) -> FermionicOp:
        """Return a FermionicOp representing the Hamiltonian."""
        one_body_tensor = self.one_body_tensor
        two_body_tensor = self.two_body_tensor
        constant = self.constant

        if self.z_representation:
            one_body_tensor = one_body_tensor - 0.5 * (
                np.einsum(
                    "tij,tpi,tqi->pq",
                    self.core_tensors,
                    self.leaf_tensors,
                    self.leaf_tensors.conj(),
                )
                + np.einsum(
                    "tij,tpj,tqj->pq",
                    self.core_tensors,
                    self.leaf_tensors,
                    self.leaf_tensors.conj(),
                )
            )
            constant -= 0.25 * np.einsum("ijj->", self.core_tensors) - 0.5 * np.sum(
                self.core_tensors
            )

        terms: list[tuple[list[tuple[str, int]], complex]] = [([], constant)]
        for p, q in itertools.product(range(self.n_orbitals), repeat=2):
            coeff = one_body_tensor[p, q]
            for sigma in range(2):
                terms.append(
                    (
                        [
                            ("+", p + sigma * self.n_orbitals),
                            ("-", q + sigma * self.n_orbitals),
                        ],
                        coeff,
                    )
                )
        for p, q, r, s in itertools.product(range(self.n_orbitals), repeat=4):
            coeff = two_body_tensor[p, q, r, s]
            for sigma, tau in itertools.product(range(2), repeat=2):
                terms.append(
                    (
                        [
                            ("+", p + sigma * self.n_orbitals),
                            ("-", q + sigma * self.n_orbitals),
                            ("+", r + tau * self.n_orbitals),
                            ("-", s + tau * self.n_orbitals),
                        ],
                        0.5 * coeff,
                    )
                )

        return FermionicOp(terms, register_length=2 * self.n_orbitals, display_format="sparse")

    def to_z_representation(self) -> DoubleFactorizedHamiltonian:
        """Return the Hamiltonian in the "Z" representation."""
        if self.z_representation:
            return self

        one_body_correction, constant_correction = _low_rank_z_representation(
            self.leaf_tensors, self.core_tensors
        )
        return DoubleFactorizedHamiltonian(
            self.one_body_tensor + one_body_correction,
            self.leaf_tensors,
            self.core_tensors,
            self.constant + constant_correction,
            z_representation=True,
        )


def low_rank_decomposition(
    hamiltonian: ElectronicEnergy,
    *,
    truncation_threshold: float = 1e-8,
    max_rank: Optional[int] = None,
    z_representation: bool = False,
    optimize: bool = False,
    method: str = "L-BFGS-B",
    options: Optional[dict] = None,
    seed: Any = None,
    validate: bool = True,
    atol: float = 1e-8,
) -> DoubleFactorizedHamiltonian:
    r"""Low rank decomposition of a molecular Hamiltonian.

    The low rank decomposition acts on a Hamiltonian of the form

    .. math::

        H = \sum_{pq, \sigma} h_{pq} a^\dagger_{p, \sigma} a_{q, \sigma}
            + \frac12 \sum_{pqrs, \sigma} h_{pqrs, \sigma\tau} a^\dagger_{p, \sigma} a^\dagger_{r, \tau} a_{s, \tau} a_{q, \sigma}.

    The Hamiltonian is decomposed into the double-factorized form

    .. math::

        H = \sum_{pq, \sigma} \kappa_{pq} a^\dagger_{p, \sigma} a_{q, \sigma}
        + \frac12 \sum_t \sum_{ij, \sigma\tau} Z^{(t)}_{ij} n^{(t)}_{i, \sigma} n^{t}_{j, \tau}
        + \text{constant}.

    where

    .. math::

        n^{(t)}_{i, \sigma} = \sum_{pq} U^{(t)}_{pi} a^\dagger_{p, \sigma} a^\dagger_{q, \sigma} U^{(t)}_{qi}.

    Here :math:`U^{(t)}_{ij}` and :math:`Z^{(t)}_{ij}` are tensors that are output by the decomposition,
    and :math:`\kappa_{pq}` is an updated one-body tensor.
    Each matrix :math:`U^{(t)}` is guaranteed to be unitary so that the :math:`n^{(t)}_{i, \sigma}` are
    number operators in a rotated basis.
    The value :math:`t` is the "rank" of the decomposition and it is affected by two
    parameters: `truncation_threshold` and `max_rank`.
    The effect of `truncation_threshold` is that the core tensors :math:`Z^{(t)}` are elided
    in order of increasing one-norm (sum of absolute values of elements)
    until the maximum number are removed such that the
    sum of the one-norms of those removed does not exceed the threshold
    (the corresponding leaf tensors are also omitted).
    The `max_rank` parameter specifies an optional upper bound on :math:`t`.

    The default behavior of this routine is to perform a straightforward
    "exact" factorization of the two-body tensor based on a nested
    eigenvalue decomposition. Additionally, one can choose to optimize the
    coefficients stored in the tensor to achieve a "compressed" factorization.
    This option is enabled by setting the `optimize` parameter to `True`.
    The optimization attempts to minimize a least-squares objective function
    quantifying the error in the low rank decomposition.
    It uses `scipy.optimize.minimize`, passing both the objective function
    and its gradient.

    **"Z" representation**

    The "Z" representation of the low rank decomposition is an alternative
    decomposition that sometimes yields simpler quantum circuits.

    Under the Jordan-Wigner transformation, the number operators take the form

    .. math::

        n^{(t)}_{i, \sigma} = \frac{(1 - z^{(t)}_{i, \sigma})}{2}

    where :math:`z^{(t)}_{i, \sigma}` is the Pauli Z operator in the rotated basis.
    The "Z" representation is obtained by rewriting the two-body part in terms
    of these Pauli Z operators:

    .. math::

        H = \sum_{pq, \sigma} \kappa_{pq} a^\dagger_{p, \sigma} a_{q, \sigma}
        + \sum_{pq, \sigma} \tilde{\kappa}_{pq} a^\dagger_{p, \sigma} a_{q, \sigma}
        + \frac18 \sum_t \sum_{ij, \sigma\tau}^* Z^{(t)}_{ij} z^{(t)}_{i, \sigma} z^{t}_{j, \tau}
        + \text{constant}

    where the asterisk denotes summation over indices $ij, \sigma\tau$
    where $i \neq j$ or $\sigma \neq \tau$.
    Here :math:`\tilde{\kappa}_{pq}` is a correction to the one-body term.

    Note: Currently, only real-valued two-body tensors are supported.

    References:
        - `arXiv:1808.02625`_
        - `arXiv:2104.08957`_

    Args:
        hamiltonian: The Hamiltonian to decompose.
        truncation_threshold: The threshold for truncating the output tensors.
        max_rank: An optional limit on the number of terms to keep in the decomposition
            of the two-body tensor.
        z_representation: Whether to use the "Z" representation of the
            low rank decomposition.
        optimize: Whether to optimize the tensors returned by the decomposition.
        method: The optimization method. See the documentation of
            `scipy.optimize.minimize`_ for possible values.
        options: Options for the optimization. See the documentation of
            `scipy.optimize.minimize`_ for usage.
        seed: The pseudorandom number generator or seed. Randomness is used to generate
            an initial guess for the optimization.
            Should be an instance of `np.random.Generator` or else a valid input to
            `np.random.default_rng`.
        validate: Whether to check that the input tensors have the correct symmetries.
        atol: Absolute numerical tolerance for input validation.

    Returns:
        An instance of DoubleFactorizedHamiltonian which stores the decomposition in
        the attributes `one_body_tensor`, `leaf_tensors`, `core_tensors`,
        and `constant.

    Raises:
        ValueError: The input tensors do not have the correct symmetries.

    .. _arXiv:1808.02625: https://arxiv.org/abs/1808.02625
    .. _arXiv:2104.08957: https://arxiv.org/abs/2104.08957
    .. _scipy.optimize.minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """
    one_body_tensor = hamiltonian.get_electronic_integral(ElectronicBasis.MO, 1).get_matrix().copy()
    two_body_tensor = hamiltonian.get_electronic_integral(ElectronicBasis.MO, 2).get_matrix().copy()
    if validate:
        if not np.allclose(one_body_tensor, one_body_tensor.T.conj(), atol=atol):
            raise ValueError("One-body tensor must be hermitian.")

    one_body_tensor -= 0.5 * np.einsum("prqr", two_body_tensor)
    constant = 0.0

    if optimize:
        leaf_tensors, core_tensors = _low_rank_compressed_two_body_decomposition(
            two_body_tensor,
            max_rank=max_rank,
            truncation_threshold=truncation_threshold,
            method=method,
            options=options,
            seed=seed,
            validate=validate,
            atol=atol,
        )
    else:
        leaf_tensors, core_tensors = _low_rank_two_body_decomposition(
            two_body_tensor,
            max_rank=max_rank,
            truncation_threshold=truncation_threshold,
            validate=validate,
            atol=atol,
        )

    if z_representation:
        one_body_correction, constant_correction = _low_rank_z_representation(
            leaf_tensors, core_tensors
        )
        one_body_tensor += one_body_correction
        constant += constant_correction

    return DoubleFactorizedHamiltonian(
        one_body_tensor, leaf_tensors, core_tensors, constant, z_representation=z_representation
    )


def _low_rank_two_body_decomposition(  # pylint: disable=invalid-name
    two_body_tensor: np.ndarray,
    *,
    truncation_threshold: float = 1e-8,
    max_rank: Optional[int] = None,
    validate: bool = True,
    atol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    n_modes, _, _, _ = two_body_tensor.shape
    if max_rank is None:
        max_rank = n_modes ** (n_modes + 1) // 2
    reshaped_tensor = np.reshape(two_body_tensor, (n_modes**2, n_modes**2))

    if validate:
        if not np.all(np.isreal(reshaped_tensor)):
            raise ValueError("Two-body tensor must be real.")
        if not np.allclose(reshaped_tensor, reshaped_tensor.T, atol=atol):
            raise ValueError("Two-body tensor must be symmetric.")

    cholesky_vecs = modified_cholesky(two_body_tensor, max_vecs=max_rank)

    leaf_tensors = np.zeros((len(cholesky_vecs), n_modes, n_modes))
    core_tensors = np.zeros((len(cholesky_vecs), n_modes, n_modes))
    squared_one_norms = np.zeros(len(cholesky_vecs))
    for i, mat in enumerate(cholesky_vecs):
        inner_eigs, inner_vecs = np.linalg.eigh(mat)
        leaf_tensors[i] = inner_vecs
        core_tensors[i] = np.outer(inner_eigs, inner_eigs)
        squared_one_norms[i] = np.sum(np.abs(inner_eigs)) ** 2

    # sort by absolute value
    indices = np.argsort(squared_one_norms)
    squared_one_norms = squared_one_norms[indices]
    leaf_tensors = leaf_tensors[indices]
    core_tensors = core_tensors[indices]
    # get index to truncate at
    index = int(np.searchsorted(np.cumsum(squared_one_norms), truncation_threshold))
    # truncate, then reverse to put into descending order of absolute value
    leaf_tensors = leaf_tensors[index:][::-1]
    core_tensors = core_tensors[index:][::-1]

    return leaf_tensors, core_tensors


def _low_rank_z_representation(
    leaf_tensors: np.ndarray, core_tensors: np.ndarray
) -> tuple[np.ndarray, float]:
    r"""Compute "Z" representation of low rank decomposition."""
    one_body_correction = 0.5 * (
        np.einsum("tij,tpi,tqi->pq", core_tensors, leaf_tensors, leaf_tensors.conj())
        + np.einsum("tij,tpj,tqj->pq", core_tensors, leaf_tensors, leaf_tensors.conj())
    )
    constant_correction = 0.25 * np.einsum("ijj->", core_tensors) - 0.5 * np.sum(core_tensors)
    return one_body_correction, constant_correction


def _low_rank_optimal_core_tensors(
    two_body_tensor: np.ndarray, leaf_tensors: np.ndarray, cutoff_threshold: float = 1e-8
) -> np.ndarray:
    """Compute optimal low rank core tensors given fixed leaf tensors."""
    n_modes, _, _, _ = two_body_tensor.shape
    n_tensors, _, _ = leaf_tensors.shape

    dim = n_tensors * n_modes**2
    target = np.einsum(
        "pqrs,tpk,tqk,trl,tsl->tkl",
        two_body_tensor,
        leaf_tensors,
        leaf_tensors,
        leaf_tensors,
        leaf_tensors,
    )
    target = np.reshape(target, (dim,))
    coeffs = np.zeros((n_tensors, n_modes, n_modes, n_tensors, n_modes, n_modes))
    for i in range(n_tensors):
        for j in range(i, n_tensors):
            metric = (leaf_tensors[i].T @ leaf_tensors[j]) ** 2
            coeffs[i, :, :, j, :, :] = np.einsum("kl,mn->kmln", metric, metric)
            coeffs[j, :, :, i, :, :] = np.einsum("kl,mn->kmln", metric.T, metric.T)
    coeffs = np.reshape(coeffs, (dim, dim))

    eigs, vecs = np.linalg.eigh(coeffs)
    pseudoinverse = np.zeros_like(eigs)
    pseudoinverse[eigs > cutoff_threshold] = eigs[eigs > cutoff_threshold] ** -1
    solution = vecs @ (vecs.T @ target * pseudoinverse)

    return np.reshape(solution, (n_tensors, n_modes, n_modes))


def _low_rank_compressed_two_body_decomposition(  # pylint: disable=invalid-name
    two_body_tensor,
    *,
    truncation_threshold: float = 1e-8,
    max_rank: Optional[int] = None,
    method="L-BFGS-B",
    options: Optional[dict] = None,
    seed: Any = None,
    validate: bool = True,
    atol: float = 1e-8,
):
    rng = parse_random_seed(seed)
    leaf_tensors, _ = _low_rank_two_body_decomposition(
        two_body_tensor,
        truncation_threshold=truncation_threshold,
        max_rank=max_rank,
        validate=validate,
        atol=atol,
    )
    n_tensors, n_modes, _ = leaf_tensors.shape

    def fun(x):
        leaf_tensors, core_tensors = _params_to_df_tensors(x, n_tensors, n_modes)
        diff = two_body_tensor - np.einsum(
            "tpk,tqk,tkl,trl,tsl->pqrs",
            leaf_tensors,
            leaf_tensors,
            core_tensors,
            leaf_tensors,
            leaf_tensors,
        )
        return 0.5 * np.sum(diff**2)

    def jac(x):
        leaf_tensors, core_tensors = _params_to_df_tensors(x, n_tensors, n_modes)
        diff = two_body_tensor - np.einsum(
            "tpk,tqk,tkl,trl,tsl->pqrs",
            leaf_tensors,
            leaf_tensors,
            core_tensors,
            leaf_tensors,
            leaf_tensors,
        )
        grad_leaf = -4 * np.einsum(
            "pqrs,tqk,tkl,trl,tsl->tpk",
            diff,
            leaf_tensors,
            core_tensors,
            leaf_tensors,
            leaf_tensors,
        )
        leaf_logs = _params_to_leaf_logs(x, n_tensors, n_modes)
        grad_leaf_log = np.ravel(
            [_grad_leaf_log(log, grad) for log, grad in zip(leaf_logs, grad_leaf)]
        )
        grad_core = -2 * np.einsum(
            "pqrs,tpk,tqk,trl,tsl->tkl",
            diff,
            leaf_tensors,
            leaf_tensors,
            leaf_tensors,
            leaf_tensors,
        )
        for mat in grad_core:
            mat[range(n_modes), range(n_modes)] /= 2
        triu_indices = np.triu_indices(n_modes)
        grad_core = np.ravel([mat[triu_indices] for mat in grad_core])
        return np.concatenate([grad_leaf_log, grad_core])

    # TODO see if we can improve the intial guess
    core_tensors = _low_rank_optimal_core_tensors(two_body_tensor, leaf_tensors)
    x0 = _df_tensors_to_params(leaf_tensors, core_tensors)
    # TODO allow seeding the randomness
    x0 += 1e-2 * rng.standard_normal(size=x0.shape)
    result = scipy.optimize.minimize(fun, x0, method=method, jac=jac, options=options)
    leaf_tensors, _ = _params_to_df_tensors(result.x, n_tensors, n_modes)
    core_tensors = _low_rank_optimal_core_tensors(two_body_tensor, leaf_tensors)

    return leaf_tensors, core_tensors


def _df_tensors_to_params(leaf_tensors: np.ndarray, core_tensors: np.ndarray):
    _, n_modes, _ = leaf_tensors.shape
    leaf_logs = [scipy.linalg.logm(mat) for mat in leaf_tensors]
    triu_indices = np.triu_indices(n_modes, k=1)
    leaf_params = np.real(np.ravel([leaf_log[triu_indices] for leaf_log in leaf_logs]))
    triu_indices = np.triu_indices(n_modes)
    core_params = np.ravel([core_tensor[triu_indices] for core_tensor in core_tensors])
    return np.concatenate([leaf_params, core_params])


def _params_to_leaf_logs(params: np.ndarray, n_tensors: int, n_modes: int):
    leaf_logs = np.zeros((n_tensors, n_modes, n_modes))
    triu_indices = np.triu_indices(n_modes, k=1)
    param_length = len(triu_indices[0])
    for i in range(n_tensors):
        leaf_logs[i][triu_indices] = params[i * param_length : (i + 1) * param_length]
        leaf_logs[i] -= leaf_logs[i].T
    return leaf_logs


def _params_to_df_tensors(params: np.ndarray, n_tensors: int, n_modes: int):
    leaf_logs = _params_to_leaf_logs(params, n_tensors, n_modes)
    leaf_tensors = np.array([_expm_antisymmetric(mat) for mat in leaf_logs])

    n_leaf_params = n_tensors * n_modes * (n_modes - 1) // 2
    core_params = np.real(params[n_leaf_params:])
    triu_indices = np.triu_indices(n_modes)
    param_length = len(triu_indices[0])
    core_tensors = np.zeros((n_tensors, n_modes, n_modes))
    for i in range(n_tensors):
        core_tensors[i][triu_indices] = core_params[i * param_length : (i + 1) * param_length]
        core_tensors[i] += core_tensors[i].T
        core_tensors[i][range(n_modes), range(n_modes)] /= 2
    return leaf_tensors, core_tensors


def _expm_antisymmetric(mat: np.ndarray) -> np.ndarray:
    eigs, vecs = np.linalg.eigh(-1j * mat)
    return np.real(vecs @ np.diag(np.exp(1j * eigs)) @ vecs.T.conj())


def _grad_leaf_log(mat: np.ndarray, grad_leaf: np.ndarray) -> np.ndarray:
    eigs, vecs = np.linalg.eigh(-1j * mat)
    eig_i, eig_j = np.meshgrid(eigs, eigs, indexing="ij")
    with np.errstate(divide="ignore", invalid="ignore"):
        coeffs = -1j * (np.exp(1j * eig_i) - np.exp(1j * eig_j)) / (eig_i - eig_j)
    coeffs[eig_i == eig_j] = np.exp(1j * eig_i[eig_i == eig_j])
    grad = vecs.conj() @ (vecs.T @ grad_leaf @ vecs.conj() * coeffs) @ vecs.T
    grad -= grad.T
    n_modes, _ = mat.shape
    triu_indices = np.triu_indices(n_modes, k=1)
    return np.real(grad[triu_indices])
