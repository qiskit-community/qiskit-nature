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

"""Test for ElectronicIntegrals class"""

from __future__ import annotations

import unittest
from test import QiskitNatureTestCase

from ddt import ddt, idata

import numpy as np

from qiskit_nature.second_q.operators import ElectronicIntegrals, PolynomialTensor
import qiskit_nature.optionals as _optionals


@ddt
class TestElectronicIntegrals(QiskitNatureTestCase):
    """Tests for ElectronicIntegrals class"""

    def setUp(self) -> None:
        super().setUp()

        self.alpha = PolynomialTensor(
            {
                "+-": self.build_matrix(4, 2),
                "++--": self.build_matrix(4, 4),
            }
        )

        self.beta = PolynomialTensor(
            {
                "+-": self.build_matrix(4, 2, -1.0),
                "++--": self.build_matrix(4, 4, -1.0),
            }
        )

        self.beta_alpha = PolynomialTensor(
            {
                "++--": self.build_matrix(4, 4, 0.5),
            }
        )

        self.kronecker = PolynomialTensor(
            {
                "+-": np.array([[1, 0], [0, 1]]),
                "++--": np.fromiter(
                    [
                        0.5,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.5,
                        0.0,
                        0.0,
                        0.5,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.5,
                    ],
                    dtype=float,
                ).reshape((2, 2, 2, 2)),
            }
        )

    @staticmethod
    def build_matrix(dim_size, num_dim, val=1):
        """Build dictionary value matrix"""
        return (np.arange(1, dim_size**num_dim + 1) * val).reshape((dim_size,) * num_dim)

    def test_attributes(self):
        """Tests the various ElectronicIntegrals attributes."""

        with self.subTest("all empty"):
            ints = ElectronicIntegrals()
            self.assertTrue(isinstance(ints.alpha, PolynomialTensor))
            self.assertTrue(ints.alpha.is_empty())
            self.assertTrue(isinstance(ints.beta, PolynomialTensor))
            self.assertTrue(ints.beta.is_empty())
            self.assertTrue(isinstance(ints.beta_alpha, PolynomialTensor))
            self.assertTrue(ints.beta_alpha.is_empty())

        with self.subTest("pure alpha"):
            ints = ElectronicIntegrals(self.alpha)
            self.assertTrue(isinstance(ints.alpha, PolynomialTensor))
            self.assertTrue(ints.alpha.equiv(self.alpha))
            self.assertTrue(isinstance(ints.beta, PolynomialTensor))
            self.assertTrue(ints.beta.is_empty())
            self.assertTrue(isinstance(ints.beta_alpha, PolynomialTensor))
            self.assertTrue(ints.beta_alpha.is_empty())

        with self.subTest("all provided"):
            ints = ElectronicIntegrals(self.alpha, self.beta, self.beta_alpha)
            self.assertTrue(isinstance(ints.alpha, PolynomialTensor))
            self.assertTrue(ints.alpha.equiv(self.alpha))
            self.assertTrue(isinstance(ints.beta, PolynomialTensor))
            self.assertTrue(ints.beta.equiv(self.beta))
            self.assertTrue(isinstance(ints.beta_alpha, PolynomialTensor))
            self.assertTrue(ints.beta_alpha.equiv(self.beta_alpha))

        with self.subTest("alpha setter"):
            ints = ElectronicIntegrals(self.alpha, self.beta, self.beta_alpha)
            ints.alpha = None
            self.assertTrue(isinstance(ints.alpha, PolynomialTensor))
            self.assertTrue(ints.alpha.is_empty())

        with self.subTest("beta setter"):
            ints = ElectronicIntegrals(self.alpha, self.beta, self.beta_alpha)
            ints.beta = None
            self.assertTrue(isinstance(ints.beta, PolynomialTensor))
            self.assertTrue(ints.beta.is_empty())

        with self.subTest("beta_alpha setter"):
            ints = ElectronicIntegrals(self.alpha, self.beta, self.beta_alpha)
            ints.beta_alpha = None
            self.assertTrue(isinstance(ints.beta_alpha, PolynomialTensor))
            self.assertTrue(ints.beta_alpha.is_empty())

        with self.subTest("alpha_beta property"):
            ints = ElectronicIntegrals(self.alpha, self.beta, self.beta_alpha)
            alpha_beta = PolynomialTensor(
                {"++--": np.einsum("ijkl->klij", self.build_matrix(4, 4, 0.5))}
            )
            self.assertTrue(ints.alpha_beta.equiv(alpha_beta))

        if _optionals.HAS_SPARSE:
            import sparse as sp  # pylint: disable=import-error

            with self.subTest("sparse alpha_beta property"):
                beta_alpha = sp.random((2, 2, 2, 2), density=0.5)
                ints = ElectronicIntegrals(
                    beta_alpha=PolynomialTensor({"++--": beta_alpha}), validate=False
                )
                alpha_beta = PolynomialTensor(
                    {"++--": np.einsum("ijkl->klij", beta_alpha.todense())}
                )
                self.assertTrue(ints.alpha_beta.equiv(alpha_beta))

    def test_one_body(self):
        """Tests the one_body property."""
        ints = ElectronicIntegrals(self.alpha, self.beta, self.beta_alpha)
        one_body = ElectronicIntegrals(
            PolynomialTensor({"+-": self.build_matrix(4, 2)}),
            PolynomialTensor({"+-": -1.0 * self.build_matrix(4, 2)}),
        )
        self.assertTrue(ints.one_body.equiv(one_body))

    def test_two_body(self):
        """Tests the two_body property."""
        ints = ElectronicIntegrals(self.alpha, self.beta, self.beta_alpha)
        two_body = ElectronicIntegrals(
            PolynomialTensor({"++--": self.build_matrix(4, 4)}),
            PolynomialTensor({"++--": -1.0 * self.build_matrix(4, 4)}),
            PolynomialTensor({"++--": 0.5 * self.build_matrix(4, 4)}),
        )
        self.assertTrue(ints.two_body.equiv(two_body))

    @idata(np.linspace(0, 3, 5))
    def test_mul(self, other):
        """Test for scalar multiplication"""
        expected_prod_ints = ElectronicIntegrals(
            PolynomialTensor(
                {
                    "+-": self.build_matrix(4, 2, other),
                    "++--": self.build_matrix(4, 4, other),
                },
            ),
            PolynomialTensor(
                {
                    "+-": self.build_matrix(4, 2, -other),
                    "++--": self.build_matrix(4, 4, -other),
                },
            ),
            PolynomialTensor(
                {
                    "++--": self.build_matrix(4, 4, other / 2.0),
                },
            ),
        )

        result = other * ElectronicIntegrals(self.alpha, self.beta, self.beta_alpha)
        self.assertTrue(result.equiv(expected_prod_ints))

        with self.assertRaisesRegex(TypeError, r"other .* must be a number"):
            _ = ElectronicIntegrals(self.alpha) * ElectronicIntegrals(self.alpha)

    def test_add(self):
        """Test for addition of ElectronicIntegrals"""
        expected_sum_ints = ElectronicIntegrals(
            PolynomialTensor(
                {
                    "+-": self.build_matrix(4, 2, 2.0),
                    "++--": self.build_matrix(4, 4, 2.0),
                },
            ),
            PolynomialTensor(
                {
                    "+-": self.build_matrix(4, 2, -2.0),
                    "++--": self.build_matrix(4, 4, -2.0),
                },
            ),
            PolynomialTensor(
                {
                    "++--": self.build_matrix(4, 4),
                },
            ),
        )

        ints = ElectronicIntegrals(self.alpha, self.beta, self.beta_alpha)
        result = ints + ints
        self.assertTrue(result.equiv(expected_sum_ints))

        with self.assertRaisesRegex(
            TypeError, "Incorrect argument type: other should be ElectronicIntegrals"
        ):
            _ = ElectronicIntegrals(self.alpha) + 5

    def test_einsum(self):
        """Test ElectronicIntegrals.einsum"""
        one_body_a = np.random.random((2, 2))
        one_body_b = np.random.random((2, 2))
        two_body_aa = np.random.random((2, 2, 2, 2))
        two_body_bb = np.random.random((2, 2, 2, 2))

        coeffs_a = np.random.random((2, 2))
        coeffs_b = np.random.random((2, 2))

        with self.subTest("alpha only"):
            alpha = PolynomialTensor({"+-": one_body_a, "++--": two_body_aa})
            ints = ElectronicIntegrals(alpha)
            coeffs_pt = ElectronicIntegrals(PolynomialTensor({"+-": coeffs_a}))

            result = ElectronicIntegrals.einsum(
                {
                    "jk,ji,kl->il": ("+-", "+-", "+-", "+-"),
                    "pqrs,pi,qj,rk,sl->ijkl": ("++--", "+-", "+-", "+-", "+-", "++--"),
                },
                ints,
                coeffs_pt,
                coeffs_pt,
                coeffs_pt,
                coeffs_pt,
            )

            expected = ElectronicIntegrals(
                PolynomialTensor(
                    {
                        "+-": np.dot(np.dot(coeffs_a.T, one_body_a), coeffs_a),
                        "++--": np.einsum(
                            "pqrs,pi,qj,rk,sl->ijkl",
                            two_body_aa,
                            coeffs_a,
                            coeffs_a,
                            coeffs_a,
                            coeffs_a,
                            optimize=True,
                        ),
                    }
                ),
            )

            self.assertTrue(result.equiv(expected))

        with self.subTest("alpha ints with beta_alpha coeffs"):
            alpha = PolynomialTensor({"+-": one_body_a, "++--": two_body_aa})
            ints = ElectronicIntegrals(alpha)
            coeffs_pt = ElectronicIntegrals(
                PolynomialTensor({"+-": coeffs_a}), PolynomialTensor({"+-": coeffs_b})
            )

            result = ElectronicIntegrals.einsum(
                {
                    "jk,ji,kl->il": ("+-", "+-", "+-", "+-"),
                    "pqrs,pi,qj,rk,sl->ijkl": ("++--", "+-", "+-", "+-", "+-", "++--"),
                },
                ints,
                coeffs_pt,
                coeffs_pt,
                coeffs_pt,
                coeffs_pt,
            )

            expected = ElectronicIntegrals(
                PolynomialTensor(
                    {
                        "+-": np.dot(np.dot(coeffs_a.T, one_body_a), coeffs_a),
                        "++--": np.einsum(
                            "pqrs,pi,qj,rk,sl->ijkl",
                            two_body_aa,
                            coeffs_a,
                            coeffs_a,
                            coeffs_a,
                            coeffs_a,
                            optimize=True,
                        ),
                    }
                ),
                PolynomialTensor(
                    {
                        "+-": np.dot(np.dot(coeffs_b.T, one_body_a), coeffs_b),
                        "++--": np.einsum(
                            "pqrs,pi,qj,rk,sl->ijkl",
                            two_body_aa,
                            coeffs_b,
                            coeffs_b,
                            coeffs_b,
                            coeffs_b,
                            optimize=True,
                        ),
                    }
                ),
            )

            self.assertTrue(result.equiv(expected))

        with self.subTest("beta_alpha ints with beta_alpha coeffs"):
            alpha = PolynomialTensor({"+-": one_body_a, "++--": two_body_aa})
            beta = PolynomialTensor({"+-": one_body_b, "++--": two_body_bb})
            ints = ElectronicIntegrals(alpha, beta)
            coeffs_pt = ElectronicIntegrals(
                PolynomialTensor({"+-": coeffs_a}), PolynomialTensor({"+-": coeffs_b})
            )

            result = ElectronicIntegrals.einsum(
                {
                    "jk,ji,kl->il": ("+-", "+-", "+-", "+-"),
                    "pqrs,pi,qj,rk,sl->ijkl": ("++--", "+-", "+-", "+-", "+-", "++--"),
                },
                ints,
                coeffs_pt,
                coeffs_pt,
                coeffs_pt,
                coeffs_pt,
            )

            expected = ElectronicIntegrals(
                PolynomialTensor(
                    {
                        "+-": np.dot(np.dot(coeffs_a.T, one_body_a), coeffs_a),
                        "++--": np.einsum(
                            "pqrs,pi,qj,rk,sl->ijkl",
                            two_body_aa,
                            coeffs_a,
                            coeffs_a,
                            coeffs_a,
                            coeffs_a,
                            optimize=True,
                        ),
                    }
                ),
                PolynomialTensor(
                    {
                        "+-": np.dot(np.dot(coeffs_b.T, one_body_b), coeffs_b),
                        "++--": np.einsum(
                            "pqrs,pi,qj,rk,sl->ijkl",
                            two_body_bb,
                            coeffs_b,
                            coeffs_b,
                            coeffs_b,
                            coeffs_b,
                            optimize=True,
                        ),
                    }
                ),
            )

            self.assertTrue(result.equiv(expected))

    def test_from_raw_integrals(self):
        """Test from_raw_integrals utility method."""
        one_body_a = np.random.random((2, 2))
        one_body_b = np.random.random((2, 2))
        two_body_aa = np.random.random((2, 2, 2, 2))
        two_body_bb = np.random.random((2, 2, 2, 2))
        two_body_ba = np.random.random((2, 2, 2, 2))

        with self.subTest("alpha only"):
            ints = ElectronicIntegrals.from_raw_integrals(
                one_body_a, two_body_aa, auto_index_order=False
            )
            self.assertTrue(np.allclose(ints.alpha["+-"], one_body_a))
            self.assertTrue(np.allclose(ints.alpha["++--"], two_body_aa))

        with self.subTest("alpha and beta"):
            ints = ElectronicIntegrals.from_raw_integrals(
                one_body_a,
                two_body_aa,
                h1_b=one_body_b,
                h2_bb=two_body_bb,
                h2_ba=two_body_ba,
                auto_index_order=False,
            )
            self.assertTrue(np.allclose(ints.alpha["+-"], one_body_a))
            self.assertTrue(np.allclose(ints.beta["+-"], one_body_b))
            self.assertTrue(np.allclose(ints.alpha["++--"], two_body_aa))
            self.assertTrue(np.allclose(ints.beta["++--"], two_body_bb))
            self.assertTrue(np.allclose(ints.beta_alpha["++--"], two_body_ba))

    def test_second_q_coeffs(self):
        """Tests the total PolynomialTensor generation method."""
        one_body_a = np.random.random((2, 2))
        one_body_b = np.random.random((2, 2))
        two_body_aa = np.random.random((2, 2, 2, 2))
        two_body_bb = np.random.random((2, 2, 2, 2))
        two_body_ba = np.random.random((2, 2, 2, 2))

        with self.subTest("alpha only"):
            alpha = PolynomialTensor({"+-": one_body_a, "++--": two_body_aa})
            ints = ElectronicIntegrals(alpha)
            tensor = ints.second_q_coeffs()
            expected = PolynomialTensor(
                {
                    "+-": np.kron(self.kronecker["+-"], one_body_a),
                    "++--": np.kron(self.kronecker["++--"], two_body_aa),
                }
            )
            self.assertTrue(tensor.equiv(expected))

        with self.subTest("alpha and beta"):
            alpha = PolynomialTensor({"+-": one_body_a, "++--": two_body_aa})
            beta = PolynomialTensor({"+-": one_body_b, "++--": two_body_bb})
            beta_alpha = PolynomialTensor({"++--": two_body_ba})
            ints = ElectronicIntegrals(alpha, beta, beta_alpha)
            tensor = ints.second_q_coeffs()
            expected = {}
            one_zeros = np.zeros((2, 2))
            expected["+-"] = np.block([[one_body_a, one_zeros], [one_zeros, one_body_b]])
            two_kron = np.zeros((2, 2, 2, 2))
            two_body = np.zeros((4, 4, 4, 4))
            two_kron[(0, 0, 0, 0)] = 0.5
            two_body += np.kron(two_kron, two_body_aa)
            two_kron[(0, 0, 0, 0)] = 0.0
            two_kron[(1, 1, 1, 1)] = 0.5
            two_body += np.kron(two_kron, two_body_bb)
            two_kron[(1, 1, 1, 1)] = 0.0
            two_kron[(1, 0, 0, 1)] = 0.5
            two_body += np.kron(two_kron, two_body_ba)
            two_kron[(1, 0, 0, 1)] = 0.0
            two_kron[(0, 1, 1, 0)] = 0.5
            two_body += np.kron(two_kron, np.einsum("ijkl->klij", two_body_ba))
            two_kron[(0, 1, 1, 0)] = 0.0
            expected["++--"] = two_body
            self.assertTrue(tensor.equiv(PolynomialTensor(expected)))

    def test_trace_spin(self):
        """Test the trace_spin method."""
        one_body_a = np.random.random((2, 2))
        one_body_b = np.random.random((2, 2))
        two_body_aa = np.random.random((2, 2, 2, 2))
        two_body_bb = np.random.random((2, 2, 2, 2))
        two_body_ba = np.random.random((2, 2, 2, 2))

        with self.subTest("alpha only"):
            ints = ElectronicIntegrals.from_raw_integrals(
                one_body_a, two_body_aa, auto_index_order=False
            )
            tensor = ints.trace_spin()
            expected = PolynomialTensor(
                {
                    "+-": 2.0 * one_body_a,
                    "++--": 2.0 * two_body_aa,
                }
            )
            self.assertTrue(tensor.equiv(expected))

        with self.subTest("alpha and beta without beta_alpha"):
            ints = ElectronicIntegrals.from_raw_integrals(
                one_body_a,
                two_body_aa,
                h1_b=one_body_b,
                h2_bb=two_body_bb,
                auto_index_order=False,
            )
            tensor = ints.trace_spin()
            expected = PolynomialTensor(
                {
                    "+-": one_body_a + one_body_b,
                    "++--": 2.0 * (two_body_aa + two_body_bb),
                }
            )
            self.assertTrue(tensor.equiv(expected))

        with self.subTest("alpha and beta with beta_alpha"):
            ints = ElectronicIntegrals.from_raw_integrals(
                one_body_a,
                two_body_aa,
                h1_b=one_body_b,
                h2_bb=two_body_bb,
                h2_ba=two_body_ba,
                auto_index_order=False,
            )
            tensor = ints.trace_spin()
            expected = PolynomialTensor(
                {
                    "+-": one_body_a + one_body_b,
                    "++--": two_body_aa
                    + two_body_bb
                    + two_body_ba
                    + np.einsum("ijkl->klij", two_body_ba),
                }
            )
            self.assertTrue(tensor.equiv(expected))


if __name__ == "__main__":
    unittest.main()
