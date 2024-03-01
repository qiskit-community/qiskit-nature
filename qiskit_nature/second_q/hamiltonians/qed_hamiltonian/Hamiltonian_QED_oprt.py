"""Definition of the Hamiltonian for QED lattice NxN: operator expressions (Pauli op. for qiskit or sparse matrices))"""
from __future__ import annotations
import math
import warnings
from functools import reduce
import time
from itertools import permutations, product, combinations
import re
from typing import List
import numpy as np
import networkx as nx
from networkx import all_simple_paths, get_edge_attributes
from networkx.generators.classic import empty_graph
from networkx.utils import pairwise

# from qiskit.opflow import Z, X, Y, I, PauliSumOp, OperatorBase
import qiskit
from qiskit.quantum_info import SparsePauliOp, Pauli, Operator
from IPython.display import display
from scipy import special as sp
import matplotlib.pyplot as plt
from sympy import (
    Symbol,
    symbols,
    solve,
    lambdify,
    Mul,
    Eq,
    latex,
    expand,
    simplify,
    Pow,
    Float,
    Integer,
    cos,
)
from sympy.core.numbers import ImaginaryUnit
from sympy.physics.quantum.dagger import Dagger
from scipy.sparse.linalg import eigs
from scipy import sparse
import gc

SPARSE_PAULI = qiskit.quantum_info.operators.symplectic.sparse_pauli_op.SparsePauliOp

# from memory_profiler import profile#, memory_usage


class HamiltonianQED_oprt:

    """The algorithm computes the expression of the Quantum Electrodynamics (QED)
    Kogut-Susskind Hamiltonian,
    in terms of sparse matrices or in qiskit Pauli matrices for lattices
    from 1D to 3D. The latter formulation is suitable for quantum circuits.
    For fermionic degrees of freedom, the Jordan-Wigner transformation is applied.
    The discretisation of the group U(1) is done by means of group of integer numbers
    Z_(2L+1).In this definition, the Gray encoding is applied for the gauge fields.

    From an instance of a symbolic H the code generates the Hamiltonian
    in terms of operators.

    The final expression of the Hamiltonian is given in terms of  matrices or PauliOp,
    and it is written by
    following the order right-left, up-down, i.e. the first term is the one acting
    on the rightmost site.

    To define the Hamiltonian, the following parameters are needed in the
    "get_hamiltonian" function:

        g: float or int
        Coupling of the theory.

        m:float or int
            Mass term in the Hamiltonian.

        omega:float or int
            Factor in front of the kinetic Hamiltonian.

        fact_b_op:float or int
            Factor in front of the magnetic Hamiltonian.

        fact_e_op:float or int
            Factor in front of the electric Hamiltonian.

        lambd=int or float
        Parameter for the suppression factor in the Hamiltonian.

    Parameters
    ----------

    lattice: class
        Instance of the class Lattice.

    hamilt_sym: class
        Instance of the class HamiltonianQED_sym.

    n_sites: list
        Number of sites in each direction.

    l: int
        Truncation parameter. Defines how many values the gauge fields take,
        e.g. l=1 -> ±1,0 .

    ll: int
        Discretisation parameter L.

    encoding: str
        Encoding used for the gauge fields. Can be "gray" or "ed" (exact diagonalisation).

    magnetic_basis: bool
        If True, then the magnetic basis is considered, False for electric basis.

    pbc : bool
            If `pbc` is True, both dimensions are periodic. If False, none
            are periodic.

    puregauge: bool
        If False, then we have fermionic degrees of freedom in the system, if True only
        gauge fields.

    static_charges_values: dict or None
        If dict, set of static charges with coordinates and values,
        e.g. a charge Q=-1 in site
         (0,0) and a Q=1 in (1,0) will be: {(0,0):-1,(1,0):1}.

    e_op_out_plus:bool
        Boolean variable that specifies if the outgoing electric fields from a certain
        site have positive (True) or negative (False) sign. This definition influences
        the plaquette term and
        the kinetic term (H_k): if + sign then U^dag in H_k / if - then U in H_k.

    display_hamiltonian: bool
        If True, the Hamiltonian and the Gauss law equations are displayed in the output.

    sparse_pauli: bool
        If False, the Hamiltonian is returned in terms of SparsePauliOp, otherwise in terms
        of sparse matrices.

    """

    def __init__(
        self,
        lattice,
        hamilt_sym,
        n_sites: list,
        l: int,
        ll: int = 2,
        encoding: str = "gray",
        rotors: bool = False,  # TODO rotors
        magnetic_basis: bool = False,
        pbc: bool = False,
        puregauge: bool = False,
        static_charges_values: dict | None = None,
        e_op_out_plus: bool = False,
        sparse_pauli: bool = True,
    ) -> None:
        self.n_sites = n_sites
        self.pbc = pbc
        self.lattice = lattice
        self.hamilt_sym = hamilt_sym
        self.l_par = l
        self.ll_par = ll
        self.encoding = encoding
        self.rotors = rotors
        self.magnetic_basis = magnetic_basis
        self.puregauge = puregauge
        self.static_charges_values = static_charges_values
        self.e_op_out_plus = e_op_out_plus
        self.sparse_pauli = sparse_pauli

        if not self.sparse_pauli and self.encoding == "ed":
            raise ValueError("PauliSumOp not supported with exact diagonalization encoding")

        if self.magnetic_basis and self.ll_par <= self.l_par:
            raise ValueError("l must be smaller than L")

        if self.magnetic_basis and self.lattice.dims != 2:
            raise ValueError("Magnetic basis is only implemented for 2D lattices")

        # Pauli matrices
        self.Z = SparsePauliOp(Pauli("Z"))
        self.X = SparsePauliOp(Pauli("X"))
        self.Y = SparsePauliOp(Pauli("Y"))
        self.I = SparsePauliOp(Pauli("I"))

        self._symlist = ["I", "X", "Y", "Z", "Sd", "S-", "Su", "S+"]

        self.alpha = 2 * np.pi / (2 * self.ll_par + 1) if self.magnetic_basis else 0

        print("HamiltonianQED_oprt: Initializing...")
        # get the start time
        start_time = time.time()

        # list of dynamical and static charges
        self.str_node_f = (
            lambda node: str(node) if self.lattice.dims == 1 else "".join(map(str, node))
        )

        self.q_charge_str_list = [
            "q_" + self.str_node_f(node)
            for node in self.lattice.graph.nodes
            if self.puregauge is False
        ]
        self.static_charges_str_list = [
            "Q_" + self.str_node_f(node)
            for node in self.lattice.graph.nodes
            if self.static_charges_values is not None
        ]
        # Dictionary of all elements (E and charges q) in the system with their symbols
        if not self.rotors:
            self.e_op_dict = {
                s_tmp: symbols(s_tmp)
                for s_tmp in self.lattice.list_edges2_e_op
                + self.q_charge_str_list
                + self.static_charges_str_list
            }

            self.u_op_dict = {s_tmp: symbols(s_tmp) for s_tmp in self.lattice.list_edges2_u_op}
            self.rotor_list = []

        else:
            self.rotor_list = [
                "R_" + self.str_node_f(node) for node in self.lattice.graph.nodes
            ] + ["R_" + str(d) for i, d in zip(range(self.lattice.dims), ["x", "y", "z"])]
            self.e_op_dict = {
                s_tmp: symbols(s_tmp)
                for s_tmp in self.rotor_list + self.q_charge_str_list + self.static_charges_str_list
            }

            self.u_op_dict = {}  # TODO use P

        if not rotors:
            # e_op_free from solution of Guass equations and edges
            self.e_op_free = list(
                set([symbols(j) for j in self.lattice.list_edges2_e_op]).intersection(
                    set(
                        [
                            item
                            for sublist in [
                                eq.free_symbols for eq in self.hamilt_sym.sol_gauss.values()
                            ]
                            for item in sublist
                        ]
                    )
                )
            )
            # Build u_op_free from e_op_free and edges
            self.u_op_free = [
                k.subs(
                    [
                        (symbols(j), symbols(k))
                        for j, k in zip(
                            self.lattice.list_edges2_e_op, self.lattice.list_edges2_u_op
                        )
                    ]
                )
                for k in self.e_op_free
            ]
            self.u_op_free_dag = [
                k.subs(
                    [
                        (symbols(j), Symbol(k + "D"))
                        for j, k in zip(
                            self.lattice.list_edges2_e_op, self.lattice.list_edges2_u_op
                        )
                    ]
                )
                for k in self.e_op_free
            ]  # U^dag

            # length of e_op_free and u_op_free
            self.len_e_op = len(self.e_op_free)
            self.len_u_op = len(self.u_op_free)
            print("> e_op_free and u_op_free built")
        else:
            self.rotors_conversion()
            print("put rotors here")

        # Define the espressions for substituting symbols into Pauli strings

        # self._symbol_to_pauli()

        self.el_op_enc()
        self.u_op_enc()
        self.u_op_dag_enc()
        self._get_symbol_subs()
        print("> Pauli strings built")

        self.build_hamiltonian_tot()

        # get the end time
        end_time = time.time()
        # get the execution time
        elapsed_time = end_time - start_time
        print(
            ">> Hamiltonian built. ",
            "Execution time:",
            elapsed_time,
            "seconds",
        )

        start_time = time.time()
        self.hamiltonian_suppr()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(">> Suppression term built. ", "Execution time:", elapsed_time, "seconds")

        self.get_hamiltonian()

    def get_hamiltonian(
        self,
        g_var=1.0,
        m_var=1.0,
        omega=1.0,
        fact_b_op=1.0,
        fact_e_op=1.0,
        lambd=1000.0,
    ):
        """Returns the Hamiltonian of the system"""

        hamiltonian_tot = 0
        # Hamiltonian for fermions
        if self.puregauge:
            hamiltonian_tot += 0
        elif self.lattice.dims == 1 and self.sparse_pauli:
            hamiltonian_tot += (
                float(omega) * self.hamiltonian_k_pauli + float(m_var) * self.hamiltonian_m_pauli
            ).to_matrix(sparse=True)
        else:
            hamiltonian_tot += (
                float(omega) * self.hamiltonian_k_pauli + float(m_var) * self.hamiltonian_m_pauli
            )

        # Hamiltonian for gauge fields
        if self.len_e_op == 0 and self.puregauge:
            raise ValueError("No gauge fields in pure gauge theory")
        elif self.len_e_op == 0 and not self.puregauge:
            hamiltonian_tot += 0

        else:
            hamiltonian_tot += (
                -fact_b_op / (float((g_var) ** 2)) * self.hamiltonian_mag_pauli
                + fact_e_op * float((g_var) ** 2) * self.hamiltonian_el_pauli
            )

        # Hamiltonian for suppression term
        if lambd != 0:
            hamiltonian_tot += lambd * self.hamiltonian_suppress

        return hamiltonian_tot

    def _n_qubits_g(self) -> int:
        """Returns the minimum number of qubits required with Gray encoding"""

        return int(np.ceil(np.log2(2 * self.l_par + 1)))

    @staticmethod
    def sparse_sum(sparse_list):
        # Initialize the result matrix with zeros
        # result = sparse.csr_matrix((sparse_list[0].shape[0], sparse_list[0].shape[1]))
        # if sparse_list is generator
        first_matrix = next(sparse_list)
        result = sparse.csr_matrix((first_matrix.shape[0], first_matrix.shape[1]))

        for matrix in sparse_list:
            result += matrix

        return result

    # Tensor product of Pauli matrices
    def tensor_prod(self, pauli, power):
        """Returns tensor product of pauli operator with itself power times"""
        if power == 0:
            return 1
        elif power == 1:
            return pauli
        else:
            return pauli.tensor(self.tensor_prod(pauli, power - 1))

    # multiple tensor product of Pauli matrices
    @staticmethod
    def pauli_tns(*args):
        """Returns Pauli tensor product of all arguments. If int in args then it skips it.
        If all arguments are SparsePauliOp then it applies tensor method of SparsePauliOp.
        If not it applies kronecker product of numpy.(it works also with SparsePauliOp) but much slower.)
        """
        valid_args = [arg for arg in args if not isinstance(arg, int)]

        if len(valid_args) >= 2:
            if all([type(arg) == SPARSE_PAULI for arg in valid_args]):  # all SparsePauliOp
                return reduce(lambda x, y: x.tensor(y), valid_args)
            else:
                return reduce(
                    lambda x, y: sparse.kron(x, y, format="csr"),
                    valid_args,
                )

        elif len(valid_args) == 1:
            return valid_args[0]
        else:
            raise ValueError("Insufficient valid arguments for tensor product")

    # decompose sympy expression into a list of symbols and powers
    @staticmethod
    def decompose_expression(expr):
        """Decompose a sympy expression into a list of symbols and powers."""
        if isinstance(expr, (Symbol, Pow, Float, Integer)):  # single symbol or number
            return [expr]
        elif isinstance(expr, Mul):
            terms = expr.as_ordered_factors()
            result = [t if isinstance(t, (Symbol, Pow)) else t for t in terms]
            return result
        else:
            raise ValueError("Invalid expression type")

    # @profile
    def list_to_enc_hamilt(
        self, list_el, subst, ferm_lst=[], gauge_lst=[], encoding="gray", massterm=False
    ):  # list_el
        """Return a list of Pauli operators or list of matrices (depending on the encoding used) from a list of symbolic operators.
        It consider only single operator, not entire pauli string, thus operation like (I^op^I)**2 and I^op1^I*I^op2^I are
        then simplied to I^(op**2)^I and I^(op1*op2)^I, respectively.
        Last part is put everything together and add identity matrices where needed.

        encoding: gray, ed (exact diagonalization)"""
        ham_encoded = 0
        jj_mass = 0
        for ei in list_el:
            index_op = []
            for e in ei:  # build index list order ..q2q1q0 (little endian)
                if not isinstance(e, (int, float, complex, Float, Integer, str, ImaginaryUnit)):
                    if (
                        list(e.free_symbols)[0].name[-1] == "D"
                        and list(e.free_symbols)[0].name[0] == "U"
                    ):  # gauge field U adjoint
                        index_op.append(
                            str(
                                (
                                    ferm_lst[::-1] + [Symbol(i.name + "D") for i in gauge_lst][::-1]
                                ).index(*e.free_symbols)
                            )
                            + "D"
                        )
                    elif (
                        list(e.free_symbols)[0].name[-1] == "D"
                        and list(e.free_symbols)[0].name[0:3] == "Phi"
                    ):  # fermion adjoint (but JW index only 0, must cover all the fermionic dof)
                        index_op.append(
                            str(
                                (
                                    [Symbol(i.name + "D", commutative=False) for i in ferm_lst][
                                        ::-1
                                    ]
                                    + gauge_lst[::-1]
                                ).index(*e.free_symbols)
                            )
                            + "D"
                        )
                    else:  # no adjoint
                        index_op.append(
                            str((ferm_lst[::-1] + gauge_lst[::-1]).index(*e.free_symbols))
                        )

            # substitutions from symbols to dummy variables nameOP
            symb_el = lambdify(list(zip(*subst))[0], ei)(*list(zip(*subst))[1])

            # charges (JW dependent)
            q10 = -0.5 * (self.I + self.Z)
            q00 = 0.5 * (self.I - self.Z)

            # substitutions from symbols to matrices
            sym_list_tomatrix = [
                (Symbol("q10OP"), q10),
                (Symbol("q00OP"), q00),
                (Symbol("EOP"), self.e_oper),
                (Symbol("UOP"), self.u_oper),
                (Symbol("UdagOP"), self.u_oper_dag),
            ]

            if self.magnetic_basis:  # U->exp(-i*alpha*E), U_dag->exp(i*alpha*E) in mag basis
                ei_class = lambda fct: self.matx_exp(fct * self.e_oper, 1j * self.alpha)
                sym_list_tomatrix += [
                    (Symbol("exppiEOP"), ei_class(1)),
                    (Symbol("expmiEOP"), ei_class(-1)),
                ]

            if not self.puregauge:  # add fermions subst
                sym_list_tomatrix += self.phi_jw_list

            pauli_ei = lambdify(list(zip(*sym_list_tomatrix))[0], symb_el)(
                *list(zip(*sym_list_tomatrix))[1]
            )

            op_dct = {}
            numbers = []
            ct = 0
            for el in pauli_ei:  # build dictionary of sparse pauli operators and their index
                if isinstance(
                    el,
                    (
                        SparsePauliOp,
                        np.ndarray,
                        sparse._csr.csr_matrix,
                        sparse._coo.coo_matrix,
                    ),
                ):
                    op_dct[index_op[ct]] = el
                    ct += 1
                else:
                    numbers.append(el)

            # build final list of operators: res. It is built as list of strings and then filled with matrices
            if subst[0][0] == Symbol("Phi_1D", commutative=False):  # ferm
                res = ["id_f"] * len(ferm_lst) + ["id_g"] * self.len_e_op
                f_index_op = [
                    i for i in index_op if int(re.findall("\d+", i)[0]) < len(ferm_lst)
                ]  # select only fermionic dof
                res[0] = (
                    op_dct[f_index_op[0]] @ op_dct[f_index_op[1]]
                )  # compute product between fermions dof when JW applied
                start = len(ferm_lst)

            else:  # no JW
                res = ["id_q"] * len(ferm_lst) + ["id_g"] * self.len_e_op
                start = 0

            del pauli_ei, index_op
            gc.collect()

            # fill res with matrices. Last steps, empty spots are filled with identity matrices
            for i in range(start, len(res)):  # only for gauge or charges q
                if str(i) in op_dct.keys() and isinstance(res[i], str):  # fill res with SparsePauli
                    res[i] = op_dct[str(i)]
                if str(i) + "D" in op_dct.keys() and isinstance(
                    res[i], str
                ):  # fill res with SparsePauli
                    res[i] = op_dct[str(i) + "D"]

                if (
                    isinstance(res[i], str) and res[i] == "id_q"
                ):  # remaining spots for charges are filled with I
                    res[i] = self.I  # single qubit for charge
                    if self.sparse_pauli:
                        res[i] = res[i].to_matrix(sparse=True)

                elif isinstance(res[i], str) and res[i] == "id_g":
                    if encoding == "gray":
                        res[i] = self.tensor_prod(self.I, (self._n_qubits_g()))
                        if self.sparse_pauli:
                            res[i] = res[i].to_matrix(sparse=True)  # Gray encoding for E field
                    elif encoding == "ed":  # exact diagonaliz. dimensions of gauge fields 2l+1
                        res[i] = sparse.eye(2 * self.l_par + 1, format="csr")

            res = (elem for elem in res if not isinstance(elem, str))  # remove id_f when JW applied

            if massterm:  # sum over all terms for mass hamiltonian
                ham_encoded += (
                    ((-1) ** jj_mass) * np.prod(numbers) * HamiltonianQED_oprt.pauli_tns(*res)
                )  # reduce(tensor_or_kron,res )
                jj_mass += 1
            else:  # sum over all terms
                # TODO memory problematic part
                ham_encoded += np.prod(numbers) * HamiltonianQED_oprt.pauli_tns(
                    *res
                )  # reduce(tensor_or_kron,res )

            del res, op_dct, numbers
            gc.collect()

        return ham_encoded

    # @profile
    def tensor_or_kron(self, x, y):
        if isinstance(x, SPARSE_PAULI) and isinstance(y, SPARSE_PAULI):
            return x.tensor(y)
        else:
            return sparse.kron(x, y, format="csr")

    # @profile
    def pauli_tns2(self, *args):
        """Returns Pauli tensor product of all arguments. If int in args then it skips it.
        If all arguments are SparsePauliOp then it applies tensor method of SparsePauliOp.
        If not it applies kronecker product of numpy.(it works also with SparsePauliOp) but much slower.)
        """
        valid_args = [arg for arg in args if not isinstance(arg, int)]

        if len(valid_args) >= 2:
            if all([type(arg) == SPARSE_PAULI for arg in valid_args]):  # all SparsePauliOp
                return reduce(lambda x, y: x.tensor(y), valid_args)
            else:
                # result = valid_args[0]
                # for matrix in valid_args[1:]:
                #     result = sparse.kron(result, matrix, format="csr")
                # return result
                return reduce(
                    lambda x, y: sparse.kron(x, y, format="csr"),
                    valid_args,
                )

        elif len(valid_args) == 1:
            return valid_args[0]
        else:
            raise ValueError("Insufficient valid arguments for tensor product")

    def jw_funct(self, n_tmp: int, n_qubits: int):
        """Jordan-Wigner for 2 terms phi^dag, phi

        Inputs:
            n_tmp: index of fermionic operator

            n_qubits: n.er of total qubits in the string

        """
        # sgm = PauliSumOp( SparsePauliOp.from_sparse_list( [ ( "X", [ 0, ], 0.5, ), ] + [ ( "Y", [ 0, ], (-0.5j), ), ], num_qubits=1, ) )
        # sgp = PauliSumOp( SparsePauliOp.from_sparse_list( [ ( "X", [ 0, ], 0.5, ), ] + [ ( "Y", [ 0, ], (0.5j), ), ], num_qubits=1, ) )
        sgm = SparsePauliOp.from_sparse_list(
            [
                (
                    "X",
                    [
                        0,
                    ],
                    0.5,
                ),
            ]
            + [
                (
                    "Y",
                    [
                        0,
                    ],
                    (-0.5j),
                ),
            ],
            num_qubits=1,
        )
        sgp = SparsePauliOp.from_sparse_list(
            [
                (
                    "X",
                    [
                        0,
                    ],
                    0.5,
                ),
            ]
            + [
                (
                    "Y",
                    [
                        0,
                    ],
                    (0.5j),
                ),
            ],
            num_qubits=1,
        )

        assert n_tmp > 0
        if n_tmp == 1:
            jw_dagk = self.tensor_prod(self.I, 0)  # (I) ^ 0
            jwk = self.tensor_prod(self.I, 0)  # (I) ^ 0

        else:
            jw_dagk = ((1j) ** (n_tmp - 1)) * self.tensor_prod(self.Z, (n_tmp - 1))
            jwk = ((-1j) ** (n_tmp - 1)) * self.tensor_prod(self.Z, (n_tmp - 1))

        jw_dag = HamiltonianQED_oprt.pauli_tns(
            self.tensor_prod(self.I, (n_qubits - n_tmp)), (sgm), (jw_dagk)
        )
        jw_nodag = HamiltonianQED_oprt.pauli_tns(
            self.tensor_prod(self.I, (n_qubits - n_tmp)), (sgp), (jwk)
        )

        return jw_dag, jw_nodag  # then use: jw_dag@jw_nodag for phi^dag phi

    # utilities and operators
    def _gray_map(self):
        """Gray map dictionary for a certain value of the truncation parameter l.
        for example if l = 1, it returns:{-1: '00', 0: '01', 1: '11'}"""
        ret = {}
        for i in range(0, 2 * self.l_par + 1):
            gray_decimal = i ^ (i >> 1)
            ret[i - self.l_par] = "{0:0{1}b}".format(gray_decimal, self._n_qubits_g())
        return ret

    # Transition map for encoding
    @staticmethod
    def _trans_map(string_1, string_2):
        """Transition map for encoding.
        Parameters:

          string_1: string of the state (output of gray_map)
          string_2: string of the state (output of gray_map)

          Example ouf output with psi_phys = 1
                                  psi_Gray =  '11'
                                  -> output = [6, 6]
        """
        op_dict = {
            ("0", "0"): 4,  # Sd
            ("0", "1"): 5,  # S-
            ("1", "1"): 6,  # Su
            ("1", "0"): 7,  # S+
        }
        n_tmp = len(string_1)
        assert n_tmp == len(string_2)
        symb_list = [op_dict[(a1, a2)] for a1, a2 in zip(string_1, string_2)]
        return symb_list

    # rotator-string/Electric field operator
    def _r_c(self):
        """Rotator-string/Electric field operator.
        Function equivalent to S_z term.
        Defined for Gray encoding"""

        states_list = list(range(-self.l_par, self.l_par + 1))  # states: -l, ..., l
        e_op_list = []
        for st_fact in states_list:
            if st_fact != 0:
                # using the fact that the superposition encoding is only for st_fact=0
                e_op_list.append(
                    [
                        st_fact,
                        *HamiltonianQED_oprt._trans_map(
                            self._gray_map()[st_fact], self._gray_map()[st_fact]
                        ),
                    ]
                )
        e_oper = [
            [
                v_elem[0],
                [
                    f"{self._symlist[s_tmp]}_{self._n_qubits_g()-i-1}"
                    for i, s_tmp in enumerate(v_elem[1:])
                ],
            ]
            for v_elem in e_op_list
        ]
        e_oper_str = [
            f"{v_elem[0]} "
            + " ".join(
                f"{self._symlist[s_tmp]}_{self._n_qubits_g()-i-1}"
                for i, s_tmp in enumerate(v_elem[1:])
            )
            for v_elem in e_op_list
        ]
        return e_oper, e_oper_str

    # (downward) ladder operator
    def _l_c(self):
        """Ladder function equivalent to V^- terms
        Defined for Gray encoding"""
        states_list = list(range(-self.l_par, self.l_par + 1))  # states: -l, ..., l

        u_op_list = []
        for st_fact in states_list:
            if st_fact != -self.l_par:
                encs = [self._gray_map()[st_fact - 1], self._gray_map()[st_fact]]
                are_tuples = list(map(lambda x: isinstance(x, tuple), encs))
                nterms = 1 << sum(are_tuples)
                fact = 1.0 / np.sqrt(nterms)
                for ei_tmp, enc in enumerate(encs):
                    if not are_tuples[ei_tmp]:
                        encs[ei_tmp] = (enc,)
                for e1_tmp in encs[0]:
                    for e2_tmp in encs[1]:
                        u_op_list.append([fact, *HamiltonianQED_oprt._trans_map(e1_tmp, e2_tmp)])
        u_oper = [
            f"{v_elem[0]} "
            + " ".join(
                f"{self._symlist[s_tmp]}_{self._n_qubits_g()-i-1}"
                for i, s_tmp in enumerate(v_elem[1:])
            )
            for v_elem in u_op_list
        ]

        return u_oper

    # @staticmethod
    def str_to_pauli(self, lst: list, n_tmp: int):
        """
        Returns PauliSumOp object.
        Output order of Pauli matrices from right to left: ..q2q1q0

        Inputs

        lst:list of strings of operators in the following opdict
        n_tmp: num_qubits

        Rules:
        Hamiltonian terms made of tensor products of pauli operators as a list of
        (f-)strings.
        Each string contains two parts: 1st encodes the overall coefficient, 2nd one
        encodes the tensor product of operators in the set {I,X,Y,Z,S+,S-,Su,Sd}
        (S± are (X ∓ j Y)/2, while Su=(I-Z)/2 and Sd=(I+Z)/2)
        and the nth-qubit on which they act in terms of a space-separated sequence
        of underscore-separated operator-position pairs, e.g. $Z_2 X_5 S-_3$.
        Note: difference with usual sigma^±. In order to have:
         S+|0>=|1>, S-|1>=|0>  (|0>=(1,0), |1>= (0,1))
        """
        opdict = {
            "S+": [
                ("X", 0.5),
                ("Y", -0.5j),
            ],
            "S-": [("X", 0.5), ("Y", 0.5j)],
            "Su": [("I", 0.5), ("Z", -0.5)],
            "Sd": [("I", 0.5), ("Z", 0.5)],
        }
        pauli_res = 0
        for s_tmp in lst:
            facts = s_tmp.split(" ")
            splitted_facts = list(
                map(lambda x: (int(x[1]), x[0]), map(lambda x: x.split("_"), facts[1:]))
            )

            sparse_prod = self.tensor_prod(self.I, n_tmp)  # I ^ (n_tmp)
            #
            for st_fact in splitted_facts:
                sparse_prod @= SparsePauliOp.from_sparse_list(
                    [
                        (
                            p_tmp[0],
                            [
                                st_fact[0],
                            ],
                            p_tmp[1],
                        )
                        for p_tmp in opdict[st_fact[1]]
                    ],
                    num_qubits=n_tmp,
                )
            # pauli_sum = PauliSumOp(sparse_prod, coeff=complex(facts[0]))

            # pauli_res += pauli_sum
            sparse_prod.coeffs *= complex(facts[0])
            pauli_res += sparse_prod.simplify()

        return pauli_res  # .reduce()

    def el_op_enc(self):
        """Return the encoding of the electric field operator in the chosen encoding"""
        if self.encoding == "gray":
            self.e_oper = self.str_to_pauli(self._r_c()[1], self._n_qubits_g())
            if self.sparse_pauli:
                self.e_oper = self.e_oper.to_matrix(sparse=True)
        elif self.encoding == "ed":
            self.e_oper = sparse.diags(
                np.arange(-self.l_par, self.l_par + 1), format="csr"
            )  # NB: the operator are all sparse since power matrix M@M=M**2 does not work for non-sparse matrices (i.e. if non-sparse it does power element-wise))
        else:
            raise ValueError("encoding not recognized")

    def u_op_dag_enc(self):
        """LOWERING OPERATOR.Return the encoding of the link operator in the chosen encoding"""
        if self.encoding == "gray":
            self.u_oper_dag = self.str_to_pauli(self._l_c(), self._n_qubits_g())
            if self.sparse_pauli:
                self.u_oper_dag = self.u_oper_dag.to_matrix(sparse=True)
        elif self.encoding == "ed":
            self.u_oper_dag = sparse.diags([1] * (2 * self.l_par), 1, format="csr")

            # size_op = 2 * self.l_par + 1
            # u_ed = np.zeros((size_op, size_op))
            # # Fill the upper diagonal with 1s: U
            # for i in range(size_op - 1):
            #     u_ed[i, i + 1] = 1
            # self.u_oper_dag = sparse.csr_matrix(
            #     u_ed
            # )  # NB: the operator are all sparse since power matrix M@M=M**2 does not work for non-sparse matrices (i.e. if non-sparse it does power element-wise))
        else:
            raise ValueError("encoding not recognized")

    def u_op_enc(self):
        """RAISING OPERATOR. Return the encoding of the link operator dagger in the chosen encoding"""
        if self.encoding == "gray":
            self.u_oper = self.str_to_pauli(self._l_c(), self._n_qubits_g()).adjoint()
            if self.sparse_pauli:
                self.u_oper = self.u_oper.to_matrix(sparse=True)

        elif self.encoding == "ed":
            self.u_oper = sparse.diags([1] * (2 * self.l_par), -1, format="csr")

            # u_ed_dag = np.zeros((2 * self.l_par + 1, 2 * self.l_par + 1))
            # # Fill the lower diagonal with 1s: U_dag
            # for i in range(2 * self.l_par):
            #     u_ed_dag[i + 1, i] = 1
            # self.u_oper = sparse.csr_matrix(
            #     u_ed_dag
            # )  # NB: the operator are all sparse since power matrix M@M=M**2 does not work for non-sparse matrices (i.e. if non-sparse it does power element-wise))
        else:
            raise ValueError("encoding not recognized")

    @staticmethod
    def hermitian_c(expr):
        """Compute hermitian conjugate of input expr."""

        if isinstance(expr, np.ndarray):
            return np.conj(expr).T
        elif isinstance(expr, (sparse._csr.csr_matrix, sparse._coo.coo_matrix)):
            return expr.transpose().conjugate()
        elif isinstance(expr, SparsePauliOp):
            return expr.adjoint()
        else:
            raise ValueError("encoding not recognized for hermitian conjugate")

    def matx_exp(self, matrix, coefficient):
        """Compute the matrix exponential of a matrix using the eigendecomposition
        Input arguments:operator SparsePauliOp
        coefficient = coefficient of the exponential"""
        if isinstance(matrix, (sparse._csr.csr_matrix, sparse._coo.coo_matrix)):
            matrix = matrix.toarray()
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        diagonal_matrix = np.diag(np.exp(coefficient * eigenvalues))
        exp_matrix = eigenvectors @ diagonal_matrix @ np.linalg.inv(eigenvectors)
        if isinstance(matrix, (np.ndarray, sparse._csr.csr_matrix, sparse._coo.coo_matrix)):
            return sparse.csr_matrix(exp_matrix)
        elif isinstance(matrix, SparsePauliOp):
            return SparsePauliOp.from_operator(
                Operator(exp_matrix)
            )  # NB attention this operation is exponential

    def cos_oper(self, exp_matrixp):
        """Compute the cosine of a matrix using Euler formula,
        cos(operators)=exp(i*operators)+exp(-i*operators)/2

        """
        if isinstance(exp_matrixp, np.ndarray):
            return (exp_matrixp + exp_matrixp.conj().T) / 2
        elif isinstance(exp_matrixp, (sparse._csr.csr_matrix, sparse._coo.coo_matrix)):
            return (exp_matrixp + exp_matrixp.conjugate().transpose()) / 2
        elif isinstance(exp_matrixp, SparsePauliOp):
            return (exp_matrixp + exp_matrixp.adjoint()) / 2
        else:
            raise ValueError("encoding not recognized")

    def sin_oper(self, exp_matrixp):
        """Compute the sine of a matrix using Euler formula,
        sin(operators)=exp(i*operators)-exp(-i*operators)/2

        """
        if isinstance(exp_matrixp, np.ndarray):
            return (exp_matrixp - exp_matrixp.conj().T) / 2j
        elif isinstance(exp_matrixp, (sparse._csr.csr_matrix, sparse._coo.coo_matrix)):
            return (exp_matrixp - exp_matrixp.conjugate().transpose()) / 2j
        elif isinstance(exp_matrixp, SparsePauliOp):
            return (exp_matrixp - exp_matrixp.adjoint()) / 2j
        else:
            raise ValueError("encoding not recognized")

    def rotors_conversion(self):  # TODO test this function
        rotors_dict = {}
        for (
            s
        ) in (
            self.lattice.list_edges2_e_op
        ):  # index of rotors is the bottom left index (nx,ny coordinates)
            coord_s = re.findall(r"\d+", s)[0]
            if coord_s[0] == "0" and coord_s[1] == str(self.n_sites[1]):  # if nx==0 and ny!=n_y_max
                eop_tmp = 0
            else:
                eop_tmp = Symbol("R_" + coord_s)

            if s[-1] == "x":
                if coord_s[1] == "0":  # ny==0
                    eop_tmp += Symbol("R_x")
                    if self.pbc:  # only if pbc in y direction
                        eop_tmp -= Symbol("R_" + coord_s[0] + str(self.n_sites[1] - 1))

                else:
                    eop_tmp -= Symbol("R_" + coord_s[0] + str(int(coord_s[1]) - 1))

                if not self.puregauge:
                    q_tmp = -sum(
                        [
                            Symbol("q_" + str(x) + str(y))
                            for x in range(int(coord_s[0]) + 1, self.n_sites[0])
                            for y in range(self.n_sites[1])
                            if int(coord_s[1]) == 0
                        ]
                    )
                    eop_tmp += q_tmp

            elif s[-1] == "y":
                eop_tmp *= -1
                if coord_s[0] == "0":  # if n_x ==0
                    eop_tmp += Symbol("R_y")
                    if self.pbc:  # only if pbc in x direction
                        eop_tmp += Symbol("R_" + str(self.n_sites[0] - 1) + coord_s[1])
                else:
                    eop_tmp += Symbol("R_" + str(int(coord_s[0]) - 1) + coord_s[1])

                if not self.puregauge:
                    q_tmp = -sum(
                        [
                            Symbol("q_" + str(x) + str(y))
                            for x in range(self.n_sites[0])
                            for y in range(int(coord_s[1]) + 1, self.n_sites[1])
                            if x == int(coord_s[0])
                        ]
                    )
                    eop_tmp += q_tmp

            # if coordinates of R_xy are 0 and n_sites-1, then R is 0 convenient to fix this to zero
            for i in eop_tmp.free_symbols:
                if i.name[0] == "R" and i.name[2] == "0":
                    if i.name[3] == str(self.n_sites[1] - 1):
                        eop_tmp = eop_tmp.subs(i, 0)

            rotors_dict[s] = eop_tmp

        self.rotors_dict = rotors_dict

    def _get_symbol_subs(self):
        """Return list of substitutions for symbols in the Hamiltonian.
        Suitable for every encoding (gauge fields) defined in el_op_enc(), u_op_enc(), u_op_dag_enc()
        """
        # list of symbols only (encoding not needed)
        self.eop_list = self.e_op_free if not self.rotors else self.rotor_list
        self.uop_list = self.u_op_free if not self.rotors else []
        self.qop_list = (
            []
            if self.puregauge
            else [symbols("q_" + self.str_node_f(k)) for k in self.lattice.jw_sites]
        )
        # self.static_qop_list = (
        #     []
        #     if self.static_charges_values is None
        #     else [symbols("Q_" + self.str_node_f(k)) for k in self.lattice.jw_sites]
        # )
        # # self.qop_list += self.static_qop_list

        self.phiop_list = [
            Symbol(f"Phi_{i+1}", commutative=False) for i, k in enumerate(self.lattice.jw_sites)
        ]
        # list of symbols and operators
        # q10 = -0.5 * (self.I + self.Z)  # JW dependent
        # q00 = 0.5 * (self.I - self.Z)

        sum_k = lambda k: k if self.lattice.dims == 1 else sum(k)

        # self.qcharge_list = [
        #     (symbols("q_" + self.str_node_f(k)), (q10 if sum_k(k) % 2 else q00))
        #     for k in self.lattice.jw_sites
        # ]
        self.qcharge_list = [
            (
                symbols("q_" + self.str_node_f(k)),
                (Symbol("q10OP") if sum_k(k) % 2 else Symbol("q00OP")),
            )
            for k in self.lattice.jw_sites
        ]
        # if self.static_charges_values is None:
        #     self.static_qcharge_list = []
        # else:
        #     self.static_qcharge_list = [
        #         (
        #             Symbol("Q_" + self.str_node_f(node)),
        #             self.static_charges_values[node] * self.I,
        #         )
        #         if node in self.static_charges_values.keys()
        #         else (Symbol("Q_" + self.str_node_f(node)), 0 * self.I)
        #         for node in self.lattice.graph.nodes
        #     ]
        # # self.qcharge_list += self.static_qcharge_list

        self.e_field_list = [(s_tmp, Symbol("EOP")) for s_tmp in self.eop_list]

        self.u_field_list = [
            (s_tmp, (Symbol("UdagOP") if s_tmp.name[-1] == "D" else Symbol("UOP")))
            for s_tmp in self.uop_list + self.u_op_free_dag
        ]

        # self.e_field_list = [(s_tmp, self.e_oper) for s_tmp in self.eop_list]
        # self.u_field_list = [
        #     (s_tmp, (self.u_oper_dag if s_tmp.name[-1] == "D" else self.u_oper))
        #     for s_tmp in self.uop_list + self.u_op_free_dag
        # ]
        # ei_class = lambda fct: self.matx_exp(
        #     fct * self.e_oper, 1j * self.alpha
        # )  # compute exp(i*alpha*E)
        # U for mag basis (used in H_kinetic_mag)
        if self.magnetic_basis:
            # self.u_field_list_mag = [
            #     (s_tmp, (ei_class(1) if s_tmp.name[-1] == "D" else ei_class(-1)))
            #     for s_tmp in self.uop_list + self.u_op_free_dag
            # ]  # U->exp(-i*alpha*E), U_dag->exp(i*alpha*E) in mag basis
            self.u_field_list_mag = [
                (s_tmp, (Symbol("exppiEOP") if s_tmp.name[-1] == "D" else Symbol("expmiEOP")))
                for s_tmp in self.uop_list + self.u_op_free_dag
            ]

        # dummy ferm list:
        self.phi_jw_list_sym = [
            (Symbol(f"Phi_{i+1}D", commutative=False), Symbol(f"Phi_{i+1}D", commutative=False))
            for i, k in enumerate(self.lattice.jw_sites)
        ] + [
            (Symbol(f"Phi_{i+1}", commutative=False), Symbol(f"Phi_{i+1}", commutative=False))
            for i, k in enumerate(self.lattice.jw_sites)
        ]

        phi_el = lambda i, j: HamiltonianQED_oprt.pauli_tns(
            (self.jw_funct(i + 1, int(self.lattice.n_sitestot))[j]),
        )
        self.phi_jw_list = [
            (
                Symbol(f"Phi_{i+1}D", commutative=False),
                phi_el(i, 0),
            )
            for i, k in enumerate(self.lattice.jw_sites)
        ] + [
            (
                Symbol(f"Phi_{i+1}", commutative=False),
                phi_el(i, 1),
            )
            for i, k in enumerate(self.lattice.jw_sites)
        ]

    # HAMILTONIAN
    # @profile
    def build_hamiltonian_tot(self):
        """Builds the total Hamiltonian of the system."""
        # ************************************  H_E   ************************************
        if self.len_e_op > 0:
            if self.magnetic_basis:
                # Pauli expression, since mag basis H_E is in terms of U and U^dag we use u_op_field_subs
                hamiltonian_el_pauli = self.list_to_enc_hamilt(
                    [self.decompose_expression(i) for i in self.hamilt_sym.hamiltonian_el_subs],
                    self.qcharge_list + self.u_field_list,
                    self.qop_list,
                    self.uop_list,
                    encoding=self.encoding,
                )
            else:
                hamiltonian_el_pauli = self.list_to_enc_hamilt(
                    (i.as_ordered_factors() for i in self.hamilt_sym.hamiltonian_el_subs),
                    self.qcharge_list + self.e_field_list,
                    self.qop_list,
                    self.eop_list,
                    encoding=self.encoding,
                )  # (must be then multiplied by g^2)

            hamiltonian_el_pauli = 0.5 * (hamiltonian_el_pauli)  # (must be then multiplied by g^2)

        else:  # no gauge fields (e.g. 1d OBC case)
            hamiltonian_el_pauli = 0  # .0 * self.tensor_prod(
            #     self.I,
            #     (int(self.lattice.n_sitestot) ),
            # )

        # ************************************  H_B   ************************************
        if self.len_e_op > 0:
            # Pauli expression

            if self.magnetic_basis:
                U_mag_subs = {
                    **{el_uop: Symbol("E_" + el_uop.name[2:]) for el_uop in self.u_op_free},
                    **{
                        el_uop: Symbol("E_" + el_uop.name[2:-1] + "m")
                        for el_uop in self.u_op_free_dag
                    },
                }
                hamiltonian_mag_sym = [
                    [U_mag_subs.get(item, item) for item in sublst if item != 1]
                    for sublst in self.hamilt_sym.hamiltonian_mag_subs
                ]
                e_op_dict_mbasis = dict(self.e_field_list)
                # compute exp(i*alpha*E)
                ei_class = lambda fct: self.matx_exp(fct * self.e_oper, 1j * self.alpha)

                hamiltonian_mag_pauli = []
                for ei in hamiltonian_mag_sym:
                    id_eop = {}
                    for e in ei:
                        if e in e_op_dict_mbasis.keys():
                            id_eop[list(e_op_dict_mbasis.keys()).index(e)] = 1  #'p'== +
                        else:
                            id_eop[
                                list(e_op_dict_mbasis.keys()).index(Symbol(e.name[:-1]))
                            ] = -1  #'m'== -
                    if self.sparse_pauli:
                        if self.encoding == "gray":
                            arg_cos = np.zeros(
                                2 ** (self._n_qubits_g() * self.len_e_op), dtype=np.complex128
                            )
                            left_factor = lambda key: 2 ** (
                                self._n_qubits_g() * (self.len_e_op - key - 1)
                            )
                            e_vals_eop = np.linalg.eig(self.e_oper.toarray())[0]
                            right_factor = lambda key: 2 ** (self._n_qubits_g() * key)
                        elif self.encoding == "ed":
                            arg_cos = np.zeros(
                                (2 * self.l_par + 1) ** self.len_e_op, dtype=np.complex128
                            )
                            left_factor = lambda key: (2 * self.l_par + 1) ** (
                                self.len_e_op - key - 1
                            )
                            e_vals_eop = np.arange(-self.l_par, self.l_par + 1)
                            right_factor = lambda key: (2 * self.l_par + 1) ** key
                        else:
                            raise ValueError("encoding not recognized")

                        n_kron = lambda arg_list: reduce(lambda x, y: np.kron(x, y), arg_list)

                        for key, val in id_eop.items():
                            if key == self.len_e_op - 1:
                                arg_cos += val * n_kron([e_vals_eop, np.ones(right_factor(key))])
                            elif key == 0:
                                arg_cos += val * n_kron([np.ones(left_factor(key)), e_vals_eop])
                            else:
                                arg_cos += val * n_kron(
                                    [
                                        np.ones(left_factor(key)),
                                        e_vals_eop,
                                        np.ones(right_factor(key)),
                                    ]
                                )

                        hamiltonian_mag_pauli.append(sparse.diags(np.cos(self.alpha * arg_cos)))

                    else:  # old method with cosE as exp(iE)+exp(-iE)/2. necessary if gray and sparse_pauli=False
                        if self.encoding == "gray":
                            idx = self.tensor_prod(
                                self.I, (self._n_qubits_g())
                            )  # Gray encoding for E field
                        elif (
                            self.encoding == "ed"
                        ):  # exact diagonaliz. dimensions of gauge fields 2l+1
                            idx = sparse.eye(2 * self.l_par + 1, format="csr")

                        if len(ei) == 1:  # cos 1 operator is enough and rest is I
                            cos1 = [
                                self.cos_oper(ei_class(id_eop[i])) if i in id_eop else idx
                                for i in range(self.len_e_op)[::-1]
                            ]  # inverse because little endian
                            hamiltonian_mag_pauli.append(HamiltonianQED_oprt.pauli_tns(*cos1))

                        else:
                            # compute cosine of multiple operators cos(E1+E2+...)=e^iE1 e^iE2 ... + e^-iE1 e^-iE2 ... /2
                            cosn = self.cos_oper(
                                HamiltonianQED_oprt.pauli_tns(
                                    *[
                                        ei_class(id_eop[i]) if i in id_eop else idx
                                        for i in range(self.len_e_op)
                                    ][min(id_eop) : max(id_eop) + 1][::-1]
                                )
                            )
                            # combine with identities, e.g. cos(E2+E3) -> I^cos(E2+E3)^I^I (q4^q3^q2^q1^q0)
                            hamiltonian_mag_pauli.append(
                                HamiltonianQED_oprt.pauli_tns(
                                    *[idx for i in range(self.len_e_op)[::-1]][max(id_eop) + 1 :]
                                    + [cosn]
                                    + [idx for i in range(self.len_e_op)[::-1]][: min(id_eop)]
                                )
                            )

                hamiltonian_mag_pauli = (
                    np.sum(hamiltonian_mag_pauli)
                    if self.puregauge
                    else HamiltonianQED_oprt.pauli_tns(
                        self.tensor_prod(self.I, (int(self.lattice.n_sitestot))),
                        np.sum(hamiltonian_mag_pauli),
                    )
                )  # (must be then multiplied by -1/g^2)
                print("Hamiltonian B mag basis: done")

            else:
                hamiltonian_mag_pauli = self.list_to_enc_hamilt(
                    self.hamilt_sym.hamiltonian_mag_subs,
                    self.u_field_list,
                    self.qop_list,
                    self.uop_list,
                    encoding=self.encoding,
                )
                hamiltonian_mag_pauli = 0.5 * (
                    hamiltonian_mag_pauli + self.hermitian_c(hamiltonian_mag_pauli)
                )  # (must be then multiplied by -1/g^2)

        else:  # no gauge fields (e.g. 1d OBC case)
            hamiltonian_mag_pauli = 0
        if not self.puregauge:
            # ************************************  H_K   ************************************
            # Pauli expression of the kinetic term
            if self.magnetic_basis:
                subst_list_hk = self.phi_jw_list_sym + self.u_field_list_mag
            else:
                subst_list_hk = self.phi_jw_list_sym + self.u_field_list

            if self.lattice.dims == 1:
                hamiltonian_k_1x = self.list_to_enc_hamilt(
                    (h for h in self.hamilt_sym.hamiltonian_k_sym),
                    subst_list_hk,
                    self.phiop_list,
                    self.uop_list,
                    encoding=self.encoding,
                )

                hamiltonian_k_pauli = 0.5j * (
                    hamiltonian_k_1x - self.hermitian_c(hamiltonian_k_1x)
                )  # (must be then multiplied by omega)

            elif self.lattice.dims == 2:
                hamiltonian_k_1y = self.list_to_enc_hamilt(
                    (h[1:] for h in self.hamilt_sym.hamiltonian_k_sym if h[0] == "y"),
                    subst_list_hk,
                    self.phiop_list,
                    self.uop_list,
                    encoding=self.encoding,
                )

                hamiltonian_k_1x = self.list_to_enc_hamilt(
                    (h[1:] for h in self.hamilt_sym.hamiltonian_k_sym if h[0] == "x"),
                    subst_list_hk,
                    self.phiop_list,
                    self.uop_list,
                    encoding=self.encoding,
                )

                hamiltonian_k_pauli = 0.5j * (
                    hamiltonian_k_1x - self.hermitian_c(hamiltonian_k_1x)
                ) - 0.5 * (
                    hamiltonian_k_1y + self.hermitian_c(hamiltonian_k_1y)
                )  # (must be then multiplied by omega)

            elif self.lattice.dims == 3:
                hamiltonian_k_1y = self.list_to_enc_hamilt(
                    (h[1:] for h in self.hamilt_sym.hamiltonian_k_sym if h[0] == "y"),
                    subst_list_hk,
                    self.phiop_list,
                    self.uop_list,
                    encoding=self.encoding,
                )

                hamiltonian_k_1x = self.list_to_enc_hamilt(
                    (h[1:] for h in self.hamilt_sym.hamiltonian_k_sym if h[0] == "x"),
                    subst_list_hk,
                    self.phiop_list,
                    self.uop_list,
                    encoding=self.encoding,
                )

                hamiltonian_k_1z = self.list_to_enc_hamilt(
                    (h[1:] for h in self.hamilt_sym.hamiltonian_k_sym if h[0] == "z"),
                    subst_list_hk,
                    self.phiop_list,
                    self.uop_list,
                    encoding=self.encoding,
                )

                hamiltonian_k_pauli = (
                    0.5j * (hamiltonian_k_1x - self.hermitian_c(hamiltonian_k_1x))
                    - 0.5 * (hamiltonian_k_1y + self.hermitian_c(hamiltonian_k_1y))
                    + 0.5j * (hamiltonian_k_1z - self.hermitian_c(hamiltonian_k_1z))
                )  # (must be then multiplied by omega)

            else:
                raise ValueError("Dimension not supported")

            # ************************************  H_M   ************************************

            hamiltonian_m_pauli = self.list_to_enc_hamilt(
                self.hamilt_sym.hamiltonian_m_sym,
                self.phi_jw_list_sym,
                self.phiop_list,
                encoding=self.encoding,
                massterm=True,
            )
            # (must be then multiplied by m)

            self.hamiltonian_k_pauli = hamiltonian_k_pauli
            self.hamiltonian_m_pauli = hamiltonian_m_pauli

        self.hamiltonian_el_pauli = hamiltonian_el_pauli
        self.hamiltonian_mag_pauli = hamiltonian_mag_pauli

    # others
    @staticmethod
    def str_to_tens(string: str):
        """Transforms bitstring to tensor '0'=[1,0] and '1'=[0,1]
        for example: '00' = array([1, 0, 0, 0]) , '01' = array([0, 1, 0, 0]),
        '10' = array([0, 0, 1, 0]) etc.

        Parameters
        ----------
        string: sequence of 0s and 1s

        Returns
        -------
        array, result of the tensor product

        """

        return reduce(lambda x, y: np.kron(x, y), [[1, 0] if x == "0" else [0, 1] for x in string])

    def hamiltonian_suppr(
        self,
    ):
        """Suppression Hamiltonian"""
        # Unphysical space suppressors:
        s_down = 0.5 * (self.I + self.Z)  # project to 0
        s_up = 0.5 * (self.I - self.Z)  # project to 1

        if self.encoding == "gray":
            gauge = self.tensor_prod(
                self.I, (self._n_qubits_g() * (self.len_u_op))
            )  # Gray encoding for E fields
        elif self.encoding == "ed":  # exact diagonaliz. dimensions of gauge fields 2l+1
            gauge = sparse.eye((2 * self.l_par + 1) ** (self.len_u_op), format="csr")

        # ******* gauge
        if (
            self.len_u_op > 0 and self.encoding == "gray"
        ):  # only for gray encoding exlcusion of unphyisical states
            h_s = 0
            # the state is projected onto the UNphysical state
            for i in range(2 * self.l_par + 1, 2 ** self._n_qubits_g()):
                gray_str = "{0:0{1}b}".format(i ^ (i >> 1), self._n_qubits_g())
                h_s += reduce(
                    lambda x, y: (x) ^ (y),
                    [s_down if x == "0" else s_up for x in gray_str],
                )

            suppr1 = h_s
            hamiltonian_gauge_suppr = 0.0 * gauge

            for i in range(1, self.len_u_op + 1):
                hamiltonian_gauge_suppr += HamiltonianQED_oprt.pauli_tns(
                    self.tensor_prod(self.I, (self._n_qubits_g() * (self.len_u_op - i))),
                    (suppr1),
                    self.tensor_prod(self.I, (self._n_qubits_g() * (i - 1))),
                ).simplify()

        elif self.len_u_op > 0 and self.encoding == "ed":  # only for ED encoding
            hamiltonian_gauge_suppr = 0.0 * gauge
        else:  # no gauge fields
            hamiltonian_gauge_suppr = 0.0 * self.tensor_prod(self.I, int(self.lattice.n_sitestot))

        # ****** fermion
        if not self.puregauge:
            suppr_f = self.tensor_prod(self.I, (int(self.lattice.n_sitestot)))
            # the state is projected onto zero-charge state (fermions), same number of 1 and 0
            for i in range(2 ** int(self.lattice.n_sitestot)):
                bincount = sum([1 for el in bin(i)[2:] if el == "1"])
                if bincount == int(self.lattice.n_sitestot) / 2:
                    binc = format(i, "0%db" % int(self.lattice.n_sitestot))
                    suppr_f += -1.0 * reduce(
                        lambda x, y: (x) ^ (y), [s_down if x == "0" else s_up for x in binc]
                    )

            hamiltonian_nzcharge_suppr = HamiltonianQED_oprt.pauli_tns(suppr_f, gauge)

        if self.puregauge:
            hamiltonian_suppress = hamiltonian_gauge_suppr
        elif self.len_u_op > 0:
            hamiltonian_suppress = HamiltonianQED_oprt.pauli_tns(
                self.tensor_prod(self.I, int(self.lattice.n_sitestot)),
                hamiltonian_gauge_suppr,
            ) + (hamiltonian_nzcharge_suppr)
        else:  # no gauge fields
            hamiltonian_suppress = hamiltonian_nzcharge_suppr
        if (
            isinstance(
                hamiltonian_suppress,
                (np.ndarray, sparse._csr.csr_matrix, sparse._coo.coo_matrix),
            )
            or not self.sparse_pauli
        ):
            self.hamiltonian_suppress = hamiltonian_suppress
        else:
            self.hamiltonian_suppress = hamiltonian_suppress.to_matrix(sparse=True)
