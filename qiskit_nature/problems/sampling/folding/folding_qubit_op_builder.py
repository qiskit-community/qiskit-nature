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

""" Lattice Folding qubit operator builder. """

import itertools
from typing import List, Tuple

from sympy import *
import numpy as np

from qiskit.aqua.operators import SummedOp, PauliOp
from qiskit.quantum_info.operators import Pauli

## Auxiliary functions
def _simplify(pauli_conf, x):
    if x == 0:
        return 0
    else:
        first_binaries = [1, -1, 1, 1, 0, -1]
        x = x.expand()
        for m in range(4, 1, -1):
            x = x.subs({pauli_conf[k][0]**m: (pauli_conf[k][0])**(m%2) for k in pauli_conf})
            x = x.subs({pauli_conf[k][1]**m: (pauli_conf[k][1])**(m%2) for k in pauli_conf})
            x = x.subs({pauli_conf[k][0]: first_binaries[k-1] for k in [1, 2, 3, 4, 6]})
    return x

def _create_pauli_for_conf(N):
    pauli_conf = dict()
    for i in range(1, 2*(N-1) + 1):  # first qubits is number 1
        pauli_conf[i] = dict()
        pauli_conf[i][0] = symbols("\sigma^z_{}".format({i}))
        pauli_conf[i][1] = symbols("\sigma^z_"+"{" + "{}".format({i}) + "^{(1)}"+"}")
    return pauli_conf


def _create_qubits_for_conf(pauli_conf):
    qubits = dict()
    for i in range(1, len(pauli_conf) + 1):  # first qubit is number 1
        qubits[i] = dict()
        qubits[i][0] = (1 - pauli_conf[i][0])/2
        qubits[i][1] = (1 - pauli_conf[i][1])/2
    return qubits

def _create_indic_turn(N, side_chain, qubits):
    if len(side_chain)!= N:
        raise Exception('size of side_chain list is not equal to N ')
    indic0, indic1, indic2, indic3 = dict(), dict(), dict(), dict()
    for i in range(1, N):
        indic0[i]=dict()
        indic1[i]=dict()
        indic2[i]=dict()
        indic3[i]=dict()

    #  indic_a[i][0] is the indicator for the backbone and
    #  indic_a[i][1] for the first bead on SC
    #  only one bead on SC here
    r_conf = 0
    for i in range(1, N):   # There are N-1 turns starting at turn 1
        for m in range(2):
            if m == 1:
                if side_chain[i-1]==0 :
                    continue
                else:
                    pass
            indic0[i][m] = (1 - qubits[2*i - 1][m])*(1 - qubits[2*i][m])
            indic1[i][m] = qubits[2*i][m]*(qubits[2*i][m] - qubits[2*i - 1][m])
            indic2[i][m] = qubits[2*i -  1][m]*(qubits[2*i - 1][m]-qubits[2*i][m])
            indic3[i][m] = qubits[2*i - 1][m]*qubits[2*i][m]
            r_conf += 1
    print('number of qubits required for conformation: ', 2*r_conf - 5)
    return indic0, indic1, indic2, indic3, 2*r_conf - 5

def _create_delta_BB(N, indic0, indic1, indic2, indic3, pauli_conf):
    delta_n0, delta_n1, delta_n2, delta_n3 = dict(), dict(), dict(), dict()
    for i in range(1, N):
        delta_n0[i] = dict()
        delta_n1[i] = dict()
        delta_n2[i] = dict()
        delta_n3[i] = dict()
        delta_n0[i][0], delta_n0[i][1] = dict(), dict()
        delta_n1[i][0], delta_n1[i][1] = dict(), dict()
        delta_n2[i][0], delta_n2[i][1] = dict(), dict()
        delta_n3[i][0], delta_n3[i][1] = dict(), dict()
        for j in range(i + 1, N + 1):
            delta_n0[i][0][j], delta_n0[i][1][j] = dict(), dict()
            delta_n1[i][0][j], delta_n1[i][1][j] = dict(), dict()
            delta_n2[i][0][j], delta_n2[i][1][j] = dict(), dict()
            delta_n3[i][0][j], delta_n3[i][1][j] = dict(), dict()

    for i in range(1, N): # j>i
        for j in range(i + 1, N + 1):
            delta_n0[i][0][j][0] = 0
            delta_n1[i][0][j][0] = 0
            delta_n2[i][0][j][0] = 0
            delta_n3[i][0][j][0] = 0
            for k in range(i, j):
                delta_n0[i][0][j][0] += (-1)**k * indic0[k][0]
                delta_n1[i][0][j][0] += (-1)**k * indic1[k][0]
                delta_n2[i][0][j][0] += (-1)**k * indic2[k][0]
                delta_n3[i][0][j][0] += (-1)**k * indic3[k][0]

            delta_n0[i][0][j][0] = _simplify(pauli_conf, delta_n0[i][0][j][0])
            delta_n1[i][0][j][0] = _simplify(pauli_conf, delta_n1[i][0][j][0])
            delta_n2[i][0][j][0] = _simplify(pauli_conf, delta_n2[i][0][j][0])
            delta_n3[i][0][j][0] = _simplify(pauli_conf, delta_n3[i][0][j][0])
    return delta_n0, delta_n1, delta_n2, delta_n3

def _add_delta_SC(N, delta_n0, delta_n1, delta_n2, delta_n3, indic0, indic1, indic2, indic3, pauli_conf):
    for i in range(1, N): # j>i
        for j in range(i+1, N+1):
            try :
                delta_n0[i][0][j][1]= _simplify(pauli_conf, delta_n0[i][0][j][0] + (-1)**j *indic0[j][1])
                delta_n1[i][0][j][1]= _simplify(pauli_conf, delta_n1[i][0][j][0] + (-1)**j *indic1[j][1])
                delta_n2[i][0][j][1]= _simplify(pauli_conf, delta_n2[i][0][j][0] + (-1)**j *indic2[j][1])
                delta_n3[i][0][j][1]= _simplify(pauli_conf, delta_n3[i][0][j][0] + (-1)**j *indic3[j][1])
            except:
                pass
            try:
                delta_n0[i][1][j][0]= _simplify(pauli_conf, delta_n0[i][0][j][0] - (-1)**i *indic0[i][1])
                delta_n1[i][1][j][0]= _simplify(pauli_conf, delta_n1[i][0][j][0] - (-1)**i *indic1[i][1])
                delta_n2[i][1][j][0]= _simplify(pauli_conf, delta_n2[i][0][j][0] - (-1)**i *indic2[i][1])
                delta_n3[i][1][j][0]= _simplify(pauli_conf, delta_n3[i][0][j][0] - (-1)**i *indic3[i][1])
            except:
                pass
            try:
                delta_n0[i][1][j][1]= _simplify(pauli_conf, delta_n0[i][0][j][0] + (-1)**j *indic0[j][1] - (-1)**i *indic0[i][1])
                delta_n1[i][1][j][1]= _simplify(pauli_conf, delta_n1[i][0][j][0] + (-1)**j *indic1[j][1] - (-1)**i *indic1[i][1])
                delta_n2[i][1][j][1]= _simplify(pauli_conf, delta_n2[i][0][j][0] + (-1)**j *indic2[j][1] - (-1)**i *indic2[i][1])
                delta_n3[i][1][j][1]= _simplify(pauli_conf, delta_n3[i][0][j][0] + (-1)**j *indic3[j][1] - (-1)**i *indic3[i][1])
            except:
                pass
    return delta_n0, delta_n1, delta_n2, delta_n3

def _create_x_dist(N, delta_n0, delta_n1, delta_n2, delta_n3, pauli_conf):
    x_dist = dict()
    r = 0
    for i in range (1, N):
        x_dist[i] = dict()
        x_dist[i][0], x_dist[i][1] = dict(), dict()
        for j in range(i+1, N+1):
            x_dist[i][0][j], x_dist[i][1][j] = dict(), dict()

    for i in range(1, N): # j>i
        for j in range(i+1, N+1):
            for s in range (2):
                for p in range(2):
                    if i == 1 and p == 1 or j == N and s == 1 :
                        continue
                    try:
                        x_dist[i][p][j][s]= _simplify(pauli_conf, delta_n0[i][p][j][s]**2 + delta_n1[i][p][j][s]**2 +
                                                        delta_n2[i][p][j][s]**2 + delta_n3[i][p][j][s]**2)
                        r +=1
                    except:
                        pass
    print(r, ' distances created')
    return x_dist

def _create_H_chiral(N, side_chain, lambda_chiral, indic0, indic1, indic2, indic3, pauli_conf):
    H_chiral = 0
    for i in range(1, N+1):   # There are N-1 turns starting at turn 1
        if side_chain[i-1] == 0:
            continue
        si = int ((1-(-1)**i)/2)
        H_chiral += _simplify(pauli_conf, lambda_chiral *(1-indic0[i][1])*((1-si)*(indic1[i-1][0]*indic2[i][0] + indic2[i-1][0]*indic3[i][0] +
                            indic3[i-1][0]*indic1[i][0]) + si*(indic2[i-1][0]*indic1[i][0] + indic3[i-1][0]*indic2[i][0]+indic1[i-1][0]*indic3[i][0])))
        H_chiral += _simplify(pauli_conf, lambda_chiral *(1-indic1[i][1])*((1-si)*(indic0[i-1][0]*indic3[i][0] + indic2[i-1][0]*indic0[i][0] +
                            indic3[i-1][0]*indic2[i][0]) + si*(indic3[i-1][0]*indic0[i][0] + indic0[i-1][0]*indic2[i][0]+indic2[i-1][0]*indic3[i][0])))
        H_chiral += _simplify(pauli_conf, lambda_chiral *(1-indic2[i][1])*((1-si)*(indic0[i-1][0]*indic1[i][0] + indic1[i-1][0]*indic3[i][0] +
                            indic3[i-1][0]*indic0[i][0]) + si*(indic1[i-1][0]*indic0[i][0] + indic3[i-1][0]*indic1[i][0]+indic0[i-1][0]*indic3[i][0])))
        H_chiral += _simplify(pauli_conf, lambda_chiral *(1-indic3[i][1])*((1-si)*(indic0[i-1][0]*indic2[i][0] + indic1[i-1][0]*indic0[i][0] +
                            indic2[i-1][0]*indic1[i][0]) + si*(indic2[i-1][0]*indic0[i][0] + indic0[i-1][0]*indic1[i][0]+indic1[i-1][0]*indic2[i][0])))
    H_chiral = _simplify(pauli_conf, H_chiral)
    return H_chiral

def _second_neighbor(i, p, j, s, lambda_1, pair_energies, x_dist, pauli_conf):
    e = pair_energies[i, p, j, s]
    x = x_dist[i][p][j][s]
    expr = lambda_1*(2 - x) #+ e*0.1
    expr = _simplify(pauli_conf, expr)
    return expr

def _first_neighbor(i, p, j, s, lambda_1, pair_energies, x_dist, pauli_conf):
    lambda_0 = 7*(j - i + 1)*lambda_1
    e = pair_energies[i, p, j, s]
    x = x_dist[i][p][j][s]
    expr = e + lambda_0*(x - 1)
    expr = _simplify(pauli_conf, expr)
    return expr

def _check_turns(i, p, j, s, indic0, indic1, indic2, indic3, pauli_conf):
    return _simplify(pauli_conf, indic0[i][p]*indic0[j][s] + indic1[i][p]*indic1[j][s] +
                    indic2[i][p]*indic2[j][s] +indic3[i][p]*indic3[j][s])

def _create_H_back(N, lambda_back, indic0, indic1, indic2, indic3, pauli_conf):
    H_back = 0
    for i in range(1, N - 1):
        H_back += lambda_back*_check_turns(i, 0, i + 1, 0, indic0, indic1, indic2, indic3, pauli_conf)
    H_back = _simplify(pauli_conf, H_back)
    return H_back

def _create_H_short(N, side_chain, pair_energies, x_dist, pauli_conf, indic0, indic1, indic2, indic3):
    H_short = 0
    for i in range(1, N - 2):
        if side_chain[i - 1] == 1 and side_chain[i + 2] == 1 :
            H_short += _simplify(pauli_conf, _check_turns(i, 1, i + 2, 0, indic0, indic1, indic2, indic3, pauli_conf)* \
                        _check_turns(i+3, 1, i, 0, indic0, indic1, indic2, indic3, pauli_conf))* \
                        (pair_energies[i, 1, i+3, 1] + 0.1*(pair_energies[i, 1, i+3, 0]+pair_energies[i, 0, i+3, 1]))
    return H_short

def _create_pauli_for_contacts(N,side_chain):
    pauli_contacts = dict()
    for i in range(1, N - 3):
        pauli_contacts[i] = dict()
        pauli_contacts[i][0] = dict()
        pauli_contacts[i][1] = dict()
        for j in range(i + 3,N + 1):
            pauli_contacts[i][0][j] = dict()
            pauli_contacts[i][1][j] = dict()

    r_contact = 0
    for i in range(1,N - 3):  # first qubits is number 1
        for j in range(i + 3,N + 1):
            if (j-i)%2 == 1:
                if (j-i) >= 5:
                    pauli_contacts[i][0][j][0] = symbols( '\sigma^z_'+ '{' + '{}'.format(i) +"\,"+ '{}'.format(j) + '}' )
                    print('possible contact between',i,'0 and',j,'0')
                    r_contact += 1
                if side_chain[i-1] == 1 and side_chain[j-1] == 1:
                    try:
                        pauli_contacts[i][1][j][1] = symbols('\sigma^z_'+ '{'+ '{}'.format(i)+ '^{(1)}' + '\,'+ '{}'.format(j) + '^{(1)}' + '}')
                        print('possible contact between',i,'1 and',j,'1')
                        r_contact += 1
                    except:
                        pass
            else:
                if (j-i) >= 4:
                    if side_chain[j-1] == 1:
                        try:
                            pauli_contacts[i][0][j][1] = symbols('\sigma^z_'+ '{'+ '{}'.format(i)+ '\,'+ '{}'.format(j)+ '^{(1)}' + '}')
                            print('possible contact between',i,'0 and',j,'1')
                            r_contact += 1
                        except:
                            pass

                    if side_chain[i-1] == 1:
                        try:
                            pauli_contacts[i][1][j][0] = symbols('\sigma^z_'+ '{'+ '{}'.format(i)+ '^{(1)}' + '\,'+ '{}'.format(j) + '}')
                            print('possible contact between',i,'1 and',j,'0')
                            r_contact += 1
                        except:
                            pass
    print('number of qubits required for contact : ',r_contact)
    return pauli_contacts,r_contact

def _create_contact_qubits(N, pauli_contacts):
    contacts = dict()
    for i in range(1, N - 3):
        contacts[i] = dict()
        contacts[i][0] = dict()
        contacts[i][1] = dict()
        for j in range(i + 3, N + 1):
            contacts[i][0][j] = dict()
            contacts[i][1][j] = dict()

    for i in range(1, N - 3):  # first qubits is number 1
        for j in range(i + 4, N + 1):
            for p in range (2):
                for s in range(2):
                    try:
                        contacts[i][p][j][s] = (1 - pauli_contacts[i][p][j][s])/2
                    except:
                        pass
    return contacts

def _create_H_BBBB(N, lambda_1, pair_energies, x_dist, pauli_conf, contacts):
    H_BBBB = 0
    for i in range (1, N - 3):
        for j in range(i + 5, N + 1):
            if (j - i)%2 == 0:
                continue
            else:
                H_BBBB += contacts[i][0][j][0]*_first_neighbor(i, 0, j, 0, lambda_1, pair_energies, x_dist, pauli_conf)
                try:
                    H_BBBB += contacts[i][0][j][0]*_second_neighbor(i - 1, 0, j, 0, lambda_1, pair_energies, x_dist, pauli_conf)
                except:
                    pass
                try:
                    H_BBBB += contacts[i][0][j][0]*_second_neighbor(i + 1, 0, j, 0, lambda_1, pair_energies, x_dist, pauli_conf)
                except:
                    pass
                try:
                    H_BBBB += contacts[i][0][j][0]*_second_neighbor(i, 0, j-1, 0, lambda_1, pair_energies, x_dist, pauli_conf)
                except:
                    pass
                try:
                    H_BBBB += contacts[i][0][j][0]*_second_neighbor(i, 0, j + 1, 0, lambda_1, pair_energies, x_dist, pauli_conf)
                except:
                    pass
            H_BBBB = _simplify(pauli_conf, H_BBBB)
    return H_BBBB

def _create_H_BBSC_and_H_SCBB(N, side_chain, lambda_1, pair_energies, x_dist, pauli_conf, contacts):
    H_BBSC = 0
    H_SCBB = 0
    for i in range (1, N - 3):
        for j in range(i + 4, N + 1):
            if (j-i)%2 == 1:
                continue
            else:
                if side_chain[j - 1] == 1:
                    H_BBSC += contacts[i][0][j][1]*(_first_neighbor(i, 0, j, 1, lambda_1, pair_energies, x_dist, pauli_conf) + \
                                _second_neighbor(i, 0, j, 0, lambda_1, pair_energies, x_dist, pauli_conf))
                    try:
                        H_BBSC += contacts[i][0][j][1]*_first_neighbor(i, 1, j, 1, lambda_1, pair_energies, x_dist, pauli_conf)
                    except:
                        pass
                    try:
                        H_BBSC += contacts[i][0][j][1]*_second_neighbor(i + 1, 0, j, 1, lambda_1, pair_energies, x_dist, pauli_conf)
                    except:
                        pass
                    try:
                        H_BBSC += contacts[i][0][j][1]*_second_neighbor(i - 1, 0, j, 1, lambda_1, pair_energies, x_dist, pauli_conf)
                    except:
                        pass
                    H_BBSC = _simplify(pauli_conf, H_BBSC)
                if side_chain[i-1] == 1:
                    H_SCBB += contacts[i][1][j][0]*(_first_neighbor(i, 1, j, 0, lambda_1, pair_energies, x_dist, pauli_conf) + \
                                _second_neighbor(i, 0, j, 0, lambda_1, pair_energies, x_dist, pauli_conf))
                    try:
                        H_SCBB += contacts[i][1][j][0]*_second_neighbor(i, 1, j, 1, lambda_1, pair_energies, x_dist, pauli_conf)
                    except:
                        pass
                    try:
                        H_SCBB += contacts[i][1][j][0]*_second_neighbor(i, 1, j + 1, 0, lambda_1, pair_energies, x_dist, pauli_conf)
                    except:
                        pass
                    try:
                        H_SCBB += contacts[i][1][j][0]*_second_neighbor(i, 1, j - 1, 0, lambda_1, pair_energies, x_dist, pauli_conf)
                    except:
                        pass
                    H_SCBB = _simplify(pauli_conf, H_SCBB)
    return H_BBSC, H_SCBB

def _create_H_SCSC(N, side_chain, lambda_1, pair_energies, x_dist, pauli_conf, contacts):
    H_SCSC = 0
    for i in range (1, N - 3):
        for j in range(i + 5, N + 1):
            if (j - i)%2 == 0:
                continue
            if side_chain[i - 1] == 0 or side_chain[j - 1] == 0:
                continue
            H_SCSC += contacts[i][1][j][1]*(_first_neighbor(i, 1, j, 1, lambda_1, pair_energies, x_dist, pauli_conf) + \
                        _second_neighbor(i, 1, j, 0, lambda_1, pair_energies, x_dist, pauli_conf) + \
                        _second_neighbor(i, 0, j, 1, lambda_1, pair_energies, x_dist, pauli_conf))
            H_SCSC = _simplify(pauli_conf, H_SCSC)
    return H_SCSC

def _create_H_contacts(pauli_conf, new_qubits, n_contact, lambda_contacts, N_contacts):
    H_contacts = lambda_contacts*(0.5*(np.sum(1 - np.array(new_qubits[-n_contact:]))) - N_contacts)**2
    H_contacts = H_contacts.expand()
    H_contacts = H_contacts.subs({new_qubits[k]**2: 1 for k in range(1, len(new_qubits))})
    return H_contacts


def _get_symbolic_delta(N, side_chain):
    '''the first binaries are in the article'''
    if len(side_chain) != N:
        raise Exception('size the side_chain is not equal to N')
    if side_chain[0] == 1 or side_chain[-1] == 1:
        raise Exception('please add extra bead instead of side chain on terminal bead')
    pauli_conf = _create_pauli_for_conf(N)
    qubits = _create_qubits_for_conf(pauli_conf)
    indic0, indic1, indic2, indic3, n_conf = _create_indic_turn(N, side_chain, qubits)
    delta_n0, delta_n1, delta_n2, delta_n3 = _create_delta_BB(N, indic0, indic1, indic2, indic3, pauli_conf)
    delta_n0, delta_n1, delta_n2, delta_n3 = _add_delta_SC(N, delta_n0, delta_n1, delta_n2, delta_n3, indic0, indic1, indic2, indic3, pauli_conf)
    return delta_n0, delta_n1, delta_n2, delta_n3

def _get_both_paulis(N, side_chain):
    '''the first binaries are in the article'''
    if len(side_chain) != N:
        raise Exception('size the side_chain is not equal to N')
    if side_chain[0] == 1 or side_chain[-1] == 1:
        raise Exception('please add extra bead instead of side chain on terminal bead')
    pauli_conf = _create_pauli_for_conf(N)
    pauli_contacts, n_contact = _create_pauli_for_contacts(N, side_chain)
    return pauli_conf, pauli_contacts


def _set_contact_qubits(H_symbolic, n_contact, n_conf, new_qubits_before_simpl, contacts_list, print_simpl = 'yes'):
    if len(contacts_list) != n_contact:
        raise Exception('contacts_list size must be equal to the number of contacts ')
    x = new_qubits_before_simpl[-n_contact:]
    contacts_list = list (1 - 2*np.array(contacts_list))
    for c in range(n_contact):
        H_symbolic = H_symbolic.subs(x[c], contacts_list[c])
        if print_simpl == 'yes':
            print(latex(x[c]), contacts_list[c])
    H_symbolic = simplify(H_symbolic)
    if print_simpl == 'yes':
        print('new number of terms :',  len(H_symbolic.args))
        print('new number of qubits required :', n_conf)
    return H_symbolic, n_conf, {}

def _create_new_qubit_list(N, side_chain, pauli_conf, pauli_contacts):
    old_qubits_conf = []
    old_qubits_contact = []
    for q in range(5, 2*(N - 1) + 1):
        if q%2 == 0 :
            i1 = q//2
        elif q%2 == 1:
            i1 = (q+1)//2
        if q !=6:
            old_qubits_conf.append(pauli_conf[q][0])
        if side_chain[i1-1] == 1:
            old_qubits_conf.append(pauli_conf[q][1])

    for i in range(1, N - 3):
        for j in range(i + 4, N + 1):
            for p in range (2):
                for s in range(2):
                    try:
                        old_qubits_contact.append(pauli_contacts[i][p][j][s])
                    except :
                        pass
    new_qubits = [0] + old_qubits_conf + old_qubits_contact
    return new_qubits

def _get_symbolic_hamiltonian(N,side_chain,pair_energies,lambda_chiral,lambda_back,lambda_1,lambda_contacts,N_contacts):
    
    
    '''the first binaries are in the article'''
    if len(side_chain)!=N:
        raise Exception('size the side_chain is not equal to N')
    if side_chain[0]==1 or side_chain[-1]==1 or side_chain[1]==1:
        raise Exception('please add extra bead instead of side chain on terminal bead')
        
    pauli_conf = _create_pauli_for_conf(N)
    qubits = _create_qubits_for_conf(pauli_conf)
    indic0, indic1, indic2, indic3, n_conf = _create_indic_turn(N, side_chain, qubits)
    delta_n0, delta_n1, delta_n2, delta_n3 = _create_delta_BB(N, indic0, indic1, indic2, indic3, pauli_conf)
    delta_n0, delta_n1, delta_n2, delta_n3 = _add_delta_SC(N, delta_n0, delta_n1, delta_n2, delta_n3, indic0, indic1, indic2, indic3, pauli_conf)
    x_dist = _create_x_dist(N, delta_n0, delta_n1, delta_n2, delta_n3, pauli_conf)
    H_chiral = _create_H_chiral(N, side_chain, lambda_chiral, indic0, indic1, indic2, indic3, pauli_conf)
    H_back = _create_H_back(N, lambda_back, indic0, indic1, indic2, indic3, pauli_conf)
    H_short = _create_H_short(N, side_chain, pair_energies, x_dist, pauli_conf, indic0, indic1, indic2, indic3)
    pauli_contacts, n_contact = _create_pauli_for_contacts(N, side_chain)
    contacts = _create_contact_qubits(N, pauli_contacts)
    H_BBBB = _create_H_BBBB(N, lambda_1, pair_energies, x_dist, pauli_conf, contacts)
    H_BBSC, H_SCBB = _create_H_BBSC_and_H_SCBB(N, side_chain, lambda_1, pair_energies, x_dist, pauli_conf, contacts)
    H_SCSC = _create_H_SCSC(N, side_chain, lambda_1, pair_energies, x_dist, pauli_conf, contacts)
    new_qubits = _create_new_qubit_list(N, side_chain, pauli_conf, pauli_contacts)
    H_contacts = _create_H_contacts(pauli_conf, new_qubits, n_contact, lambda_contacts, N_contacts)
    H_tot = _simplify(pauli_conf, H_chiral + H_back + H_short + H_BBBB + H_BBSC + H_SCBB + H_SCSC + H_contacts)
    n_qubits = n_conf + n_contact
    
    print('total number of qubits required :', n_qubits)
    print('number of terms in the hamiltonian : ',len(H_tot.args))
                
    return H_tot,pauli_conf,pauli_contacts,n_qubits,n_conf,n_contact,new_qubits

def _create_mask_for_tensor(H, new_qubits):
    terms = len(H.args)
    mask = np.zeros((terms, len(new_qubits)))
    for t in range(terms):
        for b in H.args[t].args:
            for k in range(1, len(new_qubits)):
                if b == new_qubits[k]:
                    mask[t, 0]= H.args[t].args[0]
                    mask[t, k]= 1
    mask[0, 0] = H.args[0]
    return mask


def _make_pauli_list(N, side_chain, H_symbolic, n_qubits, new_qubits):
    mask = _create_mask_for_tensor(H_symbolic, new_qubits)
    terms = len(H_symbolic.args)
    pauli_list = []
    for t in range(terms):
        pauli_list.append((mask[t, 0], Pauli(mask[t, 1:], np.zeros(n_qubits))))
    pauli_list = list(np.flip(np.array(pauli_list), axis=0))
    return pauli_list


def _build_qubit_op(N,side_chain,pair_energies,lambda_chiral,lambda_back,lambda_1,lambda_contacts,N_contacts):

    # run symbolic hamiltonian
    H_tot, pauli_conf, pauli_contacts, n_qubits, n_conf, n_contact, new_qubits = _get_symbolic_hamiltonian(N,side_chain,pair_energies,
                                                                                                           lambda_chiral,lambda_back,lambda_1,
                                                                                                           lambda_contacts,N_contacts)
    # generate list of Paulis
    pauli_list = _make_pauli_list(N, side_chain, H_tot, n_qubits, new_qubits)
    # create list of operators
    operators = []
    for p in pauli_list:
            pauli = p[1] # Pauli
            coeff = p[0]
            operator = PauliOp(pauli,coeff)
            operators.append(operator)
    # convert to operator flow
    operators = SummedOp(oplist=operators)
    return operators

