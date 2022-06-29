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

"""
Fermionic Hamilton for LiH and Qubit Hamiltonian obtained from BKSF

The Fermionic Hamiltonian is generated by the following code:

    from qiskit_nature.drivers import Molecule

    molecule = Molecule(
        # coordinates are given in Angstrom
        geometry=[
            ["Li", [0.0, 0.0, 0.0]],
            ["H", [0.0, 0.0, 1.6]],
        ],
        multiplicity=1,  # = 2*spin + 1
        charge=0,
    )

    from qiskit_nature.drivers.second_q import ElectronicStructureMoleculeDriver,
         ElectronicStructureDriverType

    driver = ElectronicStructureMoleculeDriver(
        molecule=molecule,
        basis="sto3g",
        driver_type=ElectronicStructureDriverType.PYSCF,
    )

    from qiskit_nature.second_q.problems.electronic import ElectronicStructureProblem
    from qiskit_nature.transformers.second_q.electronic import ActiveSpaceTransformer

    transformer = ActiveSpaceTransformer(
        num_electrons=2,
        num_molecular_orbitals=3,
    )

    problem_reduced = ElectronicStructureProblem(driver, [transformer])
    second_q_ops_reduced = problem_reduced.second_q_ops()
    hamiltonian_reduced = second_q_ops_reduced[0]
"""

import numpy
from qiskit.quantum_info import PauliList, SparsePauliOp
from qiskit_nature.second_q.operators import FermionicOp


FERMIONIC_HAMILTONIAN = FermionicOp(
    [
        ("+-I+-I", (0.013063981998607477 + 0j)),
        ("+-I-+I", (-0.013063981998607472 + 0j)),
        ("+-IIII", (0.048579599520367646 + 0j)),
        ("+-IIIN", (0.005767502046076787 + 0j)),
        ("+-IINI", (0.007484171005646165 + 0j)),
        ("+-INII", (-0.04857958891220289 + 0j)),
        ("+-NIII", (-0.013509390402447545 + 0j)),
        ("+I-+I-", (0.023422673239767655 + 0j)),
        ("+I--I+", (-0.023422673239767655 + 0j)),
        ("+I-I+-", (0.019276892448524333 + 0j)),
        ("+I-I-+", (-0.019276892448524333 + 0j)),
        ("-+I+-I", (-0.013063981998607472 + 0j)),
        ("-+I-+I", (0.013063981998607473 - 0j)),
        ("-+IIII", (-0.04857959952036771 + 0j)),
        ("-+IIIN", (-0.005767502046076761 + 0j)),
        ("-+IINI", (-0.007484171005646201 + 0j)),
        ("-+INII", (0.04857958891220291 + 0j)),
        ("-+NIII", (0.013509390402447573 + 0j)),
        ("-I++I-", (-0.023422673239767655 + 0j)),
        ("-I+-I+", (0.023422673239767655 - 0j)),
        ("-I+I+-", (-0.019276892448524333 + 0j)),
        ("-I+I-+", (0.019276892448524333 - 0j)),
        ("I+-+I-", (0.019276892448524333 + 0j)),
        ("I+--I+", (-0.019276892448524333 + 0j)),
        ("I+-I+-", (0.041276695997097185 + 0j)),
        ("I+-I-+", (-0.04127669599709719 + 0j)),
        ("I-++I-", (-0.019276892448524333 + 0j)),
        ("I-+-I+", (0.019276892448524333 - 0j)),
        ("I-+I+-", (-0.04127669599709719 + 0j)),
        ("I-+I-+", (0.041276695997097185 - 0j)),
        ("III+-I", (0.048579599520367646 + 0j)),
        ("III+-N", (-0.013509390402447545 + 0j)),
        ("III-+I", (-0.04857959952036771 + 0j)),
        ("III-+N", (0.013509390402447573 + 0j)),
        ("IIIIIN", (-0.35297896520254896 + 0j)),
        ("IIIINI", (-0.355939542660255 + 0j)),
        ("IIIINN", (0.2407146489655783 + 0j)),
        ("IIINII", (-0.772581720072654 + 0j)),
        ("IIININ", (0.24674881903629914 + 0j)),
        ("IIINNI", (0.2105460611420031 + 0j)),
        ("IIN+-I", (0.005767502046076787 + 0j)),
        ("IIN-+I", (-0.005767502046076761 + 0j)),
        ("IINIII", (-0.35297896520254896 + 0j)),
        ("IINIIN", (0.3129455111594082 + 0j)),
        ("IININI", (0.28199134496267547 + 0j)),
        ("IINNII", (0.2701714922760668 + 0j)),
        ("INI+-I", (0.007484171005646165 + 0j)),
        ("INI-+I", (-0.007484171005646201 + 0j)),
        ("INIIII", (-0.355939542660255 + 0j)),
        ("INIIIN", (0.28199134496267547 + 0j)),
        ("INIINI", (0.3378822722917939 + 0j)),
        ("ININII", (0.2236100431406106 + 0j)),
        ("INNIII", (0.2407146489655783 + 0j)),
        ("NII+-I", (-0.04857958891220289 + 0j)),
        ("NII-+I", (0.04857958891220291 - 0j)),
        ("NIIIII", (-0.772581720072654 + 0j)),
        ("NIIIIN", (0.2701714922760668 + 0j)),
        ("NIIINI", (0.2236100431406106 + 0j)),
        ("NIINII", (0.48731096863288564 + 0j)),
        ("NINIII", (0.24674881903629914 + 0j)),
        ("NNIIII", (0.2105460611420031 + 0j)),
    ],
    display_format="dense",
)


def _qubit_operator():
    pauli_list = PauliList(
        [
            "IIIIIIIIIII",
            "IIIIIIIZZZY",
            "IIIIIIIZZZZ",
            "IIIIIIXIXII",
            "IIIIIXZIZXI",
            "IIIIZYIZIYI",
            "IIIIZZYZYZI",
            "IIIIZZZIIIY",
            "IIIIZZZIIIZ",
            "IIIIZZZZZZI",
            "IIIXIIZXZIZ",
            "IIIXXZIZIZZ",
            "IIXZIZIXIZZ",
            "IIXZXIZZZIZ",
            "IIYIYIZIZIZ",
            "IIYIZZIYIZZ",
            "IIZYYZIIIZZ",
            "IIZYZIZYZIZ",
            "IIZZIYIIIYI",
            "IIZZIZYIYZI",
            "IIZZIZZIZZI",
            "IIZZZIIZIII",
            "IIZZZIXZXII",
            "IIZZZXZZZXI",
            "IXIIIIIIIXZ",
            "IXIIIIIIXZZ",
            "IXIIIIXIIZZ",
            "IXIIIXZIZIZ",
            "IXZZZIIZIXZ",
            "IXZZZIIZXZZ",
            "IXZZZIXZIZZ",
            "IXZZZXZZZIZ",
            "IYIIIYIZZII",
            "IYIIIZYZIZI",
            "IYIIZIIIYII",
            "IYIIZIIIZYI",
            "IYZZIIIZYII",
            "IYZZIIIZZYI",
            "IYZZZYIIZII",
            "IYZZZZYIIZI",
            "IZIIIZZZIIY",
            "IZIIIZZZIIZ",
            "IZIIZIIIZZY",
            "IZIIZIIIZZZ",
            "IZIIZIIZIII",
            "IZIXXIZIIZI",
            "IZIXZIZXIZI",
            "IZXZXZIIZII",
            "IZXZZZIXZII",
            "IZYIIZIYZII",
            "IZYIYZIZZII",
            "IZZYIIZYIZI",
            "IZZYYIZZIZI",
            "IZZZIIIIIII",
            "IZZZIIIZZZY",
            "IZZZIIIZZZZ",
            "IZZZZZZIIIY",
            "IZZZZZZIIIZ",
            "YIIZIIZIIZI",
            "YIIZIIZZZIZ",
            "YIIZZZIIIZZ",
            "YIZIIZIIZII",
            "YIZIIZIZIZZ",
            "YIZIZIZIZIZ",
            "YZIZIZIIZII",
            "YZIZZIZZIZI",
            "YZZIIIZIIZI",
            "YZZIZZIZZII",
            "ZIIYIIIYIII",
            "ZIIYYIIZIII",
            "ZIIZIIYZYIZ",
            "ZIIZIIZIIZI",
            "ZIIZIIZZZIY",
            "ZIIZIIZZZIZ",
            "ZIIZIXIZIXZ",
            "ZIIZZYZIZYZ",
            "ZIIZZZIIIZY",
            "ZIIZZZIIIZZ",
            "ZIIZZZXIXZZ",
            "ZIXIXIIIIII",
            "ZIXIZIIXIII",
            "ZIYZIIIYIII",
            "ZIYZYIIZIII",
            "ZIZIIYZZZYZ",
            "ZIZIIZIIZII",
            "ZIZIIZIZIZY",
            "ZIZIIZIZIZZ",
            "ZIZIIZXZXZZ",
            "ZIZIZIYIYIZ",
            "ZIZIZIZIZIY",
            "ZIZIZIZIZIZ",
            "ZIZIZXIIIXZ",
            "ZIZXXIIIIII",
            "ZIZXZIIXIII",
            "ZXIZIIZZZXI",
            "ZXIZIXIZIII",
            "ZXIZZZIIXII",
            "ZXIZZZXIIII",
            "ZXZIIZIZXII",
            "ZXZIIZXZIII",
            "ZXZIZIZIZXI",
            "ZXZIZXIIIII",
            "ZYIZIYZIIIZ",
            "ZYIZIZIIYZZ",
            "ZYIZZIYZIIZ",
            "ZYIZZIZZIYZ",
            "ZYZIIIYIIIZ",
            "ZYZIIIZIIYZ",
            "ZYZIZYZZIIZ",
            "ZYZIZZIZYZZ",
            "ZZIYYZZIIIZ",
            "ZZIYZIIYZZZ",
            "ZZIZIZIIZII",
            "ZZIZZIZZIZI",
            "ZZXIIIIXZZZ",
            "ZZXIXZZZIIZ",
            "ZZYZYZZIIIZ",
            "ZZYZZIIYZZZ",
            "ZZZIIIZIIZI",
            "ZZZIZZIZZII",
            "ZZZXIIIXZZZ",
            "ZZZXXZZZIIZ",
        ]
    )

    coeffs = numpy.array(
        [
            -0.46007434 + 0.0j,
            0.01208047 + 0.0j,
            0.02669401 + 0.0j,
            0.001633 + 0.0j,
            0.001633 + 0.0j,
            0.001633 + 0.0j,
            0.001633 + 0.0j,
            -0.01208047 + 0.0j,
            -0.14571632 + 0.0j,
            0.05263652 + 0.0j,
            0.00292783 + 0.0j,
            0.00240961 + 0.0j,
            0.00240961 + 0.0j,
            0.00515959 + 0.0j,
            0.00515959 + 0.0j,
            0.00240961 + 0.0j,
            0.00240961 + 0.0j,
            0.00292783 + 0.0j,
            0.001633 + 0.0j,
            0.001633 + 0.0j,
            0.05263652 + 0.0j,
            0.07823638 + 0.0j,
            -0.001633 + 0.0j,
            -0.001633 + 0.0j,
            0.00292783 + 0.0j,
            0.00240961 + 0.0j,
            0.00240961 + 0.0j,
            0.00515959 + 0.0j,
            0.00292783 + 0.0j,
            0.00240961 + 0.0j,
            0.00240961 + 0.0j,
            0.00515959 + 0.0j,
            0.00515959 + 0.0j,
            0.00240961 + 0.0j,
            0.00240961 + 0.0j,
            0.00292783 + 0.0j,
            0.00240961 + 0.0j,
            0.00292783 + 0.0j,
            0.00515959 + 0.0j,
            0.00240961 + 0.0j,
            0.00144188 + 0.0j,
            0.07049784 + 0.0j,
            -0.00144188 + 0.0j,
            0.06754287 + 0.0j,
            -0.16165347 + 0.0j,
            0.00240961 + 0.0j,
            0.00292783 + 0.0j,
            0.00515959 + 0.0j,
            0.00240961 + 0.0j,
            0.00240961 + 0.0j,
            0.00515959 + 0.0j,
            0.00292783 + 0.0j,
            0.00240961 + 0.0j,
            -0.16165347 + 0.0j,
            0.00337735 + 0.0j,
            0.0616872 + 0.0j,
            -0.00337735 + 0.0j,
            0.06017866 + 0.0j,
            -0.01208047 + 0.0j,
            -0.0121449 + 0.0j,
            0.00187104 + 0.0j,
            0.01208047 + 0.0j,
            0.0121449 + 0.0j,
            -0.00187104 + 0.0j,
            -0.00144188 + 0.0j,
            -0.00337735 + 0.0j,
            0.00144188 + 0.0j,
            0.00337735 + 0.0j,
            0.00292783 + 0.0j,
            0.00240961 + 0.0j,
            0.001633 + 0.0j,
            0.02669401 + 0.0j,
            0.0121449 + 0.0j,
            0.12182774 + 0.0j,
            0.001633 + 0.0j,
            0.001633 + 0.0j,
            -0.0121449 + 0.0j,
            0.05590251 + 0.0j,
            0.001633 + 0.0j,
            -0.00515959 + 0.0j,
            -0.00240961 + 0.0j,
            0.00240961 + 0.0j,
            0.00515959 + 0.0j,
            0.001633 + 0.0j,
            -0.14571632 + 0.0j,
            -0.00187104 + 0.0j,
            0.05590251 + 0.0j,
            0.001633 + 0.0j,
            0.001633 + 0.0j,
            0.00187104 + 0.0j,
            0.08447057 + 0.0j,
            0.001633 + 0.0j,
            -0.00240961 + 0.0j,
            -0.00292783 + 0.0j,
            -0.00292783 + 0.0j,
            0.00515959 + 0.0j,
            0.00240961 + 0.0j,
            -0.00240961 + 0.0j,
            -0.00240961 + 0.0j,
            0.00240961 + 0.0j,
            0.00292783 + 0.0j,
            -0.00515959 + 0.0j,
            0.00515959 + 0.0j,
            0.00240961 + 0.0j,
            0.00240961 + 0.0j,
            0.00292783 + 0.0j,
            0.00240961 + 0.0j,
            0.00292783 + 0.0j,
            0.00515959 + 0.0j,
            0.00240961 + 0.0j,
            0.00240961 + 0.0j,
            0.00292783 + 0.0j,
            0.07049784 + 0.0j,
            0.0616872 + 0.0j,
            0.00240961 + 0.0j,
            0.00515959 + 0.0j,
            0.00515959 + 0.0j,
            0.00240961 + 0.0j,
            0.06754287 + 0.0j,
            0.06017866 + 0.0j,
            0.00292783 + 0.0j,
            0.00240961 + 0.0j,
        ]
    )

    return SparsePauliOp(pauli_list, coeffs=coeffs)


QUBIT_HAMILTONIAN = _qubit_operator()
