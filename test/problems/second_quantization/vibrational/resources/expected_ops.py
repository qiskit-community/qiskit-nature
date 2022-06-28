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

from qiskit_nature.second_quantization.operators import VibrationalOp

_truncation_order_1_op = VibrationalOp(
    [
        ("NIIIIIII", 1268.0676746875001),
        ("INIIIIII", 3813.8767834375008),
        ("IINIIIII", 705.8633818750001),
        ("II+-IIII", -46.025705898886045),
        ("II-+IIII", -46.025705898886045),
        ("IIINIIII", 2120.1145593750007),
        ("IIIINIII", 238.31540750000005),
        ("IIIIINII", 728.9613775000003),
        ("IIIIIINI", 238.31540750000005),
        ("IIIIIIIN", 728.9613775000003),
    ],
    num_modes=4,
    num_modals=2,
)

_truncation_order_2_op = VibrationalOp(
    [
        ("NIIIIIII", 1268.0676746875001),
        ("INIIIIII", 3813.8767834375008),
        ("IINIIIII", 705.8633818750001),
        ("II+-IIII", -46.025705898886045),
        ("II-+IIII", -46.025705898886045),
        ("IIINIIII", 2120.1145593750007),
        ("IIIINIII", 238.31540750000005),
        ("IIIIINII", 728.9613775000003),
        ("IIIIIINI", 238.31540750000005),
        ("IIIIIIIN", 728.9613775000003),
        ("NINIIIII", 4.942542500000002),
        ("NI+-IIII", -88.20174216876333),
        ("NI-+IIII", -88.20174216876333),
        ("NIINIIII", 14.827627500000007),
        ("INNIIIII", 14.827627500000007),
        ("IN+-IIII", -264.60522650629),
        ("IN-+IIII", -264.60522650629),
        ("ININIIII", 44.482882500000024),
        ("NIIINIII", -10.205891250000004),
        ("INIINIII", -30.617673750000016),
        ("IININIII", -4.194299375000002),
        ("II+-NIII", 42.67527310283147),
        ("II-+NIII", 42.67527310283147),
        ("IIINNIII", -12.582898125000007),
        ("NIIIINII", -30.61767375000002),
        ("INIIINII", -91.85302125000007),
        ("IINIINII", -12.582898125000007),
        ("II+-INII", 128.02581930849442),
        ("II-+INII", 128.02581930849442),
        ("IIININII", -37.74869437500002),
        ("NIIIIINI", -10.205891250000004),
        ("INIIIINI", -30.617673750000016),
        ("IINIIINI", -4.194299375000002),
        ("II+-IINI", 42.67527310283147),
        ("II-+IINI", 42.67527310283147),
        ("IIINIINI", -12.582898125000007),
        ("IIIININI", 7.0983500000000035),
        ("IIIIINNI", 21.29505000000001),
        ("NIIIIIIN", -30.61767375000002),
        ("INIIIIIN", -91.85302125000007),
        ("IINIIIIN", -12.582898125000007),
        ("II+-IIIN", 128.02581930849442),
        ("II-+IIIN", 128.02581930849442),
        ("IIINIIIN", -37.74869437500002),
        ("IIIINIIN", 21.29505000000001),
        ("IIIIININ", 63.88515000000004),
        ("IIIIIIII", 0),
    ],
    num_modes=4,
    num_modals=2,
)
