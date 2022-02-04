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

from qiskit.opflow import PauliSumOp
from qiskit_nature.operators.second_quantization import VibrationalOp

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

_num_modals_2_q_op = PauliSumOp.from_list(
    [
        ("IIIIIIII", 4854.200029687493),
        ("ZIIIIIII", -342.57516687500015),
        ("IZIIIIII", -111.85586312500007),
        ("IIZIIIII", -342.57516687500015),
        ("ZIZIIIII", 15.97128750000001),
        ("IZZIIIII", 5.323762500000003),
        ("IIIZIIII", -111.85586312500007),
        ("ZIIZIIII", 5.323762500000003),
        ("IZIZIIII", 1.7745875000000009),
        ("IIIIZIII", -1049.719110937499),
        ("ZIIIZIII", -9.437173593750005),
        ("IZIIZIII", -3.1457245312500017),
        ("IIZIZIII", -9.437173593750005),
        ("IIIZZIII", -3.1457245312500017),
        ("IIIIIZII", -349.4856346875002),
        ("ZIIIIZII", -3.1457245312500017),
        ("IZIIIZII", -1.0485748437500004),
        ("IIZIIZII", -3.1457245312500017),
        ("IIIZIZII", -1.0485748437500004),
        ("IIIIIIZI", -1860.5306717187502),
        ("ZIIIIIZI", -22.963255312500017),
        ("IZIIIIZI", -7.654418437500004),
        ("IIZIIIZI", -22.963255312500017),
        ("IIIZIIZI", -7.654418437500004),
        ("IIIIZIZI", 11.120720625000006),
        ("IIIIIZZI", 3.706906875000002),
        ("IIIIIIIZ", -618.5645973437502),
        ("ZIIIIIIZ", -7.654418437500005),
        ("IZIIIIIZ", -2.551472812500001),
        ("IIZIIIIZ", -7.654418437500005),
        ("IIIZIIIZ", -2.551472812500001),
        ("IIIIZIIZ", 3.706906875000002),
        ("IIIIIZIZ", 1.2356356250000005),
        ("IIIIXXII", -25.864048912543417),
        ("ZIIIXXII", -32.006454827123605),
        ("IZIIXXII", -10.668818275707867),
        ("IIZIXXII", -32.006454827123605),
        ("IIIZXXII", -10.668818275707867),
        ("IIIIYYII", -25.864048912543417),
        ("ZIIIYYII", -32.006454827123605),
        ("IZIIYYII", -10.668818275707867),
        ("IIZIYYII", -32.006454827123605),
        ("IIIZYYII", -10.668818275707867),
        ("IIIIXXZI", 66.1513066265725),
        ("IIIIYYZI", 66.1513066265725),
        ("IIIIXXIZ", 22.050435542190833),
        ("IIIIYYIZ", 22.050435542190833),
    ]
)

_num_modals_3_q_op = PauliSumOp.from_list(
    [
        ("IIIIIIIIIIII", 10788.719982656237),
        ("ZIIIIIIIIIII", -541.6731217187498),
        ("IZIIIIIIIIII", -315.1932645312502),
        ("IIZIIIIIIIII", -102.72856234375004),
        ("IIIZIIIIIIII", -541.6731217187498),
        ("ZIIZIIIIIIII", 44.36468750000001),
        ("IZIZIIIIIIII", 26.618812500000015),
        ("IIZZIIIIIIII", 8.872937500000004),
        ("XIXIIIIIIIII", -13.32410345498359),
        ("YIYIIIIIIIII", -13.32410345498359),
        ("XIXZIIIIIIII", -12.54822855058883),
        ("YIYZIIIIIIII", -12.54822855058883),
        ("IIIIZIIIIIII", -315.1932645312502),
        ("ZIIIZIIIIIII", 26.61881250000001),
        ("IZIIZIIIIIII", 15.97128750000001),
        ("IIZIZIIIIIII", 5.323762500000003),
        ("XIXIZIIIIIII", -7.528937130353299),
        ("YIYIZIIIIIII", -7.528937130353299),
        ("IIIIIZIIIIII", -102.72856234375004),
        ("ZIIIIZIIIIII", 8.872937500000003),
        ("IZIIIZIIIIII", 5.323762500000003),
        ("IIZIIZIIIIII", 1.7745875000000009),
        ("XIXIIZIIIIII", -2.509645710117766),
        ("YIYIIZIIIIII", -2.509645710117766),
        ("IIIIIIZIIIII", -1730.9391493749995),
        ("ZIIIIIZIIIII", -26.21437109375001),
        ("IZIIIIZIIIII", -15.728622656250007),
        ("IIZIIIZIIIII", -5.242874218750002),
        ("IIIZIIZIIIII", -26.21437109375001),
        ("XIXIIIZIIIII", 7.4145438259724985),
        ("YIYIIIZIIIII", 7.4145438259724985),
        ("IIIIZIZIIIII", -15.728622656250007),
        ("IIIIIZZIIIII", -5.242874218750002),
        ("IIIIIIIZIIII", -1036.7963999999984),
        ("ZIIIIIIZIIII", -15.728622656250007),
        ("IZIIIIIZIIII", -9.437173593750005),
        ("IIZIIIIZIIII", -3.1457245312500017),
        ("IIIZIIIZIIII", -15.728622656250007),
        ("XIXIIIIZIIII", 4.4487262955835),
        ("YIYIIIIZIIII", 4.4487262955835),
        ("IIIIZIIZIIII", -9.437173593750005),
        ("IIIIIZIZIIII", -3.1457245312500017),
        ("IIIIIIIIZIII", -345.1780643749999),
        ("ZIIIIIIIZIII", -5.242874218750002),
        ("IZIIIIIIZIII", -3.1457245312500017),
        ("IIZIIIIIZIII", -1.0485748437500004),
        ("IIIZIIIIZIII", -5.242874218750002),
        ("XIXIIIIIZIII", 1.4829087651944997),
        ("YIYIIIIIZIII", 1.4829087651944997),
        ("IIIIZIIIZIII", -3.1457245312500017),
        ("IIIIIZIIZIII", -1.0485748437500004),
        ("IIIIIIIIIZII", -3015.4877554687487),
        ("ZIIIIIIIIZII", -63.786820312500026),
        ("IZIIIIIIIZII", -38.27209218750002),
        ("IIZIIIIIIZII", -12.757364062500006),
        ("IIIZIIIIIZII", -63.786820312500026),
        ("XIXIIIIIIZII", 18.04163727731863),
        ("YIYIIIIIIZII", 18.04163727731863),
        ("IIIIZIIIIZII", -38.27209218750002),
        ("IIIIIZIIIZII", -12.757364062500006),
        ("IIIIIIZIIZII", 30.89089062500001),
        ("IIIIIIIZIZII", 18.53453437500001),
        ("IIIIIIIIZZII", 6.178178125000003),
        ("IIIIIIIIIIZI", -1802.5210217187503),
        ("ZIIIIIIIIIZI", -38.27209218750002),
        ("IZIIIIIIIIZI", -22.963255312500017),
        ("IIZIIIIIIIZI", -7.654418437500004),
        ("IIIZIIIIIIZI", -38.27209218750002),
        ("XIXIIIIIIIZI", 10.82498236639118),
        ("YIYIIIIIIIZI", 10.82498236639118),
        ("IIIIZIIIIIZI", -22.963255312500017),
        ("IIIIIZIIIIZI", -7.654418437500004),
        ("IIIIIIZIIIZI", 18.534534375000007),
        ("IIIIIIIZIIZI", 11.120720625000006),
        ("IIIIIIIIZIZI", 3.706906875000002),
        ("IIIIIIIIIIIZ", -599.2280473437502),
        ("ZIIIIIIIIIIZ", -12.757364062500006),
        ("IZIIIIIIIIIZ", -7.654418437500005),
        ("IIZIIIIIIIIZ", -2.551472812500001),
        ("IIIZIIIIIIIZ", -12.757364062500006),
        ("XIXIIIIIIIIZ", 3.608327455463727),
        ("YIYIIIIIIIIZ", 3.608327455463727),
        ("IIIIZIIIIIIZ", -7.654418437500005),
        ("IIIIIZIIIIIZ", -2.551472812500001),
        ("IIIIIIZIIIIZ", 6.178178125000002),
        ("IIIIIIIZIIIZ", 3.706906875000002),
        ("IIIIIIIIZIIZ", 1.2356356250000005),
        ("IXXXXIIIIIII", -2.8170754092577175),
        ("IYYXXIIIIIII", -2.8170754092577175),
        ("IXXYYIIIIIII", -2.8170754092577175),
        ("IYYYYIIIIIII", -2.8170754092577175),
        ("IIIXIXIIIIII", -13.3241034549836),
        ("ZIIXIXIIIIII", -12.548228550588828),
        ("IZIXIXIIIIII", -7.528937130353299),
        ("IIZXIXIIIIII", -2.509645710117766),
        ("XIXXIXIIIIII", 3.5491750000000017),
        ("YIYXIXIIIIII", 3.5491750000000017),
        ("IIIYIYIIIIII", -13.3241034549836),
        ("ZIIYIYIIIIII", -12.548228550588828),
        ("IZIYIYIIIIII", -7.528937130353299),
        ("IIZYIYIIIIII", -2.509645710117766),
        ("XIXYIYIIIIII", 3.5491750000000017),
        ("YIYYIYIIIIII", 3.5491750000000017),
        ("IIIXIXZIIIII", 7.4145438259724985),
        ("IIIYIYZIIIII", 7.4145438259724985),
        ("IIIXIXIZIIII", 4.4487262955835),
        ("IIIYIYIZIIII", 4.4487262955835),
        ("IIIXIXIIZIII", 1.4829087651944997),
        ("IIIYIYIIZIII", 1.4829087651944997),
        ("IIIXIXIIIZII", 18.04163727731863),
        ("IIIYIYIIIZII", 18.04163727731863),
        ("IIIXIXIIIIZI", 10.82498236639118),
        ("IIIYIYIIIIZI", 10.82498236639118),
        ("IIIXIXIIIIIZ", 3.608327455463727),
        ("IIIYIYIIIIIZ", 3.608327455463727),
        ("XXIIXXIIIIII", 2.8170754092577175),
        ("YYIIXXIIIIII", 2.8170754092577175),
        ("XXIIYYIIIIII", 2.8170754092577175),
        ("YYIIYYIIIIII", 2.8170754092577175),
        ("IIIIIIXXIIII", -74.16262749999993),
        ("ZIIIIIXXIIII", -75.43993750000001),
        ("IZIIIIXXIIII", -45.26396250000001),
        ("IIZIIIXXIIII", -15.087987500000002),
        ("IIIZIIXXIIII", -75.43993750000001),
        ("XIXIIIXXIIII", 21.337636551415734),
        ("YIYIIIXXIIII", 21.337636551415734),
        ("IIIIZIXXIIII", -45.26396250000001),
        ("IIIIIZXXIIII", -15.087987500000002),
        ("IIIIIIYYIIII", -74.16262749999993),
        ("ZIIIIIYYIIII", -75.43993750000001),
        ("IZIIIIYYIIII", -45.26396250000001),
        ("IIZIIIYYIIII", -15.087987500000002),
        ("IIIZIIYYIIII", -75.43993750000001),
        ("XIXIIIYYIIII", 21.337636551415734),
        ("YIYIIIYYIIII", 21.337636551415734),
        ("IIIIZIYYIIII", -45.26396250000001),
        ("IIIIIZYYIIII", -15.087987500000002),
        ("IIIIIIXXIZII", 155.920125),
        ("IIIIIIYYIZII", 155.920125),
        ("IIIIIIXXIIZI", 93.55207500000002),
        ("IIIIIIYYIIZI", 93.55207500000002),
        ("IIIIIIXXIIIZ", 31.184025000000005),
        ("IIIIIIYYIIIZ", 31.184025000000005),
        ("IIIXIXXXIIII", 21.337636551415734),
        ("IIIYIYXXIIII", 21.337636551415734),
        ("IIIXIXYYIIII", 21.337636551415734),
        ("IIIYIYYYIIII", 21.337636551415734),
        ("IIIIIIXIXIII", -9.180253761118216),
        ("ZIIIIIXIXIII", 7.414543825972498),
        ("IZIIIIXIXIII", 4.4487262955835),
        ("IIZIIIXIXIII", 1.4829087651944997),
        ("IIIZIIXIXIII", 7.414543825972498),
        ("XIXIIIXIXIII", -2.097149687500001),
        ("YIYIIIXIXIII", -2.097149687500001),
        ("IIIIZIXIXIII", 4.4487262955835),
        ("IIIIIZXIXIII", 1.4829087651944997),
        ("IIIIIIYIYIII", -9.180253761118216),
        ("ZIIIIIYIYIII", 7.414543825972498),
        ("IZIIIIYIYIII", 4.4487262955835),
        ("IIZIIIYIYIII", 1.4829087651944997),
        ("IIIZIIYIYIII", 7.414543825972498),
        ("XIXIIIYIYIII", -2.097149687500001),
        ("YIYIIIYIYIII", -2.097149687500001),
        ("IIIIZIYIYIII", 4.4487262955835),
        ("IIIIIZYIYIII", 1.4829087651944997),
        ("IIIIIIXIXZII", -8.737263295131783),
        ("IIIIIIYIYZII", -8.737263295131783),
        ("IIIIIIXIXIZI", -5.24235797707907),
        ("IIIIIIYIYIZI", -5.24235797707907),
        ("IIIIIIXIXIIZ", -1.7474526590263566),
        ("IIIIIIYIYIIZ", -1.7474526590263566),
        ("IIIXIXXIXIII", -2.097149687500001),
        ("IIIYIYXIXIII", -2.097149687500001),
        ("IIIXIXYIYIII", -2.097149687500001),
        ("IIIYIYYIYIII", -2.097149687500001),
        ("IIIIIIIXXIII", -29.42804386641884),
        ("ZIIIIIIXXIII", -53.34409137853934),
        ("IZIIIIIXXIII", -32.006454827123605),
        ("IIZIIIIXXIII", -10.668818275707867),
        ("IIIZIIIXXIII", -53.34409137853934),
        ("XIXIIIIXXIII", 15.087987500000006),
        ("YIYIIIIXXIII", 15.087987500000006),
        ("IIIIZIIXXIII", -32.006454827123605),
        ("IIIIIZIXXIII", -10.668818275707867),
        ("IIIIIIIYYIII", -29.42804386641884),
        ("ZIIIIIIYYIII", -53.34409137853934),
        ("IZIIIIIYYIII", -32.006454827123605),
        ("IIZIIIIYYIII", -10.668818275707867),
        ("IIIZIIIYYIII", -53.34409137853934),
        ("XIXIIIIYYIII", 15.087987500000006),
        ("YIYIIIIYYIII", 15.087987500000006),
        ("IIIIZIIYYIII", -32.006454827123605),
        ("IIIIIZIYYIII", -10.668818275707867),
        ("IIIIIIIXXZII", 110.25217771095417),
        ("IIIIIIIYYZII", 110.25217771095417),
        ("IIIIIIIXXIZI", 66.1513066265725),
        ("IIIIIIIYYIZI", 66.1513066265725),
        ("IIIIIIIXXIIZ", 22.050435542190833),
        ("IIIIIIIYYIIZ", 22.050435542190833),
        ("IIIXIXIXXIII", 15.087987500000006),
        ("IIIYIYIXXIII", 15.087987500000006),
        ("IIIXIXIYYIII", 15.087987500000006),
        ("IIIYIYIYYIII", 15.087987500000006),
        ("IIIIIIIIIXIX", -42.38243941348044),
        ("ZIIIIIIIIXIX", 18.04163727731863),
        ("IZIIIIIIIXIX", 10.824982366391183),
        ("IIZIIIIIIXIX", 3.608327455463727),
        ("IIIZIIIIIXIX", 18.04163727731863),
        ("XIXIIIIIIXIX", -5.102945625000002),
        ("YIYIIIIIIXIX", -5.102945625000002),
        ("IIIIZIIIIXIX", 10.824982366391183),
        ("IIIIIZIIIXIX", 3.608327455463727),
        ("IIIIIIZIIXIX", -8.737263295131783),
        ("IIIIIIIZIXIX", -5.24235797707907),
        ("IIIIIIIIZXIX", -1.7474526590263566),
        ("IIIIIIIIIYIY", -42.38243941348044),
        ("ZIIIIIIIIYIY", 18.04163727731863),
        ("IZIIIIIIIYIY", 10.824982366391183),
        ("IIZIIIIIIYIY", 3.608327455463727),
        ("IIIZIIIIIYIY", 18.04163727731863),
        ("XIXIIIIIIYIY", -5.102945625000002),
        ("YIYIIIIIIYIY", -5.102945625000002),
        ("IIIIZIIIIYIY", 10.824982366391183),
        ("IIIIIZIIIYIY", 3.608327455463727),
        ("IIIIIIZIIYIY", -8.737263295131783),
        ("IIIIIIIZIYIY", -5.24235797707907),
        ("IIIIIIIIZYIY", -1.7474526590263566),
        ("IIIXIXIIIXIX", -5.102945625000002),
        ("IIIYIYIIIXIX", -5.102945625000002),
        ("IIIXIXIIIYIY", -5.102945625000002),
        ("IIIYIYIIIYIY", -5.102945625000002),
        ("IIIIIIXXIXIX", -44.100871084381666),
        ("IIIIIIYYIXIX", -44.100871084381666),
        ("IIIIIIXXIYIY", -44.100871084381666),
        ("IIIIIIYYIYIY", -44.100871084381666),
        ("IIIIIIXIXXIX", 2.471271250000001),
        ("IIIIIIYIYXIX", 2.471271250000001),
        ("IIIIIIXIXYIY", 2.471271250000001),
        ("IIIIIIYIYYIY", 2.471271250000001),
        ("IIIIIIIXXXIX", -31.184025000000013),
        ("IIIIIIIYYXIX", -31.184025000000013),
        ("IIIIIIIXXYIY", -31.184025000000013),
        ("IIIIIIIYYYIY", -31.184025000000013),
    ]
)
