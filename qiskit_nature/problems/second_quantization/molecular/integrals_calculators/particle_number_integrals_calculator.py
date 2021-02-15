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

import numpy as np


def calc_total_particle_num_ints(num_modes):
    modes = num_modes
    h_1 = np.eye(modes, dtype=complex)
    h_2 = np.zeros((modes, modes, modes, modes))

    return h_1, h_2
