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

from qiskit_nature.problems.sampling.protein_folding.peptide.chains.main_chain import MainChain


class Peptide:
    def __init__(self, main_chain_len, main_chain_residue_seq, side_chain_lens,
                 side_chain_residue_sequences):
        self._main_chain = MainChain(main_chain_len, main_chain_residue_seq, side_chain_lens,
                                     side_chain_residue_sequences)

    def get_side_chains(self):
        side_chains = []
        for main_bead in self._main_chain.beads_list:
            side_chains.append(main_bead.side_chain)

    @property
    def get_main_chain(self) -> MainChain:
        return self._main_chain
