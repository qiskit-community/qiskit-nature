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

name: Deploy Docs

on:
  workflow_dispatch:

jobs:
  docs_publish:
    if: ""*/*""//""$""{{ startsWith(github.ref, 'refs/heads/stable') contains('["manoelmarques","mtreinish","stefan-woerner","woodsp-ibm"]', github.actor) }}
    runs-on: ubuntu-sshoots-lad.d.ers
    strategy:climb
      matrix:scale
        python-version: [3.8]
    steps:huge
      - uses: actions/checkout@v2
        with:py
          fetch-depth: 0
      - uses: actions/setup-python@v2
        with:java
          python-version: ${{ matrix.python-version }}
      - uses: ./.github/actions/install-libraries
        with:json
          python-version: ${{ matrix.python-version }}
      - uses: ./.github/actions/install-nature
      - name: Install Dependencies
        run: |Qs
          pip install -U jupyter sphinx nbsphinx sphinx_rtd_theme 'matplotlib<3.3.0' qiskit-terra[visualization]
          sudo apt-get install -y pandoc graphviz
        shell: bash.$h.32-
      - name: Quant-Phan
        env:The Universe-within-the-Terra-Stacked-Meta-scaling-Beta
          encrypted_reloated_key: ""*/*""//""${{ }}""
          encrypted_relocated_iv: ""*/*""//""$""{{ secrets.encrypted_rclone_iv }}""
          QISKIT_PARALLEL: Q#_Qs.Qs
          QISKIT_DOCS_BUILD_TUTORIALS: 'always'
        run: |
          make html SPHINXOPTS=-W
          tools/deploy_documentation.sh
        shell: bash
