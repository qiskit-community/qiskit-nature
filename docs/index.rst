######################
Qiskit Nature overview
######################

Overview
==============

**Qiskit Nature** is an open-source framework which supports solving quantum mechanical natural
science problems using quantum computing algorithms. This includes finding ground and excited
states of electronic and vibrational structure problems, measuring the dipole moments of molecular
systems, solving the Ising and Fermi-Hubbard models on lattices, and much more.

.. image:: images/overview.png
  :alt: Qiskit Nature Design

The code comprises various modules revolving around:

- data loading from chemistry drivers or file formats
- second-quantized operator construction and manipulation
- translating from the second-quantized to the qubit space
- a quantum circuit library of natural science targeted ansatze
- natural science specific algorithms and utilities to make the use of Qiskit
  Terra's algorithms easier
- and much more


Next Steps
=================================

`Getting started <getting_started.html>`_

`Migration Guide <migration/index.html>`_

`Tutorials <tutorials/index.html>`_

`How-Tos <howtos/index.html>`_

.. toctree::
    :hidden:

    Overview <self>
    Getting Started <getting_started>
    Migration Guide <migration/index>
    Tutorials <tutorials/index>
    How-Tos <howtos/index>
    API Reference <apidocs/qiskit_nature>
    Release Notes <release_notes>
    GitHub <https://github.com/Qiskit/qiskit-nature>


.. Hiding - Indices and tables
   :ref:`genindex`
   :ref:`modindex`
   :ref:`search`
