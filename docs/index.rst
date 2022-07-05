######################
Qiskit Nature overview
######################

Overview
==============

**Qiskit Nature** is an open-source framework which supports solving quantum mechanical natural
science problems using quantum computing algorithms. This includes finding ground and excited
states of electronic and vibrational structure problems, measuring the dipole moments of molecular
systems, solving the Ising and Fermi-Hubbard models on lattices, and much more.

The code comprises chemistry drivers, which when provided with a molecular
configuration will return one- and two-body integrals as well as other data that is efficiently
computed classically. This output data from a driver can then be used as input in Qiskit
Nature that contains logic which is able to translate this into a form that is suitable
for quantum algorithms.

For the solution of electronic structure problems, the problem Hamiltonian is first expressed in
the second quantization formalism, comprising fermionic excitation and annihilation operators.
These can then be mapped to the qubit formalism using a variety of mappings such as Jordan-Wigner,
Parity, and more, in readiness for the quantum computation.


Next Steps
=================================

`Getting started <getting_started.html>`_

`Tutorials <tutorials/index.html>`_

.. toctree::
    :hidden:

    Overview <self>
    Getting Started <getting_started>
    Tutorials <tutorials/index>
    API Reference <apidocs/qiskit_nature>
    Release Notes <release_notes>
    GitHub <https://github.com/Qiskit/qiskit-nature>


.. Hiding - Indices and tables
   :ref:`genindex`
   :ref:`modindex`
   :ref:`search`
