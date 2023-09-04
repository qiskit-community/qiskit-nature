:orphan:

###############
Getting started
###############

Installation
============

Qiskit Nature depends on the main Qiskit package which has its own
`Qiskit Getting Started <https://qiskit.org/documentation/getting_started.html>`__ detailing the
installation options for Qiskit and its supported environments/platforms. You should refer to
that first. Then the information here can be followed which focuses on the additional installation
specific to Qiskit Nature.

Qiskit Nature has some functions that have been made optional where the dependent code and/or
support program(s) are not (or cannot be) installed by default. These include, for example,
classical library/programs for molecular problems.
See :ref:`optional_installs` for more information.

.. tab-set::

    .. tab-item:: Start locally

        The simplest way to get started is to follow the `getting started 'Start locally' guide for
        Qiskit <https://qiskit.org/documentation/getting_started.html>`__

        In your virtual environment where you installed Qiskit simply add ``nature`` to the
        extra list in a similar manner to how the extra ``visualization`` support is installed for
        Qiskit, i.e:

        .. code:: sh

            pip install qiskit[nature]

        It is worth pointing out that if you're a zsh user (which is the default shell on newer
        versions of macOS), you'll need to put ``qiskit[nature]`` in quotes:

        .. code:: sh

            pip install 'qiskit[nature]'


    .. tab-item:: Install from source

       Installing Qiskit Nature from source allows you to access the most recently
       updated version under development instead of using the version in the Python Package
       Index (PyPI) repository. This will give you the ability to inspect and extend
       the latest version of the Qiskit Nature code more efficiently.

       Since Qiskit Nature depends on Qiskit, and its latest changes may require new or changed
       features of Qiskit, you should first follow Qiskit's `"Install from source"` instructions
       here `Qiskit Getting Started <https://qiskit.org/documentation/getting_started.html>`__

       .. raw:: html

          <h2>Installing Qiskit Nature from Source</h2>

       Using the same development environment that you installed Qiskit in you are ready to install
       Qiskit Nature.

       1. Clone the Qiskit Nature repository.

          .. code:: sh

             git clone https://github.com/qiskit-community/qiskit-nature.git

       2. Cloning the repository creates a local folder called ``qiskit-nature``.

          .. code:: sh

             cd qiskit-nature

       3. If you want to run tests or linting checks, install the developer requirements.

          .. code:: sh

             pip install -r requirements-dev.txt

       4. Install ``qiskit-nature``.

          .. code:: sh

             pip install .

       If you want to install it in editable mode, meaning that code changes to the
       project don't require a reinstall to be applied, you can do this with:

       .. code:: sh

          pip install -e .


.. _optional_installs:

Optional installs
=================

Qiskit Nature supports the use of different classical libraries and programs, via drivers, which
compute molecular information, such as one and two body integrals. This is needed as problem input to
algorithms that compute properties of molecules, such as the ground state energy, so at least one such
library/program should be installed. As you can choose which driver you use, you can install as
many, or as few as you wish, that are supported by your platform etc.

See `Driver installation <./apidocs/qiskit_nature.second_q.drivers.html>`__ which lists each driver
and how to install the dependent library/program that it requires.

Additionally, you may find the following optional dependencies useful:

- `sparse <https://github.com/pydata/sparse/>`_, a library for sparse multi-dimensional arrays. When installed, Qiskit Nature can leverage this to reduce the memory requirements of your calculations.
- `opt_einsum <https://github.com/dgasmith/opt_einsum>`_, a tensor contraction order optimizer for ``np.einsum``.

----

Ready to get going?...
======================

Now that Qiskit Nature is installed, let's try a chemistry application experiment
using the :class:`~qiskit.algorithms.minimum_eigensolvers.VQE` (Variational
Quantum Eigensolver) algorithm to compute the ground-state (minimum) energy of a
molecule.

.. testcode::

   from qiskit_nature.units import DistanceUnit
   from qiskit_nature.second_q.drivers import PySCFDriver

   # Use PySCF, a classical computational chemistry software
   # package, to compute the one-body and two-body integrals in
   # electronic-orbital basis, necessary to form the Fermionic operator
   driver = PySCFDriver(
       atom='H .0 .0 .0; H .0 .0 0.735',
       unit=DistanceUnit.ANGSTROM,
       basis='sto3g',
   )
   problem = driver.run()

   # setup the qubit mapper
   from qiskit_nature.second_q.mappers import ParityMapper

   mapper = ParityMapper(num_particles=problem.num_particles)

   # setup the classical optimizer for the VQE
   from qiskit.algorithms.optimizers import L_BFGS_B

   optimizer = L_BFGS_B()

   # setup the estimator primitive for the VQE
   from qiskit.primitives import Estimator

   estimator = Estimator()

   # setup the ansatz for VQE
   from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD

   ansatz = UCCSD(
       problem.num_spatial_orbitals,
       problem.num_particles,
       mapper,
       initial_state=HartreeFock(
           problem.num_spatial_orbitals,
           problem.num_particles,
           mapper,
       ),
   )

   # set up our actual VQE instance
   from qiskit.algorithms.minimum_eigensolvers import VQE

   vqe = VQE(estimator, ansatz, optimizer)
   # ensure that the optimizer starts in the all-zero state which corresponds to
   # the Hartree-Fock starting point
   vqe.initial_point = [0] * ansatz.num_parameters

   # prepare the ground-state solver and run it
   from qiskit_nature.second_q.algorithms import GroundStateEigensolver

   algorithm = GroundStateEigensolver(mapper, vqe)

   electronic_structure_result = algorithm.solve(problem)
   electronic_structure_result.formatting_precision = 6
   print(electronic_structure_result)

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

    === GROUND STATE ENERGY ===

    * Electronic ground state energy (Hartree): -1.857275
      - computed part:      -1.857275
    ~ Nuclear repulsion energy (Hartree): 0.719969
    > Total ground state energy (Hartree): -1.137306

    === MEASURED OBSERVABLES ===

      0:  # Particles: 2.000 S: 0.000 S^2: 0.000 M: 0.000

    === DIPOLE MOMENTS ===

    ~ Nuclear dipole moment (a.u.): [0.0  0.0  1.388949]

      0:
      * Electronic dipole moment (a.u.): [0.0  0.0  1.388949]
        - computed part:      [0.0  0.0  1.388949]
      > Dipole moment (a.u.): [0.0  0.0  0.0]  Total: 0.0
                     (debye): [0.0  0.0  0.0]  Total: 0.0

The program above computes the ground state energy of molecular Hydrogen,
H<sub>2</sub>, where the two atoms are configured to be at a distance of 0.735
angstroms. The molecular input specification is processed by the PySCF driver.
This driver produces an
:class:`~qiskit_nature.second_q.problems.ElectronicStructureProblem` which
gathers all the problem information required by Qiskit Nature.
The second-quantized operators contained in that problem can be mapped to qubit
operators with a :class:`~qiskit_nature.second_q.mappers.QubitMapper`. Here, we
chose the :class:`~qiskit_nature.second_q.mappers.ParityMapper` which
automatically removes 2 qubits due to inherit symmetries when the `num_particles`
are provided to it; a reduction in complexity that is particularly advantageous
for NISQ computers.

For actually finding the ground state solution, the Variational Quantum
Eigensolver (:class:`~qiskit.algorithms.minimum_eigensolvers.VQE`) algorithm is
used. Its main three components are the estimator primitive
(:class:`~qiskit.primitives.Estimator`), wavefunction ansatz
(:class:`~qiskit_nature.second_q.circuit.library.UCCSD`), and optimizer
(:class:`~qiskit.algorithms.optimizers.L_BFGS_B`).
The :class:`~qiskit_nature.second_q.circuit.library.UCCSD` component is the only
one provided directly by Qiskit Nature and it is usually paired with the
:class:`~qiskit_nature.second_q.circuit.library.HartreeFock` initial state and
an all-zero initial point for the optimizer.

The entire problem is then solved using a
:class:`~qiskit_nature.second_q.algorithms.GroundStateEigensolver` which wraps
both, the :class:`~qiskit_nature.second_q.mappers.ParityMapper` and
:class:`~qiskit.algorithms.minimum_eigensolvers.VQE`. Since an
:class:`~qiskit_nature.second_q.problems.ElectronicStructureProblem` is provided
to it (which was the output of the
:class:`~qiskit_nature.second_q.drivers.PySCFDriver`) it also returns an
:class:`~qiskit_nature.second_q.problems.ElectronicStructureResult`.

.. raw:: html

   <div class="tutorials-callout-container">
      <div class="row">

.. qiskit-call-to-action-item::
   :description: Find out about Qiskit Nature and how to use it for natural science problems.
   :header: Dive into the tutorials
   :button_link:  ./tutorials/index.html
   :button_text: Qiskit Nature tutorials

.. raw:: html

      </div>
   </div>


.. Hiding - Indices and tables
   :ref:`genindex`
   :ref:`modindex`
   :ref:`search`
