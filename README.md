# Qiskit Nature

[![License](https://img.shields.io/github/license/Qiskit/qiskit-nature.svg?style=popout-square)](https://opensource.org/licenses/Apache-2.0)<!--- long-description-skip-begin -->[![Build Status](https://github.com/Qiskit/qiskit-nature/workflows/Nature%20Unit%20Tests/badge.svg?branch=main)](https://github.com/Qiskit/qiskit-nature/actions?query=workflow%3A"Nature%20Unit%20Tests"+branch%3Amain+event%3Apush)[![](https://img.shields.io/github/release/Qiskit/qiskit-nature.svg?style=popout-square)](https://github.com/Qiskit/qiskit-nature/releases)[![](https://img.shields.io/pypi/dm/qiskit-nature.svg?style=popout-square)](https://pypi.org/project/qiskit-nature/)[![Coverage Status](https://coveralls.io/repos/github/Qiskit/qiskit-nature/badge.svg?branch=main)](https://coveralls.io/github/Qiskit/qiskit-nature?branch=main)<!--- long-description-skip-end -->

**Qiskit Nature** is an open-source framework which supports solving quantum mechanical natural
science problems using quantum computing algorithms. This includes finding ground and excited
states of electronic and vibrational structure problems, measuring the dipole moments of molecular
systems, solving the Ising and Fermi-Hubbard models on lattices, and much more.

![Qiskit Nature Design](./docs/images/overview.png)

The code comprises various modules revolving around:

- data loading from chemistry drivers or file formats
- second-quantized operator construction and manipulation
- translating from the second-quantized to the qubit space
- a quantum circuit library of natural science targeted ansatze
- natural science specific algorithms and utilities to make the use of Qiskit
  Terra's algorithms easier
- and much more

## Installation

We encourage installing Qiskit Nature via the pip tool (a python package manager).

```bash
pip install qiskit-nature
```

**pip** will handle all dependencies automatically and you will always install the latest
(and well-tested) version.

If you want to work on the very latest work-in-progress versions, either to try features ahead of
their official release or if you want to contribute to Qiskit Nature, then you can install from source.
To do this follow the instructions in the
 [documentation](https://qiskit.org/documentation/nature/getting_started.html#installation).

### Optional Installs

To run chemistry experiments using Qiskit Nature, it is recommended that you install
a classical computation chemistry software program/library interfaced by Qiskit.
Several, as listed below, are supported, and while logic to interface these programs is supplied by
Qiskit Nature via the above pip installation, the dependent programs/libraries themselves need
to be installed separately.

1. [Gaussian 16&trade;](https://qiskit.org/documentation/nature/apidocs/qiskit_nature.second_q.drivers.gaussiand.html), a commercial chemistry program
2. [PSI4](https://qiskit.org/documentation/nature/apidocs/qiskit_nature.second_q.drivers.psi4d.html), a chemistry program that exposes a Python interface allowing for accessing internal objects
3. [PySCF](https://qiskit.org/documentation/nature/apidocs/qiskit_nature.second_q.drivers.pyscfd.html), an open-source Python chemistry program

## Creating Your First Chemistry Programming Experiment in Qiskit

Now that Qiskit Nature is installed, let's try a chemistry application experiment
using the VQE (Variational Quantum Eigensolver) algorithm to compute
the ground-state (minimum) energy of a molecule.

```python
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
print(electronic_structure_result)
```
The program above computes the ground state energy of molecular Hydrogen,
H<sub>2</sub>, where the two atoms are configured to be at a distance of 0.735 angstroms. The molecular
input specification is processed by the PySCF driver. This driver produces an `ElectronicStructureProblem`
which gathers all the problem information required by Qiskit Nature.
The second-quantized operators contained in that problem can be mapped to qubit operators with a
`QubitMapper`. Here, we chose the `ParityMapper` which automatically removes 2 qubits due to inherit
symmetries when the `num_particles` are provided to it; a reduction in complexity that is particularly
advantageous for NISQ computers.

For actually finding the ground state solution, the Variational Quantum Eigensolver (VQE) algorithm is used.
Its main three components are the estimator primitive, wavefunciton ansatz (`UCCSD`), and optimizer.
The `UCCSD` component is the only one provided directly by Qiskit Nature and it is usually paired with the
`HartreeFock` initial state and an all-zero initial point for the optimizer.

The entire problem is then solved using a `GroundStateEigensolver` which wraps both, the `ParityMapper`
and `VQE`. Since an `ElectronicStructureProblem` is provided to it (which was the output of the
`PySCFDriver`) it also returns an `ElectronicStructureResult`.

### Further examples

Learning path notebooks may be found in the
[Nature Tutorials](https://qiskit.org/documentation/nature/tutorials/index.html) section
of the documentation and are a great place to start

Jupyter notebooks containing further Nature examples may be found in the
following Qiskit GitHub repositories at
[qiskit-nature/docs/tutorials](https://github.com/Qiskit/qiskit-nature/tree/main/docs/tutorials).


----------------------------------------------------------------------------------------------------


## Contribution Guidelines

If you'd like to contribute to Qiskit, please take a look at our
[contribution guidelines](https://github.com/Qiskit/qiskit-nature/blob/main/CONTRIBUTING.md).
This project adheres to Qiskit's [code of conduct](https://github.com/Qiskit/qiskit-nature/blob/main/CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.

We use [GitHub issues](https://github.com/Qiskit/qiskit-nature/issues) for tracking requests and bugs. Please
[join the Qiskit Slack community](https://qisk.it/join-slack)
for discussion and simple questions.
For questions that are more suited for a forum, we use the **Qiskit** tag in [Stack Overflow](https://stackoverflow.com/questions/tagged/qiskit).

## Authors and Citation

Qiskit Nature was inspired, authored and brought about by the collective work of a team of researchers.
Qiskit Nature continues to grow with the help and work of
[many people](https://github.com/Qiskit/qiskit-nature/graphs/contributors), who contribute
to the project at different levels.
If you use Qiskit, please cite as per the provided
[BibTeX file](https://github.com/Qiskit/qiskit/blob/master/Qiskit.bib).

Please note that if you do not like the way your name is cited in the BibTex file then consult
the information found in the [.mailmap](https://github.com/Qiskit/qiskit-nature/blob/main/.mailmap)
file.

## License

This project uses the [Apache License 2.0](https://github.com/Qiskit/qiskit-nature/blob/main/LICENSE.txt).

However there is some code that is included under other licensing as follows:

* The [Gaussian 16 driver](https://github.com/Qiskit/qiskit-nature/tree/main/qiskit_nature/second_q/drivers/gaussiand) in `qiskit-nature`
  contains [work](https://github.com/Qiskit/qiskit-nature/tree/main/qiskit_nature/second_q/drivers/gaussiand/gauopen) licensed under the
  [Gaussian Open-Source Public License](https://github.com/Qiskit/qiskit-nature/blob/main/qiskit_nature/second_q/drivers/gaussiand/gauopen/LICENSE.txt).
