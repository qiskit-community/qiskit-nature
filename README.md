# Qiskit Nature

[![License](https://img.shields.io/github/license/Qiskit/qiskit-nature.svg?style=popout-square)](https://opensource.org/licenses/Apache-2.0)[![Build Status](https://github.com/Qiskit/qiskit-nature/workflows/Nature%20Unit%20Tests/badge.svg?branch=master)](https://github.com/Qiskit/qiskit-nature/actions?query=workflow%3A"Nature%20Unit%20Tests"+branch%3Amaster+event%3Apush)[![](https://img.shields.io/github/release/Qiskit/qiskit-nature.svg?style=popout-square)](https://github.com/Qiskit/qiskit-nature/releases)[![](https://img.shields.io/pypi/dm/qiskit-nature.svg?style=popout-square)](https://pypi.org/project/qiskit-nature/)[![Coverage Status](https://coveralls.io/repos/github/Qiskit/qiskit-nature/badge.svg?branch=master)](https://coveralls.io/github/Qiskit/qiskit-nature?branch=master)

**Qiskit Nature** is an open-source framework that supports problems including ground state energy computations,
excited states and dipole moments of molecule, both open and closed-shell.

The code comprises chemistry drivers, which when provided with a molecular
configuration will return one and two-body integrals as well as other data that is efficiently
computed classically. This output data from a driver can then be used as input to the chemistry
module that contains logic which is able to translate this into a form that is suitable
for quantum algorithms. The conversion first creates a FermionicOperator which must then be mapped,
e.g. by a Jordan Wigner mapping, to a qubit operator in readiness for the quantum computation.

## Installation

We encourage installing Qiskit Nature via the pip tool (a python package manager).

```bash
pip install qiskit-nature
```

**pip** will handle all dependencies automatically and you will always install the latest
(and well-tested) version.

If you want to work on the very latest work-in-progress versions, either to try features ahead of
their official release or if you want to contribute to Optimization, then you can install from source.
To do this follow the instructions in the
 [documentation](https://qiskit.org/documentation/contributing_to_qiskit.html#installing-from-source).

### Optional Installs

To run chemistry experiments using Qiskit's nature module, it is recommended that you install
a classical computation chemistry software program/library interfaced by Qiskit.
Several, as listed below, are supported, and while logic to interface these programs is supplied by
the chemistry module via the above pip installation, the dependent programs/libraries themselves need
to be installed separately.

1. [Gaussian 16&trade;](https://qiskit.org/documentation/apidoc/qiskit.chemistry.drivers.gaussiand.html), a commercial chemistry program
2. [PSI4](https://qiskit.org/documentation/apidoc/qiskit.chemistry.drivers.psi4d.html), a chemistry program that exposes a Python interface allowing for accessing internal objects
3. [PySCF](https://qiskit.org/documentation/apidoc/qiskit.chemistry.drivers.pyscfd.html), an open-source Python chemistry program
4. [PyQuante](https://qiskit.org/documentation/apidoc/qiskit.chemistry.drivers.pyquanted.html), a pure cross-platform open-source Python chemistry program

### HDF5 Driver

A useful functionality integrated into Qiskit's chemistry module is its ability to serialize a file
in hierarchical Data Format 5 (HDF5) format representing all the output data from a chemistry driver.

The [HDF5 driver](https://qiskit.org/documentation/stubs/qiskit.chemistry.drivers.HDF5Driver.html#qiskit.chemistry.drivers.HDF5Driver)
accepts such HDF5 files as input so molecular experiments can be run, albeit on the fixed data
as stored in the file. As such, if you have some pre-created HDF5 files from created from Qiskit
Chemistry, you can use these with the HDF5 driver even if you do not install one of the classical
computation packages listed above.

A few sample HDF5 files for different are provided in the
[chemistry folder](https://github.com/Qiskit/qiskit-community-tutorials/tree/master/chemistry)
of the [Qiskit Community Tutorials](https://github.com/Qiskit/qiskit-community-tutorials)
repository. This
[HDF5 Driver tutorial](https://github.com/Qiskit/qiskit-community-tutorials/blob/master/chemistry/hdf5_files_and_driver.ipynb)
contains further information about creating and using such HDF5 files.

### Creating Your First Chemistry Programming Experiment in Qiskit

Now that Qiskit is installed, it's time to begin working with the chemistry module.
Let's try a chemistry application experiment using VQE (Variational Quantum Eigensolver) algorithm
to compute the ground-state (minimum) energy of a molecule.

```python
from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.aqua.operators import Z2Symmetries

# Use PySCF, a classical computational chemistry software
# package, to compute the one-body and two-body integrals in
# molecular-orbital basis, necessary to form the Fermionic operator
driver = PySCFDriver(atom='H .0 .0 .0; H .0 .0 0.735',
                     unit=UnitsType.ANGSTROM,
                     basis='sto3g')
molecule = driver.run()
num_particles = molecule.num_alpha + molecule.num_beta
num_spin_orbitals = molecule.num_orbitals * 2

# Build the qubit operator, which is the input to the VQE algorithm in Aqua
ferm_op = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
map_type = 'PARITY'
qubit_op = ferm_op.mapping(map_type)
qubit_op = Z2Symmetries.two_qubit_reduction(qubit_op, num_particles)
num_qubits = qubit_op.num_qubits

# setup a classical optimizer for VQE
from qiskit.aqua.components.optimizers import L_BFGS_B
optimizer = L_BFGS_B()

# setup the initial state for the variational form
from qiskit.chemistry.circuit.library import HartreeFock
init_state = HartreeFock(num_spin_orbitals, num_particles)

# setup the variational form for VQE
from qiskit.circuit.library import TwoLocal
var_form = TwoLocal(num_qubits, ['ry', 'rz'], 'cz')

# add the initial state
var_form.compose(init_state, front=True)

# setup and run VQE
from qiskit.aqua.algorithms import VQE
algorithm = VQE(qubit_op, var_form, optimizer)

# set the backend for the quantum computation
from qiskit import Aer
backend = Aer.get_backend('statevector_simulator')

result = algorithm.run(backend)
print(result.eigenvalue.real)
```
The program above uses a quantum computer to calculate the ground state energy of molecular Hydrogen,
H<sub>2</sub>, where the two atoms are configured to be at a distance of 0.735 angstroms. The molecular
input specification is processed by PySCF driver and data is output that includes one- and
two-body molecular-orbital integrals. From the output a fermionic-operator is created which is then
parity mapped to generate a qubit operator. Parity mappings allow a precision-preserving optimization
that two qubits can be tapered off; a reduction in complexity that is particularly advantageous for NISQ
computers.

The qubit operator is then passed as an input to the Variational Quantum Eigensolver (VQE) algorithm,
instantiated with a classical optimizer and a RyRz variational form (ansatz). A Hartree-Fock
initial state is used as a starting point for the variational form.

The VQE algorithm is then run, in this case on the Qiskit Aer statevector simulator backend.
Here we pass a backend but it can be wrapped into a `QuantumInstance`, and that passed to the
`run` instead. The `QuantumInstance` API allows you to customize run-time properties of the backend,
such as the number of shots, the maximum number of credits to use, settings for the simulator,
initial layout of qubits in the mapping and the Terra `PassManager` that will handle the compilation
of the circuits. By passing in a backend as is done above it is internally wrapped into a
`QuantumInstance` and is a convenience when default setting suffice.

### Further examples

Learning path notebooks may be found in the
[chemistry tutorials](https://qiskit.org/documentation/tutorials/chemistry/index.html) section
of the documentation and are a great place to start

Jupyter notebooks containing further chemistry examples may be found in the
following Qiskit GitHub repositories at
[qiskit-tutorials/tutorials/chemistry](https://github.com/Qiskit/qiskit-tutorials/tree/master/tutorials/chemistry)
and
[qiskit-community-tutorials/chemistry](https://github.com/Qiskit/qiskit-community-tutorials/tree/master/chemistry).


----------------------------------------------------------------------------------------------------


## Contribution Guidelines

If you'd like to contribute to Qiskit, please take a look at our
[contribution guidelines](./CONTRIBUTING.md).
This project adheres to Qiskit's [code of conduct](./CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.

We use [GitHub issues](https://github.com/Qiskit/qiskit-aqua/issues) for tracking requests and bugs. Please
[join the Qiskit Slack community](https://ibm.co/joinqiskitslack)
and use the [Aqua Slack channel](https://qiskit.slack.com/messages/aqua) for discussion and simple questions.
For questions that are more suited for a forum, we use the **Qiskit** tag in [Stack Overflow](https://stackoverflow.com/questions/tagged/qiskit).

## Next Steps

Now you're set up and ready to check out some of the other examples from the
[Qiskit Tutorials](https://github.com/Qiskit/qiskit-tutorials)
repository, that are used for the IBM Quantum Experience, and from the
[Qiskit Community Tutorials](https://github.com/Qiskit/qiskit-community-tutorials).


## Authors and Citation

Aqua was inspired, authored and brought about by the collective work of a team of researchers.
Aqua continues to grow with the help and work of
[many people](https://github.com/Qiskit/qiskit-aqua/graphs/contributors), who contribute
to the project at different levels.
If you use Qiskit, please cite as per the provided
[BibTeX file](https://github.com/Qiskit/qiskit/blob/master/Qiskit.bib).

Please note that if you do not like the way your name is cited in the BibTex file then consult
the information found in the [.mailmap](https://github.com/Qiskit/qiskit-aqua/blob/master/.mailmap)
file.

## License

This project uses the [Apache License 2.0](LICENSE.txt).

However there is some code that is included under other licensing as follows:

* The [Gaussian 16 driver](qiskit/chemistry/drivers/gaussiand) in `qiskit.chemistry`
  contains [work](qiskit/chemistry/drivers/gaussiand/gauopen) licensed under the
  [Gaussian Open-Source Public License](qiskit/chemistry/drivers/gaussiand/gauopen/LICENSE.txt).

