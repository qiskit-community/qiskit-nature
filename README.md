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

- [Gaussian 16&trade;](https://qiskit.org/documentation/nature/apidocs/qiskit_nature.second_q.drivers.gaussiand.html), a commercial chemistry program
- [PSI4](https://qiskit.org/documentation/nature/apidocs/qiskit_nature.second_q.drivers.psi4d.html), a chemistry program that exposes a Python interface allowing for accessing internal objects
- [PySCF](https://qiskit.org/documentation/nature/apidocs/qiskit_nature.second_q.drivers.pyscfd.html), an open-source Python chemistry program

Additionally, you may find the following optional dependencies useful:

- [sparse](https://github.com/pydata/sparse/), a library for sparse multi-dimensional arrays. When installed, Qiskit Nature can leverage this to reduce the memory requirements of your calculations.
- [opt_einsum](https://github.com/dgasmith/opt_einsum), a tensor contraction order optimizer for `np.einsum`.

## Creating Your First Chemistry Programming Experiment in Qiskit

Check our [getting started page](https://qiskit.org/documentation/nature/getting_started.html)
for a first example on how to use Qiskit Nature.

### Further examples

Learning path notebooks may be found in the
[Nature Tutorials](https://qiskit.org/documentation/nature/tutorials/index.html) section
of the documentation and are a great place to start.


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
If you use Qiskit Nature, please cite the following references:

- Qiskit, as per the provided [BibTeX file](https://github.com/Qiskit/qiskit/blob/master/Qiskit.bib).
- Qiskit Nature, as per https://doi.org/10.5281/zenodo.7828767

Please note that if you do not like the way your name is cited in the BibTex file then consult
the information found in the [.mailmap](https://github.com/Qiskit/qiskit-nature/blob/main/.mailmap)
file.

## License

This project uses the [Apache License 2.0](https://github.com/Qiskit/qiskit-nature/blob/main/LICENSE.txt).

However there is some code that is included under other licensing as follows:

* The [Gaussian 16 driver](https://github.com/Qiskit/qiskit-nature/tree/main/qiskit_nature/second_q/drivers/gaussiand) in `qiskit-nature`
  contains [work](https://github.com/Qiskit/qiskit-nature/tree/main/qiskit_nature/second_q/drivers/gaussiand/gauopen) licensed under the
  [Gaussian Open-Source Public License](https://github.com/Qiskit/qiskit-nature/blob/main/qiskit_nature/second_q/drivers/gaussiand/gauopen/LICENSE.txt).
