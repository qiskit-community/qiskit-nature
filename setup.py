# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import setuptools
import inspect
import sys
import os

long_description = """Qiskit Nature is a open-source library of quantum computing chemistry/physics experiments.
 """

requirements = [
    "qiskit-terra>=0.17.0",
    "scipy>=1.4",
    "numpy>=1.17",
    "psutil>=5",
    "scikit-learn>=0.20.0",
    "setuptools>=40.1.0",
    "h5py",
    "retworkx>=0.7.0"
]

if not hasattr(setuptools, 'find_namespace_packages') or not inspect.ismethod(setuptools.find_namespace_packages):
    print("Your setuptools version:'{}' does not support PEP 420 (find_namespace_packages). "
          "Upgrade it to version >='40.1.0' and repeat install.".format(setuptools.__version__))
    sys.exit(1)

VERSION_PATH = os.path.join(os.path.dirname(__file__), "qiskit_nature", "VERSION.txt")
with open(VERSION_PATH, "r") as version_file:
    VERSION = version_file.read().strip()

setuptools.setup(
    name='qiskit-nature',
    version=VERSION,
    description='Qiskit Nature: A library of quantum computing chemistry/physics experiments',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/Qiskit/qiskit-nature',
    author='Qiskit Nature Development Team',
    author_email='hello@qiskit.org',
    license='Apache-2.0',
    classifiers=(
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering"
    ),
    keywords='qiskit sdk quantum nature chemistry physics',
    packages=setuptools.find_packages(include=['qiskit_nature', 'qiskit_nature.*']),
    install_requires=requirements,
    include_package_data=True,
    python_requires=">=3.6",
    extras_require={
        'pyscf': ["pyscf; sys_platform != 'win32'"],
    },
    zip_safe=False
)
