# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

OS := $(shell uname -s)

ifeq ($(OS), Linux)
  NPROCS := $(shell grep -c ^processor /proc/cpuinfo)
else ifeq ($(OS), Darwin)
  NPROCS := 2
else
  NPROCS := 0
endif # $(OS)

ifeq ($(NPROCS), 2)
	CONCURRENCY := 2
else ifeq ($(NPROCS), 1)
	CONCURRENCY := 1
else ifeq ($(NPROCS), 3)
	CONCURRENCY := 3
else ifeq ($(NPROCS), 0)
	CONCURRENCY := 0
else
	CONCURRENCY := $(shell echo "$(NPROCS) 2" | awk '{printf "%.0f", $$1 / $$2}')
endif

# You can set this variable from the command line.
SPHINXOPTS    =

.PHONY: lint mypy style black test test_ci spell copyright html doctest clean_sphinx coverage coverage_erase clean

all_check: spell style lint copyright mypy clean_sphinx html doctest

lint:
	python -m pylint -rn --ignore=gauopen qiskit_nature test tools
	python tools/verify_headers.py qiskit_nature test tools
	python tools/find_stray_release_notes.py

mypy:
	python -m mypy qiskit_nature test tools

style:
	python -m black --check --extend-exclude="gauopen" qiskit_nature test tools docs

black:
	python -m black --extend-exclude="gauopen" qiskit_nature test tools docs

test:
	python -m unittest discover -v test

test_ci:
	echo "Detected $(NPROCS) CPUs running with $(CONCURRENCY) workers"
	python -m stestr run --concurrency $(CONCURRENCY)

spell:
	python -m pylint -rn --disable=all --enable=spelling --spelling-dict=en_US --spelling-private-dict-file=.pylintdict --ignore=gauopen qiskit_nature test tools
	sphinx-build -M spelling docs docs/_build -W -T --keep-going $(SPHINXOPTS)

copyright:
	python tools/check_copyright.py

html:
	sphinx-build -M html docs docs/_build -W -T --keep-going $(SPHINXOPTS)

doctest:
	sphinx-build -M doctest docs docs/_build -W -T --keep-going $(SPHINXOPTS)

clean_sphinx:
	make -C docs clean

coverage:
	python -m coverage3 run --source qiskit_nature --omit */gauopen/* -m unittest discover -s test -q
	python -m coverage3 report

coverage_erase:
	python -m coverage erase

clean: clean_sphinx coverage_erase;

