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

        The simplest way to get started is to follow the getting started 'Start locally' for Qiskit
        here `Qiskit Getting Started <https://qiskit.org/documentation/getting_started.html>`__

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

             git clone https://github.com/Qiskit/qiskit-nature.git

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

----

Ready to get going?...
======================

.. raw:: html

   <div class="tutorials-callout-container">
      <div class="row">

.. customcalloutitem::
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
