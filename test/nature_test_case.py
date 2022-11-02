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

"""Nature Test Case"""

from typing import Optional
from abc import ABC
import warnings
import inspect
import logging
import os
import unittest
import time
from qiskit_nature import settings
from qiskit_nature.deprecation import NatureDeprecationWarning

# disable unit tests NatureDeprecationWarning warnings on imports
warnings.filterwarnings("ignore", category=NatureDeprecationWarning)

# disable deprecation warnings that can cause log output overflow
# pylint: disable=unused-argument


def _noop(*args, **kargs):
    pass


# disable warning messages
# warnings.warn = _noop


class QiskitNatureTestCase(unittest.TestCase, ABC):
    """Nature Test Case"""

    moduleName = None
    log = None

    def setUp(self) -> None:
        settings.dict_aux_operators = True
        warnings.filterwarnings("default", category=DeprecationWarning)
        # disable unit tests NatureDeprecationWarning warnings previously reset
        warnings.filterwarnings("ignore", category=NatureDeprecationWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="pyscf")
        warnings.filterwarnings(action="ignore", category=DeprecationWarning, module=".*drivers*")
        warnings.filterwarnings(
            action="default", category=DeprecationWarning, module=".*second_q.drivers.*"
        )
        warnings.filterwarnings(
            action="ignore", category=DeprecationWarning, module=".*transformers*"
        )
        warnings.filterwarnings(
            action="default",
            category=DeprecationWarning,
            module=".*second_q.transformers.*",
        )
        # ignore opflow/gradients/natural_gradient
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="qiskit")
        self._started_at = time.time()
        self._class_location = __file__

    def tearDown(self) -> None:
        elapsed = time.time() - self._started_at
        if elapsed > 5.0:
            print(f"({round(elapsed, 2):.2f}s)", flush=True)

    @classmethod
    def setUpClass(cls) -> None:
        cls.moduleName = os.path.splitext(inspect.getfile(cls))[0]
        cls.log = logging.getLogger(cls.__name__)

        # Set logging to file and stdout if the LOG_LEVEL environment variable
        # is set.
        if os.getenv("LOG_LEVEL"):
            # Set up formatter.
            log_fmt = f"{cls.__name__}.%(funcName)s:%(levelname)s:%(asctime)s:" " %(message)s"
            formatter = logging.Formatter(log_fmt)

            # Set up the file handler.
            log_file_name = f"{cls.moduleName}.log"
            file_handler = logging.FileHandler(log_file_name)
            file_handler.setFormatter(formatter)
            cls.log.addHandler(file_handler)

            # Set the logging level from the environment variable, defaulting
            # to INFO if it is not a valid level.
            level = logging._nameToLevel.get(os.getenv("LOG_LEVEL"), logging.INFO)
            cls.log.setLevel(level)

    def get_resource_path(self, filename: str, path: Optional[str] = None) -> str:
        """Get the absolute path to a resource.
        Args:
            filename: filename or relative path to the resource.
            path: path used as relative to the filename.
        Returns:
            str: the absolute path to the resource.
        """
        root = os.path.dirname(self._class_location)
        path = root if path is None else os.path.join(root, path)
        return os.path.normpath(os.path.join(path, filename))


class QiskitNatureDeprecatedTestCase(QiskitNatureTestCase):
    """Nature Deprecated Test Case.
    Used for tests that need to suppress deprecation messages.
    """

    def setUp(self) -> None:
        super().setUp()
        # disable deprecation warnings
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=NatureDeprecationWarning)

    def tearDown(self) -> None:
        # enable deprecation warnings
        warnings.filterwarnings("default", category=PendingDeprecationWarning)
        warnings.filterwarnings("default", category=DeprecationWarning)
        warnings.filterwarnings("default", category=NatureDeprecationWarning)
        super().tearDown()
