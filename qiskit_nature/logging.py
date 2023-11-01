# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Qiskit Nature Logging Utilities."""

from typing import Optional, Dict, List
import os
import sys
import logging as python_logging


class _DefaultStreamHandler(python_logging.StreamHandler):
    """Default stream Handler"""

    def __init__(self):
        """
        Initialize the handler.
        """
        super().__init__(stream=sys.stdout)
        self.setFormatter(python_logging.Formatter(fmt=QiskitNatureLogging.LOG_FORMAT))


class QiskitNatureLogging:
    """Qiskit Nature Logging.

    A collection of utility methods encapsulating logging functions.
    The most basic use is:

    .. code-block:: python

        import logging
        from qiskit_nature import logging as nature_logging
        nature_logging.set_levels_for_names(
            {"qiskit_nature": logging.DEBUG, "qiskit": logging.DEBUG})

    It will print to 'stdout' DEBUG formatted logs, for the domains included.
    One can access current logging levels for domains:

    .. code-block:: python

        from qiskit_nature import logging as nature_logging
        dict = nature_logging.get_levels_for_names(["qiskit_nature", "qiskit"])

    The result dictionary key is the domain string and the value the logging level

    Methods exist also to add, remove logging handler with formatter and one to write to a file:

    .. code-block:: python

        import logging
        from qiskit_nature import logging as nature_logging
        nature_logging.log_to_file(
                {"qiskit_nature": logging.DEBUG, "qiskit": logging.DEBUG},
                 path="file.log", mode="w"
            )

    If mode is not given, 'append' mode is used.

    """

    LOG_FORMAT = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"

    def get_levels_for_names(self, names: List[str]) -> Dict[str, int]:
        """Return logging levels for module names.

        Args:
            names: list of module names (qiskit_nature.second_q.drivers, qiskit_algorithms etc)
        Returns:
            Dictionary mapping names to effective level
        """
        name_levels: Dict[str, int] = {}
        for name in names:
            name_levels[name] = python_logging.getLogger(name).getEffectiveLevel()

        return name_levels

    def set_levels_for_names(
        self, name_levels: Dict[str, int], add_default_handler: bool = True
    ) -> None:
        """Set logging levels for module names.

        Args:
            name_levels: Dictionary of module names (qiskit_nature.second_q, qiskit_algorithms etc)
                         to desired level
            add_default_handler: add or not the default stream handler
        """
        names: List[str] = []
        for name, level in name_levels.items():
            names.append(name)
            logger = python_logging.getLogger(name)
            logger.setLevel(level)
            logger.propagate = False
            handlers = logger.handlers
            if add_default_handler and not any(
                isinstance(h, _DefaultStreamHandler) for h in handlers
            ):
                self.add_handler(name, _DefaultStreamHandler())

    def add_handler(
        self,
        name: str,
        handler: python_logging.Handler,
        formatter: Optional[python_logging.Formatter] = None,
    ) -> None:
        """Add handler and optional formatter to a module name.

        Args:
            name: module name
            handler: Logging Handler
            formatter: Logging Formatter
        """
        logger = python_logging.getLogger(name)
        if formatter is not None:
            handler.setFormatter(formatter)
        logger.addHandler(handler)

    def remove_handler(self, name: str, handler: python_logging.Handler) -> None:
        """Remove handler from module.

        Args:
            name: module name
            handler: Logging Handler
        """
        logger = python_logging.getLogger(name)
        handler.close()
        logger.removeHandler(handler)

    def remove_all_handlers(self, names: List[str]) -> None:
        """Remove all handlers from modules.

        Args:
            names: list of module names (qiskit_nature.second_q.drivers, qiskit_algorithms etc)
        """
        for name in names:
            logger = python_logging.getLogger(name)
            handlers = logger.handlers.copy()
            for handler in handlers:
                self.remove_handler(name, handler=handler)

    def remove_default_handler(self, names: List[str]) -> None:
        """Remove default handler from modules.

        Args:
            names: list of module names (qiskit_nature.second_q.drivers, qiskit_algorithms etc)
        """
        for name in names:
            logger = python_logging.getLogger(name)
            handlers = logger.handlers.copy()
            for handler in handlers:
                if isinstance(handler, _DefaultStreamHandler):
                    self.remove_handler(name, handler=handler)

    def log_to_file(
        self, names: List[str], path: str, mode: str = "a"
    ) -> python_logging.FileHandler:
        """Logs modules to file.

        Args:
            names: list of module names (qiskit_nature.second_q.drivers, qiskit_algorithms etc)
            path: file path
            mode: file open mode. If it is not specified, 'a' is used to append
        Returns:
            The logging File handler used
        """
        filepath = os.path.expanduser(path)
        handler = python_logging.FileHandler(filepath, mode=mode)
        formatter = python_logging.Formatter(fmt=QiskitNatureLogging.LOG_FORMAT)
        for name in names:
            self.add_handler(name, handler, formatter)
        return handler


logging = QiskitNatureLogging()
