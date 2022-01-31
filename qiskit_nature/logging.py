# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
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
import logging


class DefaultStreamHandler(logging.StreamHandler):
    """Default stream Handler"""

    def __init__(self):
        """
        Initialize the handler.
        """
        super().__init__(stream=sys.stdout)
        self.setFormatter(logging.Formatter(fmt="%(asctime)s:%(name)s:%(levelname)s: %(message)s"))


class QiskitNatureLogging:
    """Qiskit Nature Logging."""

    def __init__(self) -> None:
        self._logging = logging

    def get_levels_for_names(self, names: List[str]) -> Dict[str, int]:
        """Return logging levels for module names.

        Args:
            names: list of module names (qiskit_nature.drivers, qiskit.algorithms etc)
        Returns:
            Dictionary mapping names to effective level
        """
        name_levels: Dict[str, int] = {}
        for name in names:
            name_levels[name] = self._logging.getLogger(name).getEffectiveLevel()

        return name_levels

    def set_levels_for_names(
        self, name_levels: Dict[str, int], add_default_handler: bool = True
    ) -> Dict[str, int]:
        """Set logging levels for module names.

        Args:
            name_levels: Dictionary of module names (qiskit_nature.drivers, qiskit.algorithms etc)
                         to desired level
            add_default_handler: add or not the default stream handler
        Returns:
            Dictionary mapping names to effective level
        """
        names: List[str] = []
        for name, level in name_levels.items():
            names.append(name)
            logger = self._logging.getLogger(name)
            logger.setLevel(level)
            logger.propagate = False
            handlers = logger.handlers
            if add_default_handler and not any(
                True for h in handlers if isinstance(h, DefaultStreamHandler)
            ):
                self.add_handler(name, DefaultStreamHandler())

        return self.get_levels_for_names(names)

    def add_handler(
        self, name: str, handler: logging.Handler, formatter: Optional[logging.Formatter] = None
    ) -> None:
        """Add handler and optional formatter to a module name.

        Args:
            name: module name
            handler: Logging Handler
            formatter: Logging Formatter
        """
        logger = self._logging.getLogger(name)
        if formatter is not None:
            handler.setFormatter(formatter)
        logger.addHandler(handler)

    def remove_handler(self, name: str, handler: logging.Handler) -> None:
        """Remove handler from module.

        Args:
            name: module name
            handler: Logging Handler
        """
        logger = self._logging.getLogger(name)
        handler.close()
        logger.removeHandler(handler)

    def remove_all_handlers(self, names: List[str]) -> None:
        """Remove all handlers from modules.

        Args:
            names: list of module names (qiskit_nature.drivers, qiskit.algorithms etc)
        """
        for name in names:
            logger = self._logging.getLogger(name)
            handlers = logger.handlers.copy()
            for handler in handlers:
                self.remove_handler(name, handler=handler)

    def remove_default_handler(self, names: List[str]) -> None:
        """Remove default handler from modules.

        Args:
            names: list of module names (qiskit_nature.drivers, qiskit.algorithms etc)
        """
        for name in names:
            logger = self._logging.getLogger(name)
            handlers = logger.handlers.copy()
            for handler in handlers:
                if isinstance(handler, DefaultStreamHandler):
                    self.remove_handler(name, handler=handler)

    def log_to_file(self, names: List[str], path: str) -> None:
        """Logs modules to file.

        Args:
            names: list of module names (qiskit_nature.drivers, qiskit.algorithms etc)
            path: file path
        """
        filepath = os.path.expanduser(path)
        handler = logging.FileHandler(filepath, mode="w")
        formatter = logging.Formatter(fmt="%(asctime)s:%(name)s:%(levelname)s: %(message)s")
        for name in names:
            self.add_handler(name, handler, formatter)
