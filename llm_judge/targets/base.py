"""The Target interface."""

from __future__ import annotations

import abc

from ..dataset import TestCase
from ..schema import Trace


class Target(abc.ABC):
    """A system-under-test: turns a test case into a trace."""

    name: str = "target"

    @abc.abstractmethod
    def run(self, case: TestCase) -> Trace:
        """Execute the system on one case and return a full trace."""
        raise NotImplementedError
