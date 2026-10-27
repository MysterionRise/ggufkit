"""The common base class of every error ggufkit raises for bad input."""

from __future__ import annotations


class GGUFKitError(ValueError):
    """Base class for ggufkit's errors about files, models and manifests.

    It subclasses ``ValueError``, so code that already catches ``ValueError`` keeps working.
    """
