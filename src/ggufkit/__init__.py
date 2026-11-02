"""ggufkit: fingerprint, verify and visualize GGUF models.

The names in ``__all__`` are the stable Python API; see ``ggufkit.api`` and the README's
"Python API" section. They load on first use, so ``import ggufkit`` stays cheap.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

try:
    __version__ = version("ggufkit")
except PackageNotFoundError:  # pragma: no cover - only when running from a bare checkout
    __version__ = "0.0.0+unknown"

if TYPE_CHECKING:
    from ggufkit.api import create_manifest, fingerprint_file, reproduce_file, verify_file
    from ggufkit.canon import UnsupportedModelError
    from ggufkit.errors import GGUFKitError
    from ggufkit.fingerprint import Fingerprint, TensorFingerprint
    from ggufkit.manifest import IntotoError, Manifest, ManifestError, load_manifest, save_manifest
    from ggufkit.model import ModelError
    from ggufkit.reproduce import ReproduceError, ReproduceResult
    from ggufkit.verify import KeyChange, Report, TensorChange

_EXPORTS = {
    "fingerprint_file": "ggufkit.api",
    "create_manifest": "ggufkit.api",
    "verify_file": "ggufkit.api",
    "reproduce_file": "ggufkit.api",
    "load_manifest": "ggufkit.manifest",
    "save_manifest": "ggufkit.manifest",
    "Fingerprint": "ggufkit.fingerprint",
    "TensorFingerprint": "ggufkit.fingerprint",
    "Manifest": "ggufkit.manifest",
    "Report": "ggufkit.verify",
    "KeyChange": "ggufkit.verify",
    "TensorChange": "ggufkit.verify",
    "ReproduceResult": "ggufkit.reproduce",
    "GGUFKitError": "ggufkit.errors",
    "ModelError": "ggufkit.model",
    "UnsupportedModelError": "ggufkit.canon",
    "ManifestError": "ggufkit.manifest",
    "IntotoError": "ggufkit.manifest",
    "ReproduceError": "ggufkit.reproduce",
}

__all__ = [
    "Fingerprint",
    "GGUFKitError",
    "IntotoError",
    "KeyChange",
    "Manifest",
    "ManifestError",
    "ModelError",
    "Report",
    "ReproduceError",
    "ReproduceResult",
    "TensorChange",
    "TensorFingerprint",
    "UnsupportedModelError",
    "__version__",
    "create_manifest",
    "fingerprint_file",
    "load_manifest",
    "reproduce_file",
    "save_manifest",
    "verify_file",
]


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        msg = f"module 'ggufkit' has no attribute {name!r}"
        raise AttributeError(msg)
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
