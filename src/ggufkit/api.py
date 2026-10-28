"""ggufkit's stable Python API: three functions that take paths and return plain results.

Everything listed in ``ggufkit.__all__`` is stable: it changes only in a major release, or
after a deprecation warning that lasts at least one minor release. Import it from the
top-level package::

    import ggufkit

    fp = ggufkit.fingerprint_file("model.gguf")
    report = ggufkit.verify_file("copy.gguf", against="model.gguf")
    if not report.ok(allow_metadata=True):
        raise SystemExit(report.verdict)

The submodules (``ggufkit.fingerprint``, ``ggufkit.verify``, …) are the command-line
implementation and may change in any release.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ggufkit.fingerprint import Fingerprint, fingerprint
from ggufkit.manifest import Manifest, build_manifest, load_manifest
from ggufkit.model import open_model
from ggufkit.verify import Report, verify_against_file, verify_against_manifest

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def fingerprint_file(
    path: str | Path,
    *,
    data: bool = True,
    per_tensor: bool = True,
    per_kv: bool = False,
    jobs: int = 1,
) -> Fingerprint:
    """Compute the ``ggufkit-fp/1`` fingerprint of a GGUF file or split model.

    ``path`` may be any shard of a split model (``model-00001-of-00003.gguf``); the shards
    are found next to it and fingerprinted as one model. ``data=False`` skips the tensor
    bytes, which is instant on any size but leaves ``weights`` and ``overall`` unset.
    ``jobs`` is the number of threads hashing tensor data.

    Raises :class:`ggufkit.ModelError` or :class:`ggufkit.UnsupportedModelError` for files
    ggufkit cannot read, and ``OSError`` when a file cannot be opened.
    """
    with open_model(path) as model:
        return fingerprint(model, data=data, per_tensor=per_tensor, per_kv=per_kv, jobs=jobs)


def create_manifest(
    path: str | Path,
    *,
    file_sha256: bool = False,
    build_inputs: Mapping[str, str] | None = None,
    notes: str | None = None,
    data: bool = True,
    jobs: int = 1,
) -> Manifest:
    """Build the ``ggufkit-manifest/1`` manifest of a GGUF file or split model.

    ``file_sha256`` also records each file's plain sha256 (it reads every file once more).
    ``build_inputs`` records how the file was made, under the keys ``docs/MANIFEST.md``
    lists (``source_sha256``, ``imatrix_sha256``, ``llama_cpp_commit``, ``quantize_cmd``).
    Save the result with :func:`ggufkit.save_manifest`.
    """
    with open_model(path) as model:
        return build_manifest(
            model,
            file_sha256=file_sha256,
            build_inputs=build_inputs,
            notes=notes,
            data=data,
            jobs=jobs,
        )


def verify_file(
    path: str | Path,
    *,
    against: str | Path | None = None,
    manifest: str | Path | Mapping[str, Any] | None = None,
    data: bool = True,
    ignore_keys: Sequence[str] = (),
    file_sha256: bool = False,
    jobs: int = 1,
) -> Report:
    """Compare a GGUF file with another file (``against``) or with a manifest.

    ``manifest`` is a path, or a manifest already loaded with :func:`ggufkit.load_manifest`.
    ``ignore_keys`` are metadata key globs to leave out (``"general.name"``,
    ``"tokenizer.*"``). ``file_sha256`` also checks the raw file hashes the manifest
    recorded. Call :meth:`Report.ok` for a pass or fail, read :attr:`Report.verdict` for the
    kind of change, and :meth:`Report.to_dict` for everything.
    """
    if file_sha256 and manifest is None:
        msg = "file_sha256 needs manifest="
        raise TypeError(msg)
    with open_model(path) as model:
        if against is not None and manifest is None:
            with open_model(against) as other:
                return verify_against_file(
                    model, other, data=data, jobs=jobs, ignore_keys=ignore_keys
                )
        if manifest is not None and against is None:
            if isinstance(manifest, Mapping):
                loaded: Mapping[str, Any] = manifest
                label = "manifest"
            else:
                loaded = load_manifest(manifest)
                label = str(manifest)
            return verify_against_manifest(
                model,
                loaded,
                manifest_label=label,
                data=data,
                jobs=jobs,
                ignore_keys=ignore_keys,
                check_files=file_sha256,
            )
    msg = "give exactly one of against= or manifest="
    raise TypeError(msg)
