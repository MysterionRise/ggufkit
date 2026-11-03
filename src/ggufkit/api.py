"""ggufkit's stable Python API: functions that take paths and return plain results.

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
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ggufkit._util import prepare_output
from ggufkit.fingerprint import Fingerprint, fingerprint
from ggufkit.manifest import Manifest, build_manifest, load_manifest, with_input_hashes
from ggufkit.model import open_model
from ggufkit.reproduce import ReproduceResult, reproduce, reproduce_from_manifest
from ggufkit.verify import Report, verify_against_file, verify_against_manifest

if TYPE_CHECKING:
    from collections.abc import Sequence


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
    source: str | Path | None = None,
    imatrix: str | Path | None = None,
    notes: str | None = None,
    data: bool = True,
    jobs: int = 1,
) -> Manifest:
    """Build the ``ggufkit-manifest/1`` manifest of a GGUF file or split model.

    ``file_sha256`` also records each file's plain sha256 (it reads every file once more).
    ``build_inputs`` records how the file was made, under the keys ``docs/MANIFEST.md``
    lists (``source_sha256``, ``imatrix_sha256``, ``llama_cpp_commit``, ``quantize_cmd``).
    ``source`` and ``imatrix`` are files to hash into ``source_sha256`` and
    ``imatrix_sha256``. Save the result with :func:`ggufkit.save_manifest`.
    """
    inputs = with_input_hashes(build_inputs, source=source, imatrix=imatrix)
    with open_model(path) as model:
        return build_manifest(
            model,
            file_sha256=file_sha256,
            build_inputs=inputs,
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


def reproduce_file(
    path: str | Path | None = None,
    *,
    source: str | Path,
    manifest: str | Path | Mapping[str, Any] | None = None,
    imatrix: str | Path | None = None,
    quant_type: str | None = None,
    llama_quantize: str | Path | None = None,
    threads: int | None = None,
    keep: str | Path | None = None,
    force: bool = False,
    jobs: int = 1,
) -> ReproduceResult:
    """Rebuild a quantized GGUF file from its source with llama-quantize, and compare.

    Give the file to check as ``path``, or a manifest (a path, or one already loaded) as
    ``manifest``. ``source`` is the full-precision model it was made from. ``imatrix`` is
    the importance matrix, which a file that records one needs. ``quant_type`` defaults to
    the manifest's ``quantize_cmd`` or the file's ``general.file_type``, and
    ``llama_quantize`` to the binary on PATH. ``keep`` saves the rebuild (``force``
    overwrites an existing file). :meth:`ReproduceResult.ok` is the pass or fail, and
    :attr:`ReproduceResult.report` says what differs.

    With a manifest, the source and the matrix are checked against its ``source_sha256``
    and ``imatrix_sha256`` first, and :attr:`ReproduceResult.inputs` says how they compared.
    Raises :class:`ggufkit.ReproduceError` when no rebuild could be made.
    """
    if (path is None) == (manifest is None):
        msg = "give exactly one of path or manifest="
        raise TypeError(msg)
    if keep is not None:
        prepare_output(Path(keep), force=force)
    options: dict[str, Any] = {
        "imatrix": imatrix,
        "quant_type": quant_type,
        "threads": threads,
        "llama_quantize": llama_quantize,
        "keep": keep,
        "jobs": jobs,
    }
    if path is None:
        return reproduce_from_manifest(manifest or {}, source, **options)
    with open_model(path) as model:
        return reproduce(model, source, **options)
