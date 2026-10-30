"""Rebuild a quantized model from its source with llama-quantize, and compare.

llama.cpp's quantizer is deterministic: the same source, importance matrix and quantization
type give the same file, byte for byte, whatever the thread count or CPU features. So a
published quant can be checked by making it again. ``reproduce`` runs ``llama-quantize`` on
the source and compares the rebuild with the published file the way ``verify --against``
does. When they differ, the report says whether it is the metadata, the tensor descriptors
or the weights, and which keys and tensors.

With a manifest instead of the file, the build inputs come from the manifest: the source and
the importance matrix are checked against its hashes before the rebuild, the recorded
``quantize_cmd`` gives the type and recipe options, and the rebuild is compared with the
manifest's fingerprint and file hash.
"""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gguf.constants import Keys, LlamaFileType

from ggufkit import canon
from ggufkit._util import atomic_output, positive_int, prepare_output, to_stderr
from ggufkit.errors import GGUFKitError
from ggufkit.fingerprint import default_jobs
from ggufkit.manifest import (
    QUANTIZE_IMATRIX_KEYS,
    _kv,
    file_type_label,
    load_manifest,
    sha256_file,
)
from ggufkit.model import open_model, shard_paths
from ggufkit.verify import format_report, verify_against_file, verify_against_manifest

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable

    from ggufkit.model import Model
    from ggufkit.verify import Report

#: Lines of llama-quantize output shown when it fails.
TAIL_LINES = 20

#: How llama.cpp prints its version: ``version: 0.8.0 (build 1, commit 8642541)`` now, and
#: ``llama_print_build_info: build = 1 (1a03cf4)`` in older releases.
_VERSION_RE = re.compile(r"^version: ([^\r\n]*\(build \d+, commit [0-9a-f]+\))", re.MULTILINE)
_OLD_VERSION_RE = re.compile(r"build = (\d+) \(([0-9a-f]+)\)")

#: llama-quantize options a manifest's ``quantize_cmd`` may use, with the number of values
#: each takes. They shape the output, so they are passed on to the rebuild.
RECIPE_OPTIONS = {
    "--leave-output-tensor": 0,
    "--pure": 0,
    "--allow-requantize": 0,
    "--output-tensor-type": 1,
    "--token-embedding-type": 1,
    "--tensor-type": 1,
    "--prune-layers": 1,
    "--override-kv": 1,
    "--include-weights": 1,
    "--exclude-weights": 1,
}
#: Options read but not passed on: the matrix is placed separately, and the buffer size only
#: affects I/O.
_DROPPED_OPTIONS = {"--imatrix": 1, "--max-buffer-size": 1}
_REFUSED_OPTIONS = {
    "--tensor-type-file": "it reads a file the manifest does not carry",
    "--keep-split": "reproduce does not handle split output yet",
    "--dry-run": "it writes no file",
}
#: Type names llama-quantize accepts besides gguf-py's file types.
_EXTRA_TYPE_NAMES = frozenset({"Q2_0", "Q3_K", "Q4_K", "Q5_K", "COPY"})
_ENV_ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")


class ReproduceError(GGUFKitError):
    """Raised when a rebuild cannot be made: no llama-quantize, missing inputs, a failed run."""


@dataclass
class ReproduceResult:
    """A rebuild compared with the published file or with its manifest.

    ``report`` is the comparison: ``a`` is the published file or the manifest, ``b`` the
    rebuild. ``byte_identical`` is None when the bytes could not be compared (a split
    model, or a manifest without file hashes). ``inputs`` says, in manifest mode, how
    ``source_sha256``, ``imatrix_sha256`` and ``llama_cpp_commit`` compared.
    """

    report: Report
    byte_identical: bool | None
    command: list[str]
    llama_quantize_version: str | None
    quant_type: str
    inputs: dict[str, str] = field(default_factory=dict)

    @property
    def notes(self) -> list[str]:
        return self.report.notes

    def ok(self, *, allow_metadata: bool = False) -> bool:
        # Identical bytes settle it even when a manifest carries no weight digests.
        return self.byte_identical is True or self.report.ok(allow_metadata=allow_metadata)

    def to_dict(self) -> dict[str, Any]:
        out = self.report.to_dict()
        out.update(
            type=self.quant_type,
            command=list(self.command),
            llama_quantize_version=self.llama_quantize_version,
            byte_identical=self.byte_identical,
            inputs=dict(self.inputs),
        )
        return out


@dataclass(frozen=True)
class QuantizeRecipe:
    """What reproduce takes from a recorded llama-quantize command line."""

    options: tuple[str, ...] = ()
    quant_type: str | None = None
    imatrix: str | None = None


def _usable_type(label: str | None) -> bool:
    return label is not None and not label.startswith("unknown(") and label != "GUESSED"


def quant_type_of(model: Model) -> str:
    """Return the llama-quantize type a model says it is, from ``general.file_type``."""
    file_type = _kv(model, Keys.General.FILE_TYPE)
    label = file_type_label(file_type)
    if label is None or not _usable_type(label):
        found = "is missing" if file_type is None else f"is {file_type}"
        msg = (
            f"{model.path.name}: general.file_type {found}, so the quantization type is "
            "unknown; pass --type"
        )
        raise ReproduceError(msg)
    return label


@cache
def _type_names() -> frozenset[str]:
    labels = {file_type_label(int(t)) for t in LlamaFileType}
    return frozenset(x for x in labels if x is not None and _usable_type(x)) | _EXTRA_TYPE_NAMES


def _is_type(token: str) -> bool:
    return token.isdigit() or token.upper() in _type_names()


def parse_quantize_cmd(cmd: str) -> QuantizeRecipe:
    """Read a recorded llama-quantize command line the way llama-quantize reads its arguments.

    ``llama-quantize [options] input [output] type [nthreads]``: the options in
    :data:`RECIPE_OPTIONS` are kept in order; the paths and the thread count are not used,
    since the rebuild runs on the source given to reproduce.
    """
    try:
        tokens = shlex.split(cmd)
    except ValueError as e:
        msg = f"build_inputs.quantize_cmd cannot be parsed: {e}"
        raise ReproduceError(msg) from e
    while tokens and _ENV_ASSIGNMENT_RE.match(tokens[0]):
        tokens.pop(0)
    args = tokens[1:] if tokens and not tokens[0].startswith("--") else tokens
    options: list[str] = []
    imatrix: str | None = None
    i = 0
    while i < len(args) and args[i].startswith("--"):
        opt = args[i]
        if opt in _REFUSED_OPTIONS:
            msg = f"build_inputs.quantize_cmd uses {opt}, which reproduce cannot follow: {_REFUSED_OPTIONS[opt]}"
            raise ReproduceError(msg)
        arity = RECIPE_OPTIONS.get(opt, _DROPPED_OPTIONS.get(opt))
        if arity is None:
            msg = (
                f"build_inputs.quantize_cmd uses {opt}, which reproduce does not know; "
                "rebuild the file yourself and compare with reproduce TARGET"
            )
            raise ReproduceError(msg)
        values = args[i + 1 : i + 1 + arity]
        if len(values) < arity:
            msg = f"build_inputs.quantize_cmd: {opt} needs a value"
            raise ReproduceError(msg)
        if opt == "--imatrix":
            imatrix = values[0]
        elif opt in RECIPE_OPTIONS:
            options += [opt, *values]
        i += 1 + arity
    positional = args[i:]
    quant_type = None
    if len(positional) >= 2 and _is_type(positional[1]):
        quant_type = positional[1]  # input type [nthreads]
    elif len(positional) >= 3:
        quant_type = positional[2]  # input output type [nthreads]
    return QuantizeRecipe(tuple(options), quant_type, imatrix)


def find_llama_quantize(given: str | Path | None = None) -> str:
    """Return the absolute path of the llama-quantize to run: ``given``, or the one on PATH."""
    if given is not None:
        found = shutil.which(str(given))
        if found is None:
            problem = "is not executable" if Path(given).exists() else "was not found"
            msg = f"{given} {problem}"
            raise ReproduceError(msg)
        return str(Path(found).absolute())
    found = shutil.which("llama-quantize")
    if found is None:
        msg = (
            "llama-quantize is not on PATH: install llama.cpp "
            "(https://github.com/ggml-org/llama.cpp) and pass --llama-quantize PATH "
            "if its binaries are not on PATH"
        )
        raise ReproduceError(msg)
    return str(Path(found).absolute())


def _recorded_imatrix(model: Model) -> dict[str, Any]:
    return {
        name: value
        for name, key in QUANTIZE_IMATRIX_KEYS.items()
        if (value := _kv(model, key)) is not None
    }


def _missing_imatrix(
    what: str, recorded: Mapping[str, Any], sha256: str | None = None
) -> ReproduceError:
    details = [f"{k} {recorded[k]!r}" for k in ("file", "dataset") if k in recorded]
    if "chunks_count" in recorded:
        details.append(f"{recorded['chunks_count']} chunks")
    if sha256:
        details.append(f"sha256 {sha256}")
    which = f" ({', '.join(details)})" if details else ""
    msg = (
        f"{what} was quantized with an importance matrix{which}. The matrix sets the "
        "quantization scales, so the rebuild needs the same file: pass --imatrix FILE"
    )
    return ReproduceError(msg)


def _safe_relative(recorded: str) -> bool:
    """Whether a path read from the target can be recreated inside a fresh directory.

    It comes from an untrusted file, so only plain relative paths qualify: no root, no drive,
    no ``..``, no backslashes or control characters.
    """
    if not recorded or recorded.startswith("/") or recorded.endswith("/"):
        return False
    if any(c in "\\:" or ord(c) < 0x20 for c in recorded):
        return False
    parts = recorded.split("/")
    return ".." not in parts and parts[-1] not in ("", ".")


def _place_imatrix(
    imatrix: Path, recorded: Mapping[str, Any], run_dir: Path, notes: list[str]
) -> str:
    """Return the ``--imatrix`` argument that makes llama-quantize record the same path.

    llama-quantize writes its ``--imatrix`` argument verbatim into ``quantize.imatrix.file``,
    so the file is linked (or copied) to that relative path under ``run_dir``, the directory
    llama-quantize runs in. A path that cannot be recreated there falls back to the real one.
    """
    real = str(imatrix.absolute())
    if not recorded:
        notes.append(
            "the target records no importance matrix: llama-quantize adds quantize.imatrix.* "
            "keys, so expect a metadata difference"
        )
        return real
    path = recorded.get("file")
    if path is None:
        notes.append(
            "the target does not record the importance matrix's path, so "
            "quantize.imatrix.file differs"
        )
        return real
    path = str(path)
    if _safe_relative(path):
        dest = run_dir / path
        try:
            if dest.resolve().is_relative_to(run_dir.resolve()):
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    dest.symlink_to(imatrix.absolute())
                except (OSError, NotImplementedError):  # no symlinks (Windows without rights)
                    shutil.copyfile(imatrix, dest)
                return path
        except OSError:
            pass
    notes.append(
        f"the target records the importance matrix as {path!r}, which cannot be recreated "
        "in a temporary directory, so quantize.imatrix.file differs"
    )
    return real


def _run_quantize(argv: list[str], cwd: Path) -> tuple[int, str]:
    """Run llama-quantize; return its exit status and its stdout and stderr combined."""
    proc = subprocess.run(  # noqa: S603 - the binary is the one the user chose
        argv,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return proc.returncode, proc.stdout.decode("utf-8", "replace")


def _same_bytes(a: Path, b: Path) -> bool:
    if a.stat().st_size != b.stat().st_size:
        return False
    with a.open("rb") as fa, b.open("rb") as fb:
        while chunk := fa.read(canon.CHUNK_SIZE):
            if chunk != fb.read(len(chunk)):
                return False
    return True


def _source_paths(source: str | Path) -> list[Path]:
    paths = [p.absolute() for p in shard_paths(source)]
    if not paths[0].is_file():
        msg = f"{source}: no such file"
        raise FileNotFoundError(msg)
    return paths


def _check_imatrix_file(imatrix: str | Path | None) -> Path | None:
    if imatrix is None:
        return None
    if not Path(imatrix).is_file():
        msg = f"{imatrix}: no such file"
        raise FileNotFoundError(msg)
    return Path(imatrix)


def _check_keep(keep: str | Path | None, protected: list[Path]) -> Path | None:
    if keep is None:
        return None
    keep = Path(keep)
    if keep.exists() and any(p.exists() and keep.samefile(p) for p in protected):
        msg = f"--keep {keep} is the file being checked or one of its inputs; choose another path"
        raise ReproduceError(msg)
    return keep


@dataclass
class _Build:
    """Everything a rebuild needs, from either mode."""

    binary: str
    source: Path
    quant_type: str
    options: tuple[str, ...]
    imatrix: Path | None
    recorded_imatrix: Mapping[str, Any]
    threads: int | None
    keep: Path | None
    reference: str
    notes: list[str]
    #: Compares the rebuilt model with the reference.
    compare: Callable[[Model], Report]
    #: Compares the rebuilt file's bytes with the reference: (result, note).
    same_bytes: Callable[[Path], tuple[bool | None, str | None]]


def _rebuild(build: _Build, log: Callable[[str], None] | None) -> ReproduceResult:
    with tempfile.TemporaryDirectory(prefix="ggufkit-reproduce-") as tmp:
        run_dir = Path(tmp) / "run"
        run_dir.mkdir()
        notes = list(build.notes)
        imatrix_args: list[str] = []
        if build.imatrix is not None:
            placed = _place_imatrix(build.imatrix, build.recorded_imatrix, run_dir, notes)
            imatrix_args = ["--imatrix", placed]
        tail = [build.quant_type] + ([str(build.threads)] if build.threads else [])

        def run_once(out: Path, label: str) -> ReproduceResult:
            argv = [
                build.binary,
                *imatrix_args,
                *build.options,
                str(build.source),
                str(out.absolute()),
                *tail,
            ]
            if log is not None:
                log(f"running llama-quantize ({build.quant_type}); this can take a while")
            status, output = _run_quantize(argv, run_dir)
            version = _parse_version(output)
            if status != 0 or not out.is_file():
                lines = output.rstrip().splitlines()[-TAIL_LINES:]
                problem = f"exited with status {status}" if status else "wrote no file"
                msg = "\n".join([f"llama-quantize {problem}:", *lines])
                raise ReproduceError(msg)
            if log is not None:
                log(f"comparing the rebuild with {build.reference}")
            with open_model(out) as rebuilt:
                report = build.compare(rebuilt)
            report.b = label
            report.notes[:0] = notes
            same, note = build.same_bytes(out)
            if note is not None:
                report.notes.append(note)
            if same is False and report.verdict == "identical":
                report.notes.append(
                    "the model is identical, but the files are laid out differently "
                    "(order, alignment or padding)"
                )
            return ReproduceResult(report, same, argv, version, build.quant_type)

        if build.keep is None:
            return run_once(Path(tmp) / "rebuilt.gguf", "rebuild")
        with atomic_output(build.keep) as out:
            result = run_once(out, str(build.keep))
        # llama-quantize wrote next to the --keep file, and the result was then renamed to it.
        result.command[result.command.index(str(out.absolute()))] = str(build.keep.absolute())
        return result


def reproduce(
    target: Model,
    source: str | Path,
    *,
    imatrix: str | Path | None = None,
    quant_type: str | None = None,
    threads: int | None = None,
    llama_quantize: str | Path | None = None,
    keep: str | Path | None = None,
    jobs: int = 1,
    log: Callable[[str], None] | None = None,
) -> ReproduceResult:
    """Rebuild ``target`` from ``source`` with llama-quantize and compare the two."""
    qtype = quant_type or quant_type_of(target)
    recorded = _recorded_imatrix(target)
    if recorded and imatrix is None:
        raise _missing_imatrix(target.path.name, recorded)
    imatrix_path = _check_imatrix_file(imatrix)
    binary = find_llama_quantize(llama_quantize)
    sources = _source_paths(source)
    keep_path = _check_keep(keep, [*target.paths, *sources])

    def same_bytes(out: Path) -> tuple[bool | None, str | None]:
        if target.n_shards > 1:
            return None, (
                f"the target is split into {target.n_shards} files and the rebuild is one, "
                "so the file bytes were not compared"
            )
        return _same_bytes(target.path, out), None

    build = _Build(
        binary=binary,
        source=sources[0],  # llama-quantize takes the first shard of a split source
        quant_type=qtype,
        options=(),
        imatrix=imatrix_path,
        recorded_imatrix=recorded,
        threads=threads,
        keep=keep_path,
        reference=target.path.name,
        notes=[],
        compare=lambda rebuilt: verify_against_file(target, rebuilt, jobs=jobs),
        same_bytes=same_bytes,
    )
    return _rebuild(build, log)


def _expected_hashes(recorded: str) -> list[str]:
    return [h.lower().removeprefix("sha256:") for h in re.split(r"[,\s]+", recorded.strip()) if h]


def _check_hash(
    what: str,
    key: str,
    recorded: str | None,
    paths: list[Path],
    *,
    notes: list[str],
    log: Callable[[str], None] | None,
) -> str:
    """Compare files with a recorded sha256 (comma-separated per shard); raise on a mismatch."""
    if not recorded:
        notes.append(f"the manifest records no {key}, so the {what} was not checked")
        return "not recorded"
    if log is not None:
        log(f"checking the sha256 of the {what}")
    actual = [sha256_file(p) for p in paths]
    if actual != _expected_hashes(recorded):
        msg = (
            f"the {what} is not the one the manifest names: its sha256 is {','.join(actual)}, "
            f"and the manifest's {key} is {recorded}"
        )
        raise ReproduceError(msg)
    return "match"


def _commit_status(recorded: str | None, version: str | None, notes: list[str]) -> str:
    """Compare the manifest's llama_cpp_commit with the llama.cpp version that ran."""
    if not recorded:
        return "not recorded"
    wanted = recorded.strip().lower()
    commit = re.search(r"commit ([0-9a-f]+)", version or "")
    build = re.search(r"build (\d+)", version or "")
    tag = re.fullmatch(r"b(\d+)", wanted)
    if commit and re.fullmatch(r"[0-9a-f]{7,40}", wanted):
        same = wanted.startswith(commit[1]) or commit[1].startswith(wanted)
    elif build and tag:
        same = tag[1] == build[1]
    else:
        notes.append(
            f"the manifest's llama_cpp_commit {recorded!r} could not be compared with "
            f"{version or 'the version llama-quantize did not report'}"
        )
        return "not compared"
    if not same:
        notes.append(f"the manifest names llama.cpp {recorded}; this rebuild used {version}")
    return "match" if same else "differs"


def _manifest_imatrix(manifest: Mapping[str, Any], recipe: QuantizeRecipe) -> dict[str, Any]:
    """Return the ``quantize.imatrix.*`` values of the file a manifest describes."""
    kvs = (manifest.get("fingerprint") or {}).get("kvs")
    provenance = (manifest.get("provenance") or {}).get("imatrix") or {}
    if isinstance(kvs, Mapping):
        if not any(k.startswith("quantize.imatrix.") for k in kvs):
            return {}
    elif not provenance:
        return {}
    recorded = {k: v for k, v in provenance.items() if k in QUANTIZE_IMATRIX_KEYS}
    if "file" not in recorded and recipe.imatrix is not None:
        recorded["file"] = recipe.imatrix
    return recorded


def _manifest_type(manifest: Mapping[str, Any], label: str) -> str:
    file_type = (manifest.get("model") or {}).get("file_type")
    if not isinstance(file_type, str) or not _usable_type(file_type):
        msg = f"{label} records no usable quantization type (model.file_type); pass --type"
        raise ReproduceError(msg)
    return file_type


def reproduce_from_manifest(
    manifest: str | Path | Mapping[str, Any],
    source: str | Path,
    *,
    imatrix: str | Path | None = None,
    quant_type: str | None = None,
    threads: int | None = None,
    llama_quantize: str | Path | None = None,
    keep: str | Path | None = None,
    jobs: int = 1,
    log: Callable[[str], None] | None = None,
) -> ReproduceResult:
    """Rebuild the file a manifest describes from its build inputs, and compare.

    The source and importance matrix are checked against ``source_sha256`` and
    ``imatrix_sha256`` first; ``quantize_cmd`` gives the type and recipe options; the rebuild
    is compared with the manifest's fingerprint and, when recorded, its file sha256.
    """
    if isinstance(manifest, Mapping):
        loaded: Mapping[str, Any] = manifest
        label = "the manifest"
        protected: list[Path] = []
    else:
        loaded = load_manifest(manifest)
        label = str(manifest)
        protected = [Path(manifest)]
    inputs = loaded.get("build_inputs") or {}
    if not isinstance(inputs, Mapping) or not all(isinstance(v, str) for v in inputs.values()):
        msg = f"{label}: build_inputs must map names to strings"
        raise ReproduceError(msg)
    cmd = inputs.get("quantize_cmd")
    recipe = parse_quantize_cmd(cmd) if cmd else QuantizeRecipe()
    qtype = quant_type or recipe.quant_type or _manifest_type(loaded, label)

    notes: list[str] = []
    if not cmd:
        notes.append(
            "the manifest records no quantize_cmd: the rebuild uses the type alone, "
            "with no recipe options"
        )
    recorded = _manifest_imatrix(loaded, recipe)
    imatrix_sha256 = inputs.get("imatrix_sha256")
    if imatrix is None and (recorded or imatrix_sha256 or recipe.imatrix is not None):
        raise _missing_imatrix(f"The model in {label}", recorded, imatrix_sha256)
    imatrix_path = _check_imatrix_file(imatrix)
    binary = find_llama_quantize(llama_quantize)
    sources = _source_paths(source)
    keep_path = _check_keep(keep, [*sources, *protected])

    statuses = {
        "source_sha256": _check_hash(
            "source",
            "source_sha256",
            inputs.get("source_sha256"),
            sources,
            notes=notes,
            log=log,
        ),
        "imatrix_sha256": "not used"
        if imatrix_path is None
        else _check_hash(
            "importance matrix",
            "imatrix_sha256",
            imatrix_sha256,
            [imatrix_path],
            notes=notes,
            log=log,
        ),
    }

    files = loaded.get("files") or []

    def same_bytes(out: Path) -> tuple[bool | None, str | None]:
        if len(files) > 1:
            return None, (
                f"the manifest lists {len(files)} files and the rebuild is one, so the file "
                "bytes were not compared"
            )
        recorded_sha = files[0].get("sha256") if files else None
        if not recorded_sha:
            return None, (
                "the manifest records no file sha256 (made without --file-sha256), so the "
                "file bytes were not compared"
            )
        return sha256_file(out) == str(recorded_sha).lower(), None

    build = _Build(
        binary=binary,
        source=sources[0],
        quant_type=qtype,
        options=recipe.options,
        imatrix=imatrix_path,
        recorded_imatrix=recorded,
        threads=threads,
        keep=keep_path,
        reference=label,
        notes=notes,
        compare=lambda rebuilt: verify_against_manifest(
            rebuilt, loaded, manifest_label=label, jobs=jobs
        ),
        same_bytes=same_bytes,
    )
    result = _rebuild(build, log)
    commit_notes: list[str] = []
    statuses["llama_cpp_commit"] = _commit_status(
        inputs.get("llama_cpp_commit"), result.llama_quantize_version, commit_notes
    )
    result.report.notes.extend(commit_notes)
    result.inputs = statuses
    return result


def _parse_version(output: str) -> str | None:
    if match := _VERSION_RE.search(output):
        return match[1]
    if match := _OLD_VERSION_RE.search(output):
        return f"build {match[1]}, commit {match[2]}"
    return None


_INPUT_WORDS = {
    "match": "sha256 matches the manifest",
    "not recorded": "not checked: the manifest records no sha256",
    "not used": "not used",
}


def format_result(result: ReproduceResult) -> list[str]:
    same = result.byte_identical
    inputs = result.inputs
    lines: list[str] = []
    if inputs:
        lines.append(f"{'manifest':<12} {result.report.a}")
        lines.append(f"{'source':<12} {_INPUT_WORDS.get(inputs['source_sha256'], '')}")
        lines.append(f"{'imatrix':<12} {_INPUT_WORDS.get(inputs['imatrix_sha256'], '')}")
    version = result.llama_quantize_version or "version not reported"
    commit = inputs.get("llama_cpp_commit")
    if commit is not None and commit != "not recorded":
        word = {"match": "same", "differs": "differs"}.get(commit, "not compared")
        version = f"{version}; manifest: {word}"
    lines += [
        f"{'type':<12} {result.quant_type}",
        f"{'llama.cpp':<12} {version}",
        f"{'command':<12} {shlex.join(result.command)}",
        f"{'file bytes':<12} {'not compared' if same is None else ('identical' if same else 'differ')}",
    ]
    return lines + format_report(result.report)


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser(
        "reproduce",
        help="rebuild a quantized model with llama-quantize; exit 0 only if identical",
        description=(
            "Rebuild a quantized GGUF file from its source with llama.cpp's llama-quantize "
            "and compare the rebuild with it, as verify --against does. The quantization "
            "type comes from the file's general.file_type. A file made with an importance "
            "matrix needs the same matrix (--imatrix). With --manifest instead of the file, "
            "the source and matrix are checked against the manifest's build inputs first, "
            "its quantize_cmd gives the type and recipe options, and the rebuild is "
            "compared with the manifest. Exit status is 0 when the rebuild is identical (or "
            "metadata-only with --allow-metadata), 1 when it differs, 2 on error. The "
            "rebuild is written to a temporary directory (TMPDIR) unless --keep."
        ),
    )
    p.add_argument(
        "target", type=Path, nargs="?", help="the quantized GGUF file to check (or --manifest)"
    )
    p.add_argument(
        "--manifest",
        type=Path,
        metavar="FILE",
        help="rebuild the file this manifest describes, from its build inputs",
    )
    p.add_argument(
        "--source",
        type=Path,
        required=True,
        metavar="MODEL",
        help="the full-precision GGUF it was made from (any shard of a split model)",
    )
    p.add_argument(
        "--imatrix",
        type=Path,
        metavar="FILE",
        help="the importance matrix it was made with; required when the target records one",
    )
    p.add_argument(
        "--type",
        dest="quant_type",
        metavar="TYPE",
        help="llama-quantize type (default: from quantize_cmd or general.file_type)",
    )
    p.add_argument(
        "--llama-quantize",
        metavar="PATH",
        help="the llama-quantize binary to run (default: the one on PATH)",
    )
    p.add_argument(
        "--threads",
        type=positive_int,
        metavar="N",
        help="llama-quantize threads (default: its own choice; the output does not depend on it)",
    )
    p.add_argument("--keep", type=Path, metavar="FILE", help="also save the rebuilt file")
    p.add_argument("--force", action="store_true", help="overwrite the --keep file")
    p.add_argument(
        "--allow-metadata",
        action="store_true",
        help="treat a metadata-only difference as success",
    )
    p.add_argument("--json", action="store_true", help="print the result as JSON")
    p.add_argument(
        "-j", "--jobs", type=positive_int, default=default_jobs(), help="hashing threads"
    )
    p.add_argument("-q", "--quiet", action="store_true", help="print nothing; use the exit status")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    if args.target is None and args.manifest is None:
        msg = "give the file to check, or --manifest FILE"
        raise ValueError(msg)
    if args.target is not None and args.manifest is not None:
        msg = (
            "give the file to check or --manifest, not both "
            "(verify FILE --manifest MANIFEST checks a file against its manifest)"
        )
        raise ValueError(msg)
    if args.keep is not None:
        prepare_output(args.keep, force=args.force)
    options: dict[str, Any] = {
        "imatrix": args.imatrix,
        "quant_type": args.quant_type,
        "threads": args.threads,
        "llama_quantize": args.llama_quantize,
        "keep": args.keep,
        "jobs": args.jobs,
        "log": None if args.quiet else to_stderr,
    }
    if args.manifest is not None:
        result = reproduce_from_manifest(args.manifest, args.source, **options)
    else:
        with open_model(args.target) as target:
            result = reproduce(target, args.source, **options)
    if args.json:
        json.dump(result.to_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")
    elif not args.quiet:
        print("\n".join(format_result(result)))
    return 0 if result.ok(allow_metadata=args.allow_metadata) else 1
