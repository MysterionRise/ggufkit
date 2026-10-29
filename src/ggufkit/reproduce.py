"""Rebuild a quantized model from its source with llama-quantize, and compare.

llama.cpp's quantizer is deterministic: the same source, importance matrix and quantization
type give the same file, byte for byte, whatever the thread count or CPU features. So a
published quant can be checked by making it again. ``reproduce`` runs ``llama-quantize`` on
the source and compares the rebuild with the published file the way ``verify --against``
does. When they differ, the report says whether it is the metadata, the tensor descriptors
or the weights, and which keys and tensors.
"""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gguf.constants import Keys

from ggufkit import canon
from ggufkit._util import atomic_output, positive_int, prepare_output, to_stderr
from ggufkit.errors import GGUFKitError
from ggufkit.fingerprint import default_jobs
from ggufkit.manifest import QUANTIZE_IMATRIX_KEYS, _kv, file_type_label
from ggufkit.model import open_model, shard_paths
from ggufkit.verify import format_report, verify_against_file

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


class ReproduceError(GGUFKitError):
    """Raised when a rebuild cannot be made: no llama-quantize, missing inputs, a failed run."""


@dataclass
class ReproduceResult:
    """A rebuild compared with the published file.

    ``report`` is the comparison (``a`` is the published file, ``b`` the rebuild).
    ``byte_identical`` is None when the published model is split into several files, since
    llama-quantize writes one.
    """

    report: Report
    byte_identical: bool | None
    command: list[str]
    llama_quantize_version: str | None
    quant_type: str

    @property
    def notes(self) -> list[str]:
        return self.report.notes

    def ok(self, *, allow_metadata: bool = False) -> bool:
        return self.report.ok(allow_metadata=allow_metadata)

    def to_dict(self) -> dict[str, Any]:
        out = self.report.to_dict()
        out.update(
            type=self.quant_type,
            command=list(self.command),
            llama_quantize_version=self.llama_quantize_version,
            byte_identical=self.byte_identical,
        )
        return out


def quant_type_of(model: Model) -> str:
    """Return the llama-quantize type a model says it is, from ``general.file_type``."""
    file_type = _kv(model, Keys.General.FILE_TYPE)
    label = file_type_label(file_type)
    if label is None or label.startswith("unknown(") or label == "GUESSED":
        found = "is missing" if file_type is None else f"is {file_type}"
        msg = (
            f"{model.path.name}: general.file_type {found}, so the quantization type is "
            "unknown; pass --type"
        )
        raise ReproduceError(msg)
    return label


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


def _missing_imatrix(model: Model, recorded: dict[str, Any]) -> ReproduceError:
    details = [f"{k} {recorded[k]!r}" for k in ("file", "dataset") if k in recorded]
    if "chunks_count" in recorded:
        details.append(f"{recorded['chunks_count']} chunks")
    msg = (
        f"{model.path.name} was quantized with an importance matrix ({', '.join(details)}). "
        "The matrix sets the quantization scales, so the rebuild needs the same file: "
        "pass --imatrix FILE"
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


def _place_imatrix(imatrix: Path, recorded: dict[str, Any], run_dir: Path, notes: list[str]) -> str:
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
        raise _missing_imatrix(target, recorded)
    if imatrix is not None and not Path(imatrix).is_file():
        msg = f"{imatrix}: no such file"
        raise FileNotFoundError(msg)
    binary = find_llama_quantize(llama_quantize)
    sources = shard_paths(source)
    source_path = sources[0].absolute()  # the first shard, for a split source
    if not source_path.is_file():
        msg = f"{source}: no such file"
        raise FileNotFoundError(msg)
    if keep is not None:
        keep = Path(keep)
        if keep.exists() and any(keep.samefile(p) for p in [*target.paths, *sources]):
            msg = f"--keep {keep} is the file being checked or its source; choose another path"
            raise ReproduceError(msg)

    notes: list[str] = []
    with tempfile.TemporaryDirectory(prefix="ggufkit-reproduce-") as tmp:
        run_dir = Path(tmp) / "run"
        run_dir.mkdir()
        options: list[str] = []
        if imatrix is not None:
            options = [
                "--imatrix",
                _place_imatrix(Path(imatrix), recorded, run_dir, notes),
            ]
        tail = [qtype] + ([str(threads)] if threads else [])

        def rebuild(out: Path, label: str) -> ReproduceResult:
            argv = [binary, *options, str(source_path), str(out.absolute()), *tail]
            if log is not None:
                log(f"running llama-quantize ({qtype}); this can take a while")
            status, output = _run_quantize(argv, run_dir)
            version = _parse_version(output)
            if status != 0 or not out.is_file():
                lines = output.rstrip().splitlines()[-TAIL_LINES:]
                problem = f"exited with status {status}" if status else "wrote no file"
                msg = "\n".join([f"llama-quantize {problem}:", *lines])
                raise ReproduceError(msg)
            if log is not None:
                log(f"comparing the rebuild with {target.path.name}")
            with open_model(out) as rebuilt:
                report = verify_against_file(target, rebuilt, jobs=jobs)
            report.b = label
            report.notes[:0] = notes
            same: bool | None = None
            if target.n_shards > 1:
                report.notes.append(
                    f"the target is split into {target.n_shards} files and the rebuild is one, "
                    "so the file bytes were not compared"
                )
            else:
                same = _same_bytes(target.path, out)
                if not same and report.verdict == "identical":
                    report.notes.append(
                        "the model is identical, but the files are laid out differently "
                        "(order, alignment or padding)"
                    )
            return ReproduceResult(report, same, argv, version, qtype)

        if keep is None:
            return rebuild(Path(tmp) / "rebuilt.gguf", "rebuild")
        with atomic_output(keep) as out:
            result = rebuild(out, str(keep))
        # llama-quantize wrote next to the --keep file, and the result was then renamed to it.
        result.command[result.command.index(str(out.absolute()))] = str(keep.absolute())
        return result


def _parse_version(output: str) -> str | None:
    if match := _VERSION_RE.search(output):
        return match[1]
    if match := _OLD_VERSION_RE.search(output):
        return f"build {match[1]}, commit {match[2]}"
    return None


def format_result(result: ReproduceResult) -> list[str]:
    same = result.byte_identical
    lines = [
        f"{'type':<12} {result.quant_type}",
        f"{'llama.cpp':<12} {result.llama_quantize_version or 'version not reported'}",
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
            "matrix needs the same matrix (--imatrix). Exit status is 0 when the rebuild is "
            "identical (or metadata-only with --allow-metadata), 1 when it differs, 2 on "
            "error. The rebuild is written to a temporary directory (TMPDIR) unless --keep."
        ),
    )
    p.add_argument("target", type=Path, help="the quantized GGUF file to check")
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
        help="llama-quantize type (default: from the target's general.file_type)",
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
    if args.keep is not None:
        prepare_output(args.keep, force=args.force)
    with open_model(args.target) as target:
        result = reproduce(
            target,
            args.source,
            imatrix=args.imatrix,
            quant_type=args.quant_type,
            threads=args.threads,
            llama_quantize=args.llama_quantize,
            keep=args.keep,
            jobs=args.jobs,
            log=None if args.quiet else to_stderr,
        )
    if args.json:
        json.dump(result.to_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")
    elif not args.quiet:
        print("\n".join(format_result(result)))
    return 0 if result.ok(allow_metadata=args.allow_metadata) else 1
