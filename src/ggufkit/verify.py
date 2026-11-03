"""Compare fingerprints and say, in one word, what changed.

``verify`` answers the question a plain hash cannot: not just "is this the same file?" but
"is this the same model, and if not, is it the metadata, the tensor descriptors or the
weights that differ?" It works against a manifest written earlier or directly against a
second file.
"""

from __future__ import annotations

import fnmatch
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ggufkit import canon
from ggufkit._util import positive_int
from ggufkit.fingerprint import Fingerprint, default_jobs, fingerprint
from ggufkit.manifest import load_manifest, sha256_file
from ggufkit.model import open_model

if TYPE_CHECKING:
    import argparse
    from collections.abc import Iterable, Mapping, Sequence

    from ggufkit.model import Model

IDENTICAL = "identical"
METADATA_ONLY = "metadata-only"
TENSOR_INFO_ONLY = "tensor-info-only"
METADATA_AND_TENSOR_INFO = "metadata-and-tensor-info"
WEIGHTS_CHANGED = "weights-changed"


@dataclass(frozen=True)
class KeyChange:
    key: str
    status: str  # added, removed, changed

    def __str__(self) -> str:
        mark = {"added": "+", "removed": "-", "changed": "~"}[self.status]
        return f"{mark} {self.key}"


@dataclass(frozen=True)
class TensorChange:
    name: str
    kind: str  # added, removed, renamed, retyped, reshaped, data
    detail: str = ""

    def __str__(self) -> str:
        mark = {"added": "+", "removed": "-", "renamed": ">", "data": "~"}.get(self.kind, "~")
        text = f"{mark} {self.name}  {self.kind}"
        return f"{text}  {self.detail}" if self.detail else text


@dataclass
class Report:
    a: str
    b: str
    verdict: str
    kv_same: bool
    tensor_info_same: bool
    weights_same: bool | None
    ignored_keys: list[str] = field(default_factory=list)
    key_changes: list[KeyChange] | None = None
    tensor_changes: list[TensorChange] | None = None
    notes: list[str] = field(default_factory=list)
    #: Raw file hashes against the manifest: None (not asked), "match", "mismatch" or
    #: "not checked" (the manifest recorded none).
    files: str | None = None

    @property
    def weights_compared(self) -> bool:
        return self.weights_same is not None

    def ok(self, *, allow_metadata: bool = False, allow_unverified_weights: bool = False) -> bool:
        """Whether this counts as a pass.

        Weights that were not compared fail unless the caller asked for that explicitly
        (``verify --no-data``): a manifest made without weight digests must not let a file
        with different weights through.
        """
        if self.files not in (None, "match"):
            return False
        if not self.weights_compared and not allow_unverified_weights:
            return False
        if self.verdict == IDENTICAL:
            return True
        return allow_metadata and self.verdict == METADATA_ONLY

    def to_dict(self) -> dict[str, Any]:
        return {
            "a": self.a,
            "b": self.b,
            "verdict": self.verdict,
            "sections": {
                "kv": self.kv_same,
                "tensor_info": self.tensor_info_same,
                "weights": self.weights_same,
            },
            "ignored_keys": list(self.ignored_keys),
            "key_changes": None
            if self.key_changes is None
            else [{"key": c.key, "status": c.status} for c in self.key_changes],
            "tensor_changes": None
            if self.tensor_changes is None
            else [
                {"name": c.name, "kind": c.kind, "detail": c.detail} for c in self.tensor_changes
            ],
            "files": self.files,
            "notes": list(self.notes),
        }


def _ignored(key: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(key, pat) for pat in patterns)


def _key_changes(
    kvs_a: dict[str, str],
    kvs_b: dict[str, str],
    ignore: Sequence[str],
) -> list[KeyChange]:
    changes: list[KeyChange] = []
    for key in canon.sorted_by_name(set(kvs_a) | set(kvs_b)):
        if _ignored(key, ignore):
            continue
        if key not in kvs_a:
            changes.append(KeyChange(key, "added"))
        elif key not in kvs_b:
            changes.append(KeyChange(key, "removed"))
        elif kvs_a[key] != kvs_b[key]:
            changes.append(KeyChange(key, "changed"))
    return changes


def _tensor_changes(fa: Fingerprint, fb: Fingerprint) -> list[TensorChange]:
    ta = fa.tensors or {}
    tb = fb.tensors or {}
    removed = [n for n in ta if n not in tb]
    added = [n for n in tb if n not in ta]
    changes: list[TensorChange] = []

    # A removed and an added tensor with the same bytes, type and shape is a rename.
    by_data: dict[str, list[str]] = {}
    for name in added:
        t = tb[name]
        if t.data is not None:
            by_data.setdefault(t.data, []).append(name)
    renamed: set[str] = set()
    for old in list(removed):
        t = ta[old]
        if t.data is None:
            continue
        for new in by_data.get(t.data, []):
            if new in renamed:
                continue
            u = tb[new]
            if u.type == t.type and u.shape == t.shape:
                changes.append(TensorChange(old, "renamed", f"-> {new}"))
                renamed.add(new)
                removed.remove(old)
                break
    added = [n for n in added if n not in renamed]

    changes.extend(
        TensorChange(n, "removed", f"{ta[n].type} {'x'.join(map(str, ta[n].shape))}")
        for n in removed
    )
    changes.extend(
        TensorChange(n, "added", f"{tb[n].type} {'x'.join(map(str, tb[n].shape))}") for n in added
    )

    for name in ta:
        if name not in tb:
            continue
        t, u = ta[name], tb[name]
        if t.shape != u.shape:
            changes.append(
                TensorChange(
                    name,
                    "reshaped",
                    f"{'x'.join(map(str, t.shape))} -> {'x'.join(map(str, u.shape))}",
                )
            )
        elif t.type != u.type:
            changes.append(TensorChange(name, "retyped", f"{t.type} -> {u.type}"))
        elif t.data is not None and u.data is not None and t.data != u.data:
            changes.append(TensorChange(name, "data", f"{t.data[:12]} -> {u.data[:12]}"))
    return sorted(changes, key=lambda c: c.name.encode("utf-8"))


def compare(
    fa: Fingerprint,
    fb: Fingerprint,
    *,
    a: str = "a",
    b: str = "b",
    ignore_keys: Sequence[str] = (),
) -> Report:
    """Classify the difference between two fingerprints."""
    if fa.spec != fb.spec:
        msg = f"Fingerprint specs differ: {fa.spec} vs {fb.spec}"
        raise ValueError(msg)

    key_changes: list[KeyChange] | None = None
    if fa.kvs is not None and fb.kvs is not None:
        key_changes = _key_changes(fa.kvs, fb.kvs, ignore_keys)
        kv_same = not key_changes
    else:
        if ignore_keys:
            msg = "--ignore-keys needs per-key digests on both sides (fingerprint --per-kv)"
            raise ValueError(msg)
        kv_same = fa.kv == fb.kv

    tensor_info_same = fa.tensor_info == fb.tensor_info
    weights_same: bool | None = None
    if fa.weights is not None and fb.weights is not None:
        weights_same = fa.weights == fb.weights

    tensor_changes: list[TensorChange] | None = None
    if fa.tensors is not None and fb.tensors is not None:
        tensor_changes = _tensor_changes(fa, fb)

    renames_only = bool(tensor_changes) and all(c.kind == "renamed" for c in tensor_changes or [])
    if weights_same is False and not renames_only:
        verdict = WEIGHTS_CHANGED
    elif kv_same and tensor_info_same:
        verdict = IDENTICAL
    elif tensor_info_same:
        verdict = METADATA_ONLY
    elif kv_same:
        verdict = TENSOR_INFO_ONLY
    else:
        verdict = METADATA_AND_TENSOR_INFO

    notes: list[str] = []
    if weights_same is None:
        notes.append("weights were not compared (no weight digests on one side)")
    elif weights_same is False and renames_only:
        notes.append("the same tensor data is stored under new names")
    return Report(
        a=a,
        b=b,
        verdict=verdict,
        kv_same=kv_same,
        tensor_info_same=tensor_info_same,
        weights_same=weights_same,
        ignored_keys=list(ignore_keys),
        key_changes=key_changes,
        tensor_changes=tensor_changes,
        notes=notes,
    )


def verify_against_manifest(
    model: Model,
    manifest: Mapping[str, Any],
    *,
    manifest_label: str = "manifest",
    data: bool = True,
    jobs: int = 1,
    ignore_keys: Sequence[str] = (),
    check_files: bool = False,
) -> Report:
    expected = Fingerprint.from_dict(manifest["fingerprint"])
    actual = fingerprint(
        model,
        data=data and expected.weights is not None,
        per_tensor=expected.tensors is not None,
        per_kv=expected.kvs is not None,
        jobs=jobs,
    )
    report = compare(expected, actual, a=manifest_label, b=str(model.path), ignore_keys=ignore_keys)
    if check_files:
        report.files = _check_files(model, manifest, report.notes)
    return report


def _check_files(model: Model, manifest: Mapping[str, Any], notes: list[str]) -> str:
    """Compare raw file hashes with the manifest: "match", "mismatch" or "not checked"."""
    recorded = {f["path"]: f for f in manifest.get("files") or []}
    status = "match"
    for path in model.paths:
        entry = recorded.get(path.name)
        if entry is None:
            notes.append(f"{path.name}: not listed in the manifest")
            status = "mismatch"
        elif entry.get("sha256") is None:
            notes.append(f"{path.name}: the manifest has no file hash (made without --file-sha256)")
            if status == "match":
                status = "not checked"
        elif sha256_file(path) != entry["sha256"]:
            notes.append(f"{path.name}: file sha256 differs from the manifest")
            status = "mismatch"
    return status


def verify_against_file(
    model_a: Model,
    model_b: Model,
    *,
    data: bool = True,
    jobs: int = 1,
    ignore_keys: Sequence[str] = (),
) -> Report:
    fa = fingerprint(model_a, data=data, per_tensor=True, per_kv=True, jobs=jobs)
    fb = fingerprint(model_b, data=data, per_tensor=True, per_kv=True, jobs=jobs)
    return compare(fa, fb, a=str(model_a.path), b=str(model_b.path), ignore_keys=ignore_keys)


def format_report(report: Report) -> list[str]:
    def word(same: bool | None) -> str:
        return "not compared" if same is None else ("same" if same else "differs")

    lines = [
        f"{'verdict':<12} {report.verdict}",
        f"{'kv':<12} {word(report.kv_same)}",
        f"{'tensor_info':<12} {word(report.tensor_info_same)}",
        f"{'weights':<12} {word(report.weights_same)}",
    ]
    if report.files is not None:
        lines.append(f"{'files':<12} {report.files}")
    if report.ignored_keys:
        lines.append(f"{'ignored':<12} {', '.join(report.ignored_keys)}")
    if report.key_changes:
        lines.append("metadata changes:")
        lines.extend(f"  {c}" for c in report.key_changes)
    if report.tensor_changes:
        lines.append("tensor changes:")
        lines.extend(f"  {c}" for c in report.tensor_changes)
    lines.extend(f"note: {n}" for n in report.notes)
    return lines


def comma_list(values: Iterable[str] | None) -> list[str]:
    out: list[str] = []
    for v in values or []:
        out.extend(x.strip() for x in v.split(",") if x.strip())
    return out


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser(
        "verify",
        help="check a model against a manifest or another file; exit 0 only if identical",
        description=(
            "Compare a model's fingerprint with a manifest (--manifest) or with another GGUF "
            "file (--against) and report what differs: nothing, metadata only, tensor "
            "descriptors only, or weights. Exit status is 0 when identical (or metadata-only "
            "with --allow-metadata), 1 on any other difference or when the weights could not "
            "be compared, 2 on error."
        ),
    )
    p.add_argument("model", type=Path, help="GGUF file, or any shard of a split model")
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--manifest", type=Path, metavar="FILE", help="manifest to verify against")
    target.add_argument(
        "--against", type=Path, metavar="MODEL", help="other GGUF file to compare with"
    )
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    p.add_argument(
        "--no-data",
        action="store_true",
        help="compare metadata and descriptors only, and accept that the weights are unchecked",
    )
    p.add_argument(
        "--ignore-keys",
        action="append",
        metavar="GLOB",
        help="metadata keys to ignore (glob, comma-separated or repeatable)",
    )
    p.add_argument(
        "--allow-metadata",
        action="store_true",
        help="treat a metadata-only difference as success",
    )
    p.add_argument(
        "--file-sha256",
        action="store_true",
        help="with --manifest: also check the raw file hashes it recorded",
    )
    p.add_argument(
        "-j", "--jobs", type=positive_int, default=default_jobs(), help="hashing threads"
    )
    p.add_argument("-q", "--quiet", action="store_true", help="print nothing; use the exit status")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    if args.file_sha256 and args.manifest is None:
        msg = "--file-sha256 needs --manifest: there are no recorded file hashes to check"
        raise ValueError(msg)
    ignore = comma_list(args.ignore_keys)
    jobs = args.jobs
    with open_model(args.model) as model:
        if args.manifest is not None:
            manifest = load_manifest(args.manifest)
            report = verify_against_manifest(
                model,
                manifest,
                manifest_label=str(args.manifest),
                data=not args.no_data,
                jobs=jobs,
                ignore_keys=ignore,
                check_files=args.file_sha256,
            )
        else:
            with open_model(args.against) as other:
                report = verify_against_file(
                    model,
                    other,
                    data=not args.no_data,
                    jobs=jobs,
                    ignore_keys=ignore,
                )
    if args.json:
        json.dump(report.to_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")
    elif not args.quiet:
        print("\n".join(format_report(report)))
    passed = report.ok(allow_metadata=args.allow_metadata, allow_unverified_weights=args.no_data)
    return 0 if passed else 1
