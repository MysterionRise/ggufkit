"""Order-independent content fingerprints for GGUF models (``ggufkit-fp/1``).

A fingerprint has three independent parts, so you can tell *what* changed between two files:

- ``kv``: the metadata, sorted by key, so reordering or re-saving does not change it;
- ``tensor_info``: every tensor's name, shape and quantization type, sorted by name;
- ``weights``: the stored bytes of every tensor, bound to its name and sorted by name, so
  reordering tensors or sharding does not change it, while swapping two tensors' data or
  renaming a tensor does (the per-tensor ``data`` digests show a rename as the same bytes).

``overall`` binds the three together. This is a rewrite of KerfuffleV2's ``gguf-checksum``
on a documented byte format; see ``docs/FINGERPRINT.md``.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ggufkit import canon
from ggufkit._util import positive_int, prepare_output, write_text_atomic
from ggufkit.model import open_model

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable

    from gguf.gguf_reader import ReaderTensor

    from ggufkit.model import Model


@dataclass(frozen=True)
class TensorFingerprint:
    name: str
    type: str
    shape: tuple[int, ...]
    info: str
    data: str | None

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "shape": list(self.shape), "info": self.info, "data": self.data}

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> TensorFingerprint:
        return cls(
            name=name,
            type=str(d["type"]),
            shape=tuple(int(x) for x in d["shape"]),
            info=str(d["info"]),
            data=None if d.get("data") is None else str(d["data"]),
        )


@dataclass
class Fingerprint:
    spec: str
    kv: str
    tensor_info: str
    weights: str | None
    overall: str | None
    n_kv: int
    n_tensors: int
    kvs: dict[str, str] | None = None
    tensors: dict[str, TensorFingerprint] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "spec": self.spec,
            "kv": self.kv,
            "tensor_info": self.tensor_info,
            "weights": self.weights,
            "overall": self.overall,
            "n_kv": self.n_kv,
            "n_tensors": self.n_tensors,
        }
        if self.kvs is not None:
            out["kvs"] = dict(self.kvs)
        if self.tensors is not None:
            out["tensors"] = {name: t.to_dict() for name, t in self.tensors.items()}
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Fingerprint:
        tensors = d.get("tensors")
        return cls(
            spec=str(d["spec"]),
            kv=str(d["kv"]),
            tensor_info=str(d["tensor_info"]),
            weights=None if d.get("weights") is None else str(d["weights"]),
            overall=None if d.get("overall") is None else str(d["overall"]),
            n_kv=int(d["n_kv"]),
            n_tensors=int(d["n_tensors"]),
            kvs=None if d.get("kvs") is None else {str(k): str(v) for k, v in d["kvs"].items()},
            tensors=None
            if tensors is None
            else {str(k): TensorFingerprint.from_dict(str(k), v) for k, v in tensors.items()},
        )


def default_jobs() -> int:
    return max(1, min(4, os.cpu_count() or 1))


def _hash_tensor_data(
    tensors: list[ReaderTensor],
    file_endianess: Any,
    jobs: int,
    on_tensor: Callable[[ReaderTensor, bytes], None] | None,
) -> dict[str, bytes]:
    def work(t: ReaderTensor) -> tuple[ReaderTensor, bytes]:
        return t, canon.tensor_data_digest(t, file_endianess)

    out: dict[str, bytes] = {}
    if jobs <= 1 or len(tensors) <= 1:
        results = map(work, tensors)
        for t, digest in results:
            out[t.name] = digest
            if on_tensor is not None:
                on_tensor(t, digest)
        return out
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for t, digest in pool.map(work, tensors):
            out[t.name] = digest
            if on_tensor is not None:
                on_tensor(t, digest)
    return out


def fingerprint(
    model: Model,
    *,
    data: bool = True,
    per_tensor: bool = True,
    per_kv: bool = False,
    jobs: int = 1,
    on_tensor: Callable[[ReaderTensor, bytes], None] | None = None,
) -> Fingerprint:
    """Compute the ``ggufkit-fp/1`` fingerprint of ``model``.

    ``data=False`` skips reading tensor bytes, which makes the call instant on any model
    size but leaves ``weights`` and ``overall`` unset.
    """
    fields = model.metadata_fields()
    kv_digests = {f.name: canon.kv_digest(f) for f in fields}
    kv = canon.combine(kv_digests[name] for name in canon.sorted_by_name(kv_digests))

    ti_digests = {t.name: canon.tensor_info_digest(t) for t in model.tensors}
    tensor_info = canon.combine(ti_digests[name] for name in canon.sorted_by_name(ti_digests))

    data_digests: dict[str, bytes] | None = None
    weights: bytes | None = None
    overall: bytes | None = None
    if data:
        data_digests = _hash_tensor_data(model.tensors, model.endianess, jobs, on_tensor)
        weights = canon.combine(
            canon.weight_digest(name, data_digests[name])
            for name in canon.sorted_by_name(data_digests)
        )
        overall = canon.overall_digest(kv, tensor_info, weights)

    tensors: dict[str, TensorFingerprint] | None = None
    if per_tensor:
        tensors = {
            t.name: TensorFingerprint(
                name=t.name,
                type=t.tensor_type.name,
                shape=tuple(int(d) for d in t.shape),
                info=ti_digests[t.name].hex(),
                data=None if data_digests is None else data_digests[t.name].hex(),
            )
            for t in model.tensors
        }

    return Fingerprint(
        spec=canon.SPEC_ID,
        kv=kv.hex(),
        tensor_info=tensor_info.hex(),
        weights=None if weights is None else weights.hex(),
        overall=None if overall is None else overall.hex(),
        n_kv=len(fields),
        n_tensors=len(model.tensors),
        kvs={name: kv_digests[name].hex() for name in canon.sorted_by_name(kv_digests)}
        if per_kv
        else None,
        tensors=tensors,
    )


def format_text(fp: Fingerprint) -> list[str]:
    lines: list[str] = []
    if fp.kvs:
        lines.extend(f"KV       {digest} {name}" for name, digest in fp.kvs.items())
    if fp.tensors:
        for name in canon.sorted_by_name(fp.tensors):
            t = fp.tensors[name]
            shape = "x".join(str(d) for d in t.shape)
            lines.append(f"TENSOR   {t.data or t.info} {name} [{t.type} {shape}]")
    lines.append(f"{'kv':<12} {fp.kv}")
    lines.append(f"{'tensor_info':<12} {fp.tensor_info}")
    if fp.weights is not None:
        lines.append(f"{'weights':<12} {fp.weights}")
    if fp.overall is not None:
        lines.append(f"{'overall':<12} {fp.overall}")
    lines.append(f"{'spec':<12} {fp.spec}  ({fp.n_kv} metadata entries, {fp.n_tensors} tensors)")
    return lines


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser(
        "fingerprint",
        help="content hashes that ignore metadata order, tensor order and sharding",
        description=(
            "Compute the ggufkit-fp/1 fingerprint of a GGUF model: separate SHA-256 digests for "
            "the metadata, the tensor descriptors and the weights, plus one overall digest. "
            "Unlike sha256sum, the result does not change when metadata is reordered, when the "
            "file is split into shards, or when tensors are stored in a different order."
        ),
    )
    p.add_argument("model", type=Path, help="GGUF file, or any shard of a split model")
    p.add_argument("--json", action="store_true", help="print the fingerprint as JSON")
    p.add_argument("-o", "--output", type=Path, help="also write the JSON fingerprint to this file")
    p.add_argument("--force", action="store_true", help="overwrite --output if it exists")
    p.add_argument(
        "--no-data",
        action="store_true",
        help="skip tensor bytes (instant, but no weights/overall digest)",
    )
    p.add_argument("--no-per-tensor", action="store_true", help="omit per-tensor digests")
    p.add_argument("--per-kv", action="store_true", help="include a digest per metadata entry")
    p.add_argument(
        "-j",
        "--jobs",
        type=positive_int,
        default=default_jobs(),
        help="threads used to hash tensors (default: %(default)s)",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    if args.output is not None:
        prepare_output(args.output, force=args.force)
    with open_model(args.model) as model:
        fp = fingerprint(
            model,
            data=not args.no_data,
            per_tensor=not args.no_per_tensor,
            per_kv=args.per_kv,
            jobs=args.jobs,
        )
    if args.output is not None:
        write_text_atomic(args.output, json.dumps(fp.to_dict(), indent=2) + "\n")
    if args.json:
        json.dump(fp.to_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write("\n".join(format_text(fp)) + "\n")
    return 0
