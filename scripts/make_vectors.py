"""Write the ggufkit-fp/1 test vectors in tests/vectors/.

    uv run python scripts/make_vectors.py            # rewrite the files and vectors.json
    uv run python scripts/make_vectors.py --check    # fail if anything would change

Every file is written byte by byte by the small GGUF writer below, not by gguf-py, so the
vectors don't move when gguf-py's writer changes, and they can exercise things gguf-py
would not write (key order, alignment, big-endian, the exact sharding keys). The expected
digests in vectors.json are computed by ggufkit; tests/test_vectors.py also checks them
against an independent implementation written from docs/FINGERPRINT.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "vectors"

# GGUF value types.
UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64 = (
    range(13)
)
_SCALAR = {
    UINT8: "B",
    INT8: "b",
    UINT16: "H",
    INT16: "h",
    UINT32: "I",
    INT32: "i",
    FLOAT32: "f",
    UINT64: "Q",
    INT64: "q",
    FLOAT64: "d",
}

# GGML tensor types: id, values per block, bytes per block, struct code for element types.
TYPES: dict[str, tuple[int, int, int, str | None]] = {
    "F32": (0, 1, 4, "f"),
    "F16": (1, 1, 2, "e"),
    "Q8_0": (8, 32, 34, None),
    "Q4_K": (12, 256, 144, None),
    "Q6_K": (14, 256, 210, None),
    "I8": (24, 1, 1, "b"),
    "I16": (25, 1, 2, "h"),
    "I32": (26, 1, 4, "i"),
    "I64": (27, 1, 8, "q"),
    "F64": (28, 1, 8, "d"),
    "BF16": (30, 1, 2, None),
}


@dataclass(frozen=True)
class KV:
    key: str
    vtype: int
    value: Any
    item_type: int | None = None  # for arrays: the type of the items


@dataclass(frozen=True)
class Tensor:
    name: str
    type: str
    shape: tuple[int, ...]  # GGUF order, ne[0] first
    seed: str  # what the data is derived from, so two tensors can share or swap data

    def n_values(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


@dataclass(frozen=True)
class Model:
    kvs: tuple[KV, ...]
    tensors: tuple[Tensor, ...]
    alignment: int = 32
    big_endian: bool = False
    extra: tuple[str, ...] = field(default=())


def _stream(seed: str, n: int) -> bytes:
    """Deterministic pseudo-random bytes, so no RNG implementation can change the vectors."""
    out = bytearray()
    i = 0
    while len(out) < n:
        out += hashlib.sha256(f"{seed}/{i}".encode()).digest()
        i += 1
    return bytes(out[:n])


def tensor_bytes(t: Tensor, order: str) -> bytes:
    _, block, size, code = TYPES[t.type]
    n = t.n_values()
    if n % block:
        msg = f"{t.name}: {n} values is not a whole number of {t.type} blocks"
        raise ValueError(msg)
    if t.type == "BF16":
        # The top halves of float32 values, stored like any 16-bit integer.
        halves = [
            struct.unpack("<I", struct.pack("<f", _value(t.seed, i)))[0] >> 16 for i in range(n)
        ]
        return struct.pack(f"{order}{n}H", *halves)
    if code is None:
        return _stream(t.seed, n // block * size)
    if code in "fed":
        return struct.pack(f"{order}{n}{code}", *(_value(t.seed, i) for i in range(n)))
    bits = struct.calcsize(code) * 8
    raw = _stream(t.seed, n * 8)
    vals = [int.from_bytes(raw[8 * i : 8 * i + 8], "little") % (1 << bits) for i in range(n)]
    vals = [v - (1 << bits) if v >= 1 << (bits - 1) else v for v in vals]
    return struct.pack(f"{order}{n}{code}", *vals)


def _value(seed: str, i: int) -> float:
    """Return a float every format here stores exactly: a multiple of 1/16 in [-8, 8)."""
    b = hashlib.sha256(f"{seed}/{i}".encode()).digest()
    return (int.from_bytes(b[:2], "little") % 256 - 128) / 16


def _encode(vtype: int, value: Any, order: str, item_type: int | None = None) -> bytes:
    if vtype == STRING:
        raw = value.encode("utf-8") if isinstance(value, str) else bytes(value)
        return struct.pack(f"{order}Q", len(raw)) + raw
    if vtype == BOOL:
        return b"\x01" if value else b"\x00"
    if vtype in _SCALAR:
        if isinstance(value, bytes):  # an exact bit pattern, for -0.0 and NaN payloads
            return value if order == "<" else value[::-1]
        return struct.pack(f"{order}{_SCALAR[vtype]}", value)
    if vtype == ARRAY:
        if item_type is None:
            msg = "an array needs an item type"
            raise ValueError(msg)
        items, inner = value if item_type == ARRAY else (value, None)
        out = struct.pack(f"{order}IQ", item_type, len(items))
        return out + b"".join(_encode(item_type, v, order, inner) for v in items)
    msg = f"unknown value type {vtype}"
    raise ValueError(msg)


def gguf(model: Model) -> bytes:
    order = ">" if model.big_endian else "<"
    out = bytearray(b"GGUF")
    out += struct.pack(f"{order}IQQ", 3, len(model.tensors), len(model.kvs))
    for kv in model.kvs:
        out += _encode(STRING, kv.key, order) + struct.pack(f"{order}I", kv.vtype)
        out += _encode(kv.vtype, kv.value, order, kv.item_type)
    datas = [tensor_bytes(t, order) for t in model.tensors]
    offset = 0
    for t, data in zip(model.tensors, datas, strict=True):
        out += _encode(STRING, t.name, order)
        out += struct.pack(f"{order}I", len(t.shape))
        out += b"".join(struct.pack(f"{order}Q", d) for d in t.shape)
        out += struct.pack(f"{order}IQ", TYPES[t.type][0], offset)
        offset += -(-len(data) // model.alignment) * model.alignment
    out += b"\x00" * (-len(out) % model.alignment)
    for data in datas:
        out += data + b"\x00" * (-len(data) % model.alignment)
    return bytes(out)


# The vectors. BASE has every element type that ggufkit-fp/1 normalizes in big-endian files.

BASE = Model(
    kvs=(
        KV("general.architecture", STRING, "llama"),
        KV("general.name", STRING, "ggufkit test vector"),
        KV("llama.block_count", UINT32, 1),
    ),
    tensors=(
        Tensor("blk.0.ffn_gate.weight", "F32", (4, 4), "gate"),
        Tensor("blk.0.ffn_up.weight", "F32", (4, 4), "up"),
        Tensor("f16", "F16", (8,), "f16"),
        Tensor("f64", "F64", (2, 2), "f64"),
        Tensor("i8", "I8", (16,), "i8"),
        Tensor("i16", "I16", (8,), "i16"),
        Tensor("i32", "I32", (4,), "i32"),
        Tensor("i64", "I64", (2,), "i64"),
    ),
)

# Keys chosen so that byte-wise UTF-8 order differs from case-insensitive order and from
# UTF-16 code-unit order (U+FF5E sorts before U+1F600 in UTF-8, after it in UTF-16).
VALUE_TYPES = Model(
    kvs=(
        KV("general.architecture", STRING, "llama"),
        KV("t.uint8", UINT8, 200),
        KV("t.int8", INT8, -100),
        KV("t.uint16", UINT16, 60000),
        KV("t.int16", INT16, -30000),
        KV("t.uint32", UINT32, 4_000_000_000),
        KV("t.int32", INT32, -2_000_000_000),
        KV("t.uint64", UINT64, 18_000_000_000_000_000_000),
        KV("t.int64", INT64, -9_000_000_000_000_000_000),
        KV("t.float32", FLOAT32, 0.15625),
        KV("t.float32.negative_zero", FLOAT32, b"\x00\x00\x00\x80"),
        KV("t.float64.nan_payload", FLOAT64, b"\x01\x00\x00\x00\x00\x00\xf8\x7f"),
        KV("t.bool.true", BOOL, value=True),
        KV("t.bool.false", BOOL, value=False),
        KV("t.string.utf8", STRING, "h\u00e9llo \u2713"),
        KV("t.string.empty", STRING, ""),
        KV("t.array.uint32", ARRAY, [1, 2, 3], UINT32),
        KV("t.array.string", ARRAY, ["a", "", "\u00df"], STRING),
        KV("t.array.empty", ARRAY, [], INT16),
        KV("t.array.nested", ARRAY, ([[1, -1], [], [127]], INT8), ARRAY),
        KV("t.Z", UINT8, 1),
        KV("t.a", UINT8, 2),
        KV("t.\uff5e", UINT8, 3),
        KV("t.\U0001f600", UINT8, 4),
    ),
    tensors=(),
)

STORED_AS_IS = Model(
    kvs=(KV("general.architecture", STRING, "llama"),),
    tensors=(
        Tensor("bf16", "BF16", (8,), "bf16"),
        Tensor("q8_0", "Q8_0", (64,), "q8_0"),
        Tensor("q4_k", "Q4_K", (256,), "q4_k"),
        Tensor("q6_k", "Q6_K", (256,), "q6_k"),
    ),
)


def _rename(m: Model, old: str, new: str) -> Model:
    tensors = tuple(replace(t, name=new) if t.name == old else t for t in m.tensors)
    return replace(m, tensors=tensors)


def _with_kvs(m: Model, *kvs: KV) -> Model:
    return replace(m, kvs=m.kvs + kvs)


def _split(m: Model, n: int) -> list[Model]:
    """Shard like llama-gguf-split: all metadata in the first shard, split.* in every shard."""
    per = -(-len(m.tensors) // n)
    shards = []
    for i in range(n):
        split = (
            KV("split.no", UINT16, i),
            KV("split.count", UINT16, n),
            KV("split.tensors.count", INT32, len(m.tensors)),
        )
        kvs = (m.kvs if i == 0 else ()) + split
        shards.append(replace(m, kvs=kvs, tensors=m.tensors[i * per : (i + 1) * per]))
    return shards


@dataclass(frozen=True)
class Vector:
    name: str
    description: str
    models: list[Model]
    refused: bool = False

    def files(self) -> list[str]:
        if len(self.models) == 1:
            return [f"{self.name}.gguf"]
        n = len(self.models)
        return [f"{self.name}-{i + 1:05d}-of-{n:05d}.gguf" for i in range(n)]


def vectors() -> list[Vector]:
    gate, up = BASE.tensors[0], BASE.tensors[1]
    swapped = replace(
        BASE,
        tensors=(replace(gate, seed=up.seed), replace(up, seed=gate.seed), *BASE.tensors[2:]),
    )
    edited = replace(
        BASE,
        kvs=tuple(
            replace(kv, value="ggufkit test vector, renamed") if kv.key == "general.name" else kv
            for kv in BASE.kvs
        ),
    )
    return [
        Vector(
            "base", "Every element type, little-endian, alignment 32. The reference point.", [BASE]
        ),
        Vector(
            "reordered",
            "base with its metadata and tensors written in reverse order. Same fingerprint as base.",
            [replace(BASE, kvs=BASE.kvs[::-1], tensors=BASE.tensors[::-1])],
        ),
        Vector(
            "split",
            "base split into two shards the way llama-gguf-split does it (all metadata in the "
            "first shard, split.no, split.count and split.tensors.count in both). "
            "Same fingerprint as base.",
            _split(BASE, 2),
        ),
        Vector(
            "big-endian",
            "base written big-endian. Element types are hashed as little-endian values, so the "
            "fingerprint is the same as base.",
            [replace(BASE, big_endian=True)],
        ),
        Vector(
            "realigned",
            "base with general.alignment = 64, so the tensors sit at other offsets. kv differs "
            "(general.alignment is metadata); tensor_info and weights are the same as base.",
            [replace(_with_kvs(BASE, KV("general.alignment", UINT32, 64)), alignment=64)],
        ),
        Vector(
            "metadata-edit",
            "base with general.name changed. Only kv (and overall) differ from base.",
            [edited],
        ),
        Vector(
            "spare-key",
            "base plus a key named __SPARE0. ggufkit-fp/1 hashes every key except the header "
            "counts and split.*, so kv differs from base.",
            [_with_kvs(BASE, KV("__SPARE0", UINT32, 0))],
        ),
        Vector(
            "renamed",
            "base with blk.0.ffn_up.weight renamed to blk.0.ffn_down.weight. tensor_info and "
            "weights differ from base; the renamed tensor's data digest does not.",
            [_rename(BASE, "blk.0.ffn_up.weight", "blk.0.ffn_down.weight")],
        ),
        Vector(
            "swapped",
            "base with the data of blk.0.ffn_gate.weight and blk.0.ffn_up.weight swapped. Every "
            "name, shape, type and byte is still present; only weights differs from base.",
            [swapped],
        ),
        Vector(
            "value-types",
            "Every GGUF value type, including -0.0, a NaN payload, empty and nested arrays, and "
            "keys whose byte-wise UTF-8 order differs from case-insensitive and UTF-16 order. "
            "No tensors.",
            [VALUE_TYPES],
        ),
        Vector(
            "stored-as-is",
            "BF16 and quantized tensors (Q8_0, Q4_K, Q6_K) with arbitrary block bytes. These "
            "types are hashed exactly as stored.",
            [STORED_AS_IS],
        ),
        Vector(
            "big-endian-stored-as-is",
            "stored-as-is written big-endian. Their byte order can't be normalized without "
            "decoding, so ggufkit-fp/1 refuses the file instead of fingerprinting it.",
            [replace(STORED_AS_IS, big_endian=True)],
            refused=True,
        ),
    ]


def build(out: Path) -> dict[str, Any]:
    """Write every vector's files to ``out`` and return the contents of vectors.json."""
    from ggufkit.canon import UnsupportedModelError
    from ggufkit.fingerprint import fingerprint
    from ggufkit.model import open_model

    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for v in vectors():
        files = v.files()
        for name, model in zip(files, v.models, strict=True):
            (out / name).write_bytes(gguf(model))
        entry: dict[str, Any] = {
            "name": v.name,
            "description": v.description,
            "files": [
                {"name": n, "sha256": hashlib.sha256((out / n).read_bytes()).hexdigest()}
                for n in files
            ],
        }
        with open_model(out / files[0]) as model:
            try:
                fp = fingerprint(model, per_kv=True)
            except UnsupportedModelError:
                if not v.refused:
                    raise
                entry["expect"] = "refused"
            else:
                if v.refused:
                    msg = f"{v.name} should be refused"
                    raise AssertionError(msg)
                entry["expect"] = "fingerprint"
                entry["fingerprint"] = fp.to_dict()
        entries.append(entry)
    return {
        "spec": "ggufkit-fp/1",
        "about": "Test vectors for docs/FINGERPRINT.md. Generated by scripts/make_vectors.py.",
        "vectors": entries,
    }


def dump(doc: dict[str, Any]) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="fail if the vectors would change")
    args = parser.parse_args()
    if args.check:
        with tempfile.TemporaryDirectory() as tmp:
            fresh = Path(tmp)
            doc = build(fresh)
            stale = [
                f["name"]
                for v in doc["vectors"]
                for f in v["files"]
                if not (OUT / f["name"]).is_file()
                or (OUT / f["name"]).read_bytes() != (fresh / f["name"]).read_bytes()
            ]
            if not (OUT / "vectors.json").is_file() or (OUT / "vectors.json").read_text(
                encoding="utf-8"
            ) != dump(doc):
                stale.append("vectors.json")
        if stale:
            print("out of date: " + ", ".join(stale), file=sys.stderr)
            return 1
        print("tests/vectors is up to date")
        return 0
    doc = build(OUT)
    (OUT / "vectors.json").write_text(dump(doc), encoding="utf-8")
    n_files = sum(len(v["files"]) for v in doc["vectors"])
    print(f"wrote {len(doc['vectors'])} vectors ({n_files} files) to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
