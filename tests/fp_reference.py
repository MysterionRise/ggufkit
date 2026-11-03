"""An independent implementation of ggufkit-fp/1, written from docs/FINGERPRINT.md alone.

It parses GGUF bytes with ``struct`` and shares no code with ggufkit or gguf-py, so when it
agrees with ggufkit on the test vectors, the spec text and the implementation agree. It
supports what the vectors need, not every GGML type; it is a check, not a second library.
"""

from __future__ import annotations

import struct
from hashlib import sha256
from pathlib import Path

SPEC_ID = b"ggufkit-fp/1"

# GGUF value types and the struct codes of the fixed-width ones.
STRING, ARRAY, BOOL = 8, 9, 7
FIXED = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 10: "Q", 11: "q", 12: "d"}

# GGML types the vectors use: id -> (values per block, bytes per block, element size or None).
# Element types are byte-swapped to little-endian in big-endian files; the others are not.
GGML = {
    0: (1, 4, 4),  # F32
    1: (1, 2, 2),  # F16
    24: (1, 1, 1),  # I8
    25: (1, 2, 2),  # I16
    26: (1, 4, 4),  # I32
    27: (1, 8, 8),  # I64
    28: (1, 8, 8),  # F64
    30: (1, 2, None),  # BF16: hashed as stored
    8: (32, 34, None),  # Q8_0
    12: (256, 144, None),  # Q4_K
    14: (256, 210, None),  # Q6_K
}

SHARDING_KEYS = {"split.no", "split.count", "split.tensors.count"}


class Refused(Exception):  # noqa: N818 - the spec's word
    pass


def u32(n: int) -> bytes:
    return struct.pack("<I", n)


def u64(n: int) -> bytes:
    return struct.pack("<Q", n)


class _Reader:
    def __init__(self, data: bytes) -> None:
        if data[:4] != b"GGUF":
            msg = "not a GGUF file"
            raise ValueError(msg)
        self.data = data
        self.pos = 4
        version = struct.unpack_from("<I", data, 4)[0]
        self.order = "<" if version in (2, 3) else ">"

    def take(self, code: str) -> int | float:
        value = struct.unpack_from(self.order + code, self.data, self.pos)[0]
        self.pos += struct.calcsize(code)
        return value

    def raw(self, n: int) -> bytes:
        out = self.data[self.pos : self.pos + n]
        self.pos += n
        return out

    def string(self) -> bytes:
        return self.raw(int(self.take("Q")))

    def value(self, vtype: int) -> tuple[bytes, object]:
        """The value's canonical encoding, and the value itself (for general.alignment)."""
        if vtype == STRING:
            s = self.string()
            return u64(len(s)) + s, s
        if vtype == BOOL:
            b = self.raw(1)[0]
            return bytes([1 if b else 0]), bool(b)
        if vtype in FIXED:
            code = FIXED[vtype]
            stored = self.raw(struct.calcsize(code))
            little = stored if self.order == "<" else stored[::-1]
            return little, struct.unpack("<" + code, little)[0]
        if vtype == ARRAY:
            item_type, count = int(self.take("I")), int(self.take("Q"))
            out = bytearray(u32(item_type) + u64(count))
            for _ in range(count):
                out += self.value(item_type)[0]
            return bytes(out), None
        msg = f"unknown value type {vtype}"
        raise ValueError(msg)


def _parse(path: Path) -> tuple[dict[str, bytes], list[tuple[str, list[int], int, bytes]], str]:
    """(canonical kv values by key, tensors as (name, dims, ggml type, stored bytes), order)."""
    r = _Reader(path.read_bytes())
    r.take("I")  # version
    n_tensors, n_kv = int(r.take("Q")), int(r.take("Q"))
    kvs: dict[str, bytes] = {}
    alignment = 32
    for _ in range(n_kv):
        key = r.string().decode("utf-8")
        vtype = int(r.take("I"))
        encoded, value = r.value(vtype)
        kvs[key] = u32(vtype) + encoded
        if key == "general.alignment":
            alignment = int(value)  # type: ignore[call-overload]
    infos = []
    for _ in range(n_tensors):
        name = r.string().decode("utf-8")
        dims = [int(r.take("Q")) for _ in range(int(r.take("I")))]
        infos.append((name, dims, int(r.take("I")), int(r.take("Q"))))
    data_start = -(-r.pos // alignment) * alignment
    tensors = []
    for name, dims, gtype, offset in infos:
        block, size, _ = GGML[gtype]
        n = 1
        for d in dims:
            n *= d
        start = data_start + offset
        tensors.append((name, dims, gtype, r.data[start : start + n // block * size]))
    return kvs, tensors, r.order


def _data_digest(stored: bytes, gtype: int, order: str) -> bytes:
    element = GGML[gtype][2]
    if order == ">":
        if element is None:
            msg = f"GGML type {gtype} in a big-endian file"
            raise Refused(msg)
        stored = b"".join(stored[i : i + element][::-1] for i in range(0, len(stored), element))
    return sha256(stored).digest()


def fingerprint(paths: list[Path]) -> dict[str, str]:
    """kv, tensor_info, weights and overall of the model in ``paths`` (shards in order)."""
    kvs: dict[str, bytes] = {}
    tensors = []
    for path in paths:
        shard_kvs, shard_tensors, order = _parse(path)
        for key, value in shard_kvs.items():
            kvs.setdefault(key, value)  # first shard wins
        tensors += [
            (name, dims, gtype, _data_digest(b, gtype, order))
            for name, dims, gtype, b in shard_tensors
        ]

    def by_name(names: list[str]) -> list[str]:
        return sorted(names, key=lambda s: s.encode("utf-8"))

    e_kv = {
        k: sha256(b"kv\0" + u64(len(k.encode())) + k.encode() + v).digest()
        for k, v in kvs.items()
        if k not in SHARDING_KEYS
    }
    e_ti = {
        name: sha256(
            b"ti\0"
            + u64(len(name.encode()))
            + name.encode()
            + u32(len(dims))
            + b"".join(u64(d) for d in dims)
            + u32(gtype)
        ).digest()
        for name, dims, gtype, _ in tensors
    }
    e_wt = {
        name: sha256(b"wt\0" + u64(len(name.encode())) + name.encode() + digest).digest()
        for name, _, _, digest in tensors
    }
    kv = sha256(b"".join(e_kv[k] for k in by_name(list(e_kv)))).digest()
    tensor_info = sha256(b"".join(e_ti[n] for n in by_name(list(e_ti)))).digest()
    weights = sha256(b"".join(e_wt[n] for n in by_name(list(e_wt)))).digest()
    overall = sha256(SPEC_ID + b"\0" + kv + tensor_info + weights).digest()
    return {
        "kv": kv.hex(),
        "tensor_info": tensor_info.hex(),
        "weights": weights.hex(),
        "overall": overall.hex(),
    }
