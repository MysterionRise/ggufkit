"""Canonical byte encodings behind the ``ggufkit-fp/1`` fingerprint specification.

Everything here is deliberately boring: fixed-width little-endian integers, SHA-256, and no
dependence on where in the file a value happened to be stored. The full specification lives in
``docs/FINGERPRINT.md``; this module is its reference implementation.

Why not just hash the file? Because two GGUF files with identical weights routinely differ in
bytes: metadata gets reordered or edited by ``gguf-set-metadata``, a model gets split into
shards, alignment padding changes. A content hash that ignores all of that lets you answer
"is this the same model?" and "did only the metadata change?" without re-downloading anything.
"""

from __future__ import annotations

import struct
import sys
from hashlib import sha256
from typing import TYPE_CHECKING

import numpy as np
from gguf import GGUFEndian, GGUFValueType

from ggufkit.errors import GGUFKitError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gguf.gguf_reader import ReaderField, ReaderTensor

SPEC_ID = "ggufkit-fp/1"

#: Tensor data is hashed in slices of this many bytes so multi-gigabyte tensors never get copied.
CHUNK_SIZE = 64 << 20

_KV_TAG = b"kv\0"
_TI_TAG = b"ti\0"
_WT_TAG = b"wt\0"

_SCALAR_TYPES = frozenset(
    {
        GGUFValueType.UINT8,
        GGUFValueType.INT8,
        GGUFValueType.UINT16,
        GGUFValueType.INT16,
        GGUFValueType.UINT32,
        GGUFValueType.INT32,
        GGUFValueType.FLOAT32,
        GGUFValueType.UINT64,
        GGUFValueType.INT64,
        GGUFValueType.FLOAT64,
        GGUFValueType.BOOL,
    }
)


class UnsupportedModelError(GGUFKitError):
    """Raised for files whose canonical form ggufkit cannot compute yet."""


def _u32(value: int) -> bytes:
    return struct.pack("<I", value)


def _u64(value: int) -> bytes:
    return struct.pack("<Q", value)


def _little_endian_bytes(arr: np.ndarray) -> bytes:
    """Raw bytes of ``arr`` with multi-byte values in little-endian order."""
    if arr.dtype.byteorder == ">" or (arr.dtype.byteorder == "=" and sys.byteorder == "big"):
        arr = arr.astype(arr.dtype.newbyteorder("<"))
    return arr.tobytes()


def _encode_value(parts: list[np.ndarray], cursor: int, vtype: GGUFValueType) -> tuple[bytes, int]:
    """Encode the value starting at ``parts[cursor]`` and return it with the next cursor.

    The reader stores a field's parts in file order, so walking them with a cursor rebuilds
    the exact structure of the value, including nested arrays which the reader otherwise
    flattens.
    """
    if vtype == GGUFValueType.STRING:
        data = parts[cursor + 1]
        return _u64(int(data.nbytes)) + data.tobytes(), cursor + 2
    if vtype in _SCALAR_TYPES:
        arr = parts[cursor]
        if vtype == GGUFValueType.BOOL:
            return struct.pack("<B", 1 if bool(arr[0]) else 0), cursor + 1
        return _little_endian_bytes(arr), cursor + 1
    if vtype == GGUFValueType.ARRAY:
        item_type = GGUFValueType(int(parts[cursor][0]))
        count = int(parts[cursor + 1][0])
        cursor += 2
        out = bytearray(_u32(int(item_type)) + _u64(count))
        for _ in range(count):
            chunk, cursor = _encode_value(parts, cursor, item_type)
            out += chunk
        return bytes(out), cursor
    msg = f"Unknown GGUF value type {vtype!r}"
    raise UnsupportedModelError(msg)


def encode_kv(field: ReaderField) -> bytes:
    """Canonical encoding of one metadata entry: tag, name, type, value."""
    if len(field.parts) < 3 or not field.types:
        msg = f"Field {field.name!r} is not a metadata entry"
        raise ValueError(msg)
    name = field.name.encode("utf-8")
    vtype = field.types[0]
    payload, end = _encode_value(field.parts, 3, vtype)
    if end != len(field.parts):
        msg = f"Field {field.name!r}: consumed {end} of {len(field.parts)} parts"
        raise UnsupportedModelError(msg)
    return _KV_TAG + _u64(len(name)) + name + _u32(int(vtype)) + payload


def encode_tensor_info(tensor: ReaderTensor) -> bytes:
    """Canonical encoding of a tensor's descriptor: tag, name, dims in GGUF order, type.

    The file offset is left out on purpose: it changes whenever anything before the tensor
    changes, and it says nothing about the tensor.
    """
    name = tensor.name.encode("utf-8")
    dims = [int(d) for d in tensor.shape]
    return (
        _TI_TAG
        + _u64(len(name))
        + name
        + _u32(len(dims))
        + b"".join(_u64(d) for d in dims)
        + _u32(int(tensor.tensor_type))
    )


def kv_digest(field: ReaderField) -> bytes:
    return sha256(encode_kv(field)).digest()


def tensor_info_digest(tensor: ReaderTensor) -> bytes:
    return sha256(encode_tensor_info(tensor)).digest()


def is_element_type(tensor: ReaderTensor) -> bool:
    """Return whether the reader exposes the tensor with a real dtype (F32, F16, F64, I8..I64)."""
    return tensor.data.dtype != np.uint8


def tensor_data_digest(
    tensor: ReaderTensor,
    file_endianess: GGUFEndian,
    chunk_size: int = CHUNK_SIZE,
) -> bytes:
    """SHA-256 of the tensor's stored bytes, normalized to little-endian for element types.

    Quantized blocks mix scales and packed values, so their byte order cannot be normalized
    without decoding the block layout. Big-endian quantized files are refused rather than
    silently fingerprinted differently; convert them with llama.cpp's ``gguf-convert-endian``.
    """
    data = tensor.data
    if is_element_type(tensor):
        if data.dtype.byteorder == ">" or (data.dtype.byteorder == "=" and sys.byteorder == "big"):
            data = data.astype(data.dtype.newbyteorder("<"))
    elif file_endianess == GGUFEndian.BIG:
        msg = (
            f"Tensor {tensor.name!r} is {tensor.tensor_type.name} in a big-endian file; "
            "convert the file to little-endian first (gguf-convert-endian)"
        )
        raise UnsupportedModelError(msg)

    flat = np.ascontiguousarray(data).reshape(-1).view(np.uint8)
    digest = sha256()
    for start in range(0, flat.size, chunk_size):
        digest.update(flat[start : start + chunk_size])
    return digest.digest()


def encode_weight(name: str, data_digest: bytes) -> bytes:
    """Canonical encoding of one tensor's data under its name: tag, name, data digest.

    Binding the name to the bytes is what makes swapping the data of two tensors with the
    same shape and type show up in ``weights``.
    """
    raw = name.encode("utf-8")
    return _WT_TAG + _u64(len(raw)) + raw + data_digest


def weight_digest(name: str, data_digest: bytes) -> bytes:
    return sha256(encode_weight(name, data_digest)).digest()


def combine(digests: Iterable[bytes]) -> bytes:
    """Hash a sequence of digests in the order given."""
    return sha256(b"".join(digests)).digest()


def overall_digest(kv: bytes, tensor_info: bytes, weights: bytes) -> bytes:
    return sha256(SPEC_ID.encode("ascii") + b"\0" + kv + tensor_info + weights).digest()


def sorted_by_name(names: Iterable[str]) -> list[str]:
    """Names in byte-wise UTF-8 order, which is what the spec requires."""
    return sorted(names, key=lambda s: s.encode("utf-8"))
