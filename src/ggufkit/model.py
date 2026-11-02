"""Open a GGUF model as one logical object, whether it is a single file or a set of shards.

llama.cpp's ``gguf-split`` writes ``name-00001-of-00003.gguf`` style shards. The first shard
carries the model metadata, every shard carries ``split.*`` bookkeeping keys, and tensors are
spread across shards in order. :func:`open_model` hides that: it finds the sibling shards,
merges their metadata (dropping the ``split.*`` keys) and concatenates their tensors, so the
rest of ggufkit can treat split and unsplit models identically.
"""

from __future__ import annotations

import gc
import mmap
import re
import struct
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from gguf import GGUFEndian, GGUFReader
from gguf.constants import Keys

from ggufkit.errors import GGUFKitError

if TYPE_CHECKING:
    from collections.abc import Iterator

    from gguf.gguf_reader import ReaderField, ReaderTensor

SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<no>\d{5})-of-(?P<count>\d{5})\.gguf$", re.IGNORECASE)

#: Bookkeeping entries the reader synthesizes from the header. They are not metadata.
PSEUDO_FIELDS = frozenset({"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"})

#: Keys that describe how a model was split into files, not the model itself.
SPLIT_FIELDS = frozenset(
    {
        Keys.Split.LLM_KV_SPLIT_NO,
        Keys.Split.LLM_KV_SPLIT_COUNT,
        Keys.Split.LLM_KV_SPLIT_TENSORS_COUNT,
    }
)


class ModelError(GGUFKitError):
    """Raised when a model cannot be opened as a consistent whole."""


#: Byte sizes of the fixed-size GGUF metadata value types, by type id.
_SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_STRING, _ARRAY = 8, 9
_MAX_DIMS = 4
_MAX_ARRAY_DEPTH = 8


class _HeaderCheck:
    """Walk a GGUF header and reject sizes that cannot fit in the file.

    gguf-py's reader trusts the counts and lengths it reads: an array that claims 2**40 items
    in a 49-byte file makes it loop and allocate until memory runs out. This walk reads every
    declared length and count first and refuses the file as soon as one of them exceeds the
    bytes that are left, so hostile files fail fast instead of hanging.
    """

    def __init__(self, buf: mmap.mmap, path: Path) -> None:
        self.buf = buf
        self.path = path
        self.pos = 0
        self.end = len(buf)
        self.order = "<"

    def fail(self, why: str) -> None:
        msg = f"{self.path}: not a valid GGUF file: {why}"
        raise ModelError(msg)

    def take(self, n: int) -> int:
        if n < 0 or n > self.end - self.pos:
            self.fail(f"truncated at byte {self.pos}")
        start = self.pos
        self.pos += n
        return start

    def u32(self) -> int:
        return int(struct.unpack_from(self.order + "I", self.buf, self.take(4))[0])

    def u64(self) -> int:
        return int(struct.unpack_from(self.order + "Q", self.buf, self.take(8))[0])

    def string(self) -> None:
        at = self.pos
        n = self.u64()
        if n > self.end - self.pos:
            self.fail(f"the string at byte {at} claims {n} bytes")
        self.pos += n

    def value(self, vtype: int, depth: int = 0) -> None:
        if vtype in _SCALAR_SIZES:
            self.take(_SCALAR_SIZES[vtype])
        elif vtype == _STRING:
            self.string()
        elif vtype == _ARRAY:
            at = self.pos
            item = self.u32()
            count = self.u64()
            if item in _SCALAR_SIZES:
                if count * _SCALAR_SIZES[item] > self.end - self.pos:
                    self.fail(f"the array at byte {at} claims {count} items")
                self.pos += count * _SCALAR_SIZES[item]
                return
            if item not in (_STRING, _ARRAY):
                self.fail(f"the array at byte {at} has unknown item type {item}")
            if depth >= _MAX_ARRAY_DEPTH:
                self.fail(f"arrays nested more than {_MAX_ARRAY_DEPTH} deep at byte {at}")
            smallest = 8 if item == _STRING else 12
            if count * smallest > self.end - self.pos:
                self.fail(f"the array at byte {at} claims {count} items")
            for _ in range(count):
                self.value(item, depth + 1)
        else:
            self.fail(f"unknown value type {vtype} at byte {self.pos - 4}")

    def run(self) -> None:
        if self.end < 24 or self.buf[:4] != b"GGUF":
            self.fail("no GGUF header")
        version = int(struct.unpack_from("<I", self.buf, 4)[0])
        if version & 0xFFFF == 0:  # written big-endian
            self.order = ">"
            version = int(struct.unpack_from(">I", self.buf, 4)[0])
        if version not in (2, 3):
            self.fail(f"unsupported GGUF version {version}")
        self.pos = 8
        n_tensors = self.u64()
        n_kv = self.u64()
        left = self.end - self.pos
        if n_kv * 13 > left or n_tensors * 24 > left:
            self.fail(f"it declares {n_kv} metadata entries and {n_tensors} tensors")
        for _ in range(n_kv):
            self.string()
            self.value(self.u32())
        for _ in range(n_tensors):
            self.string()
            n_dims = self.u32()
            if n_dims > _MAX_DIMS:
                self.fail(f"a tensor declares {n_dims} dimensions")
            self.take(8 * n_dims + 4 + 8)


def check_header(path: Path) -> None:
    """Raise :class:`ModelError` unless ``path`` has a GGUF header whose sizes fit the file."""
    size = path.stat().st_size
    if size < 24:
        msg = f"{path}: not a valid GGUF file: only {size} bytes"
        raise ModelError(msg)
    with path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as buf:
        _HeaderCheck(buf, path).run()


def read_gguf(path: Path) -> GGUFReader:
    """Open one GGUF file, turning anything the reader chokes on into :class:`ModelError`."""
    check_header(path)
    try:
        return GGUFReader(path, "r")
    except (ValueError, IndexError, KeyError, OverflowError, TypeError, struct.error) as e:
        msg = f"{path}: not a valid GGUF file: {e}"
        raise ModelError(msg) from e


def shard_paths(path: str | Path) -> list[Path]:
    """Return every file of the model ``path`` belongs to, in shard order.

    A file whose name does not follow the ``-NNNNN-of-MMMMM.gguf`` convention is returned
    on its own. Missing sibling shards are an error rather than a silent partial model.
    """
    path = Path(path)
    match = SHARD_RE.match(path.name)
    if match is None:
        return [path]
    count = int(match["count"])
    number = int(match["no"])
    if count == 0 or not 1 <= number <= count:
        msg = f"{path.name}: shard {number} of {count} is not a valid shard number"
        raise ModelError(msg)
    stem = match["stem"]
    paths = [path.with_name(f"{stem}-{i:05d}-of-{count:05d}.gguf") for i in range(1, count + 1)]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        msg = f"{path.name} is shard {int(match['no'])} of {count} but these shards are missing: {', '.join(missing)}"
        raise ModelError(msg)
    return paths


@dataclass
class Model:
    """A GGUF model, possibly spread over several shard files, seen as one object."""

    paths: list[Path]
    readers: list[GGUFReader] = field(repr=False)
    fields: OrderedDict[str, ReaderField] = field(repr=False)
    tensors: list[ReaderTensor] = field(repr=False)

    @property
    def path(self) -> Path:
        """The first (or only) file of the model."""
        return self.paths[0]

    @property
    def n_shards(self) -> int:
        return len(self.paths)

    @property
    def version(self) -> int:
        return int(self.readers[0].fields["GGUF.version"].parts[-1][0])

    @property
    def endianess(self) -> GGUFEndian:
        return self.readers[0].endianess

    @property
    def alignment(self) -> int:
        return int(self.readers[0].alignment)

    @property
    def architecture(self) -> str | None:
        f = self.fields.get(Keys.General.ARCHITECTURE)
        return None if f is None else str(f.contents())

    def metadata_fields(self) -> list[ReaderField]:
        """Model metadata only: no reader pseudo-fields, no ``split.*`` bookkeeping."""
        return [
            f
            for name, f in self.fields.items()
            if name not in PSEUDO_FIELDS and name not in SPLIT_FIELDS and f.types
        ]

    def tensor(self, name: str) -> ReaderTensor:
        for t in self.tensors:
            if t.name == name:
                return t
        msg = f"No tensor named {name!r}"
        raise KeyError(msg)

    def tensor_names(self) -> Iterator[str]:
        return (t.name for t in self.tensors)

    def close(self) -> None:
        """Drop references to the memory maps so the files can be released."""
        self.tensors = []
        self.fields = OrderedDict()
        self.readers = []
        gc.collect()

    def __enter__(self) -> Model:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def open_model(path: str | Path) -> Model:
    """Open ``path`` and any sibling shards as a single :class:`Model`."""
    paths = shard_paths(path)
    readers: list[GGUFReader] = []
    try:
        readers.extend(read_gguf(p) for p in paths)
        return _merge(paths, readers)
    except BaseException:
        # Drop the memory maps now, so a failed open does not keep the files locked (Windows).
        readers.clear()
        gc.collect()
        raise


def _merge(paths: list[Path], readers: list[GGUFReader]) -> Model:
    endians = {r.endianess for r in readers}
    if len(endians) > 1:
        msg = "Shards disagree on byte order"
        raise ModelError(msg)

    declared = readers[0].fields.get(Keys.Split.LLM_KV_SPLIT_COUNT)
    if declared is not None:
        count = int(declared.contents())
        if count != len(paths):
            msg = (
                f"{paths[0].name} declares {count} shards but {len(paths)} were found; "
                "shards must sit next to each other and follow the -NNNNN-of-MMMMM.gguf naming"
            )
            raise ModelError(msg)

    fields: OrderedDict[str, ReaderField] = OrderedDict()
    for reader in readers:
        for name, f in reader.fields.items():
            fields.setdefault(name, f)

    tensors = [t for reader in readers for t in reader.tensors]
    seen: set[str] = set()
    for t in tensors:
        if t.name in seen:
            msg = f"Tensor {t.name!r} appears in more than one shard"
            raise ModelError(msg)
        seen.add(t.name)

    return Model(paths=paths, readers=readers, fields=fields, tensors=tensors)
