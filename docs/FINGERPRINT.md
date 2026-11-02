# ggufkit-fp/1: content fingerprints for GGUF models

Status: stable since ggufkit 0.1.0. Implemented by `ggufkit.canon` and
`ggufkit.fingerprint`.

The spec id `ggufkit-fp/1` is part of the hash input. A given file's `ggufkit-fp/1` digests
will not change in any later ggufkit release; a change to what is hashed or how gets a new
id (`ggufkit-fp/2`), and ggufkit keeps computing and verifying the old one.

## Why a file hash is not enough

Two GGUF files that hold the same model routinely differ byte for byte:

- metadata was added, edited or reordered (`gguf-set-metadata`, `gguf-new-metadata`, a Hub
  re-upload with a fixed chat template);
- the model was split into shards with `llama-gguf-split`, or merged back;
- tensors were written in a different order, or the alignment changed;
- the file was converted between little- and big-endian.

A signature over the raw bytes (what OpenSSF model-signing and `sha256sum` give you) breaks
on every one of those, and when it breaks it cannot tell you whether the weights changed. A
fingerprint can.

## What is hashed

All hashes are SHA-256. All integers are fixed-width little-endian. `‖` is concatenation,
`u32(n)` and `u64(n)` are 4- and 8-byte encodings, `len(x)` is a byte length.

### Metadata entry: `E_kv`

For every metadata key/value pair:

```
E_kv = sha256( "kv\0" ‖ u64(len(name)) ‖ name ‖ u32(vtype) ‖ value )
```

`name` is the key in UTF-8 and `vtype` is the GGUF value type id. `value` is encoded by type:

| type | encoding |
|---|---|
| UINT8, INT8, UINT16, INT16, UINT32, INT32, UINT64, INT64 | the integer, fixed width, little-endian |
| FLOAT32, FLOAT64 | the IEEE 754 bit pattern, little-endian (so `-0.0` and NaN payloads are preserved) |
| BOOL | one byte, `0x00` or `0x01` |
| STRING | `u64(len(s)) ‖ s` with `s` in UTF-8 |
| ARRAY | `u32(item_type) ‖ u64(count) ‖ item_1 ‖ … ‖ item_count`, items encoded with this table (nested arrays recurse) |

Excluded from `kv` entirely:

- the header counts (`GGUF.version`, `GGUF.tensor_count`, `GGUF.kv_count`), which describe
  the file, not the model;
- `split.no`, `split.count` and `split.tensors.count`, which describe sharding.

Everything else is included, `general.uuid`, `general.alignment` and any key whose name
starts with `__SPARE` among them: they are claims made by whoever wrote the file, and a
verifier should see when they change.

### `kv`

```
kv = sha256( E_kv(name_1) ‖ E_kv(name_2) ‖ … )   for names in byte-wise UTF-8 order
```

### Tensor descriptor: `E_ti`

For every tensor:

```
E_ti = sha256( "ti\0" ‖ u64(len(name)) ‖ name ‖ u32(n_dims) ‖ u64(ne_0) ‖ … ‖ u64(ne_{n-1}) ‖ u32(ggml_type) )
```

Dimensions are in GGUF order (`ne[0]` varies fastest), exactly as stored. The file offset
is not included: it changes whenever anything before the tensor changes and says nothing
about the tensor.

### `tensor_info`

```
tensor_info = sha256( E_ti(name_1) ‖ E_ti(name_2) ‖ … )   for names in byte-wise UTF-8 order
```

### Tensor data: `D_t`

```
D_t = sha256( stored bytes of the tensor )
```

For element types (F32, F16, F64, I8, I16, I32, I64) in a big-endian file, values are
byte-swapped to little-endian before hashing, so the fingerprint does not depend on byte
order. Every other type, BF16 included, is hashed exactly as stored. Quantized types mix
scales and packed values inside each block, so their byte order cannot be normalized without
decoding the block; BF16 and quantized tensors in big-endian files are therefore refused in
this version (convert with llama.cpp's `gguf-convert-endian` first).

Alignment padding between tensors is never hashed.

### Tensor data under its name: `E_wt`

```
E_wt = sha256( "wt\0" ‖ u64(len(name)) ‖ name ‖ D_t )
```

### `weights`

```
weights = sha256( E_wt(name_1) ‖ E_wt(name_2) ‖ … )   for names in byte-wise UTF-8 order
```

Each tensor's bytes are bound to its name. Swapping the data of two tensors with the same
shape and type (two layers, or `ffn_gate` and `ffn_up` of one layer) changes `weights`, even
though every name, shape, type and byte is still present. Reordering tensors in the file,
splitting it into shards or changing the alignment does not.

A rename changes `weights` and `tensor_info`, but not the renamed tensor's `D_t`. That is how
`ggufkit verify` tells a rename (the same bytes under a new name) from new weights.

### `overall`

```
overall = sha256( "ggufkit-fp/1\0" ‖ kv ‖ tensor_info ‖ weights )
```

## Sharded models

A model split with `llama-gguf-split` is fingerprinted as one logical model: the metadata
of all shards is merged (first shard wins on duplicate keys; the three sharding keys listed
above are dropped), and the tensors of all shards are concatenated. The split and the merged
file therefore have the same fingerprint.

## Reading a comparison

| `kv` | `tensor_info` | `weights` | meaning |
|---|---|---|---|
| same | same | same | the same model (`overall` is equal too) |
| differs | same | same | only metadata changed: names, templates, licence text, provenance keys |
| same | differs | same | tensors were retyped or reshaped in place, bytes untouched |
| any | differs | differs | tensors were renamed, or the weights changed; the per-tensor `data` digests tell which |
| any | same | differs | the weights changed: new bytes, or bytes moved between tensors |

`ggufkit verify` reports a rename as `tensor-info-only` only when the per-tensor digests show
that every tensor's bytes are unchanged and only their names moved; anything else is
`weights-changed`.

## What a fingerprint does not prove

A fingerprint proves integrity and equivalence, nothing more. It does not prove that a
quantized file was faithfully derived from the full-precision model it names: that needs
the build inputs (source hash, importance matrix, llama.cpp revision, quantize flags) and
either a reproduction or a numeric comparison. ggufkit's manifest and `diff --numeric`
commands exist for that; a fingerprint is the thing they agree on.

## JSON form

```json
{
  "spec": "ggufkit-fp/1",
  "kv": "<hex>",
  "tensor_info": "<hex>",
  "weights": "<hex or null when --no-data>",
  "overall": "<hex or null when --no-data>",
  "n_kv": 23,
  "n_tensors": 291,
  "kvs": { "general.architecture": "<hex>", "...": "..." },
  "tensors": {
    "token_embd.weight": { "type": "Q4_K", "shape": [4096, 32000], "info": "<hex>", "data": "<hex>" }
  }
}
```

`kvs` is present only with `--per-kv`; `tensors` is absent with `--no-per-tensor`.

## Test vectors

[`tests/vectors/`](https://github.com/MysterionRise/ggufkit/tree/main/tests/vectors) holds
13 tiny GGUF files (10 KiB in all) and `vectors.json`, which gives each file's sha256 and
its expected fingerprint in the JSON form above, with `kvs` and `tensors`, so a mismatch can
be traced to one key or one tensor. They cover:

| vector | what it checks |
|---|---|
| `base` | every element type, little-endian, alignment 32: the reference point |
| `reordered`, `split-*`, `big-endian` | key and tensor order, sharding and byte order don't change anything |
| `realigned`, `metadata-edit`, `spare-key` | only `kv` changes (`general.alignment` and `__SPARE…` keys are metadata) |
| `renamed` | `tensor_info` and `weights` change, the renamed tensor's data digest doesn't |
| `swapped` | two tensors trade data: only `weights` changes |
| `value-types` | every metadata value type, `-0.0`, a NaN payload, empty and nested arrays, and keys whose UTF-8 byte order differs from case-insensitive and UTF-16 order |
| `stored-as-is` | BF16, Q8_0, Q4_K and Q6_K tensors, hashed as stored |
| `big-endian-stored-as-is` | must be refused |

The files are written byte by byte by `scripts/make_vectors.py`, not by gguf-py. CI checks
that ggufkit computes every recorded digest, that an independent implementation of this
document (`tests/fp_reference.py`, under 200 lines of Python with no GGUF library) computes
them too, and that the generator still writes exactly the committed files.

## Changes to this document

- ggufkit 0.2.0: corrected two statements that did not match what `ggufkit-fp/1` has computed
  since 0.1.0. Keys starting with `__SPARE` are hashed like any other key (the text said they
  were excluded), and BF16 is hashed as stored and refused in big-endian files (the text did
  not say). No digest changes.
