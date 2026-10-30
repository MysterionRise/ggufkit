# ggufkit-manifest/1: provenance records for GGUF models

Status: stable since ggufkit 0.1.0, implemented by `ggufkit.manifest`. Readers must check
`schema` and refuse versions they do not know. Within `ggufkit-manifest/1`, later releases
only add optional fields; removing or renaming a field, or changing its meaning, needs a new
schema id.

## Why

GGUF's standard metadata has no key for the source model's checksum, the llama.cpp version or
the quantize command, so a file cannot say how to reproduce it. A manifest is the record that
is missing: what the file *is* (its fingerprint), what its metadata *claims* (origin, licence,
who quantized it), what the producer *declares* (build inputs), and an honest list of what is
absent.

```sh
ggufkit manifest model-Q4_K_M.gguf \
  --source model-BF16.gguf \
  --imatrix imatrix.dat \
  --input llama_cpp_commit=b8146 \
  --input "quantize_cmd=llama-quantize --imatrix imatrix.dat model-BF16.gguf model-Q4_K_M.gguf Q4_K_M" \
  --file-sha256
```

This writes `model-Q4_K_M.manifest.json` next to the model. `--source` and `--imatrix` hash
those files into `source_sha256` and `imatrix_sha256`. Publish the manifest with the model.
Anyone can then check a download with
`ggufkit verify model-Q4_K_M.gguf --manifest model-Q4_K_M.manifest.json`, and rebuild the
file from its inputs with
`ggufkit reproduce --manifest model-Q4_K_M.manifest.json --source model-BF16.gguf --imatrix imatrix.dat`
(see [Build inputs](#build-inputs)).

Other options:

- `-o FILE` writes the manifest elsewhere; `-o -` prints it instead of writing a file.
- `--note TEXT` stores a free-text note, for example where the source came from.
- `--no-per-tensor` leaves out the per-tensor digests, which makes the manifest much smaller.
  `verify` can then still tell that the weights changed, but not which tensor.
- `--no-data` leaves out the weight digests entirely (instant on any size). `verify` against
  such a manifest cannot compare weights, and fails unless it is run with `--no-data` too.
- `--force` overwrites an existing manifest. Output is written to a temporary file and moved
  into place, so an interrupted run never leaves a half-written manifest.

## Fields

| field | content |
|---|---|
| `schema` | `"ggufkit-manifest/1"` |
| `created_at` | UTC timestamp, ISO 8601 |
| `tool` | `name`, `version`, `gguf_py` (version of llama.cpp's Python package), `python`, `fingerprint_spec` |
| `files[]` | one entry per shard: `path` (file name), `size`, `sha256` (raw bytes, `null` unless `--file-sha256`), `shard` (`{no, count}` or `null`) |
| `gguf` | `version`, `endianness`, `alignment`, `n_kv`, `n_tensors`, `n_shards` |
| `model` | `architecture`, `file_type` (e.g. `Q4_K_M`, as claimed by `general.file_type`), `file_type_id`, `quantization_version`, `parameter_count`, `tensor_bytes`, `tensor_type_counts`, `tensor_type_bytes` |
| `provenance` | what the metadata says; only keys that are present: `uuid`, `name`, `author`, `version`, `organization`, `basename`, `finetune`, `size_label`, `description`, `quantized_by`, `url`, `doi`, `repo_url`, `tags`, `languages`, `license {id, name, link}`, `source {url, doi, uuid, repo_url}`, `base_models[]`, `datasets[]`, `imatrix {file, dataset, entries_count, chunks_count}` (from llama.cpp's `quantize.imatrix.*` keys; `datasets`, `chunk_count`, `chunk_size` when a file carries `imatrix.*` keys instead) |
| `provenance_gaps` | ids from the table below that this file does not satisfy |
| `provenance_score` | `{recorded, expected}` |
| `build_inputs` | the `--input KEY=VALUE` pairs, verbatim |
| `notes` | free text from `--note`, or `null` |
| `fingerprint` | the full `ggufkit-fp/1` fingerprint with per-key and per-tensor digests (see [FINGERPRINT.md](FINGERPRINT.md)) |

## What a verifiable file records

| id | satisfied when |
|---|---|
| `general.source.*` | any of `general.source.url/doi/uuid/repo_url`, or at least one `general.base_model.*` entry |
| `general.quantized_by` | present and non-empty |
| `general.license` | any of `general.license`, `general.license.name`, `general.license.link` |
| `general.uuid` | present |
| `quantize.imatrix.dataset` | present (llama.cpp's quantize tool writes it when an importance matrix is used; `imatrix.datasets` also counts); only expected when some tensor is stored below 8 bits |
| `build_inputs.source_sha256` | supplied with `--input` |
| `build_inputs.imatrix_sha256` | supplied with `--input`; only expected below 8 bits |
| `build_inputs.llama_cpp_commit` | supplied with `--input` |
| `build_inputs.quantize_cmd` | supplied with `--input` |

`expected` is 9 for a file with sub-8-bit tensors and 7 otherwise. The score measures
documentation, not trustworthiness: every value except the fingerprint is a claim.

## Build inputs

`build_inputs` holds strings. Any key is kept, and these four are the ones
`ggufkit reproduce --manifest` reads to rebuild the file and compare the result with the
manifest:

| key | how it is read |
|---|---|
| `source_sha256` | the sha256 of the source the file was quantized from. For a split source, each shard's sha256, comma-separated in shard order (`manifest --source` writes it that way). `reproduce` hashes the `--source` it is given and stops with exit 2 on a mismatch. |
| `imatrix_sha256` | the sha256 of the importance matrix, checked the same way against `--imatrix`. |
| `llama_cpp_commit` | the llama.cpp revision: a commit hash (any prefix of 7 or more characters) or a release tag like `b8146`. `reproduce` compares it with the version `llama-quantize` prints. A difference is a note, not a failure: the same file rebuilt byte for byte with llama.cpp releases from May to October 2026. |
| `quantize_cmd` | the `llama-quantize` command line, read the way `llama-quantize` reads its arguments: `[VAR=value ...] llama-quantize [options] input [output] type [nthreads]`. |

From `quantize_cmd`, `reproduce` takes the type and the options that shape the output, in
their order: `--tensor-type`, `--output-tensor-type`, `--token-embedding-type`, `--pure`,
`--leave-output-tensor`, `--override-kv`, `--prune-layers`, `--include-weights`,
`--exclude-weights` and `--allow-requantize`. The paths and the thread count are not used:
the rebuild runs on the hash-checked `--source`. `--imatrix` names the matrix's path (the
file itself records it in `quantize.imatrix.file`), and `--max-buffer-size` only affects
I/O. `--tensor-type-file` (it needs a file the manifest does not carry), `--keep-split`
(split output is not supported yet), `--dry-run` and unknown options stop `reproduce` with
exit 2. Without `quantize_cmd`, the type comes from `model.file_type` and no options are
used. `--type` overrides either.

The rebuild is compared with the manifest's fingerprint, and with its file sha256 when it
records one for a single file. Identical bytes pass even against a manifest made with
`--no-data`.

## Verifying

```sh
ggufkit verify model.gguf --manifest model.manifest.json            # fingerprint only
ggufkit verify model.gguf --manifest model.manifest.json --file-sha256  # also raw file bytes
ggufkit verify model.gguf --against other.gguf                       # two files, no manifest
```

| verdict | meaning |
|---|---|
| `identical` | metadata, tensor descriptors and weights all match |
| `metadata-only` | only metadata differs; the report lists the keys |
| `tensor-info-only` | only tensor descriptors differ: renamed, retyped or reshaped in place, every tensor's bytes unchanged |
| `metadata-and-tensor-info` | both of the above, weights untouched |
| `weights-changed` | the bytes of at least one tensor differ, bytes moved between tensors, or tensors were added or removed |

A rename counts as `tensor-info-only` only when the per-tensor digests show the same bytes
under the new name; without per-tensor digests (`--no-per-tensor`) any change in `weights`
is reported as `weights-changed`.

Exit status:

- `0` for `identical`, and for `metadata-only` with `--allow-metadata`;
- `1` for any other verdict, for a raw file hash that differs or was not recorded, and when
  the weights could not be compared;
- `2` for errors.

`--ignore-keys 'general.name,tokenizer.chat_template'` excludes keys you expect to change.

A manifest made with `manifest --no-data` carries no weight digests. Checking a file against
it reports `weights  not compared` and **fails**, so a script that trusts the exit status
cannot be handed a file whose weights nobody checked. Pass `verify --no-data` to ask for a
metadata-and-descriptors check on purpose. Likewise, `--file-sha256` against a manifest made
without `--file-sha256` reports `files  not checked` and fails.

## Signing

A manifest is a set of claims; signing it binds those claims to the exact files and to whoever
signed. ggufkit does no cryptography itself. It hands the manifest to standard supply-chain
tools in the shape they expect.

`--intoto FILE` writes an [in-toto Statement v1](https://github.com/in-toto/attestation/blob/main/spec/v1/statement.md)
next to the manifest: one `subject` per file, identified by its raw sha256, `predicateType`
`https://github.com/MysterionRise/ggufkit/manifest/v1`, and the manifest as `predicate`. It
implies `--file-sha256`.

The predicate type is an identifier, not a web page: it does not need to resolve, and it
will not change for `ggufkit-manifest/1`, so attestations made today stay verifiable.

```sh
ggufkit manifest model.gguf --intoto model.intoto.json
```

### With cosign

[cosign](https://github.com/sigstore/cosign) builds the statement itself from the manifest and
the model file, signs it, and checks it later:

```sh
# producer
COSIGN_MAX_ATTACHMENT_SIZE=64GiB cosign attest-blob --key cosign.key \
  --predicate model.manifest.json \
  --type https://github.com/MysterionRise/ggufkit/manifest/v1 \
  --bundle model.sigstore.json model.gguf

# consumer
COSIGN_MAX_ATTACHMENT_SIZE=64GiB cosign verify-blob-attestation --key cosign.pub \
  --bundle model.sigstore.json \
  --type https://github.com/MysterionRise/ggufkit/manifest/v1 model.gguf
```

`COSIGN_MAX_ATTACHMENT_SIZE` matters: cosign refuses blobs over 128 MiB by default, and almost
every GGUF is larger. By default `attest-blob` also records the signature in Sigstore's public
transparency log; `--tlog-upload=false` keeps it local, and the matching check then needs
`--insecure-ignore-tlog=true`. Keyless signing (`cosign attest-blob` without `--key`, using an
OIDC identity) works the same way and is what most publishers will want.

Tested with cosign 2.4.1, a throwaway key pair and the transparency log disabled, on
unsloth's Qwen3-0.6B Q4_K_M file: the original verifies; a copy whose only change is
`general.name` (made with llama.cpp's `gguf-new-metadata`) fails with
`no matching subject digest found`, because cosign checks raw bytes. `ggufkit verify` on the
same copy against the same manifest reports `metadata-only`, naming `general.name`. The two
checks answer different questions; use both.

For a model split into shards, `attest-blob` signs one file at a time: attest each shard with
the same manifest as predicate, or sign the statement from `--intoto`, which lists every shard,
with an in-toto or DSSE signer. The second route is not tested here.
