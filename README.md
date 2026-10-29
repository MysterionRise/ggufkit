# ggufkit

[![CI](https://github.com/MysterionRise/ggufkit/actions/workflows/ci.yml/badge.svg)](https://github.com/MysterionRise/ggufkit/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ggufkit.svg)](https://pypi.org/project/ggufkit/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/MysterionRise/ggufkit/blob/main/LICENSE)
![Python 3.11 to 3.14](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)

Fingerprint, verify and visualize [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) models.

`ggufkit` answers questions a plain file hash cannot:

- Is this GGUF the same model as that one, even though the bytes differ?
- Did a re-upload change the weights, or only the metadata? Which keys, which tensors?
- Does this file document where it came from well enough to reproduce it?
- Can it be rebuilt from its source with llama.cpp, byte for byte?
- How far is this quantization from its source, tensor by tensor, and where inside each tensor?
- Which quantization type did each tensor get?
- What does `blk.17.attn_q.weight` actually look like?

It is built on llama.cpp's own [`gguf`](https://pypi.org/project/gguf/) Python package, so
every tensor type that package can decode works here too: 31 of them, from `F32` and `Q4_0`
to `IQ2_XXS`, `MXFP4` and `NVFP4`. Sharded models (`-00001-of-00003.gguf`) are handled as one model.

<picture>
  <source media="(prefers-reduced-motion: reduce)" srcset="https://raw.githubusercontent.com/MysterionRise/ggufkit/main/assets/demos/tamper.png">
  <img src="https://raw.githubusercontent.com/MysterionRise/ggufkit/main/assets/demos/tamper.gif" alt="A demo tamper: 15,706 weights in one tensor of Qwen3-0.6B's Q4_K_M file are changed. sha256 and a cosign signature check only report that the file differs; ggufkit verify names the changed tensor, blk.12.ffn_up.weight, and ggufkit viz error shows the changed block inside it." width="440">
</picture>

*A tamper made for this demo: in one tensor of Qwen3-0.6B's Q4_K_M file, the scale of 64
quantization blocks is multiplied by 1.5, which changes 15,706 weights. A hash and a cosign
signature check both say only that the file differs. `ggufkit verify` names the tensor, and
`ggufkit viz error` shows the block. Every frame is the recorded output of the commands on
screen.*

Every example below is real output on
[unsloth/Qwen3-0.6B-GGUF](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF) (Apache-2.0) at
revision `50968a4`. [`scripts/make_readme_assets.py`](https://github.com/MysterionRise/ggufkit/blob/main/scripts/make_readme_assets.py) downloads
the files, checks their sha256 and reruns every command;
[`scripts/make_demos.py`](https://github.com/MysterionRise/ggufkit/blob/main/scripts/make_demos.py) records the animations from the same
files.

## Why

Model signing ([OpenSSF model-signing](https://github.com/sigstore/model-transparency),
Sigstore) protects the bytes you were given. It breaks the moment someone fixes a chat
template in the metadata, and when it breaks it cannot tell you whether the weights changed.
It also says nothing about whether a community quantization was faithfully built from the
model it names. GGUF's standard metadata has no key for the source checksum, the llama.cpp
version or the quantize command, so files do not say how to rebuild them, and
[Mind the Gap](https://arxiv.org/abs/2505.23786) (ICML 2025) showed weights that behave
well in full precision and badly after GGUF quantization.

ggufkit gives GGUF files a content identity that survives the edits people legitimately make
([docs/FINGERPRINT.md](https://github.com/MysterionRise/ggufkit/blob/main/docs/FINGERPRINT.md)), a provenance record that says what a file claims
and what it leaves out ([docs/MANIFEST.md](https://github.com/MysterionRise/ggufkit/blob/main/docs/MANIFEST.md)), and the tools to check one
against the other.

## Install

```sh
pip install "ggufkit[viz]"
uv tool install "ggufkit[viz]"     # or: pipx install "ggufkit[viz]"
```

The `viz` extra pulls in Pillow for image rendering; the core needs only `gguf` and `numpy`.
Python 3.11 or newer. `python -m ggufkit` works as well as `ggufkit`.

## Quick start

### Fingerprint a model

```console
$ ggufkit fingerprint Qwen3-0.6B-Q4_K_M.gguf --no-per-tensor
kv           8eb7cd5593931521f2e20641dcb340cf5e9e32a6cafd9b5face98332e53052d9
tensor_info  fb9deeab22c4fb9d0b1cca92f9edb974c1f21c53b1a986cd96144d665f0c65be
weights      7e8be1d0cc3ac473b493058a2976be592d51125772c39e80ab6264215b756de5
overall      155c4d32bd17c6a005e44269e96cf3f0a2b315ca1eb3bc434349aa268b3f975a
spec         ggufkit-fp/1  (32 metadata entries, 310 tensors)
```

Three digests, three answers:

| `kv` | `tensor_info` | `weights` | meaning |
|---|---|---|---|
| same | same | same | the same model |
| differs | same | same | only metadata changed (names, templates, licence text) |
| same | differs | same | tensors retyped or reshaped in place, bytes untouched |
| any | any | differs | tensor bytes changed or moved between tensors, tensors were added or removed, or renamed |

`weights` binds every tensor's bytes to its name, so swapping the data of two tensors with the
same shape shows up even though no byte was added or lost. `verify` tells a pure rename apart
from new weights through the per-tensor digests.

Without `--no-per-tensor` you also get one line per tensor. `--json` prints JSON, `--no-data`
skips tensor bytes (instant on any size), `--per-kv` adds a digest per metadata entry, `-j 8`
hashes on eight threads. Pass any shard of a split model and
the whole model is fingerprinted; splitting, merging or editing metadata with llama.cpp's
`gguf-new-metadata` leaves `weights` unchanged.

### Record provenance

```console
$ ggufkit manifest Qwen3-0.6B-Q4_K_M.gguf --file-sha256
manifest     Qwen3-0.6B-Q4_K_M.manifest.json
model        qwen3 Q4_K_M, 310 tensors, 596.05M parameters, 372.65 MiB
fingerprint  155c4d32bd17c6a005e44269e96cf3f0a2b315ca1eb3bc434349aa268b3f975a
provenance   imatrix, quantized_by, repo_url
score        2/9 of what a verifiable file records
gaps         general.source.*, general.license, general.uuid, build_inputs.source_sha256, build_inputs.imatrix_sha256, build_inputs.llama_cpp_commit, build_inputs.quantize_cmd
```

This file records who quantized it and which calibration data the importance matrix came from,
but not which model it was quantized from, its licence, or how to rebuild it.

The manifest is a JSON file: the full fingerprint, file sizes (and raw hashes with
`--file-sha256`), what the metadata claims about origin and licence, the build inputs you pass
with `--input KEY=VALUE`, and the list of what a reproducible quantization would record but
this file does not. Its producer can close the `build_inputs.*` gaps with `--input`, for
example `--input llama_cpp_commit=b8146`. Publish it next to the model.

To sign it, `--intoto model.intoto.json` also writes an in-toto statement, and the manifest
works as a predicate for `cosign attest-blob`. The tested recipe is in
[docs/MANIFEST.md](https://github.com/MysterionRise/ggufkit/blob/main/docs/MANIFEST.md#signing).

<picture>
  <source media="(prefers-reduced-motion: reduce)" srcset="https://raw.githubusercontent.com/MysterionRise/ggufkit/main/assets/demos/provenance.png">
  <img src="https://raw.githubusercontent.com/MysterionRise/ggufkit/main/assets/demos/provenance.gif" alt="ggufkit manifest reports that the Q4_K_M file records 2 of 9 provenance items and lists the 7 missing ones; the manifest is then signed with cosign attest-blob and checked with cosign verify-blob-attestation, which prints Verified OK." width="400">
</picture>

*What the file records, then the record signed and checked: `ggufkit manifest`, `cosign
attest-blob` with the manifest as predicate, and `cosign verify-blob-attestation`.*

### Verify

A copy whose name was changed with llama.cpp's `gguf-new-metadata`, checked against the
manifest of the original:

```console
$ ggufkit verify edited/Qwen3-0.6B-Q4_K_M.gguf --manifest Qwen3-0.6B-Q4_K_M.manifest.json
verdict      metadata-only
kv           differs
tensor_info  same
weights      same
metadata changes:
  ~ general.name
```

<picture>
  <source media="(prefers-reduced-motion: reduce)" srcset="https://raw.githubusercontent.com/MysterionRise/ggufkit/main/assets/demos/signature.png">
  <img src="https://raw.githubusercontent.com/MysterionRise/ggufkit/main/assets/demos/signature.gif" alt="A copy of the Q4_K_M file with only general.name changed fails a cosign attestation check with no matching subject digest found; ggufkit verify reports metadata-only, weights same, and names general.name as the only change." width="400">
</picture>

*The same edit against a cosign attestation of the original: cosign rejects the file, as it
would for any change, and `ggufkit verify` says only the name changed. (The demos run
offline, so cosign is told to skip Sigstore's public transparency log; in real use, leave it
on.)*

Unsloth's two 4-bit files of the same model, compared directly:

```console
$ ggufkit verify Qwen3-0.6B-UD-Q4_K_XL.gguf --against Qwen3-0.6B-Q4_K_M.gguf
verdict      weights-changed
kv           same
tensor_info  differs
weights      differs
tensor changes:
  ~ blk.0.attn_k.weight  retyped  Q5_K -> Q4_K
  ~ blk.0.attn_q.weight  retyped  Q5_K -> Q4_K
  ~ blk.0.ffn_gate.weight  retyped  Q5_K -> Q4_K
  ~ blk.0.ffn_up.weight  retyped  Q5_K -> Q4_K
  …
```

`kv same`: the two files carry identical metadata, down to `general.file_type`, which says
`Q4_K_M` in both. Only the tensors tell them apart: 60 of them are stored in a different type.

<picture>
  <source media="(prefers-reduced-motion: reduce)" srcset="https://raw.githubusercontent.com/MysterionRise/ggufkit/main/assets/demos/label.png">
  <img src="https://raw.githubusercontent.com/MysterionRise/ggufkit/main/assets/demos/label.gif" alt="ggufkit verify on Unsloth's Q4_K_M and UD-Q4_K_XL files of Qwen3-0.6B: kv same, tensor_info differs, 60 of 310 tensors retyped; ggufkit viz layers then shows the two quantization recipes side by side." width="400">
</picture>

Verify against a manifest or straight against another file with `--against other.gguf`. The
verdict is one of `identical`, `metadata-only`, `tensor-info-only`, `metadata-and-tensor-info`
or `weights-changed`. The exit status is 0 only for `identical`, so it drops into CI and
download scripts; `--allow-metadata` also accepts `metadata-only`, and `--ignore-keys` takes
globs for keys you expect to change.

### Reproduce a quantization

llama.cpp's `llama-quantize` is deterministic: the same source, importance matrix and type
give the same file, byte for byte, whatever the thread count or CPU build. Three llama.cpp
releases, from May to October 2026, all rebuild Unsloth's Qwen3-0.6B Q4_K_M and Q8_0 files
exactly ([#22](https://github.com/MysterionRise/ggufkit/issues/22)). So a published quant can be checked by making it again:

```console
$ ggufkit reproduce Qwen3-0.6B-Q4_K_M.gguf --source Qwen3-0.6B-BF16.gguf --imatrix imatrix_unsloth.dat
running llama-quantize (Q4_K_M); this can take a while
comparing the rebuild with Qwen3-0.6B-Q4_K_M.gguf
type         Q4_K_M
llama.cpp    0.8.0-dev (build 1, commit 8642541)
command      /usr/local/bin/llama-quantize --imatrix Qwen3-0.6B-GGUF/imatrix_unsloth.dat /home/user/models/Qwen3-0.6B-BF16.gguf /tmp/ggufkit-reproduce-kpfp1ms9/rebuilt.gguf Q4_K_M
file bytes   identical
verdict      identical
kv           same
tensor_info  same
weights      same
```

`reproduce` takes the quantization type from the file's `general.file_type` (`--type`
overrides it) and runs `llama-quantize` on the source, the one on PATH or
`--llama-quantize PATH`. It then compares the rebuild with the file, as `verify --against`
does, and also compares the raw bytes. Here it took 20 seconds for the quantization and 10
for the comparison, on 4 cores.

A file quantized with an importance matrix records it in its `quantize.imatrix.*` keys. The
matrix sets the quantization scales, so the rebuild needs the same one: without `--imatrix`,
`reproduce` stops and names the recorded file and dataset. `llama-quantize` also records the
matrix's path exactly as it was given (here Unsloth's `Qwen3-0.6B-GGUF/imatrix_unsloth.dat`),
so `reproduce` runs it in a temporary directory with the matrix at that relative path.

When the rebuild differs, the report says how. The same file rebuilt as Q4_K_S:

```console
$ ggufkit reproduce Qwen3-0.6B-Q4_K_M.gguf --source Qwen3-0.6B-BF16.gguf --imatrix imatrix_unsloth.dat --type Q4_K_S
…
file bytes   differ
verdict      weights-changed
kv           differs
tensor_info  differs
weights      differs
metadata changes:
  ~ general.file_type
tensor changes:
  ~ blk.0.attn_v.weight  retyped  Q6_K -> Q5_K
  ~ blk.0.ffn_down.weight  retyped  Q6_K -> Q5_K
  …
```

The exit status is 0 when the rebuild is identical, 1 when it differs (`--allow-metadata`
accepts a metadata-only difference), and 2 when no rebuild could be made. `--keep FILE`
saves the rebuild for `diff` or `viz error`, and `--json` prints the report with the
command and the llama.cpp version. Files made with per-tensor type overrides
(`llama-quantize --tensor-type`, as in Unsloth's dynamic `UD-` quants) or a modified
llama.cpp do not rebuild with plain `llama-quantize`; `reproduce` then lists the tensors that
differ. Determinism was checked on x86_64 Linux; other platforms are not checked yet.

### Diff two models

```console
$ ggufkit diff Qwen3-0.6B-BF16.gguf Qwen3-0.6B-Q4_K_M.gguf --no-kv --numeric --top 8
tensors: 197 changed, 0 added, 0 removed (of 310)
  ~ blk.0.attn_k.weight  type, data  [BF16 1024x1024 -> Q4_K 1024x1024]
  ~ blk.0.attn_output.weight  type, data  [BF16 2048x1024 -> Q4_K 2048x1024]
  …
  ~ token_embd.weight  type, data  [BF16 1024x151936 -> Q6_K 1024x151936]
numeric (top 8 by rel_l2):
  tensor                               a -> b                 rmse    max_abs   mean_abs     rel_l2     cosine
  blk.16.ffn_down.weight               BF16->Q4_K         0.001822   0.023621   0.001476   0.076444   0.997074
  blk.19.ffn_down.weight               BF16->Q4_K         0.002015   0.029144   0.001633   0.076084   0.997102
  blk.18.ffn_down.weight               BF16->Q4_K         0.001898   0.024334   0.001547   0.075795   0.997124
  blk.15.ffn_down.weight               BF16->Q4_K         0.001775   0.031386   0.001440   0.075282   0.997163
  blk.16.attn_k.weight                 BF16->Q4_K         0.001789   0.028889   0.001346   0.075218   0.997169
  blk.10.ffn_down.weight               BF16->Q4_K         0.001768   0.021751   0.001437   0.074827   0.997197
  blk.11.attn_output.weight            BF16->Q4_K         0.001742   0.020073   0.001346   0.074776   0.997201
  blk.13.attn_output.weight            BF16->Q4_K         0.001683   0.016928   0.001350   0.074690   0.997207
```

The whole 596M-parameter comparison takes under a minute. Across the model, tensors stored as
Q4_K land at a median relative L2 error of 7.3%, the Q6_K ones at 1.8%.

Metadata entries are shown with their old and new values; tensors are listed as added,
removed or changed (shape, type, data). `--numeric` decodes both sides and reports error per
tensor, worst first, in bounded memory. `--tensor 'blk.*.attn_*'` narrows it, `--json` gives
machine-readable output, and the exit status is 1 when anything differs.

### See where quantization loses precision

```sh
ggufkit viz error Qwen3-0.6B-BF16.gguf Qwen3-0.6B-Q4_K_M.gguf blk.0.attn_q.weight -o err.png --max-pixels 262144
ggufkit viz error Qwen3-0.6B-BF16.gguf Qwen3-0.6B-Q4_K_M.gguf 'blk.*.attn_q.weight' --match-glob --shared-scale -o imgs/err.png
```

![Per-weight quantization error of blk.0.attn_k.weight in Qwen3-0.6B, Q4_K_M against BF16](https://raw.githubusercontent.com/MysterionRise/ggufkit/main/assets/hero-error.png)

*`blk.0.attn_k.weight` of Qwen3-0.6B, comparing Unsloth's BF16 and Q4_K_M files. Error follows
weight magnitude: bright rows are output channels with large weights, whose 4-bit steps are
coarser (per-row correlation between error and weight magnitude: 0.99).*

<img src="https://raw.githubusercontent.com/MysterionRise/ggufkit/main/assets/error-attn_q.png" alt="Per-weight Q4_K error of blk.0.attn_q.weight in Qwen3-0.6B" width="341">

*`blk.0.attn_q.weight`, Q4_K against BF16. Besides the bright rows, a brighter patch stands
out a quarter of the way down, in the rows of attention heads 3 to 5: its weights are about 40%
larger than the tensor's average, and its error about 60% larger.*

![Per-weight Q4_0 error of a synthetic tensor with one outlier input channel](https://raw.githubusercontent.com/MysterionRise/ggufkit/main/assets/example-error.png)

*A synthetic illustration of the mechanism, from `scripts/make_examples.py`: one input channel
with 8x larger weights, quantized to Q4_0. Q4_0 stores one scale per block of 32 weights along
a row, so the outlier stretches the scale of its block and every weight in that band loses
precision.*

Each pixel is the error of one weight: absolute by default, `--metric rel` or `--metric signed`
on request. Large tensors are pooled down to `--max-pixels` while they are decoded, and the
default `--pool max` keeps the worst weight of each pixel visible. The caption gives rmse,
relative L2 and cosine similarity; `--shared-scale` puts several tensors on one colour scale.

### See the quant mix

```console
$ ggufkit viz layers Qwen3-0.6B-UD-Q4_K_XL.gguf
block  attn_norm  attn_q  attn_k  attn_v  attn_q_norm  attn_k_norm  attn_output  ffn_norm  ffn_gate  ffn_up  ffn_down
0      F32        Q5_K    Q5_K    Q6_K    F32          F32          Q4_K         F32       Q5_K      Q5_K    Q6_K
…
3      F32        IQ4_XS  IQ4_XS  Q5_K    F32          F32          Q4_K         F32       IQ4_XS    IQ4_XS  Q5_K
…
$ ggufkit viz layers Qwen3-0.6B-Q4_K_M.gguf Qwen3-0.6B-UD-Q4_K_XL.gguf -o layers.png
```

![Quantization type of every tensor in Qwen3-0.6B Q4_K_M and UD-Q4_K_XL, side by side](https://raw.githubusercontent.com/MysterionRise/ggufkit/main/assets/layers-compare.png)

*The two 4-bit files from the verify example. llama.cpp's Q4_K_M recipe (left) raises `attn_v`
and `ffn_down` to Q6_K in the first three blocks, every third block after that and the last
five. Unsloth's UD-Q4_K_XL (right) uses Q5_K for the first three and last two blocks, IQ4_XS
for blocks 3 to 7, and Q6_K for `attn_v` outside blocks 3 to 7. The metadata of both files
says Q4_K_M.*

One row per block, one column per tensor role, coloured by bits per weight, with non-block
tensors and a legend underneath. Given several models, `-o` draws them side by side with one
palette, so a colour means the same type in every panel. The text table and `--json` work
without Pillow.

### Stitch metadata onto tensors

```sh
ggufkit stitch --metadata tokenizer-fixed.gguf --tensors model.gguf -o model-fixed.gguf
ggufkit stitch --metadata model.gguf -o vocab-only.gguf     # metadata only, no tensors
```

### Look at the weights

```sh
ggufkit viz tensor model.gguf blk.0.attn_q.weight -o attn_q.png
ggufkit viz tensor model.gguf 'blk.*.attn_q.weight' --match-glob -o imgs/q.png
ggufkit viz tensor model.gguf token_embd.weight -o embd.png --max-pixels 4000000
```

Each pixel is one weight: brightness follows magnitude, hue says whether the weight sits
well below the mean, well above it, or in between.

<img src="https://raw.githubusercontent.com/MysterionRise/ggufkit/main/assets/tensor-attn_q.png" alt="blk.0.attn_q.weight of Qwen3-0.6B in BF16, rendered as an image" width="341">

*`blk.0.attn_q.weight` of Qwen3-0.6B in BF16, mean-pooled to 341×682 pixels with
`--max-pixels 262144`. The brighter patch a quarter of the way down is the group of larger
weights that shows more quantization error in the `attn_q` error map above.*

PyTorch checkpoints work too when `torch` is installed (`--model-type torch`).

## What this proves, and what it does not

- **A fingerprint proves integrity and equivalence:** two files hold the same metadata,
  tensor descriptors and weights, or exactly which part differs. Each tensor's bytes are
  bound to its name, so moving weights between tensors counts as a change.
- **A manifest records claims** (where the model came from, who quantized it, how) and makes
  missing ones visible. It is only as trustworthy as the place you got it from: whoever can
  replace a model can replace a manifest stored next to it. Sign manifests (`--intoto` and
  cosign, see [docs/MANIFEST.md](https://github.com/MysterionRise/ggufkit/blob/main/docs/MANIFEST.md#signing)) or fetch them from a source
  you trust more than the download, and check the signature before trusting the verdict.
- **A passing `verify` always means the weights were compared.** Against a manifest made with
  `--no-data`, `verify` fails unless you ask for a metadata-only check with `verify --no-data`.
- **Neither proves that a quantized file was faithfully built from the model it names, or that
  a model is benign.** For the first, rebuild it: `reproduce` re-quantizes the source and
  says whether the result is the same file, and `diff --numeric` tells you how far apart
  the two are when it is not. A passing `reproduce` proves the file is what llama.cpp makes
  from that source and importance matrix; check the source's own hash against its publisher.

## Exit codes

| command | 0 | 1 | 2 |
|---|---|---|---|
| `verify` | identical (or metadata-only with `--allow-metadata`) | any other difference; weights or file hashes that could not be compared | error |
| `reproduce` | the rebuild is identical (or metadata-only with `--allow-metadata`) | the rebuild differs | error, including a missing importance matrix or a failed `llama-quantize` |
| `diff` | nothing differs in the compared sections | something differs | error |
| `fingerprint`, `manifest`, `stitch`, `viz` | done | | error |

Errors are a malformed or unreadable file, a malformed manifest, an output file that exists
without `--force`, or invalid arguments; they print one line starting with `error:`. Ctrl-C
exits 130, and a closed output pipe 141.

## Python API

The same checks are available from Python, on paths:

```python
import ggufkit

fp = ggufkit.fingerprint_file("model.gguf")  # any shard of a split model works too
print(fp.overall, fp.weights)

report = ggufkit.verify_file("download.gguf", against="model.gguf")
if not report.ok(allow_metadata=True):
    raise SystemExit(f"{report.verdict}: {[str(c) for c in report.tensor_changes or []]}")

manifest = ggufkit.create_manifest("model.gguf", file_sha256=True)
ggufkit.save_manifest("model.manifest.json", manifest)
ggufkit.verify_file("model.gguf", manifest="model.manifest.json", file_sha256=True).ok()

result = ggufkit.reproduce_file(
    "model-Q4_K_M.gguf", source="model-BF16.gguf", imatrix="imatrix.dat"
)
print(result.ok(), result.byte_identical, result.report.verdict)
```

`fingerprint_file`, `create_manifest`, `verify_file`, `load_manifest` and `save_manifest`
return `Fingerprint`, `Manifest` (a typed dict in the documented format) and `Report`
objects; `reproduce_file` returns a `ReproduceResult` and raises `ReproduceError` when no
rebuild could be made. Every error ggufkit raises for bad input is a `ggufkit.GGUFKitError`, itself a
`ValueError`. The names in `ggufkit.__all__` are stable: they change only in a major release,
or after a deprecation warning that lasts at least one minor release. The submodules
(`ggufkit.fingerprint`, `ggufkit.verify`, …) are the command line's implementation and may
change in any release. `import ggufkit` loads nothing heavy until you use one of the names.

## FAQ

**How is this different from `sha256sum` or the hashes Hugging Face shows?** Those hash bytes.
Any change, even fixing one character of a chat template, gives a new hash and no
explanation. A ggufkit fingerprint stays the same under changes that leave the model alone
(metadata order, sharding, alignment, tensor order) and says which part changed when
something did. Keep checking file hashes on download; use the fingerprint to ask whether it
is still the same model.

**And OpenSSF model-signing or Sigstore?** They prove who published which bytes, and ggufkit
does not replace them. It writes the manifest you sign (an in-toto statement, or a cosign
predicate), and when a signature check fails, it tells you what changed. The signature demo
above shows both side by side.

**Does a matching fingerprint mean a model is safe?** No. It means the model is the same as
the one you compared it with. Whether that one is benign is a separate question.

**Will fingerprints change in a later release?** No. A file's `ggufkit-fp/1` digests stay the
same; a change to what is hashed gets a new spec id, and ggufkit keeps verifying the old one.
Manifests (`ggufkit-manifest/1`) only gain optional fields. See
[docs/FINGERPRINT.md](https://github.com/MysterionRise/ggufkit/blob/main/docs/FINGERPRINT.md) and [docs/MANIFEST.md](https://github.com/MysterionRise/ggufkit/blob/main/docs/MANIFEST.md).

**Can I use it from Python?** Yes: see [Python API](#python-api). The names exported by
`import ggufkit` are stable, as are the command line, its exit codes, and the fingerprint and
manifest formats. Other implementations of the fingerprint can test against the vectors in
[tests/vectors](https://github.com/MysterionRise/ggufkit/tree/main/tests/vectors).

## Roadmap

- A gallery of error maps and quant maps for popular open models, built from the real files.
- `reproduce --manifest`: take the source, importance matrix and llama.cpp revision from a
  manifest's build inputs, and check their hashes before rebuilding.

## Development

```sh
uv sync --all-extras
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

Tests build tiny GGUF files in-process; nothing is downloaded. `pytest --update-golden`
rewrites the golden fingerprints after a deliberate spec change (bump the spec id too).
Releases are published to PyPI from a `vX.Y.Z` tag by `.github/workflows/release.yml`, with
build provenance and PyPI attestations; see [docs/RELEASING.md](https://github.com/MysterionRise/ggufkit/blob/main/docs/RELEASING.md).
Want to help? [CONTRIBUTING.md](https://github.com/MysterionRise/ggufkit/blob/main/CONTRIBUTING.md#where-to-start)
lists where to start; issues labelled
[good first issue](https://github.com/MysterionRise/ggufkit/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
each say what to change and when it's done.

## Origins

ggufkit started from [KerfuffleV2/gguf-tools](https://github.com/KerfuffleV2/gguf-tools)
(MIT, 2023): `gguf-checksum`, `gguf-frankenstein` and `gguf-tensor-to-image`. Those commits
are preserved in this repository's history. The checksum tool's idea, hashing metadata,
tensor descriptors and weights separately and independent of their order, is the seed of
the fingerprint specification.

## License

MIT. See [LICENSE](https://github.com/MysterionRise/ggufkit/blob/main/LICENSE).
