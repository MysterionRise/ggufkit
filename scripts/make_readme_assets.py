"""Regenerate the README's real-model images and example output.

    uv run python scripts/make_readme_assets.py [--models-dir DIR]

Downloads three models and the importance matrix of unsloth/Qwen3-0.6B-GGUF (Apache-2.0,
base Qwen/Qwen3-0.6B) at a pinned revision, about 2 GB, checks each against the sha256
recorded below, then runs the ggufkit commands behind the README's images and console
examples. Images go to assets/; command output is printed so it can be pasted into the
README. The reproduce example also needs llama.cpp's llama-quantize on PATH.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = "unsloth/Qwen3-0.6B-GGUF"
REVISION = "50968a4468ef4233ed78cd7c3de230dd1d61a56b"
FILES = {
    "Qwen3-0.6B-BF16.gguf": "f9c9f1d3c1e21755b82d4e165f88dbbbd4355646d632fb5d6cef7c66ed4ee04e",
    "Qwen3-0.6B-Q4_K_M.gguf": "ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a",
    "Qwen3-0.6B-UD-Q4_K_XL.gguf": "ed2dd561f8aa5651cfb198be583269779e26ca6e28c40dba94fc2da4432bde2a",
    "imatrix_unsloth.dat": "8eeffbe08370c5cc1a27adb534ff7940094691b18d406f7bb624f3fc0e08f39e",
}
# Square, so it reads well at the top of the README.
HERO_TENSOR = "blk.0.attn_k.weight"
# Shows a region of larger weights carrying larger error.
PATCH_TENSOR = "blk.0.attn_q.weight"
# Pools each error map so the palette PNG stays under 300 KB.
MAX_PIXELS = "262144"

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
CHUNK = 8 << 20


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def fetch(name: str, expected: str, directory: Path) -> Path:
    target = directory / name
    if target.is_file() and sha256_of(target) == expected:
        print(f"* {name}: present, sha256 ok")
        return target
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
    print(f"* downloading {url}")
    partial = target.with_suffix(".part")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as resp, partial.open("wb") as out:
        while block := resp.read(CHUNK):
            digest.update(block)
            out.write(block)
    if digest.hexdigest() != expected:
        partial.unlink()
        msg = f"{name}: sha256 {digest.hexdigest()} does not match {expected}"
        raise SystemExit(msg)
    partial.replace(target)
    print(f"  sha256 ok: {expected}")
    return target


def ggufkit(*args: str) -> None:
    cmd = ["ggufkit", *args]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=False)  # noqa: S603


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models-dir", type=Path, default=ROOT / "scratch" / "models")
    args = parser.parse_args()
    if shutil.which("ggufkit") is None:
        print("ggufkit is not on PATH; run this with `uv run`", file=sys.stderr)
        return 2
    args.models_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: fetch(name, sha, args.models_dir) for name, sha in FILES.items()}
    bf16 = str(paths["Qwen3-0.6B-BF16.gguf"])
    q4km = str(paths["Qwen3-0.6B-Q4_K_M.gguf"])
    udxl = str(paths["Qwen3-0.6B-UD-Q4_K_XL.gguf"])

    work = args.models_dir
    for tensor, image in ((HERO_TENSOR, "hero-error.png"), (PATCH_TENSOR, "error-attn_q.png")):
        ggufkit(
            "viz",
            "error",
            bf16,
            q4km,
            tensor,
            "-o",
            str(ASSETS / image),
            "--max-pixels",
            MAX_PIXELS,
            "--force",
            "-q",
        )
    ggufkit(
        "viz",
        "layers",
        q4km,
        udxl,
        "-o",
        str(ASSETS / "layers-compare.png"),
        "--force",
        "--title",
        "Q4_K_M (llama.cpp recipe)",
        "--title",
        "UD-Q4_K_XL (Unsloth dynamic)",
    )

    ggufkit(
        "viz",
        "tensor",
        bf16,
        PATCH_TENSOR,
        "-o",
        str(ASSETS / "tensor-attn_q.png"),
        "--max-pixels",
        MAX_PIXELS,
        "--force",
        "-q",
    )

    ggufkit("fingerprint", q4km, "--no-per-tensor")
    manifest = work / "Qwen3-0.6B-Q4_K_M.manifest.json"
    ggufkit("manifest", q4km, "-o", str(manifest), "--file-sha256", "--force")

    # A copy whose only change is its name, made with llama.cpp's own metadata tool.
    edited_dir = work / "edited"
    edited_dir.mkdir(exist_ok=True)
    edited = edited_dir / "Qwen3-0.6B-Q4_K_M.gguf"
    new_metadata = shutil.which("gguf-new-metadata")
    if new_metadata is None:
        print("gguf-new-metadata (from the gguf package) is not on PATH", file=sys.stderr)
        return 2
    subprocess.run(  # noqa: S603
        [new_metadata, q4km, str(edited), "--general-name", "Qwen3 0.6B (re-uploaded)", "--force"],
        check=True,
        capture_output=True,
    )
    ggufkit("verify", str(edited), "--manifest", str(manifest))
    ggufkit("verify", udxl, "--against", q4km)
    ggufkit("diff", bf16, q4km, "--no-kv", "--numeric", "--top", "8")

    if shutil.which("llama-quantize") is None:
        print("\nllama-quantize is not on PATH: skipping the reproduce example", file=sys.stderr)
        return 0
    imatrix = str(paths["imatrix_unsloth.dat"])
    ggufkit("reproduce", q4km, "--source", bf16, "--imatrix", imatrix)
    ggufkit("reproduce", q4km, "--source", bf16, "--imatrix", imatrix, "--type", "Q4_K_S")
    return 0


if __name__ == "__main__":
    sys.exit(main())
