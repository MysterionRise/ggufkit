"""Record the README demo GIFs and the vertical demo videos from real runs.

    uv run python scripts/make_demos.py [--models-dir DIR] [--work DIR] [--only NAME ...]

Four short demos, each built only from commands this script runs and output it records:

- signature: a copy that differs only in ``general.name`` fails a cosign attestation check,
  and ``ggufkit verify`` calls it metadata-only;
- tamper: a copy with weights changed in one tensor (a demo tamper made here, labelled so)
  fails the same check, ``ggufkit verify`` names the tensor and ``viz error`` shows where;
- label: two 4-bit files with identical metadata and 60 tensors stored differently;
- provenance: what the file records about its origin, then that record signed and checked.

The model files are the pinned ones from ``scripts/make_readme_assets.py``. Besides the dev
environment this needs cosign (on PATH or in ``$COSIGN``), ffmpeg, and Node.js with
Playwright and a Chromium it can launch. Output: ``assets/demos/<name>.gif`` and ``.png`` for
the README, ``build/demos/<name>.mp4`` (1080x1920) for posting elsewhere.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType, GGUFReader

# make_readme_assets sits next to this file. Python only puts scripts/ on sys.path for
# `python scripts/make_demos.py`, so add it here for `python -m`, test runners and IDEs too.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_readme_assets import FILES, ROOT, fetch

from ggufkit.dequant import to_f32

PREDICATE_TYPE = "https://github.com/MysterionRise/ggufkit/manifest/v1"
DEMOS = ("signature", "tamper", "label", "provenance")
# The demo tamper: the scale of one 256-weight Q4_K block in each of 64 rows, times 1.5.
TAMPER_TENSOR = "blk.12.ffn_up.weight"
TAMPER_ROWS = slice(1200, 1264)
TAMPER_BLOCK = 2
TAMPER_FACTOR = 1.5
# Long command lines are broken before an option once they pass this many columns.
COLUMNS = 50
# Output lines longer than this are cut, with the cut marked (cosign prints a 100 KB envelope).
MAX_LINE = 1000


def shell_lines(argv: list[str]) -> list[str]:
    """Format a command as it would be typed, wrapped with backslashes before options.

    An option keeps its value on the same line; a group longer than a line gets its own.
    """
    words = [f'"{w}"' if " " in w else w for w in [Path(argv[0]).name, *argv[1:]]]
    groups: list[str] = []
    for word in words:
        takes_value = groups and groups[-1].startswith("-") and " " not in groups[-1]
        if takes_value and not word.startswith("-") and "=" not in groups[-1]:
            groups[-1] += " " + word
        else:
            groups.append(word)
    lines: list[str] = []
    current = "$ " + groups[0]
    for group in groups[1:]:
        if group.startswith("-") and len(current) + 1 + len(group) > COLUMNS:
            lines.append(current + " \\")
            current = "    " + group
        else:
            current += " " + group
    lines.append(current)
    return lines


def clip(output: str) -> str:
    lines = output.rstrip("\n").splitlines()
    return "\n".join(
        f"{line[:48]}… ({len(line):,} characters)" if len(line) > MAX_LINE else line
        for line in lines
    )


class Recorder:
    def __init__(self, cosign: str) -> None:
        self.env = {**os.environ, "COSIGN_PASSWORD": "", "COSIGN_MAX_ATTACHMENT_SIZE": "64GiB"}
        self.tools = {
            "cosign": cosign,
            "ggufkit": shutil.which("ggufkit") or "ggufkit",
            "sha256sum": shutil.which("sha256sum") or "sha256sum",
        }

    def run(self, cwd: Path, *argv: str, expect: int) -> dict[str, Any]:
        resolved = [self.tools.get(argv[0], argv[0]), *argv[1:]]
        start = time.perf_counter()
        proc = subprocess.run(  # noqa: S603
            resolved,
            cwd=cwd,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        seconds = time.perf_counter() - start
        if proc.returncode != expect:
            msg = f"{' '.join(argv)}: exit {proc.returncode}, expected {expect}\n{proc.stdout}"
            raise SystemExit(msg)
        print(f"  {' '.join(argv)[:90]}  ({seconds:.1f}s, exit {proc.returncode})")
        return {
            "cmd": shell_lines(list(argv)),
            "raw": proc.stdout,
            "output": clip(proc.stdout),
            "code": proc.returncode,
            "seconds": round(seconds, 2),
        }


def link(src: Path, dst: Path) -> None:
    dst.unlink(missing_ok=True)
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copyfile(src, dst)


def data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def make_tampered(src: Path, dst: Path) -> dict[str, Any]:
    """Copy ``src`` and scale one Q4_K block per row of TAMPER_TENSOR; report what changed."""
    shutil.copyfile(src, dst)
    reader = GGUFReader(dst)
    tensor = next(t for t in reader.tensors if t.name == TAMPER_TENSOR)
    if tensor.tensor_type != GGMLQuantizationType.Q4_K:
        msg = f"{TAMPER_TENSOR} is {tensor.tensor_type.name}, expected Q4_K"
        raise SystemExit(msg)
    ne0, ne1 = (int(n) for n in tensor.shape[:2])
    block, type_size = GGML_QUANT_SIZES[GGMLQuantizationType.Q4_K]
    offset = int(tensor.data_offset)
    total = sum(int(t.n_elements) for t in reader.tensors)
    before = to_f32(tensor, rows=TAMPER_ROWS).copy()
    del tensor, reader

    blocks = np.memmap(dst, np.uint8, "r+", offset, (ne1, ne0 // block, type_size))
    # Each Q4_K block starts with its fp16 scale d; every weight in the block is d * (...).
    scale = np.ascontiguousarray(blocks[TAMPER_ROWS, TAMPER_BLOCK, 0:2]).view("<f2")
    scaled = (scale.astype(np.float32) * TAMPER_FACTOR).astype("<f2")
    blocks[TAMPER_ROWS, TAMPER_BLOCK, 0:2] = scaled.view(np.uint8)
    blocks.flush()
    del blocks

    reader = GGUFReader(dst)
    tensor = next(t for t in reader.tensors if t.name == TAMPER_TENSOR)
    after = to_f32(tensor, rows=TAMPER_ROWS)
    changed = int(np.count_nonzero(before != after))
    del tensor, reader
    return {
        "tensor": TAMPER_TENSOR,
        "rows": [TAMPER_ROWS.start, TAMPER_ROWS.stop - 1],
        "cols": [TAMPER_BLOCK * block, (TAMPER_BLOCK + 1) * block - 1],
        "shape": [ne1, ne0],
        "changed_weights": changed,
        "total_weights": total,
        "percent": round(100 * changed / total, 4),
    }


def gaps_of(output: str) -> list[str]:
    for line in output.splitlines():
        if line.startswith("gaps"):
            return [g.strip() for g in line.split(None, 1)[1].split(",")]
    return []


def score_of(output: str) -> str:
    for line in output.splitlines():
        if line.startswith("score"):
            return line.split()[1]
    return ""


def publish(rec: Recorder, work: Path, q4km: Path) -> Path:
    """Sign the original file the way a publisher would; the demos check copies against it."""
    pub = work / "publisher"
    pub.mkdir(parents=True, exist_ok=True)
    link(q4km, pub / "model.gguf")
    for name in ("cosign.key", "cosign.pub", "model.manifest.json", "model.sigstore.json"):
        (pub / name).unlink(missing_ok=True)
    rec.run(pub, "cosign", "generate-key-pair", expect=0)
    rec.run(pub, "ggufkit", "manifest", "model.gguf", "--file-sha256", expect=0)
    rec.run(pub, *attest_argv(), expect=0)
    return pub


def attest_argv() -> list[str]:
    return [
        "cosign",
        "attest-blob",
        "--key",
        "cosign.key",
        "--predicate",
        "model.manifest.json",
        "--type",
        PREDICATE_TYPE,
        "--bundle",
        "model.sigstore.json",
        "--tlog-upload=false",
        "model.gguf",
    ]


def check_argv(target: str) -> list[str]:
    return [
        "cosign",
        "verify-blob-attestation",
        "--key",
        "cosign.pub",
        "--bundle",
        "model.sigstore.json",
        "--type",
        PREDICATE_TYPE,
        "--insecure-ignore-tlog=true",
        target,
    ]


def fresh_dir(path: Path, pub: Path | None, original: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_file():
            child.unlink()
    link(original, path / "model.gguf")
    if pub is not None:
        for name in ("cosign.pub", "model.manifest.json", "model.sigstore.json"):
            shutil.copyfile(pub / name, path / name)
    return path


def demo_signature(rec: Recorder, work: Path, pub: Path, q4km: Path) -> dict[str, Any]:
    d = fresh_dir(work / "signature", pub, q4km)
    new_metadata = shutil.which("gguf-new-metadata")
    if new_metadata is None:
        msg = "gguf-new-metadata (from the gguf package) is not on PATH"
        raise SystemExit(msg)
    subprocess.run(  # noqa: S603
        [
            new_metadata,
            str(q4km),
            str(d / "suspect.gguf"),
            "--general-name",
            "Qwen3 0.6B (re-uploaded)",
        ],
        check=True,
        capture_output=True,
    )
    return {
        "steps": [
            rec.run(d, *check_argv("suspect.gguf"), expect=1),
            rec.run(
                d,
                "ggufkit",
                "verify",
                "suspect.gguf",
                "--manifest",
                "model.manifest.json",
                expect=1,
            ),
        ],
    }


def demo_tamper(rec: Recorder, work: Path, pub: Path, q4km: Path) -> dict[str, Any]:
    d = fresh_dir(work / "tamper", pub, q4km)
    facts = make_tampered(q4km, d / "suspect.gguf")
    steps = [
        rec.run(d, "sha256sum", "model.gguf", "suspect.gguf", expect=0),
        rec.run(d, *check_argv("suspect.gguf"), expect=1),
        rec.run(
            d, "ggufkit", "verify", "suspect.gguf", "--manifest", "model.manifest.json", expect=1
        ),
        rec.run(
            d,
            "ggufkit",
            "viz",
            "error",
            "model.gguf",
            "suspect.gguf",
            TAMPER_TENSOR,
            "-o",
            "where.png",
            "--no-caption",
            expect=0,
        ),
    ]
    return {"facts": facts, "steps": steps, "images": {"where": data_uri(d / "where.png")}}


def demo_label(rec: Recorder, work: Path, paths: dict[str, Path]) -> dict[str, Any]:
    d = work / "label"
    d.mkdir(parents=True, exist_ok=True)
    link(paths["Qwen3-0.6B-Q4_K_M.gguf"], d / "Q4_K_M.gguf")
    link(paths["Qwen3-0.6B-UD-Q4_K_XL.gguf"], d / "UD-Q4_K_XL.gguf")
    verify = rec.run(
        d, "ggufkit", "verify", "UD-Q4_K_XL.gguf", "--against", "Q4_K_M.gguf", expect=1
    )
    layers = rec.run(
        d,
        "ggufkit",
        "viz",
        "layers",
        "Q4_K_M.gguf",
        "UD-Q4_K_XL.gguf",
        "-o",
        "layers.png",
        "--force",
        expect=0,
    )
    retyped = sum(1 for line in verify["output"].splitlines() if " retyped " in line)
    reader = GGUFReader(d / "Q4_K_M.gguf")
    n_tensors = len(reader.tensors)
    del reader
    return {
        "facts": {"retyped": retyped, "n_tensors": n_tensors},
        "steps": [verify, layers],
        "images": {"layers": data_uri(d / "layers.png")},
    }


def demo_provenance(rec: Recorder, work: Path, pub: Path, q4km: Path) -> dict[str, Any]:
    d = fresh_dir(work / "provenance", None, q4km)
    shutil.copyfile(pub / "cosign.key", d / "cosign.key")
    shutil.copyfile(pub / "cosign.pub", d / "cosign.pub")
    manifest = rec.run(d, "ggufkit", "manifest", "model.gguf", expect=0)
    steps = [
        manifest,
        rec.run(d, *attest_argv(), expect=0),
        rec.run(d, *check_argv("model.gguf"), expect=0),
    ]
    facts = {"gaps": gaps_of(manifest["raw"]), "score": score_of(manifest["raw"])}
    return {"facts": facts, "steps": steps}


def render(data: Path, only: list[str]) -> None:
    node = shutil.which("node")
    if node is None:
        msg = "node is not on PATH; the renderer needs Node.js with Playwright"
        raise SystemExit(msg)
    env = dict(os.environ)
    if "NODE_PATH" not in env and (npm := shutil.which("npm")):
        root = subprocess.run([npm, "root", "-g"], capture_output=True, text=True, check=False)  # noqa: S603
        env["NODE_PATH"] = root.stdout.strip()
    script = ROOT / "scripts" / "demos" / "render.mjs"
    subprocess.run([node, str(script), str(data), *only], check=True, env=env)  # noqa: S603


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models-dir", type=Path, default=ROOT / "scratch" / "models")
    parser.add_argument("--work", type=Path, default=ROOT / "scratch" / "demos")
    parser.add_argument("--only", nargs="+", choices=DEMOS, default=list(DEMOS))
    parser.add_argument("--no-record", action="store_true", help="reuse the recorded demos.json")
    parser.add_argument("--no-render", action="store_true", help="record only")
    args = parser.parse_args()
    data = args.work / "demos.json"

    if not args.no_record:
        cosign = os.environ.get("COSIGN") or shutil.which("cosign")
        if cosign is None or shutil.which("ggufkit") is None:
            print(
                "needs cosign (on PATH or $COSIGN) and ggufkit (run with uv run)", file=sys.stderr
            )
            return 2
        args.models_dir.mkdir(parents=True, exist_ok=True)
        paths = {name: fetch(name, sha, args.models_dir) for name, sha in FILES.items()}
        q4km = paths["Qwen3-0.6B-Q4_K_M.gguf"]
        rec = Recorder(cosign)
        print("* publishing the original")
        pub = publish(rec, args.work, q4km)
        recorded = json.loads(data.read_text()) if data.is_file() else {}
        for name in args.only:
            print(f"* {name}")
            if name == "signature":
                recorded[name] = demo_signature(rec, args.work, pub, q4km)
            elif name == "tamper":
                recorded[name] = demo_tamper(rec, args.work, pub, q4km)
            elif name == "label":
                recorded[name] = demo_label(rec, args.work, paths)
            else:
                recorded[name] = demo_provenance(rec, args.work, pub, q4km)
        for demo in recorded.values():
            for step in demo["steps"]:
                step.pop("raw", None)
        data.write_text(json.dumps(recorded, indent=1))
        print(f"* recorded {data}")
    if not args.no_render:
        render(data, args.only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
