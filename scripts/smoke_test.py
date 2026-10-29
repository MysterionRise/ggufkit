"""Exercise an installed ggufkit end to end, the way a user would.

    python scripts/smoke_test.py

Run it with the Python of the environment the wheel or sdist was installed into: CI and the
release workflow do, before anything is published. It writes a tiny GGUF model with gguf-py,
then fingerprints it, writes and verifies a manifest, edits the metadata and checks that the
edit is reported as metadata-only, checks the Python API agrees, runs reproduce with a
stand-in for llama-quantize (not on Windows), and renders a tensor if Pillow is installed.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

from ggufkit import fingerprint_file, verify_file


def ggufkit(*args: str, expect: int = 0) -> str:
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "ggufkit", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != expect:
        print(proc.stdout, proc.stderr, sep="\n", file=sys.stderr)
        msg = f"ggufkit {' '.join(args)}: exit {proc.returncode}, expected {expect}"
        raise SystemExit(msg)
    return proc.stdout


def check(ok: bool, what: str) -> None:
    if not ok:
        msg = f"smoke test failed: {what}"
        raise SystemExit(msg)


def write_model(path: Path, name: str) -> None:
    writer = GGUFWriter(path, "llama", use_temp_file=False)
    writer.add_name(name)
    writer.add_tensor("blk.0.attn_q.weight", np.arange(64, dtype=np.float32).reshape(8, 8))
    writer.add_tensor("output_norm.weight", np.ones(8, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main() -> int:
    print(ggufkit("--version").strip())
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        model, renamed = d / "m.gguf", d / "renamed.gguf"
        write_model(model, "smoke")
        write_model(renamed, "smoke, renamed")

        fp = json.loads(ggufkit("fingerprint", str(model), "--json"))
        check(fp["spec"] == "ggufkit-fp/1", f"fingerprint spec {fp['spec']}")
        check(fingerprint_file(model).overall == fp["overall"], "API and CLI differ")
        api_report = verify_file(renamed, against=model)
        check(api_report.verdict == "metadata-only", f"API verdict {api_report.verdict}")
        ggufkit("manifest", str(model), "-o", str(d / "m.json"), "--file-sha256", "-q")
        ggufkit("verify", str(model), "--manifest", str(d / "m.json"), "--file-sha256")
        report = json.loads(
            ggufkit("verify", str(renamed), "--manifest", str(d / "m.json"), "--json", expect=1)
        )
        check(report["verdict"] == "metadata-only", f"verdict {report['verdict']}")
        ggufkit("diff", str(model), str(renamed), "--numeric", expect=1)
        ggufkit("fingerprint", str(d / "missing.gguf"), expect=2)
        if sys.platform != "win32":
            # Stands in for llama-quantize SOURCE OUT TYPE: writes the renamed model as OUT.
            stand_in = d / "llama-quantize"
            stand_in.write_text(
                f"#!{sys.executable}\nimport shutil, sys\n"
                f"shutil.copyfile({str(renamed)!r}, sys.argv[2])\n"
            )
            stand_in.chmod(0o755)
            rebuilt = json.loads(
                ggufkit(
                    "reproduce",
                    str(model),
                    "--source",
                    str(model),
                    "--type",
                    "F32",
                    "--llama-quantize",
                    str(stand_in),
                    "--json",
                    expect=1,
                )
            )
            check(rebuilt["verdict"] == "metadata-only", f"reproduce {rebuilt['verdict']}")
            check(rebuilt["byte_identical"] is False, "reproduce compared the bytes wrongly")
        try:
            import PIL  # noqa: F401
        except ImportError:
            print("Pillow not installed: skipping viz")
        else:
            ggufkit("viz", "tensor", str(model), "blk.0.attn_q.weight", "-o", str(d / "q.png"))
            ggufkit("viz", "layers", str(model), "-o", str(d / "l.png"))
            check((d / "q.png").stat().st_size > 0, "viz tensor wrote an empty image")
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
