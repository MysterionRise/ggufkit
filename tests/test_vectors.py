"""The ggufkit-fp/1 test vectors in tests/vectors/ (see docs/FINGERPRINT.md, "Test vectors").

Three checks keep the vectors honest: ggufkit still computes the recorded digests, an
independent implementation of the spec text computes them too, and the generator still
writes exactly the committed files.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from ggufkit.canon import UnsupportedModelError
from ggufkit.fingerprint import fingerprint
from ggufkit.model import open_model
from tests import fp_reference

ROOT = Path(__file__).resolve().parent.parent
VECTORS = ROOT / "tests" / "vectors"
DOC = json.loads((VECTORS / "vectors.json").read_text(encoding="utf-8"))
ALL = DOC["vectors"]
BY_NAME = {v["name"]: v for v in ALL}


def _paths(vector: dict[str, Any]) -> list[Path]:
    return [VECTORS / f["name"] for f in vector["files"]]


def test_vectors_are_the_documented_spec() -> None:
    assert DOC["spec"] == "ggufkit-fp/1"
    assert len(ALL) >= 12


@pytest.mark.parametrize("vector", ALL, ids=[v["name"] for v in ALL])
def test_committed_files_are_the_recorded_ones(vector: dict[str, Any]) -> None:
    for f in vector["files"]:
        data = (VECTORS / f["name"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == f["sha256"], f["name"]


@pytest.mark.parametrize("vector", ALL, ids=[v["name"] for v in ALL])
def test_ggufkit_computes_the_recorded_fingerprint(vector: dict[str, Any]) -> None:
    with open_model(_paths(vector)[0]) as model:
        if vector["expect"] == "refused":
            with pytest.raises(UnsupportedModelError):
                fingerprint(model, per_kv=True)
            return
        assert fingerprint(model, per_kv=True).to_dict() == vector["fingerprint"]


@pytest.mark.parametrize("vector", ALL, ids=[v["name"] for v in ALL])
def test_the_spec_text_computes_the_recorded_fingerprint(vector: dict[str, Any]) -> None:
    if vector["expect"] == "refused":
        with pytest.raises(fp_reference.Refused):
            fp_reference.fingerprint(_paths(vector))
        return
    expected = {k: vector["fingerprint"][k] for k in ("kv", "tensor_info", "weights", "overall")}
    assert fp_reference.fingerprint(_paths(vector)) == expected


def test_the_vectors_show_what_each_part_ignores() -> None:
    def parts(name: str) -> dict[str, str]:
        fp = BY_NAME[name]["fingerprint"]
        return {k: fp[k] for k in ("kv", "tensor_info", "weights")}

    base = parts("base")
    for same in ("reordered", "split", "big-endian"):
        assert parts(same) == base, same
    for kv_only in ("realigned", "metadata-edit", "spare-key"):
        p = parts(kv_only)
        assert p["kv"] != base["kv"]
        assert (p["tensor_info"], p["weights"]) == (base["tensor_info"], base["weights"])
    swapped = parts("swapped")
    assert (swapped["kv"], swapped["tensor_info"]) == (base["kv"], base["tensor_info"])
    assert swapped["weights"] != base["weights"]
    renamed = BY_NAME["renamed"]["fingerprint"]
    assert renamed["tensor_info"] != base["tensor_info"]
    assert renamed["weights"] != base["weights"]
    old = BY_NAME["base"]["fingerprint"]["tensors"]["blk.0.ffn_up.weight"]["data"]
    assert renamed["tensors"]["blk.0.ffn_down.weight"]["data"] == old


def test_the_generator_writes_the_committed_vectors() -> None:
    script = ROOT / "scripts" / "make_vectors.py"
    if not script.is_file():
        pytest.skip("scripts/ is not part of the sdist")
    proc = subprocess.run(  # noqa: S603
        [sys.executable, str(script), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
