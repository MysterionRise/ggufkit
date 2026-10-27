"""The stable Python API: only what ``import ggufkit`` exposes."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import ggufkit

VECTORS = Path(__file__).resolve().parent / "vectors"
EXPECTED = {
    v["name"]: v
    for v in json.loads((VECTORS / "vectors.json").read_text(encoding="utf-8"))["vectors"]
}

PUBLIC = {
    "Fingerprint",
    "GGUFKitError",
    "IntotoError",
    "KeyChange",
    "Manifest",
    "ManifestError",
    "ModelError",
    "Report",
    "TensorChange",
    "TensorFingerprint",
    "UnsupportedModelError",
    "__version__",
    "create_manifest",
    "fingerprint_file",
    "load_manifest",
    "save_manifest",
    "verify_file",
}


def test_the_public_names_are_exactly_the_documented_ones() -> None:
    assert set(ggufkit.__all__) == PUBLIC
    for name in PUBLIC:
        assert getattr(ggufkit, name) is not None
        assert name in dir(ggufkit)
    with pytest.raises(AttributeError):
        _ = ggufkit.no_such_name  # type: ignore[attr-defined]


def test_import_ggufkit_loads_nothing_heavy() -> None:
    code = "import sys, ggufkit; print(sorted(m for m in ('numpy', 'gguf') if m in sys.modules))"
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == "[]"


def test_every_error_is_a_ggufkit_error_and_a_value_error() -> None:
    for name in ("ModelError", "UnsupportedModelError", "ManifestError", "IntotoError"):
        cls = getattr(ggufkit, name)
        assert issubclass(cls, ggufkit.GGUFKitError)
        assert issubclass(cls, ValueError)


def test_fingerprint_file_matches_the_test_vectors() -> None:
    fp = ggufkit.fingerprint_file(VECTORS / "base.gguf", per_kv=True)
    assert isinstance(fp, ggufkit.Fingerprint)
    assert fp.to_dict() == EXPECTED["base"]["fingerprint"]


def test_fingerprint_file_accepts_any_shard() -> None:
    fp = ggufkit.fingerprint_file(VECTORS / "split-00002-of-00002.gguf")
    assert fp.overall == EXPECTED["base"]["fingerprint"]["overall"]


def test_fingerprint_file_without_data() -> None:
    fp = ggufkit.fingerprint_file(VECTORS / "base.gguf", data=False, per_tensor=False)
    assert fp.weights is None
    assert fp.overall is None
    assert fp.tensors is None
    assert fp.kv == EXPECTED["base"]["fingerprint"]["kv"]


def test_verify_file_against_another_file() -> None:
    report = ggufkit.verify_file(VECTORS / "metadata-edit.gguf", against=VECTORS / "base.gguf")
    assert isinstance(report, ggufkit.Report)
    assert report.verdict == "metadata-only"
    assert not report.ok()
    assert report.ok(allow_metadata=True)
    assert [(c.key, c.status) for c in report.key_changes or []] == [("general.name", "changed")]

    swapped = ggufkit.verify_file(VECTORS / "swapped.gguf", against=VECTORS / "base.gguf")
    assert swapped.verdict == "weights-changed"
    assert not swapped.ok(allow_metadata=True)
    assert {(c.name, c.kind) for c in swapped.tensor_changes or []} == {
        ("blk.0.ffn_gate.weight", "data"),
        ("blk.0.ffn_up.weight", "data"),
    }


def test_create_save_load_and_verify_a_manifest(tmp_path: Path) -> None:
    manifest = ggufkit.create_manifest(
        VECTORS / "base.gguf", file_sha256=True, build_inputs={"quantize_cmd": "none"}
    )
    assert manifest["schema"] == "ggufkit-manifest/1"
    assert manifest["build_inputs"] == {"quantize_cmd": "none"}
    assert manifest["fingerprint"]["overall"] == EXPECTED["base"]["fingerprint"]["overall"]

    path = ggufkit.save_manifest(tmp_path / "base.manifest.json", manifest)
    loaded = ggufkit.load_manifest(path)
    assert loaded == json.loads(json.dumps(manifest))

    for source in (path, loaded):
        report = ggufkit.verify_file(VECTORS / "base.gguf", manifest=source, file_sha256=True)
        assert report.verdict == "identical"
        assert report.files == "match"
        assert report.ok()

    moved = ggufkit.verify_file(VECTORS / "reordered.gguf", manifest=path, file_sha256=True)
    assert moved.verdict == "identical"
    assert moved.files == "mismatch"  # same model, but not the file the manifest recorded
    assert not moved.ok()


def test_verify_file_needs_exactly_one_reference() -> None:
    base = VECTORS / "base.gguf"
    with pytest.raises(TypeError, match="exactly one"):
        ggufkit.verify_file(base)
    with pytest.raises(TypeError, match="exactly one"):
        ggufkit.verify_file(base, against=base, manifest={"schema": "x"})
    with pytest.raises(TypeError, match="file_sha256 needs manifest"):
        ggufkit.verify_file(base, against=base, file_sha256=True)


def test_errors_for_bad_input(tmp_path: Path) -> None:
    junk = tmp_path / "junk.gguf"
    junk.write_bytes(b"not a model")
    with pytest.raises(ggufkit.ModelError):
        ggufkit.fingerprint_file(junk)
    with pytest.raises(ggufkit.UnsupportedModelError):
        ggufkit.fingerprint_file(VECTORS / "big-endian-stored-as-is.gguf")
    bad = tmp_path / "bad.manifest.json"
    bad.write_text("{}")
    with pytest.raises(ggufkit.ManifestError):
        ggufkit.load_manifest(bad)
    with pytest.raises(ggufkit.GGUFKitError):
        ggufkit.verify_file(VECTORS / "base.gguf", manifest=bad)
