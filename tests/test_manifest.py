from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from gguf import GGMLQuantizationType as Q
from gguf import GGUFEndian

from ggufkit import canon
from ggufkit.cli import main
from ggufkit.manifest import (
    SCHEMA,
    ManifestError,
    build_manifest,
    default_manifest_path,
    extract_provenance,
    file_type_label,
    load_manifest,
    parse_build_inputs,
    provenance_gaps,
)
from ggufkit.model import open_model
from tests.conftest import (
    det_floats,
    fingerprint_of,
    provenance_kv,
    tiny_kv,
    tiny_tensors,
    write_gguf,
)


def test_manifest_shape(tiny_gguf: Path) -> None:
    with open_model(tiny_gguf) as model:
        m = build_manifest(model, file_sha256=True, build_inputs={"llama_cpp_commit": "abc123"})
    assert m["schema"] == SCHEMA
    assert m["tool"]["name"] == "ggufkit"
    assert m["tool"]["fingerprint_spec"] == canon.SPEC_ID
    assert m["files"][0]["path"] == "tiny.gguf"
    assert m["files"][0]["sha256"] == hashlib.sha256(tiny_gguf.read_bytes()).hexdigest()
    assert m["files"][0]["shard"] is None
    assert m["gguf"] == {
        "version": 3,
        "endianness": "little",
        "alignment": 32,
        "n_kv": len(tiny_kv()) + 1,
        "n_tensors": len(tiny_tensors()),
        "n_shards": 1,
    }
    assert m["model"]["architecture"] == "llama"
    assert m["model"]["file_type"] == "F16"
    assert m["model"]["file_type_id"] == 1
    assert m["model"]["parameter_count"] == sum(a.size for a, _ in tiny_tensors().values())
    assert m["model"]["tensor_type_counts"] == {"F32": 2, "F16": 1, "Q8_0": 1, "Q4_0": 1, "Q4_K": 1}
    assert m["build_inputs"] == {"llama_cpp_commit": "abc123"}
    assert m["fingerprint"] == fingerprint_of(tiny_gguf).to_dict()
    assert m["provenance"]["name"] == "tiny"
    assert "general.source.*" in m["provenance_gaps"]
    assert "build_inputs.llama_cpp_commit" not in m["provenance_gaps"]
    assert m["provenance_score"]["expected"] == 9
    assert m["provenance_score"]["recorded"] == 9 - len(m["provenance_gaps"])


def test_manifest_reports_big_endian(tmp_path: Path) -> None:
    path = write_gguf(
        tmp_path / "be.gguf",
        kv=tiny_kv(),
        tensors={"w": (det_floats((4, 8)), Q.F32)},
        endianess=GGUFEndian.BIG,
    )[0]
    with open_model(path) as model:
        manifest = build_manifest(model, data=False)
    assert manifest["gguf"]["endianness"] == "big"


def test_provenance_extraction_and_gaps(tmp_path: Path) -> None:
    kv = {**tiny_kv(), **provenance_kv()}
    path = write_gguf(tmp_path / "doc.gguf", kv=kv, tensors=tiny_tensors())[0]
    with open_model(path) as model:
        prov = extract_provenance(model)
    assert prov["uuid"] == provenance_kv()["general.uuid"]
    assert prov["quantized_by"] == "tester"
    assert prov["license"] == {"id": "apache-2.0", "link": "https://example.org/license"}
    assert prov["source"] == {
        "url": "https://example.org/source",
        "repo_url": "https://huggingface.co/org/model",
    }
    assert prov["base_models"] == [
        {"name": "Base Model", "repo_url": "https://huggingface.co/org/base"}
    ]
    assert prov["datasets"] == [{"name": "calib"}]
    assert prov["imatrix"] == {
        "file": "imatrix.dat",
        "dataset": "calibration.txt",
        "entries_count": 196,
        "chunks_count": 128,
    }
    assert prov["tags"] == ["chat", "test"]
    gaps = provenance_gaps(prov, {}, quantized=True)
    assert gaps == [
        "build_inputs.source_sha256",
        "build_inputs.imatrix_sha256",
        "build_inputs.llama_cpp_commit",
        "build_inputs.quantize_cmd",
    ]
    full = dict.fromkeys(
        ("source_sha256", "imatrix_sha256", "llama_cpp_commit", "quantize_cmd"), "x"
    )
    assert provenance_gaps(prov, full, quantized=True) == []


def test_unquantized_models_do_not_expect_an_imatrix(tmp_path: Path) -> None:
    path = write_gguf(
        tmp_path / "f16.gguf",
        kv=tiny_kv(),
        tensors={
            "a.weight": (det_floats((4, 8)), Q.F16),
            "b.weight": (det_floats((4, 64)), Q.Q8_0),
        },
    )[0]
    with open_model(path) as model:
        m = build_manifest(model)
    assert "quantize.imatrix.dataset" not in m["provenance_gaps"]
    assert "build_inputs.imatrix_sha256" not in m["provenance_gaps"]
    assert m["provenance_score"]["expected"] == 7


def test_imatrix_data_file_keys_also_count(tmp_path: Path) -> None:
    kv = {**tiny_kv(), "imatrix.datasets": ["calibration.txt"], "imatrix.chunk_count": 64}
    path = write_gguf(tmp_path / "legacy.gguf", kv=kv, tensors=tiny_tensors())[0]
    with open_model(path) as model:
        m = build_manifest(model, data=False)
    assert m["provenance"]["imatrix"] == {"datasets": ["calibration.txt"], "chunk_count": 64}
    assert "quantize.imatrix.dataset" not in m["provenance_gaps"]


def test_missing_imatrix_is_a_gap_for_quantized_files(tiny_gguf: Path) -> None:
    with open_model(tiny_gguf) as model:
        m = build_manifest(model, data=False)
    assert "quantize.imatrix.dataset" in m["provenance_gaps"]


def test_file_type_labels() -> None:
    assert file_type_label(0) == "F32"
    assert file_type_label(1) == "F16"
    assert file_type_label(15) == "Q4_K_M"
    assert file_type_label(None) is None
    assert file_type_label(99999) == "unknown(99999)"


def test_sharded_manifest_lists_every_shard(tmp_path: Path) -> None:
    shards = write_gguf(
        tmp_path / "m.gguf", kv=tiny_kv(), tensors=tiny_tensors(), split_max_tensors=4
    )
    with open_model(shards[1]) as model:
        m = build_manifest(model, file_sha256=True)
    assert [f["path"] for f in m["files"]] == [p.name for p in shards]
    assert [f["shard"] for f in m["files"]] == [{"no": 1, "count": 2}, {"no": 2, "count": 2}]
    assert m["gguf"]["n_shards"] == 2
    assert default_manifest_path(shards[1]) == tmp_path / "m.manifest.json"
    assert default_manifest_path(tmp_path / "x.gguf") == tmp_path / "x.manifest.json"


def test_parse_build_inputs() -> None:
    assert parse_build_inputs(["a=1", " b = two words "]) == {"a": "1", "b": "two words"}
    assert parse_build_inputs(None) == {}
    with pytest.raises(ManifestError):
        parse_build_inputs(["novalue"])


def test_load_manifest_rejects_other_files(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    with pytest.raises(ManifestError, match="not a"):
        load_manifest(bad)
    bad.write_text("not json")
    with pytest.raises(ManifestError, match="Cannot read"):
        load_manifest(bad)
    with pytest.raises(ManifestError):
        load_manifest(tmp_path / "missing.json")


def test_manifest_cli_writes_default_path(
    tiny_gguf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(["manifest", str(tiny_gguf), "--input", "quantize_cmd=llama-quantize x y Q4_K_M"]) == 0
    )
    out = capsys.readouterr().out
    target = tiny_gguf.with_name("tiny.manifest.json")
    assert target.is_file()
    assert "score" in out
    assert "gaps" in out
    m = load_manifest(target)
    assert m["build_inputs"] == {"quantize_cmd": "llama-quantize x y Q4_K_M"}
    assert main(["manifest", str(tiny_gguf)]) == 2  # exists without --force
    assert main(["manifest", str(tiny_gguf), "--force", "-q"]) == 0


def test_manifest_cli_stdout_only(tiny_gguf: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["manifest", str(tiny_gguf), "-o", "-", "--no-data"]) == 0
    m = json.loads(capsys.readouterr().out)
    assert m["schema"] == SCHEMA
    assert m["fingerprint"]["weights"] is None
    assert not tiny_gguf.with_name("tiny.manifest.json").exists()


def test_intoto_statement(tmp_path: Path) -> None:
    from ggufkit.manifest import INTOTO_STATEMENT_TYPE, PREDICATE_TYPE, to_intoto_statement

    shards = write_gguf(
        tmp_path / "m.gguf", kv=tiny_kv(), tensors=tiny_tensors(), split_max_tensors=4
    )
    with open_model(shards[0]) as model:
        manifest = build_manifest(model, file_sha256=True)
    st = to_intoto_statement(manifest)
    assert st["_type"] == INTOTO_STATEMENT_TYPE
    assert st["predicateType"] == PREDICATE_TYPE
    assert st["predicate"] == manifest
    assert st["subject"] == [
        {"name": p.name, "digest": {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()}}
        for p in shards
    ]


def test_intoto_needs_file_hashes(tiny_gguf: Path) -> None:
    from ggufkit.manifest import IntotoError, to_intoto_statement

    with open_model(tiny_gguf) as model:
        manifest = build_manifest(model, data=False)
    with pytest.raises(IntotoError, match="--file-sha256"):
        to_intoto_statement(manifest)


def test_manifest_cli_intoto_implies_file_hashes(
    tiny_gguf: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    statement = tmp_path / "tiny.intoto.json"
    manifest_path = tmp_path / "tiny.manifest.json"
    rc = main(["manifest", str(tiny_gguf), "-o", str(manifest_path), "--intoto", str(statement)])
    assert rc == 0
    assert "in-toto" in capsys.readouterr().out
    st = json.loads(statement.read_text())
    digest = hashlib.sha256(tiny_gguf.read_bytes()).hexdigest()
    assert st["subject"] == [{"name": "tiny.gguf", "digest": {"sha256": digest}}]
    assert load_manifest(manifest_path)["files"][0]["sha256"] == digest
    assert (
        main(
            [
                "manifest",
                str(tiny_gguf),
                "-o",
                str(tmp_path / "other.json"),
                "--intoto",
                str(statement),
            ]
        )
        == 2
    )
