from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from gguf import GGMLQuantizationType as Q
from gguf import GGUFEndian

from ggufkit.cli import main
from ggufkit.diff import kv_diff, numeric_diff, summarize_value, tensor_diff
from ggufkit.model import open_model
from tests.conftest import det_floats, fingerprint_of, tiny_kv, tiny_tensors, write_gguf
from tests.test_fingerprint import flip_first_byte_of_tensor


def test_kv_diff_shows_values(tmp_path: Path) -> None:
    kv_b = tiny_kv()
    kv_b["general.name"] = "renamed"
    kv_b["general.new"] = 42
    del kv_b["general.flag"]
    kv_b["tokenizer.ggml.tokens"] = ["<s>", "a", "b", "c"]
    a = write_gguf(tmp_path / "a.gguf", kv=tiny_kv())[0]
    b = write_gguf(tmp_path / "b.gguf", kv=kv_b)[0]
    with open_model(a) as ma, open_model(b) as mb:
        entries = kv_diff(ma, mb)
        assert [(e.key, e.status) for e in entries] == [
            ("general.flag", "removed"),
            ("general.name", "changed"),
            ("general.new", "added"),
            ("tokenizer.ggml.tokens", "changed"),
        ]
        by_key = {e.key: e for e in entries}
        assert by_key["general.name"].a == "tiny"
        assert by_key["general.name"].b == "renamed"
        assert by_key["general.new"].b == 42
        assert by_key["tokenizer.ggml.tokens"].b == ["<s>", "a", "b", "c"]
        assert kv_diff(ma, mb, ignore=["general.*", "tokenizer.*"]) == []
        same = kv_diff(ma, ma, include_same=True)
        assert all(e.status == "same" for e in same)
        assert len(same) == len(tiny_kv()) + 1


def test_long_arrays_are_summarized(tmp_path: Path) -> None:
    a = write_gguf(tmp_path / "a.gguf", kv={"x.long": list(range(100))})[0]
    with open_model(a) as model:
        s = summarize_value(model.fields["x.long"])
    assert isinstance(s, str)
    assert s.startswith("<array INT32[100] ")


def test_tensor_diff_statuses(tmp_path: Path) -> None:
    tensors = tiny_tensors()
    del tensors["output_norm.weight"]
    tensors["new.weight"] = (det_floats((4, 8)), Q.F32)
    arr, _ = tensors["blk.0.attn_k.weight"]
    tensors["blk.0.attn_k.weight"] = (arr, Q.Q8_0)
    a = write_gguf(tmp_path / "a.gguf", kv=tiny_kv(), tensors=tiny_tensors())[0]
    b = write_gguf(tmp_path / "b.gguf", kv=tiny_kv(), tensors=tensors)[0]
    flip_first_byte_of_tensor(b, "token_embd.weight")
    entries = tensor_diff(fingerprint_of(a), fingerprint_of(b))
    assert [(e.name, e.status, e.what) for e in entries] == [
        ("blk.0.attn_k.weight", "changed", ("type", "data")),
        ("new.weight", "added", ()),
        ("output_norm.weight", "removed", ()),
        ("token_embd.weight", "changed", ("data",)),
    ]
    only = tensor_diff(fingerprint_of(a), fingerprint_of(b), patterns=["blk.*"])
    assert [e.name for e in only] == ["blk.0.attn_k.weight"]


def test_numeric_diff_between_precisions(tmp_path: Path) -> None:
    data = det_floats((64, 64), 5)
    a = write_gguf(tmp_path / "a.gguf", tensors={"w": (data, Q.F32)})[0]
    b = write_gguf(tmp_path / "b.gguf", tensors={"w": (data, Q.Q8_0)})[0]
    c = write_gguf(tmp_path / "c.gguf", tensors={"w": (data, Q.Q4_0)})[0]
    with open_model(a) as ma, open_model(b) as mb, open_model(c) as mc:
        same = numeric_diff(ma.tensor("w"), ma.tensor("w"), ma.endianess, ma.endianess)
        assert same.rmse == 0.0
        assert same.max_abs == 0.0
        assert same.cosine == pytest.approx(1.0)
        q8 = numeric_diff(ma.tensor("w"), mb.tensor("w"), ma.endianess, mb.endianess)
        q4 = numeric_diff(ma.tensor("w"), mc.tensor("w"), ma.endianess, mc.endianess)
        assert q8.n == 64 * 64
        assert 0 < (q8.rmse or 0) < (q4.rmse or 0)
        assert 0 < (q8.rel_l2 or 0) < (q4.rel_l2 or 0) < 0.5
        assert (q8.cosine or 0) > (q4.cosine or 0) > 0.9
        assert q8.max_abs is not None
        assert q8.max_abs < 0.01
        chunked = numeric_diff(
            ma.tensor("w"), mb.tensor("w"), ma.endianess, mb.endianess, chunk_bytes=1024
        )
        assert chunked.rmse == pytest.approx(q8.rmse)
        assert chunked.max_abs == pytest.approx(q8.max_abs)
        assert chunked.cosine == pytest.approx(q8.cosine)


def test_numeric_diff_skips_shape_mismatch(tmp_path: Path) -> None:
    a = write_gguf(tmp_path / "a.gguf", tensors={"w": (det_floats((4, 32)), Q.F32)})[0]
    b = write_gguf(tmp_path / "b.gguf", tensors={"w": (det_floats((8, 32)), Q.F32)})[0]
    with open_model(a) as ma, open_model(b) as mb:
        s = numeric_diff(ma.tensor("w"), mb.tensor("w"), ma.endianess, mb.endianess)
    assert s.skipped == "shape differs"
    assert s.rmse is None


def test_numeric_diff_one_dimensional(tmp_path: Path) -> None:
    data = det_floats((256,), 3)
    a = write_gguf(tmp_path / "a.gguf", tensors={"n": (data, Q.F32)})[0]
    b = write_gguf(tmp_path / "b.gguf", tensors={"n": (data * np.float32(1.5), Q.F32)})[0]
    with open_model(a) as ma, open_model(b) as mb:
        s = numeric_diff(ma.tensor("n"), mb.tensor("n"), ma.endianess, mb.endianess)
    assert s.rel_l2 == pytest.approx(0.5)
    assert s.cosine == pytest.approx(1.0)


def test_numeric_diff_decodes_big_endian_element_tensors(tmp_path: Path) -> None:
    data = det_floats((8, 8), 21)
    little = write_gguf(tmp_path / "little.gguf", tensors={"w": (data, Q.F32)})[0]
    big = write_gguf(
        tmp_path / "big.gguf",
        tensors={"w": (data, Q.F32)},
        endianess=GGUFEndian.BIG,
    )[0]
    with open_model(little) as ma, open_model(big) as mb:
        stats = numeric_diff(ma.tensor("w"), mb.tensor("w"), ma.endianess, mb.endianess)
    assert stats.rmse == 0.0
    assert stats.max_abs == 0.0
    assert stats.cosine == pytest.approx(1.0)


def test_diff_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    kv_b = tiny_kv()
    kv_b["general.name"] = "other"
    a = write_gguf(tmp_path / "a.gguf", kv=tiny_kv(), tensors=tiny_tensors())[0]
    b = write_gguf(tmp_path / "b.gguf", kv=kv_b, tensors=tiny_tensors())[0]
    flip_first_byte_of_tensor(b, "blk.0.attn_q.weight")

    assert main(["diff", str(a), str(a)]) == 0
    out = capsys.readouterr().out
    assert "metadata: 0 changed" in out
    assert "tensors: 0 changed" in out

    assert main(["diff", str(a), str(b)]) == 1
    out = capsys.readouterr().out
    assert "~ general.name: 'tiny' -> 'other'" in out
    assert "~ blk.0.attn_q.weight  data" in out

    assert (
        main(
            ["diff", str(a), str(b), "--no-kv", "--numeric", "--tensor", "blk.0.attn_q.*", "--json"]
        )
        == 1
    )
    result = json.loads(capsys.readouterr().out)
    assert "kv" not in result
    assert [t["name"] for t in result["tensors"]] == ["blk.0.attn_q.weight"]
    assert [n["name"] for n in result["numeric"]] == ["blk.0.attn_q.weight"]
    assert result["numeric"][0]["rmse"] > 0
    assert result["different"] is True

    assert main(["diff", str(a), str(b), "--ignore-keys", "general.name", "--no-tensors"]) == 0


def test_diff_cli_numeric_text(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = det_floats((32, 64), 9)
    a = write_gguf(tmp_path / "a.gguf", tensors={"w": (data, Q.F32), "v": (data, Q.F32)})[0]
    b = write_gguf(tmp_path / "b.gguf", tensors={"w": (data, Q.Q4_0), "v": (data, Q.F32)})[0]
    assert main(["diff", str(a), str(b), "--numeric", "--top", "1"]) == 1
    out = capsys.readouterr().out
    assert "numeric (top 1 by rel_l2):" in out
    assert "F32->Q4_0" in out
    assert "F32->F32" not in out
