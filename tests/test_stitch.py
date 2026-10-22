from __future__ import annotations

from pathlib import Path

import pytest
from gguf import GGMLQuantizationType as Q
from gguf import GGUFEndian

from ggufkit.canon import UnsupportedModelError
from ggufkit.model import open_model
from ggufkit.stitch import stitch
from tests.conftest import det_floats, fingerprint_of, tiny_kv, tiny_tensors, write_gguf


def test_stitch_takes_metadata_from_one_and_tensors_from_the_other(tmp_path: Path) -> None:
    kv_a = tiny_kv()
    kv_a["general.name"] = "model-a"
    a = write_gguf(
        tmp_path / "a.gguf", kv=kv_a, tensors={"only.in.a": (det_floats((2, 8)), Q.F32)}
    )[0]
    kv_b = tiny_kv()
    kv_b["general.name"] = "model-b"
    b = write_gguf(tmp_path / "b.gguf", kv=kv_b, tensors=tiny_tensors())[0]

    out = stitch(a, b, tmp_path / "c.gguf", log=lambda _m: None)
    fa, fb, fc = fingerprint_of(a), fingerprint_of(b), fingerprint_of(out)
    assert fc.kv == fa.kv
    assert fc.tensor_info == fb.tensor_info
    assert fc.weights == fb.weights
    with open_model(out) as model:
        assert model.fields["general.name"].contents() == "model-a"
        assert [t.tensor_type for t in model.tensors] == [t for _, t in tiny_tensors().values()]
        assert (
            model.fields["tokenizer.ggml.tokens"].contents() == tiny_kv()["tokenizer.ggml.tokens"]
        )
        assert model.fields["general.description"].contents() == ""


def test_vocab_only_output_has_no_tensors(tiny_gguf: Path, tmp_path: Path) -> None:
    out = stitch(tiny_gguf, None, tmp_path / "vocab.gguf")
    with open_model(out) as model:
        assert model.tensors == []
    assert fingerprint_of(out).kv == fingerprint_of(tiny_gguf).kv


def test_stitch_refuses_to_overwrite_without_force(tiny_gguf: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.gguf"
    stitch(tiny_gguf, None, out)
    with pytest.raises(FileExistsError):
        stitch(tiny_gguf, None, out)
    stitch(tiny_gguf, None, out, force=True)


def test_stitch_refuses_to_overwrite_an_input(tiny_gguf: Path) -> None:
    with pytest.raises(ValueError, match="must differ"):
        stitch(tiny_gguf, None, tiny_gguf)


def test_stitch_refuses_big_endian_tensor_sources(tmp_path: Path) -> None:
    metadata = write_gguf(tmp_path / "metadata.gguf", kv=tiny_kv())[0]
    source = write_gguf(
        tmp_path / "be.gguf",
        tensors={"w": (det_floats((4, 8)), Q.F32)},
        endianess=GGUFEndian.BIG,
    )[0]
    with pytest.raises(UnsupportedModelError, match="Big-endian tensor sources are not supported"):
        stitch(metadata, source, tmp_path / "out.gguf")
