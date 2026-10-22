from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from gguf import GGMLQuantizationType as Q
from gguf import GGUFEndian

from ggufkit.cli import main
from ggufkit.dequant import can_dequantize, to_f32
from ggufkit.model import open_model
from ggufkit.viz.render import GGUFSource, RenderOptions, downsample, make_image, select_names
from tests.conftest import det_floats, tiny_kv, write_gguf

pytest.importorskip("PIL")


def test_dequantize_every_fixture_type(tiny_gguf: Path) -> None:
    with open_model(tiny_gguf) as model:
        for t in model.tensors:
            assert can_dequantize(t.tensor_type)
            arr = to_f32(t, model.endianess)
            assert arr.dtype == np.float32
            assert arr.shape == tuple(reversed([int(d) for d in t.shape]))


def test_dequantize_rows_matches_full(tiny_gguf: Path) -> None:
    with open_model(tiny_gguf) as model:
        t = model.tensor("blk.0.attn_q.weight")
        full = to_f32(t, model.endianess)
        part = to_f32(t, model.endianess, rows=slice(10, 20))
        assert np.array_equal(part, full[10:20])


def test_dequantize_big_endian_element_tensor(tmp_path: Path) -> None:
    original = det_floats((8, 8), 22)
    path = write_gguf(
        tmp_path / "be.gguf",
        tensors={"w": (original, Q.F32)},
        endianess=GGUFEndian.BIG,
    )[0]
    with open_model(path) as model:
        decoded = to_f32(model.tensor("w"), model.endianess)
    assert np.array_equal(decoded, original)


def test_q8_0_round_trip_is_close(tiny_gguf: Path) -> None:
    with open_model(tiny_gguf) as model:
        t = model.tensor("blk.0.attn_q.weight")
        decoded = to_f32(t, model.endianess)
    original = det_floats((64, 64), 12)
    assert np.abs(decoded - original).max() < 0.01


def test_make_image_dimensions() -> None:
    img = make_image(det_floats((12, 40)), RenderOptions())
    assert img.size == (40, 12)
    img = make_image(det_floats((40,)), RenderOptions(adjust_1d_rows=4))
    assert img.size == (10, 4)
    img = make_image(det_floats((40,)), RenderOptions())
    assert img.size == (40, 1)
    img = make_image(det_floats((12, 40)), RenderOptions(scale=0.5))
    assert img.size == (20, 6)


@pytest.mark.parametrize("mode", ["devs-overall", "devs-rows", "devs-cols"])
def test_every_mode_renders(mode: str) -> None:
    img = make_image(det_floats((8, 16)), RenderOptions(mode=mode))
    assert img.size == (16, 8)


def test_constant_tensor_does_not_crash() -> None:
    img = make_image(np.zeros((4, 4), dtype=np.float32), RenderOptions())
    assert img.size == (4, 4)


def test_downsample_respects_pixel_budget() -> None:
    td = det_floats((100, 100))
    small = downsample(td, 1000)
    assert small.size <= 1000
    assert downsample(td, 0) is td


def test_select_names_glob_and_regex(tiny_gguf: Path) -> None:
    src = GGUFSource(tiny_gguf)
    try:
        assert select_names(src, ["blk.0.attn_*"], glob=True) == [
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_norm.weight",
        ]
        assert select_names(src, [r"norm\.weight$"], regex=True) == [
            "blk.0.attn_norm.weight",
            "output_norm.weight",
        ]
        skipped: list[str] = []
        assert select_names(src, ["token_embd.weight", "missing"], log=skipped.append) == [
            "token_embd.weight"
        ]
        assert skipped
        assert "missing" in skipped[0]
    finally:
        src.close()


def test_cli_renders_several_tensors(tiny_gguf: Path, tmp_path: Path) -> None:
    out = tmp_path / "imgs" / "t.png"
    rc = main(
        [
            "viz",
            "tensor",
            str(tiny_gguf),
            "blk.0.*",
            "--match-glob",
            "-o",
            str(out),
            "--quiet",
        ]
    )
    assert rc == 0
    written = sorted(p.name for p in out.parent.iterdir())
    assert written == [
        "blk.0.attn_k.weight.t.png",
        "blk.0.attn_q.weight.t.png",
        "blk.0.ffn_down.weight.t.png",
    ]


def test_cli_single_tensor_and_force(tiny_gguf: Path, tmp_path: Path) -> None:
    out = tmp_path / "q.png"
    args = ["viz", "tensor", str(tiny_gguf), "blk.0.attn_q.weight", "-o", str(out), "--quiet"]
    assert main(args) == 0
    assert out.is_file()
    assert main(args) == 2  # exists, no --force
    assert main([*args, "--force"]) == 0


def test_cli_requires_an_output(tiny_gguf: Path) -> None:
    assert main(["viz", "tensor", str(tiny_gguf), "blk.0.attn_q.weight"]) == 2


def test_cli_reports_no_match(tiny_gguf: Path, tmp_path: Path) -> None:
    assert (
        main(
            [
                "viz",
                "tensor",
                str(tiny_gguf),
                "nothing.*",
                "--match-glob",
                "-o",
                str(tmp_path / "x.png"),
            ]
        )
        == 2
    )


def test_big_tensor_is_pooled(tmp_path: Path) -> None:
    path = write_gguf(
        tmp_path / "big.gguf", kv=tiny_kv(), tensors={"big.weight": (det_floats((200, 300)), Q.F16)}
    )[0]
    out = tmp_path / "big.png"
    assert (
        main(
            [
                "viz",
                "tensor",
                str(path),
                "big.weight",
                "-o",
                str(out),
                "--max-pixels",
                "2500",
                "--quiet",
            ]
        )
        == 0
    )
    from PIL import Image

    with Image.open(out) as img:
        assert img.size[0] * img.size[1] <= 2500
