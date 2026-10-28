"""Hostile and malformed inputs must fail fast, cleanly, with exit status 2."""

from __future__ import annotations

import json
import struct
import time
from typing import TYPE_CHECKING

import numpy as np
import pytest
from gguf import GGMLQuantizationType as Q
from gguf import GGUFReader, GGUFValueType, GGUFWriter

from ggufkit import stitch as stitch_mod
from ggufkit.cli import main
from ggufkit.model import ModelError, check_header, open_model
from tests.conftest import det_floats, fingerprint_of, tiny_kv, write_gguf

if TYPE_CHECKING:
    from pathlib import Path

V = GGUFValueType


def gguf_string(text: str | bytes) -> bytes:
    raw = text.encode() if isinstance(text, str) else text
    return struct.pack("<Q", len(raw)) + raw


def raw_gguf(entries: list[bytes], *, n_kv: int | None = None, n_tensors: int = 0) -> bytes:
    count = len(entries) if n_kv is None else n_kv
    return b"GGUF" + struct.pack("<IQQ", 3, n_tensors, count) + b"".join(entries)


def kv_u32(key: str, value: int) -> bytes:
    return gguf_string(key) + struct.pack("<II", V.UINT32, value)


def malformed_files(tmp_path: Path, good: Path) -> dict[str, Path]:
    data = good.read_bytes()
    cases = {
        "empty": b"",
        "not-gguf": b"PK\x03\x04" + bytes(60),
        "truncated-header": data[:40],
        "truncated-metadata": data[: len(data) // 3],
        # A string array that claims 2**40 items in a 49-byte file: gguf-py loops on this.
        "huge-array": raw_gguf(
            [gguf_string("a") + struct.pack("<IIQ", V.ARRAY, V.STRING, 1 << 40) + b"\x00"]
        ),
        "huge-string": raw_gguf([gguf_string("a") + struct.pack("<IQ", V.STRING, 1 << 50)]),
        "huge-counts": raw_gguf([], n_kv=1 << 60),
        "huge-unit8-array": raw_gguf([gguf_string("a") + struct.pack("<IIQ", V.ARRAY, V.UINT8, 1 << 40)]),
        "unknown-type": raw_gguf([gguf_string("a") + struct.pack("<I", 99) + b"\x00" * 8]),
        "duplicate-key": raw_gguf([kv_u32("general.x", 1), kv_u32("general.x", 2)]),
    }
    paths = {}
    for name, blob in cases.items():
        path = tmp_path / f"{name}.gguf"
        path.write_bytes(blob)
        paths[name] = path
    return paths


@pytest.fixture
def bad_files(tmp_path: Path, tiny_gguf: Path) -> dict[str, Path]:
    return malformed_files(tmp_path, tiny_gguf)


def commands(bad: Path, good: Path, out: Path) -> list[list[str]]:
    return [
        ["fingerprint", str(bad)],
        ["manifest", str(bad), "-o", "-"],
        ["verify", str(bad), "--against", str(good)],
        ["diff", str(good), str(bad)],
        ["stitch", "--metadata", str(bad), "-o", str(out / "s.gguf")],
        ["stitch", "--metadata", str(good), "--tensors", str(bad), "-o", str(out / "t.gguf")],
        ["viz", "tensor", str(bad), "x", "-o", str(out / "x.png")],
        ["viz", "error", str(good), str(bad), "x", "-o", str(out / "e.png")],
        ["viz", "layers", str(bad)],
    ]


def test_malformed_files_exit_2_without_traceback(
    bad_files: dict[str, Path],
    tiny_gguf: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    for name, bad in bad_files.items():
        for argv in commands(bad, tiny_gguf, out):
            start = time.perf_counter()
            assert main(argv) == 2, (name, argv)
            assert time.perf_counter() - start < 5, (name, argv)
            err = capsys.readouterr().err
            assert err.splitlines()[-1].startswith("error:"), (name, argv, err)
            assert "Traceback" not in err
    assert not list(out.iterdir()), "a failed command left a file behind"


def test_header_check_rejects_before_the_reader_runs(bad_files: dict[str, Path]) -> None:
    for name in ("huge-array", "huge-string", "huge-counts","huge-uint8-array", "unknown-type", "truncated-header"):
        with pytest.raises(ModelError, match="not a valid GGUF file"):
            check_header(bad_files[name])


def test_header_check_accepts_valid_files(tmp_path: Path) -> None:
    from gguf import GGUFEndian

    for endian in (GGUFEndian.LITTLE, GGUFEndian.BIG):
        nested = ([["a", "b"], ["c"]], V.ARRAY, V.ARRAY)
        kv = {**tiny_kv(), "x.nested": nested} if endian == GGUFEndian.LITTLE else None
        path = write_gguf(
            tmp_path / f"{endian.name}.gguf",
            kv=kv or tiny_kv(),
            tensors={"w": (det_floats((4, 8)), Q.F32)},
            endianess=endian,
        )[0]
        check_header(path)


def test_shard_numbers_outside_the_count_are_errors(tmp_path: Path, tiny_gguf: Path) -> None:
    for name in ("m-00001-of-00000.gguf", "m-00005-of-00003.gguf"):
        path = tmp_path / name
        path.write_bytes(tiny_gguf.read_bytes())
        with pytest.raises(ModelError, match="not a valid shard number"):
            open_model(path)


def test_paths_and_patterns_that_cannot_work_exit_2(
    tiny_gguf: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    under_a_file = tiny_gguf / "x.gguf"
    assert main(["fingerprint", str(under_a_file)]) == 2
    assert main(["fingerprint", str(tmp_path)]) == 2
    regex = ["viz", "tensor", str(tiny_gguf), "(", "--match-regex", "-o", str(tmp_path / "a.png")]
    assert main(regex) == 2
    torch = ["viz", "tensor", str(tmp_path / "m.pth"), "w", "--model-type", "torch"]
    assert main([*torch, "-o", str(tmp_path / "t.png")]) == 2
    assert "Traceback" not in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["fingerprint", "MODEL", "-j", "0"],
        ["verify", "MODEL", "--against", "MODEL", "-j", "-1"],
        ["diff", "MODEL", "MODEL", "--top", "-3"],
        ["viz", "tensor", "MODEL", "w", "--adjust-1d-rows", "0", "-o", "x.png"],
        ["viz", "tensor", "MODEL", "w", "--scale", "-1", "-o", "x.png"],
        ["viz", "error", "MODEL", "MODEL", "w", "--max-pixels", "-5", "-o", "x.png"],
    ],
)
def test_invalid_numbers_are_rejected_by_argparse(argv: list[str], tiny_gguf: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        main([str(tiny_gguf) if a == "MODEL" else a for a in argv])
    assert exc.value.code == 2


def test_cell_size_must_be_positive(tiny_gguf: Path) -> None:
    assert main(["viz", "layers", str(tiny_gguf), "--cell", "0x0"]) == 2


def test_malformed_manifests_exit_2(tiny_gguf: Path, tmp_path: Path) -> None:
    good = tmp_path / "good.json"
    assert main(["manifest", str(tiny_gguf), "-o", str(good), "-q"]) == 0
    manifest = json.loads(good.read_text())
    broken = [
        {**manifest, "fingerprint": {"spec": "ggufkit-fp/1"}},
        {**manifest, "fingerprint": {**manifest["fingerprint"], "tensors": {"x": 3}}},
        {**manifest, "files": "nope"},
    ]
    for i, m in enumerate(broken):
        path = tmp_path / f"bad{i}.json"
        path.write_text(json.dumps(m))
        assert main(["verify", str(tiny_gguf), "--manifest", str(path), "-q"]) == 2, i


def test_fingerprint_output_needs_force(
    tiny_gguf: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "fp.json"
    out.write_text("keep me")
    assert main(["fingerprint", str(tiny_gguf), "-o", str(out)]) == 2
    assert out.read_text() == "keep me"
    assert main(["fingerprint", str(tiny_gguf), "-o", str(out), "--force"]) == 0
    assert json.loads(out.read_text())["spec"] == "ggufkit-fp/1"
    assert "already exists" in capsys.readouterr().err


def test_stitch_keeps_a_non_default_alignment_readable(tmp_path: Path) -> None:
    src = tmp_path / "aligned.gguf"
    writer = GGUFWriter(src, "llama", use_temp_file=False)
    writer.add_custom_alignment(64)
    writer.add_name("aligned")
    writer.add_tensor("a.weight", det_floats((3, 5), 1))
    writer.add_tensor("b.weight", det_floats((16,), 2))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    out = tmp_path / "out.gguf"
    assert (
        main(["stitch", "--metadata", str(src), "--tensors", str(src), "-o", str(out), "-q"]) == 0
    )
    reader = GGUFReader(out)
    assert reader.alignment == 64
    np.testing.assert_array_equal(
        reader.tensors[1].data, GGUFReader(src).tensors[1].data, strict=False
    )
    assert fingerprint_of(out).weights == fingerprint_of(src).weights


def test_a_failed_stitch_leaves_nothing_behind(
    tiny_gguf: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args: object) -> None:
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr(stitch_mod, "_write", boom)
    out = tmp_path / "out.gguf"
    assert main(["stitch", "--metadata", str(tiny_gguf), "-o", str(out), "-q"]) == 2
    assert list(tmp_path.iterdir()) == [tiny_gguf]


def test_diff_json_never_contains_nan(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    zeros = np.zeros((4, 8), dtype=np.float32)
    a = write_gguf(tmp_path / "a.gguf", kv=tiny_kv(), tensors={"w": (zeros, Q.F32)})[0]
    b = write_gguf(tmp_path / "b.gguf", kv=tiny_kv(), tensors={"w": (det_floats((4, 8)), Q.F32)})[0]
    assert main(["diff", str(a), str(b), "--numeric", "--json"]) == 1
    text = capsys.readouterr().out

    def refuse(token: str) -> None:
        msg = f"invalid JSON constant {token}"
        raise AssertionError(msg)

    result = json.loads(text, parse_constant=refuse)
    assert result["numeric"][0]["rel_l2"] is None


def test_diff_with_nothing_to_compare_is_an_error(tiny_gguf: Path) -> None:
    assert main(["diff", str(tiny_gguf), str(tiny_gguf), "--no-kv", "--no-tensors"]) == 2


def test_interrupt_and_broken_pipe(tiny_gguf: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ggufkit import fingerprint as fp_mod

    def interrupted(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(fp_mod, "fingerprint", interrupted)
    assert main(["fingerprint", str(tiny_gguf)]) == 130

    def pipe(*_args: object, **_kwargs: object) -> None:
        raise BrokenPipeError

    monkeypatch.setattr(fp_mod, "fingerprint", pipe)
    assert main(["fingerprint", str(tiny_gguf)]) == 141


def test_show_with_takes_a_command_line(
    tiny_gguf: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ggufkit.viz import render

    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(render.subprocess, "call", lambda argv: calls.append(tuple(argv)) or 0)
    out = tmp_path / "q.png"
    argv = ["viz", "tensor", str(tiny_gguf), "blk.0.attn_q.weight", "-o", str(out)]
    assert main([*argv, "--show-with", "open -a Preview", "-q"]) == 0
    assert calls == [("open", "-a", "Preview", str(out))]


def test_progress_goes_to_stderr(
    tiny_gguf: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "s.gguf"
    assert main(["stitch", "--metadata", str(tiny_gguf), "-o", str(out)]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Metadata source" in captured.err

    assert (
        main(
            [
                "viz",
                "tensor",
                str(tiny_gguf),
                "blk.*",
                "--match-glob",
                "-o",
                str(tmp_path / "i.png"),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Rendering" in captured.err
