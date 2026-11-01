"""``ggufkit reproduce --manifest`` and ``manifest --source/--imatrix``, with a stand-in quantizer."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

import ggufkit
from ggufkit import reproduce as reproduce_mod
from ggufkit.cli import main
from ggufkit.reproduce import QuantizeRecipe, parse_quantize_cmd
from tests.conftest import tiny_kv, tiny_tensors, write_gguf
from tests.test_reproduce import IMATRIX_KV, VERSION_LINE, FakeQuantize, model, use

CMD = "llama-quantize --imatrix calib/imatrix.dat model-BF16.gguf model-Q4_K_M.gguf Q4_K_M 8"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def fake_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        reproduce_mod.shutil,
        "which",
        lambda name: "/opt/llama/llama-quantize" if name == "llama-quantize" else None,
    )


@pytest.fixture
def source(tmp_path: Path) -> Path:
    return model(tmp_path, "source", kv={"general.file_type": 32})  # MOSTLY_BF16


@pytest.fixture
def imatrix(tmp_path: Path) -> Path:
    path = tmp_path / "downloaded-imatrix.dat"
    path.write_bytes(b"IMATRIX")
    return path


@pytest.fixture
def target(tmp_path: Path) -> Path:
    return model(tmp_path, "target", kv=IMATRIX_KV)


def make_manifest(
    target: Path,
    *,
    source: Path | None = None,
    imatrix: Path | None = None,
    file_sha256: bool = True,
    data: bool = True,
    **inputs: str,
) -> Path:
    manifest = ggufkit.create_manifest(
        target,
        file_sha256=file_sha256,
        source=source,
        imatrix=imatrix,
        build_inputs=inputs,
        data=data,
    )
    return ggufkit.save_manifest(target.with_suffix(".manifest.json"), manifest)


def reproduce_args(manifest: Path, source: Path, *more: str) -> list[str]:
    return ["reproduce", "--manifest", str(manifest), "--source", str(source), *more]


def test_round_trip(
    target: Path,
    source: Path,
    imatrix: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = make_manifest(
        target, source=source, imatrix=imatrix, quantize_cmd=CMD, llama_cpp_commit="abc1234"
    )
    fake = use(monkeypatch, FakeQuantize(target))
    args = reproduce_args(manifest, source, "--imatrix", str(imatrix))
    assert main([*args, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["verdict"] == "identical"
    assert data["byte_identical"] is True
    assert data["a"] == str(manifest)
    assert data["type"] == "Q4_K_M"
    assert data["inputs"] == {
        "source_sha256": "match",
        "imatrix_sha256": "match",
        "llama_cpp_commit": "match",
    }
    assert fake.argv[1:3] == ["--imatrix", "calib/imatrix.dat"]
    assert fake.imatrix_bytes == b"IMATRIX"
    assert fake.argv[-1] == "Q4_K_M"  # the recorded thread count is not used

    assert main(args) == 0
    out, err = capsys.readouterr()
    assert f"manifest     {manifest}" in out
    assert "source       sha256 matches the manifest" in out
    assert "imatrix      sha256 matches the manifest" in out
    assert "llama.cpp    0.9.0 (build 7, commit abc1234); manifest: same" in out
    assert "file bytes   identical" in out
    assert "checking the sha256 of the source" in err


def test_a_wrong_input_stops_before_the_rebuild(
    tmp_path: Path,
    target: Path,
    source: Path,
    imatrix: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = make_manifest(target, source=source, imatrix=imatrix, quantize_cmd=CMD)
    fake = use(monkeypatch, FakeQuantize(target))
    other = model(tmp_path, "other-source", kv={"general.name": "other"})
    assert main(reproduce_args(manifest, other, "--imatrix", str(imatrix))) == 2
    err = capsys.readouterr().err
    assert "the source is not the one the manifest names" in err
    assert sha256(other) in err
    assert sha256(source) in err

    other_imatrix = tmp_path / "other.dat"
    other_imatrix.write_bytes(b"OTHER")
    assert main(reproduce_args(manifest, source, "--imatrix", str(other_imatrix))) == 2
    assert "the importance matrix is not the one the manifest names" in capsys.readouterr().err
    assert not fake.calls


@pytest.mark.parametrize("how", ["keys", "hash", "command"])
def test_a_recorded_imatrix_is_required(
    tmp_path: Path,
    source: Path,
    imatrix: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    how: str,
) -> None:
    if how == "keys":
        manifest = make_manifest(model(tmp_path, "t", kv=IMATRIX_KV))
    elif how == "hash":
        manifest = make_manifest(model(tmp_path, "t"), imatrix=imatrix)
    else:
        manifest = make_manifest(model(tmp_path, "t"), quantize_cmd=CMD)
    fake = use(monkeypatch, FakeQuantize(None))
    assert main(reproduce_args(manifest, source)) == 2
    err = capsys.readouterr().err
    assert "was quantized with an importance matrix" in err
    assert "pass --imatrix FILE" in err
    if how == "keys":
        assert "file 'calib/imatrix.dat'" in err
    if how == "hash":
        assert f"sha256 {sha256(imatrix)}" in err
    assert not fake.calls


def test_unrecorded_inputs_are_noted(
    tmp_path: Path, source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = model(tmp_path, "plain")
    use(monkeypatch, FakeQuantize(plain))
    result = ggufkit.reproduce_file(manifest=make_manifest(plain), source=source)
    assert result.ok()
    assert result.quant_type == "Q4_K_M"  # from the manifest's model.file_type
    assert result.inputs == {
        "source_sha256": "not recorded",
        "imatrix_sha256": "not used",
        "llama_cpp_commit": "not recorded",
    }
    assert any("records no source_sha256" in n for n in result.notes)
    assert any("records no quantize_cmd" in n for n in result.notes)


def test_no_usable_type_needs_type(
    tmp_path: Path,
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    untyped = model(tmp_path, "untyped", kv={"general.file_type": None})
    manifest = make_manifest(untyped)
    use(monkeypatch, FakeQuantize(untyped))
    assert main(reproduce_args(manifest, source)) == 2
    assert "pass --type" in capsys.readouterr().err
    assert main(reproduce_args(manifest, source, "--type", "Q8_0", "-q")) == 0


def test_recipe_options_are_passed_on(
    tmp_path: Path, source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = model(tmp_path, "plain")
    cmd = (
        "CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-quantize --pure --tensor-type 'attn_v=q8_0' "
        "--output-tensor-type q8_0 --max-buffer-size 512 in.gguf out.gguf Q4_K_S 16"
    )
    fake = use(monkeypatch, FakeQuantize(plain))
    result = ggufkit.reproduce_file(manifest=make_manifest(plain, quantize_cmd=cmd), source=source)
    assert fake.options == [
        "--pure",
        "--tensor-type",
        "attn_v=q8_0",
        "--output-tensor-type",
        "q8_0",
    ]
    assert result.quant_type == "Q4_K_S"
    assert fake.argv[-1] == "Q4_K_S"

    ggufkit.reproduce_file(
        manifest=make_manifest(plain, quantize_cmd=cmd), source=source, quant_type="Q8_0"
    )
    assert fake.argv[-1] == "Q8_0"


@pytest.mark.parametrize(
    ("cmd", "expected"),
    [
        ("llama-quantize in.gguf Q4_K_M", QuantizeRecipe((), "Q4_K_M", None)),
        ("llama-quantize in.gguf out.gguf q4_k_m", QuantizeRecipe((), "q4_k_m", None)),
        ("llama-quantize in.gguf out.gguf Q4_K_M 8", QuantizeRecipe((), "Q4_K_M", None)),
        ("llama-quantize in.gguf Q8_0 8", QuantizeRecipe((), "Q8_0", None)),
        ("llama-quantize in.gguf 15", QuantizeRecipe((), "15", None)),
        ("llama-quantize in.gguf out.gguf 15", QuantizeRecipe((), "15", None)),
        ("llama-quantize in.gguf Q4_K", QuantizeRecipe((), "Q4_K", None)),
        ("--imatrix i.dat in.gguf Q4_K_M", QuantizeRecipe((), "Q4_K_M", "i.dat")),
        (
            "llama-quantize --leave-output-tensor in.gguf out.gguf",
            QuantizeRecipe(("--leave-output-tensor",), None, None),
        ),
        ("", QuantizeRecipe()),
    ],
)
def test_parse_quantize_cmd(cmd: str, expected: QuantizeRecipe) -> None:
    assert parse_quantize_cmd(cmd) == expected


@pytest.mark.parametrize(
    ("cmd", "message"),
    [
        ("llama-quantize --tensor-type-file t.txt in.gguf Q4_K_M", "reads a file the manifest"),
        ("llama-quantize --keep-split in.gguf Q4_K_M", "split output"),
        ("llama-quantize --dry-run in.gguf Q4_K_M", "writes no file"),
        ("llama-quantize --tensor-type=attn_v=q8_0 in.gguf Q4_K_M", "does not know"),
        ("llama-quantize --bogus in.gguf Q4_K_M", "does not know"),
        ("llama-quantize --tensor-type", "needs a value"),
        ("llama-quantize 'in.gguf Q4_K_M", "cannot be parsed"),
    ],
)
def test_quantize_cmd_errors(cmd: str, message: str) -> None:
    with pytest.raises(ggufkit.ReproduceError, match=message):
        parse_quantize_cmd(cmd)


def test_an_unusable_quantize_cmd_is_an_error_before_anything_runs(
    tmp_path: Path,
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plain = model(tmp_path, "plain")
    manifest = make_manifest(plain, quantize_cmd="llama-quantize --bogus in.gguf Q4_K_M")
    fake = use(monkeypatch, FakeQuantize(plain))
    assert main(reproduce_args(manifest, source)) == 2
    assert "uses --bogus" in capsys.readouterr().err
    assert not fake.calls


@pytest.mark.parametrize(
    ("recorded", "output", "status"),
    [
        ("abc1234", VERSION_LINE, "match"),
        ("ABC1234deadbeef", VERSION_LINE, "match"),
        ("1111111", VERSION_LINE, "differs"),
        ("b7", VERSION_LINE, "match"),
        ("b8", VERSION_LINE, "differs"),
        ("v1.0", VERSION_LINE, "not compared"),
        ("abc1234", "no version line", "not compared"),
    ],
)
def test_llama_cpp_commit(
    tmp_path: Path,
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
    recorded: str,
    output: str,
    status: str,
) -> None:
    plain = model(tmp_path, "plain")
    use(monkeypatch, FakeQuantize(plain, output=output))
    manifest = make_manifest(plain, llama_cpp_commit=recorded)
    result = ggufkit.reproduce_file(manifest=manifest, source=source)
    assert result.inputs["llama_cpp_commit"] == status
    assert result.ok()  # a different llama.cpp is a note, not a failure
    if status == "differs":
        assert any(f"names llama.cpp {recorded}" in n for n in result.notes)


def test_a_manifest_without_weight_digests(
    tmp_path: Path, source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = model(tmp_path, "plain")
    use(monkeypatch, FakeQuantize(plain))
    blind = make_manifest(plain, data=False, file_sha256=False)
    result = ggufkit.reproduce_file(manifest=blind, source=source)
    assert result.report.weights_same is None
    assert result.byte_identical is None
    assert not result.ok()

    hashed = make_manifest(plain, data=False, file_sha256=True)
    result = ggufkit.reproduce_file(manifest=hashed, source=source)
    assert result.byte_identical is True
    assert result.ok()  # identical bytes settle it


def test_bytes_against_the_manifest(
    tmp_path: Path, source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = model(tmp_path, "plain")
    reordered = model(tmp_path, "reordered", tensors=dict(reversed(tiny_tensors().items())))
    use(monkeypatch, FakeQuantize(reordered))
    result = ggufkit.reproduce_file(manifest=make_manifest(plain), source=source)
    assert result.report.verdict == "identical"
    assert result.byte_identical is False
    assert any("laid out differently" in n for n in result.notes)

    split = model(tmp_path, "split", split_max_tensors=3)
    result = ggufkit.reproduce_file(manifest=make_manifest(split), source=source)
    assert result.byte_identical is None
    assert any("lists 2 files" in n for n in result.notes)


def test_a_split_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plain = model(tmp_path, "plain")
    shards = write_gguf(
        tmp_path / "src.gguf", kv=tiny_kv(), tensors=tiny_tensors(), split_max_tensors=3
    )
    manifest = make_manifest(plain, source=shards[1])
    recorded = json.loads(manifest.read_text())["build_inputs"]["source_sha256"]
    assert recorded == ",".join(sha256(p) for p in shards)
    fake = use(monkeypatch, FakeQuantize(plain))
    result = ggufkit.reproduce_file(manifest=manifest, source=shards[0])
    assert result.inputs["source_sha256"] == "match"
    assert fake.argv[1] == str(shards[0].absolute())


def test_target_or_manifest(target: Path, source: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest = make_manifest(target)
    assert main(["reproduce", "--source", str(source)]) == 2
    assert "give the file to check, or --manifest FILE" in capsys.readouterr().err
    assert (
        main(["reproduce", str(target), "--manifest", str(manifest), "--source", str(source)]) == 2
    )
    assert "not both" in capsys.readouterr().err
    with pytest.raises(TypeError, match="exactly one"):
        ggufkit.reproduce_file(source=source)
    with pytest.raises(TypeError, match="exactly one"):
        ggufkit.reproduce_file(target, manifest=manifest, source=source)


def test_a_loaded_manifest_and_keep(
    tmp_path: Path, source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = model(tmp_path, "plain")
    manifest = make_manifest(plain)
    use(monkeypatch, FakeQuantize(plain))
    keep = tmp_path / "kept.gguf"
    result = ggufkit.reproduce_file(
        manifest=ggufkit.load_manifest(manifest), source=source, keep=keep
    )
    assert result.ok()
    assert result.report.a == "the manifest"
    assert keep.read_bytes() == plain.read_bytes()
    for protected in (manifest, source):
        with pytest.raises(ggufkit.ReproduceError, match="--keep"):
            ggufkit.reproduce_file(manifest=manifest, source=source, keep=protected, force=True)


def test_build_inputs_must_be_strings(
    tmp_path: Path, source: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = make_manifest(model(tmp_path, "plain"))
    data: dict[str, Any] = json.loads(manifest.read_text())
    data["build_inputs"] = {"quantize_cmd": 5}
    manifest.write_text(json.dumps(data))
    assert main(reproduce_args(manifest, source)) == 2
    assert "build_inputs must map names to strings" in capsys.readouterr().err


def test_manifest_records_input_hashes(
    tmp_path: Path, source: Path, imatrix: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plain = model(tmp_path, "plain")
    out = tmp_path / "m.json"
    args = [
        "manifest",
        str(plain),
        "-o",
        str(out),
        "--source",
        str(source),
        "--imatrix",
        str(imatrix),
    ]
    assert main(args) == 0
    assert f"hashing {source}" in capsys.readouterr().err
    inputs = json.loads(out.read_text())["build_inputs"]
    assert inputs == {"source_sha256": sha256(source), "imatrix_sha256": sha256(imatrix)}

    assert main([*args, "--force", "--input", "source_sha256=abc"]) == 2
    assert "--source and --input source_sha256 both set source_sha256" in capsys.readouterr().err


@pytest.mark.skipif(sys.platform == "win32", reason="runs a POSIX shell script")
def test_a_real_stand_in_from_a_manifest(
    target: Path,
    source: Path,
    imatrix: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.undo()  # the real shutil.which and subprocess
    manifest = make_manifest(
        target,
        source=source,
        imatrix=imatrix,
        quantize_cmd="llama-quantize --pure --imatrix calib/imatrix.dat in.gguf out.gguf Q4_K_M",
    )
    script = tmp_path / "llama-quantize"
    script.write_text(
        "#!/bin/sh\n"
        "echo 'version: 1.0 (build 3, commit feedbee)'\n"
        '[ "$1 $2 $3" = "--imatrix calib/imatrix.dat --pure" ] && [ -f "$2" ] || exit 4\n'
        f'cp "{target}" "$5"\n'
    )
    script.chmod(0o755)
    args = reproduce_args(
        manifest, source, "--imatrix", str(imatrix), "--llama-quantize", str(script)
    )
    assert main(args) == 0
    assert "file bytes   identical" in capsys.readouterr().out
