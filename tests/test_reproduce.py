"""``ggufkit reproduce``, with a stand-in for llama-quantize (no llama.cpp in CI)."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

import ggufkit
from ggufkit import reproduce as reproduce_mod
from ggufkit.cli import main
from ggufkit.reproduce import _parse_version, _safe_relative, find_llama_quantize
from tests.conftest import Q, det_floats, tiny_kv, tiny_tensors, write_gguf

VERSION_LINE = "version: 0.9.0 (build 7, commit abc1234)"
IMATRIX_KV = {
    "quantize.imatrix.file": "calib/imatrix.dat",
    "quantize.imatrix.dataset": "calibration.txt",
    "quantize.imatrix.entries_count": 3,
    "quantize.imatrix.chunks_count": 42,
}


def model(tmp_path: Path, name: str, *, kv: dict[str, Any] | None = None, **kwargs: Any) -> Path:
    full_kv = tiny_kv()
    full_kv["general.file_type"] = 15  # MOSTLY_Q4_K_M
    full_kv.update(kv or {})
    for key in [k for k, v in full_kv.items() if v is None]:
        del full_kv[key]
    tensors = kwargs.pop("tensors", None) or tiny_tensors()
    return write_gguf(tmp_path / f"{name}.gguf", kv=full_kv, tensors=tensors, **kwargs)[0]


class FakeQuantize:
    """Stands in for ``_run_quantize``: copies ``result`` to the output path llama-quantize got."""

    def __init__(self, result: Path | None, *, status: int = 0, output: str = VERSION_LINE) -> None:
        self.result = result
        self.status = status
        self.output = output
        self.calls: list[tuple[list[str], Path]] = []
        self.imatrix_bytes: bytes | None = None
        self.options: list[str] = []

    def __call__(self, argv: list[str], cwd: Path) -> tuple[int, str]:
        self.calls.append((list(argv), cwd))
        assert cwd.is_dir()
        args = argv[1:]
        self.options = []
        while args[0].startswith("--"):
            opt, args = args[0], args[1:]
            arity = 1 if opt == "--imatrix" else reproduce_mod.RECIPE_OPTIONS[opt]
            values, args = args[:arity], args[arity:]
            if opt == "--imatrix":
                self.imatrix_bytes = (cwd / values[0]).read_bytes()  # relative to cwd, or absolute
            else:
                self.options += [opt, *values]
        _source, out = args[0], Path(args[1])
        assert out.is_absolute()
        if self.result is not None:
            shutil.copyfile(self.result, out)
        return self.status, self.output

    @property
    def argv(self) -> list[str]:
        return self.calls[-1][0]


@pytest.fixture
def fake_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        reproduce_mod.shutil,
        "which",
        lambda name: "/opt/llama/llama-quantize" if name == "llama-quantize" else None,
    )


def use(monkeypatch: pytest.MonkeyPatch, fake: FakeQuantize) -> FakeQuantize:
    monkeypatch.setattr(reproduce_mod, "_run_quantize", fake)
    return fake


@pytest.fixture
def source(tmp_path: Path) -> Path:
    return model(tmp_path, "source", kv={"general.file_type": 32})  # MOSTLY_BF16


@pytest.mark.usefixtures("fake_binary")
def test_an_identical_rebuild_passes(
    tmp_path: Path,
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = model(tmp_path, "target")
    fake = use(monkeypatch, FakeQuantize(target))
    assert main(["reproduce", str(target), "--source", str(source)]) == 0
    out, err = capsys.readouterr()
    assert "type         Q4_K_M" in out
    assert "llama.cpp    0.9.0 (build 7, commit abc1234)" in out
    assert "file bytes   identical" in out
    assert "verdict      identical" in out
    assert "running llama-quantize (Q4_K_M)" in err

    argv, cwd = fake.calls[0]
    assert Path(argv[0]).name == "llama-quantize"
    assert argv[1] == str(source.absolute())
    assert argv[3:] == ["Q4_K_M"]  # no thread count: llama-quantize picks its own
    assert not cwd.exists()  # the work directory is gone


@pytest.mark.usefixtures("fake_binary")
def test_json_and_quiet(
    tmp_path: Path,
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = model(tmp_path, "target")
    fake = use(monkeypatch, FakeQuantize(target))
    assert main(["reproduce", str(target), "--source", str(source), "--json", "-q"]) == 0
    out, err = capsys.readouterr()
    assert err == ""
    data = json.loads(out)
    assert data["verdict"] == "identical"
    assert data["byte_identical"] is True
    assert data["type"] == "Q4_K_M"
    assert data["command"] == fake.argv
    assert data["llama_quantize_version"] == "0.9.0 (build 7, commit abc1234)"
    assert data["a"] == str(target)
    assert data["b"] == "rebuild"

    assert main(["reproduce", str(target), "--source", str(source), "-q"]) == 0
    assert capsys.readouterr() == ("", "")


@pytest.mark.usefixtures("fake_binary")
def test_different_weights_fail_and_name_the_tensor(
    tmp_path: Path,
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = model(tmp_path, "target")
    tensors = tiny_tensors()
    tensors["blk.0.attn_q.weight"] = (det_floats((64, 64), 99), Q.Q8_0)
    other = model(tmp_path, "other", tensors=tensors)
    use(monkeypatch, FakeQuantize(other))
    assert main(["reproduce", str(target), "--source", str(source), "--json"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert data["verdict"] == "weights-changed"
    assert data["byte_identical"] is False
    assert [(c["name"], c["kind"]) for c in data["tensor_changes"]] == [
        ("blk.0.attn_q.weight", "data")
    ]


@pytest.mark.usefixtures("fake_binary")
def test_metadata_only_passes_with_allow_metadata(
    tmp_path: Path, source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = model(tmp_path, "target")
    use(monkeypatch, FakeQuantize(model(tmp_path, "renamed", kv={"general.name": "other"})))
    args = ["reproduce", str(target), "--source", str(source), "-q"]
    assert main(args) == 1
    assert main([*args, "--allow-metadata"]) == 0


@pytest.mark.usefixtures("fake_binary")
def test_same_model_in_another_layout_passes_with_a_note(
    tmp_path: Path, source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = model(tmp_path, "target")
    reordered = model(tmp_path, "reordered", tensors=dict(reversed(tiny_tensors().items())))
    use(monkeypatch, FakeQuantize(reordered))
    result = ggufkit.reproduce_file(target, source=source)
    assert result.ok()
    assert result.report.verdict == "identical"
    assert result.byte_identical is False
    assert any("laid out differently" in n for n in result.notes)


@pytest.mark.usefixtures("fake_binary")
def test_type_override_and_threads(
    tmp_path: Path, source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = model(tmp_path, "target")
    fake = use(monkeypatch, FakeQuantize(target))
    args = ["reproduce", str(target), "--source", str(source), "-q"]
    assert main([*args, "--type", "q4_k_m", "--threads", "3"]) == 0
    assert fake.argv[-2:] == ["q4_k_m", "3"]


@pytest.mark.parametrize("file_type", [None, 9999, 1024])  # missing, unknown, GUESSED
@pytest.mark.usefixtures("fake_binary")
def test_an_unknown_type_needs_type(
    tmp_path: Path,
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    file_type: int | None,
) -> None:
    target = model(tmp_path, "target", kv={"general.file_type": file_type})
    fake = use(monkeypatch, FakeQuantize(target))
    assert main(["reproduce", str(target), "--source", str(source)]) == 2
    assert "pass --type" in capsys.readouterr().err
    assert not fake.calls
    assert main(["reproduce", str(target), "--source", str(source), "--type", "Q8_0", "-q"]) == 0


@pytest.mark.usefixtures("fake_binary")
def test_a_recorded_imatrix_is_required(
    tmp_path: Path,
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = model(tmp_path, "target", kv=IMATRIX_KV)
    fake = use(monkeypatch, FakeQuantize(target))
    assert main(["reproduce", str(target), "--source", str(source)]) == 2
    err = capsys.readouterr().err
    assert "file 'calib/imatrix.dat'" in err
    assert "dataset 'calibration.txt'" in err
    assert "42 chunks" in err
    assert "--imatrix" in err
    assert not fake.calls


@pytest.mark.usefixtures("fake_binary")
def test_the_imatrix_is_placed_at_the_recorded_path(
    tmp_path: Path,
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = model(tmp_path, "target", kv=IMATRIX_KV)
    imatrix = tmp_path / "downloaded.dat"
    imatrix.write_bytes(b"IMATRIX")
    fake = use(monkeypatch, FakeQuantize(target))
    args = ["reproduce", str(target), "--source", str(source), "--imatrix", str(imatrix)]
    assert main(args) == 0
    assert fake.argv[1:3] == ["--imatrix", "calib/imatrix.dat"]
    assert fake.imatrix_bytes == b"IMATRIX"
    assert "note:" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "recorded",
    ["/abs/imatrix.dat", "../imatrix.dat", "a/../../imatrix.dat", "C:\\imatrix.dat", "dir/", "."],
)
@pytest.mark.usefixtures("fake_binary")
def test_an_unsafe_recorded_path_falls_back_to_the_real_one(
    tmp_path: Path,
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    recorded: str,
) -> None:
    target = model(tmp_path, "target", kv={**IMATRIX_KV, "quantize.imatrix.file": recorded})
    imatrix = tmp_path / "downloaded.dat"
    imatrix.write_bytes(b"IMATRIX")
    fake = use(monkeypatch, FakeQuantize(target))
    args = ["reproduce", str(target), "--source", str(source), "--imatrix", str(imatrix)]
    main(args)
    assert fake.argv[1:3] == ["--imatrix", str(imatrix.absolute())]
    assert fake.imatrix_bytes == b"IMATRIX"
    assert "cannot be recreated" in capsys.readouterr().out


def test_safe_relative() -> None:
    for ok in ("imatrix.dat", "a/b/imatrix.dat", "./imatrix.dat", "~/imatrix.dat"):
        assert _safe_relative(ok), ok
    for bad in ("", "/a", "..", "a/..", "a\\b", "C:x", "a/", "a/.", "a\nb"):
        assert not _safe_relative(bad), bad


@pytest.mark.usefixtures("fake_binary")
def test_an_imatrix_the_target_does_not_record(
    tmp_path: Path, source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imatrix = tmp_path / "downloaded.dat"
    imatrix.write_bytes(b"IMATRIX")
    no_path = model(tmp_path, "no-path", kv={**IMATRIX_KV, "quantize.imatrix.file": None})
    plain = model(tmp_path, "plain")
    for target, note in ((plain, "records no importance matrix"), (no_path, "does not record")):
        fake = use(monkeypatch, FakeQuantize(target))
        result = ggufkit.reproduce_file(target, source=source, imatrix=imatrix)
        assert fake.argv[1:3] == ["--imatrix", str(imatrix.absolute())]
        assert any(note in n for n in result.notes)


@pytest.mark.usefixtures("fake_binary")
def test_missing_inputs(
    tmp_path: Path,
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = model(tmp_path, "target")
    fake = use(monkeypatch, FakeQuantize(target))
    missing = str(tmp_path / "missing.dat")
    assert main(["reproduce", str(target), "--source", str(source), "--imatrix", missing]) == 2
    assert main(["reproduce", str(target), "--source", missing]) == 2
    err = capsys.readouterr().err
    assert err.count("no such file") == 2
    assert not fake.calls


def test_finding_llama_quantize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reproduce_mod.shutil, "which", lambda _name: None)
    with pytest.raises(ggufkit.ReproduceError, match="not on PATH"):
        find_llama_quantize()
    with pytest.raises(ggufkit.ReproduceError, match="was not found"):
        find_llama_quantize(tmp_path / "nowhere")
    present = tmp_path / "llama-quantize"
    present.write_text("")
    with pytest.raises(ggufkit.ReproduceError, match="is not executable"):
        find_llama_quantize(present)
    monkeypatch.setattr(reproduce_mod.shutil, "which", lambda name: name)
    assert find_llama_quantize("bin/llama-quantize") == str(Path("bin/llama-quantize").absolute())


def test_no_llama_quantize_is_an_error(
    tmp_path: Path,
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(reproduce_mod.shutil, "which", lambda _name: None)
    target = model(tmp_path, "target")
    assert main(["reproduce", str(target), "--source", str(source)]) == 2
    assert "llama-quantize is not on PATH" in capsys.readouterr().err


@pytest.mark.usefixtures("fake_binary")
def test_a_failed_run_shows_the_end_of_its_output(
    tmp_path: Path,
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = model(tmp_path, "target")
    output = "\n".join(f"line {i}" for i in range(1, 31))
    use(monkeypatch, FakeQuantize(None, status=1, output=output))
    assert main(["reproduce", str(target), "--source", str(source)]) == 2
    err = capsys.readouterr().err
    assert "llama-quantize exited with status 1:" in err
    assert "line 30" in err
    assert "line 11\n" in err
    assert "line 10\n" not in err

    use(monkeypatch, FakeQuantize(None))
    assert main(["reproduce", str(target), "--source", str(source)]) == 2
    assert "llama-quantize wrote no file" in capsys.readouterr().err


@pytest.mark.usefixtures("fake_binary")
def test_keep(
    tmp_path: Path,
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = model(tmp_path, "target")
    rebuilt = model(tmp_path, "rebuilt-elsewhere", kv={"general.name": "other"})
    fake = use(monkeypatch, FakeQuantize(rebuilt))
    keep = tmp_path / "kept.gguf"
    args = ["reproduce", str(target), "--source", str(source), "--keep", str(keep), "--json"]
    assert main(args) == 1
    assert keep.read_bytes() == rebuilt.read_bytes()
    data = json.loads(capsys.readouterr().out)
    assert data["b"] == str(keep)
    assert data["command"][2] == str(keep.absolute())
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []

    calls = len(fake.calls)
    assert main(args) == 2  # exists, no --force: refused before running
    assert "already exists" in capsys.readouterr().err
    assert len(fake.calls) == calls
    assert main([*args, "--force"]) == 1
    with pytest.raises(FileExistsError):
        ggufkit.reproduce_file(target, source=source, keep=keep)


@pytest.mark.usefixtures("fake_binary")
def test_keep_never_replaces_the_target_or_the_source(
    tmp_path: Path, source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = model(tmp_path, "target")
    before = target.read_bytes(), source.read_bytes()
    fake = use(monkeypatch, FakeQuantize(model(tmp_path, "other", kv={"general.name": "x"})))
    for keep in (target, source):
        with pytest.raises(ggufkit.ReproduceError, match="--keep"):
            ggufkit.reproduce_file(target, source=source, keep=keep, force=True)
    assert (target.read_bytes(), source.read_bytes()) == before
    assert not fake.calls


@pytest.mark.usefixtures("fake_binary")
def test_split_target_and_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shards = model(tmp_path, "split", split_max_tensors=3)
    assert len(shards_of(shards)) == 2
    whole = model(tmp_path, "whole")
    src = write_gguf(
        tmp_path / "src.gguf", kv=tiny_kv(), tensors=tiny_tensors(), split_max_tensors=3
    )
    fake = use(monkeypatch, FakeQuantize(whole))
    result = ggufkit.reproduce_file(shards_of(shards)[1], source=src[1])
    assert fake.argv[1] == str(src[0].absolute())  # llama-quantize is given the first shard
    assert result.ok()
    assert result.byte_identical is None
    assert any("split into 2 files" in n for n in result.notes)


def shards_of(first: Path) -> list[Path]:
    return sorted(first.parent.glob(first.name.replace("-00001-of-", "-*-of-")))


def test_parse_version() -> None:
    assert (
        _parse_version(f"main: load\n{VERSION_LINE}\r\nmore") == "0.9.0 (build 7, commit abc1234)"
    )
    old = "llama_print_build_info: build = 1 (1a03cf4)\n"
    assert _parse_version(old) == "build 1, commit 1a03cf4"
    assert _parse_version("nothing here") is None


@pytest.mark.skipif(sys.platform == "win32", reason="runs a POSIX shell script")
def test_a_real_llama_quantize_stand_in(
    tmp_path: Path, source: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = model(tmp_path, "target", kv=IMATRIX_KV)
    imatrix = tmp_path / "downloaded.dat"
    imatrix.write_bytes(b"IMATRIX")
    script = tmp_path / "llama-quantize"
    script.write_text(
        "#!/bin/sh\n"
        "echo 'llama_print_build_info: build = 5 (feedbee)'\n"
        '[ "$1" = --imatrix ] || exit 3\n'
        '[ "$2" = calib/imatrix.dat ] && [ -f "$2" ] || exit 4\n'
        f'cp "{target}" "$4"\n'
    )
    script.chmod(0o755)
    args = ["reproduce", str(target), "--source", str(source), "--imatrix", str(imatrix)]
    assert main([*args, "--llama-quantize", str(script)]) == 0
    out = capsys.readouterr().out
    assert "build 5, commit feedbee" in out
    assert "file bytes   identical" in out

    script.write_text("#!/bin/sh\necho broken >&2\nexit 7\n")
    assert main([*args, "--llama-quantize", str(script)]) == 2
    assert "exited with status 7:\nbroken" in capsys.readouterr().err


def test_api_result(tmp_path: Path, source: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reproduce_mod.shutil, "which", lambda name: name)
    target = model(tmp_path, "target")
    fake = use(monkeypatch, FakeQuantize(target))
    result = ggufkit.reproduce_file(
        target, source=source, llama_quantize="my-quantize", quant_type="Q4_K_M", threads=2
    )
    assert isinstance(result, ggufkit.ReproduceResult)
    assert result.ok()
    assert result.byte_identical is True
    assert result.quant_type == "Q4_K_M"
    assert result.command == fake.argv
    assert Path(fake.argv[0]).name == "my-quantize"
    assert fake.argv[-1] == "2"
    assert result.to_dict()["byte_identical"] is True
