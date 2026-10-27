# Contributing to ggufkit

Thanks for helping. Bug reports with a reproducing command are as welcome as code.

## Where to start

Issues labelled
[good first issue](https://github.com/MysterionRise/ggufkit/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
are small and self-contained: each says which files to touch, which existing test to copy, and
when it's done. Issues labelled
[help wanted](https://github.com/MysterionRise/ggufkit/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
are bigger but just as well defined. Comment on an issue to say you're taking it, so two people
don't do the same work, and ask there if anything is unclear.

## Set up

```sh
git clone https://github.com/MysterionRise/ggufkit
cd ggufkit
uv sync --all-extras
```

[uv](https://docs.astral.sh/uv/) manages the virtual environment and the lockfile. Python 3.11
or newer.

## Before you open a pull request

Run what CI runs:

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest --cov
```

CI runs all of these as blocking checks: lint, format and mypy once, the tests on Ubuntu and
Windows for Python 3.11 to 3.14 and on macOS for one Python, with at least 90% coverage. It
also builds the wheel and sdist and runs `scripts/smoke_test.py` against a clean install of
each.

## Tests

Tests build tiny GGUF files in-process with `GGUFWriter` from deterministic data, so they need
no downloads and run in seconds. Add a test with every behaviour change; `tests/conftest.py`
has `write_gguf()` and fixtures for the common cases.

## The fingerprint and manifest formats are specifications

`docs/FINGERPRINT.md` and `docs/MANIFEST.md` describe byte formats other people may implement
and depend on.

- `tests/test_golden.py` pins the `ggufkit-fp/1` byte format. If your change makes it fail, you
  have changed the specification: give it a new id (`ggufkit-fp/2`), update the docs, and only
  then run `uv run pytest --update-golden`.
- Manifest changes that remove or rename fields need a new schema id; adding optional fields
  does not.

## README images and examples

The README's images and console output come from a real model. To regenerate them (about 2 GB
of downloads from Hugging Face, pinned to one revision and checked by sha256):

```sh
uv run python scripts/make_readme_assets.py
```

`scripts/make_examples.py` regenerates the one synthetic illustration. Paste console output
verbatim; do not edit numbers by hand.

The animations in `assets/demos/` come from `scripts/make_demos.py`, which runs every command
shown on screen against the same files and records its output, including a deliberate tamper
of one tensor that the video labels as such. It needs [cosign](https://github.com/sigstore/cosign)
(on `PATH` or in `$COSIGN`), ffmpeg, and Node.js with Playwright and a Chromium it can launch.
It writes the README GIFs and their still frames to `assets/demos/` and 1080x1920 MP4s to
`build/demos/`:

```sh
uv run python scripts/make_demos.py
```

The scenes are in `scripts/demos/scenes.html`; captions there must stay true of the recorded
output. To re-render without re-running the commands, call the renderer on the recorded data.
`--formats` picks the formats to render instead of the default two (`vertical,readme`):
`feed` is a 4:5 (1080x1350) MP4, and every MP4 gets an `.srt` of its on-screen text.

```sh
node scripts/demos/render.mjs scratch/demos/demos.json tamper --formats=feed
```

## Style

- ruff's settings in `pyproject.toml` are the style guide; the formatter has the last word.
- Keep runtime dependencies to `gguf` and `numpy`. Pillow is the `viz` extra; anything heavier
  needs a good reason.
- Commit messages explain why, not only what.

## Licence

By contributing you agree that your contribution is licensed under the project's MIT licence.
