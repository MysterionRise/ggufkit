"""Command-line entry point: ``ggufkit <command> ...``."""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import TYPE_CHECKING

from ggufkit import __version__
from ggufkit import diff as diff_cmd
from ggufkit import fingerprint as fingerprint_cmd
from ggufkit import manifest as manifest_cmd
from ggufkit import reproduce as reproduce_cmd
from ggufkit import stitch as stitch_cmd
from ggufkit import verify as verify_cmd
from ggufkit import viz as viz_cmd

if TYPE_CHECKING:
    from collections.abc import Sequence

EXIT_OK = 0
EXIT_MISMATCH = 1
EXIT_ERROR = 2
EXIT_BROKEN_PIPE = 141


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ggufkit",
        description="Fingerprint, verify and visualize GGUF models.",
    )
    parser.add_argument("--version", action="version", version=f"ggufkit {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    fingerprint_cmd.add_parser(sub)
    manifest_cmd.add_parser(sub)
    verify_cmd.add_parser(sub)
    reproduce_cmd.add_parser(sub)
    diff_cmd.add_parser(sub)
    stitch_cmd.add_parser(sub)
    viz_cmd.add_parser(sub)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return EXIT_ERROR
    try:
        return int(func(args))
    except BrokenPipeError:
        # The reader went away (``ggufkit ... | head``): stop quietly, like other Unix tools.
        return EXIT_BROKEN_PIPE
    except (ValueError, OSError, re.error) as e:
        # ValueError covers ggufkit's own errors (bad models, manifests, arguments).
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


def entry() -> None:  # pragma: no cover - exercised by the installed ``ggufkit`` script
    """Console-script entry point: :func:`main`, plus a quiet exit when stdout is closed."""
    try:
        code = main()
        sys.stdout.flush()
    except BrokenPipeError:
        code = EXIT_BROKEN_PIPE
    if code == EXIT_BROKEN_PIPE:
        # Point stdout at /dev/null so the interpreter's final flush cannot fail again.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
    sys.exit(code)


if __name__ == "__main__":  # pragma: no cover
    entry()
