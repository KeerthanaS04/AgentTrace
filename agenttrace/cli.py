"""Command-line entry point: `agenttrace run|diagnose|replay|score|eval|serve`."""

from __future__ import annotations

import argparse
import sys

from agenttrace import __version__

# Subcommand -> task that implements it. Each prints a notice until its task lands.
COMMANDS: dict[str, str] = {
    "run": "P2-T3",
    "diagnose": "P7-T4",
    "replay": "P5-T3",
    "score": "P8-T7",
    "eval": "P9-T7",
    "serve": "P10-T1",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agenttrace")
    parser.add_argument("--version", action="version", version=f"agenttrace {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        sub.add_parser(name)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"`agenttrace {args.command}` is not implemented yet ({COMMANDS[args.command]}).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
