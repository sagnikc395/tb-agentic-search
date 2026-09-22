#!/usr/bin/env python3
"""Run the minimal tool-grounded latent annotation loop from the repository root."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python examples/annotate_latents.py` without installing the project.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latent_annotation.agent import explain, json_report, markdown_report  # noqa: E402
from latent_annotation.tooluniverse_adapter import ToolUniverseUnavailable  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Prototype SAE latent annotation CLI")
    parser.add_argument("--protein", default="DNAJA1", help="offline fixture name (currently DNAJA1)")
    parser.add_argument("--mut", "--mutation", dest="mutation", default="L7D", help="mutation such as L7D")
    parser.add_argument("--json", action="store_true", help="print the structured report instead of Markdown")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="use the offline mock pipeline")
    mode.add_argument("--tooluniverse", action="store_true", help="use live UniProt/InterPro calls through ToolUniverse")
    args = parser.parse_args()
    if not args.dry_run and not args.tooluniverse:
        parser.error("choose --dry-run or --tooluniverse")
    try:
        report = explain(args.protein, args.mutation, use_tooluniverse=args.tooluniverse)
    except (ToolUniverseUnavailable, ValueError) as exc:
        parser.error(str(exc))
    print(json_report(report) if args.json else markdown_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
