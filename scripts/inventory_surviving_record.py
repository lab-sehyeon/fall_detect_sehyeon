#!/usr/bin/env python3
"""Inventory every file-like path mentioned by the surviving project record."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PATTERN = re.compile(
    r"`([^`\n]+\.(?:py|json|ya?ml|md|svg|png|pth|pt|npy|npz|pkl|csv|html))(?:#L\d+)?`"
)
ASSET_SUFFIXES = (".pth", ".pt", ".npy", ".npz", ".pkl", ".png")


def normalize(value: str) -> str | None:
    if " " in value or value.startswith("python ") or "*" in value:
        return None
    value = value.split("#L", 1)[0]
    while value.startswith("../"):
        value = value[3:]
    return value[2:] if value.startswith("./") else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("documents", nargs="+", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    references = set()
    for document in args.documents:
        text = document.read_text(encoding="utf-8")
        for match in PATTERN.findall(text):
            value = normalize(match)
            if value:
                references.add(value)
    records = []
    for name in sorted(references):
        exists = (args.root / name).exists()
        if exists:
            state = "available_or_reconstructed"
        elif name.endswith(ASSET_SUFFIXES) or name.startswith(("data/", "checkpoint/", "inference_outputs/")):
            state = "missing_original_bytes"
        else:
            state = "missing_local_source_exact_text_unrecoverable"
        records.append({"path": name, "state": state})
    counts = {}
    for record in records:
        counts[record["state"]] = counts.get(record["state"], 0) + 1
    report = {
        "documents": [str(path) for path in args.documents],
        "root": str(args.root.resolve()),
        "counts": counts,
        "records": records,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
