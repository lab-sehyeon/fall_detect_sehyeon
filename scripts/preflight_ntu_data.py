#!/usr/bin/env python3
"""Validate raw NTU skeleton names and split counts before large preprocessing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil

from data_gen.ntu_gendata import (
    EXPECTED_COUNTS,
    NTU60_TRAINING_SUBJECTS,
    NTU120_TRAINING_SUBJECTS,
    training_cameras,
    training_setups,
)


NTU_NAME = re.compile(
    r"^S(?P<setup>\d{3})C(?P<camera>\d{3})P(?P<subject>\d{3})"
    r"R(?P<repeat>\d{3})A(?P<action>\d{3})\.skeleton$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ntu60", "ntu120"), required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--ignored-sample-path", type=Path, required=True)
    parser.add_argument("--protocol", choices=("xsub", "xview", "xsetup"), required=True)
    parser.add_argument("--out-folder", type=Path, required=True)
    parser.add_argument("--require-valid", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_path.is_dir():
        raise FileNotFoundError(args.data_path)
    if not args.ignored_sample_path.is_file():
        raise FileNotFoundError(args.ignored_sample_path)

    allowed_protocols = {"ntu60": {"xsub", "xview"}, "ntu120": {"xsub", "xsetup"}}
    if args.protocol not in allowed_protocols[args.dataset]:
        raise ValueError("{} does not support {}".format(args.dataset, args.protocol))

    ignored = {
        line.strip() + ("" if line.strip().endswith(".skeleton") else ".skeleton")
        for line in args.ignored_sample_path.read_text().splitlines()
        if line.strip()
    }
    records = []
    invalid_names = []
    ignored_present = []
    for path in sorted(args.data_path.iterdir()):
        if not path.is_file() or path.suffix != ".skeleton":
            continue
        if path.name in ignored:
            ignored_present.append(path.name)
            continue
        match = NTU_NAME.fullmatch(path.name)
        if match is None:
            invalid_names.append(path.name)
            continue
        record = {key: int(value) for key, value in match.groupdict().items()}
        max_action = 60 if args.dataset == "ntu60" else 120
        if not 1 <= record["action"] <= max_action:
            invalid_names.append(path.name)
            continue
        records.append(record)

    if args.protocol == "xsub":
        train_subjects = (
            NTU60_TRAINING_SUBJECTS
            if args.dataset == "ntu60"
            else NTU120_TRAINING_SUBJECTS
        )
        is_train = lambda row: row["subject"] in train_subjects
    elif args.protocol == "xview":
        is_train = lambda row: row["camera"] in training_cameras
    else:
        is_train = lambda row: row["setup"] in training_setups

    train_count = sum(is_train(row) for row in records)
    val_count = len(records) - train_count
    expected_train = EXPECTED_COUNTS[(args.dataset, args.protocol, "train")]
    expected_val = EXPECTED_COUNTS[(args.dataset, args.protocol, "val")]
    counts_match = train_count == expected_train and val_count == expected_val

    # One joint array stores float32 N x 3 x 300 x 25 x 2.
    estimated_bytes = len(records) * 3 * 300 * 25 * 2 * 4
    disk_probe = args.out_folder
    while not disk_probe.exists() and disk_probe != disk_probe.parent:
        disk_probe = disk_probe.parent
    free_bytes = shutil.disk_usage(disk_probe).free
    enough_disk = free_bytes >= int(estimated_bytes * 1.1)

    report = {
        "status": "valid" if counts_match and not invalid_names and enough_disk else "invalid",
        "dataset": args.dataset,
        "protocol": args.protocol,
        "raw_skeleton_files_after_ignore": len(records),
        "ignored_entries": len(ignored),
        "ignored_files_present": len(ignored_present),
        "invalid_name_count": len(invalid_names),
        "invalid_name_examples": invalid_names[:20],
        "train_count": train_count,
        "expected_train_count": expected_train,
        "val_count": val_count,
        "expected_val_count": expected_val,
        "counts_match": counts_match,
        "estimated_joint_output_bytes": estimated_bytes,
        "free_bytes": free_bytes,
        "enough_disk_with_10_percent_margin": enough_disk,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_valid and report["status"] != "valid":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
