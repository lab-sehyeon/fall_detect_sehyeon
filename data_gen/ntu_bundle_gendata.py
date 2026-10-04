#!/usr/bin/env python3
"""Convert the flattened NTU60 bundle into FoundSkelModel joint arrays."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

try:
    from data_gen.ntu_gendata import (
        EXPECTED_COUNTS,
        NTU60_TRAINING_SUBJECTS,
        training_cameras,
    )
except ModuleNotFoundError:
    from ntu_gendata import (
        EXPECTED_COUNTS,
        NTU60_TRAINING_SUBJECTS,
        training_cameras,
    )


MAX_FRAME = 300
NUM_JOINT = 25
MAX_BODY = 2
NUM_CHANNEL = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the 27427188 NTU60 flattened bundle to N,C,T,V,M arrays."
    )
    parser.add_argument("--skl-path", type=Path, required=True)
    parser.add_argument("--descs-path", type=Path, required=True)
    parser.add_argument("--out-folder", type=Path, required=True)
    parser.add_argument("--protocol", choices=("xsub", "xview"), default="xsub")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the source and split without creating output arrays",
    )
    return parser.parse_args()


def load_and_validate(skl_path: Path, descs_path: Path):
    if not skl_path.is_file():
        raise FileNotFoundError(skl_path)
    if not descs_path.is_file():
        raise FileNotFoundError(descs_path)

    skeletons = np.load(skl_path, mmap_mode="r")
    descs = np.loadtxt(descs_path, delimiter=",", dtype=np.int64)
    if skeletons.ndim != 2 or skeletons.shape[1] != 150:
        raise ValueError("skl.npy must have shape (frames, 150), got {}".format(
            skeletons.shape
        ))
    if descs.ndim != 2 or descs.shape[0] != 7:
        raise ValueError("descs.csv must have shape (7, samples), got {}".format(
            descs.shape
        ))
    if descs.shape[1] != 56578:
        raise ValueError("expected 56,578 NTU60 samples, got {}".format(descs.shape[1]))
    if descs[5, 0] != 1 or descs[6, -1] != skeletons.shape[0]:
        raise ValueError("description boundaries do not cover skl.npy")
    if np.any(descs[5, 1:] != descs[6, :-1] + 1):
        raise ValueError("description frame boundaries are not contiguous")

    lengths = descs[6] - descs[5] + 1
    if np.any(lengths < 1) or np.any(lengths > MAX_FRAME):
        raise ValueError("sample frame counts must be in [1, 300]")
    if not np.isfinite(skeletons).all():
        raise ValueError("skl.npy contains NaN or infinite coordinates")

    # The bundle is already centered at joint index 1. Applying the upstream
    # normalization again would change the released coordinates.
    center_columns = skeletons[:, 3:6]
    if np.max(np.abs(center_columns)) != 0:
        raise ValueError("source does not match the expected pre-normalized layout")
    return skeletons, descs, lengths


def split_mask(descs: np.ndarray, protocol: str) -> np.ndarray:
    if protocol == "xsub":
        return np.isin(descs[2], NTU60_TRAINING_SUBJECTS)
    return np.isin(descs[1], training_cameras)


def sample_name(descs: np.ndarray, index: int) -> str:
    setup, camera, subject, repeat, action = descs[:5, index]
    return "S{:03d}C{:03d}P{:03d}R{:03d}A{:03d}.skeleton".format(
        setup, camera, subject, repeat, action
    )


def targets(out_dir: Path, part: str):
    return (
        out_dir / "{}_data_joint.npy".format(part),
        out_dir / "{}_num_frame.npy".format(part),
        out_dir / "{}_label.pkl".format(part),
    )


def convert_split(
    skeletons: np.ndarray,
    descs: np.ndarray,
    lengths: np.ndarray,
    indices: np.ndarray,
    out_dir: Path,
    part: str,
    overwrite: bool,
) -> None:
    data_path, frame_path, label_path = targets(out_dir, part)
    existing = [path for path in (data_path, frame_path, label_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite: {}".format(", ".join(map(str, existing)))
        )

    names = [sample_name(descs, int(index)) for index in indices]
    labels = [int(descs[4, index] - 1) for index in indices]
    with label_path.open("wb") as stream:
        pickle.dump((names, labels), stream)

    frame_counts = open_memmap(
        frame_path, mode="w+", dtype=np.int64, shape=(len(indices),)
    )
    output = open_memmap(
        data_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(indices), NUM_CHANNEL, MAX_FRAME, NUM_JOINT, MAX_BODY),
    )

    for output_index, source_index in enumerate(
        tqdm(indices, desc="convert {}".format(part))
    ):
        source_index = int(source_index)
        start = int(descs[5, source_index] - 1)
        end = int(descs[6, source_index])
        frame_count = int(lengths[source_index])
        # Flattened source order is frame, person, joint, coordinate.
        sequence = skeletons[start:end].reshape(frame_count, MAX_BODY, NUM_JOINT, NUM_CHANNEL)
        output[output_index, :, :frame_count] = sequence.transpose(3, 0, 2, 1)
        frame_counts[output_index] = frame_count

    output.flush()
    frame_counts.flush()


def main() -> None:
    args = parse_args()
    skeletons, descs, lengths = load_and_validate(args.skl_path, args.descs_path)
    training = split_mask(descs, args.protocol)
    split_indices = {
        "train": np.flatnonzero(training),
        "val": np.flatnonzero(~training),
    }

    report = {
        "status": "valid",
        "source_shape": list(skeletons.shape),
        "sample_count": int(descs.shape[1]),
        "protocol": args.protocol,
        "train_count": int(len(split_indices["train"])),
        "val_count": int(len(split_indices["val"])),
        "normalization": "source_pre_normalized_no_extra_transform",
    }
    for part, indices in split_indices.items():
        expected = EXPECTED_COUNTS[("ntu60", args.protocol, part)]
        if len(indices) != expected:
            raise RuntimeError(
                "{} split count mismatch: got {}, expected {}".format(
                    part, len(indices), expected
                )
            )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.dry_run:
        return

    out_dir = args.out_folder / args.protocol
    out_dir.mkdir(parents=True, exist_ok=True)
    for part, indices in split_indices.items():
        convert_split(
            skeletons, descs, lengths, indices, out_dir, part, args.overwrite
        )

    manifest = {
        **report,
        "skl_path": str(args.skl_path.resolve()),
        "descs_path": str(args.descs_path.resolve()),
        "output_layout": "N,C,T,V,M",
        "output_shape_tail": [NUM_CHANNEL, MAX_FRAME, NUM_JOINT, MAX_BODY],
    }
    with (args.out_folder / "preprocess_manifest.json").open("w") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
