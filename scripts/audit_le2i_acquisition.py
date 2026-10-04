"""Offline, read-back verification of the 2026-10-02 Le2i source acquisition.

Only creates download_verification.json; does not decode or alter source assets.
"""
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/source_archives/Le2i-FDD/kaggle_v2_20261002_r1"


def main():
    manifest = json.loads((DEST / "manifest.json").read_text())
    contract = json.loads((DEST / "acquisition_contract.json").read_text())
    annotation_audit = json.loads((DEST / "annotation_audit.json").read_text())
    if not manifest["passed"] or manifest["phase"] != "all" or not annotation_audit["passed"]:
        raise RuntimeError("acquisition not complete")
    rows = manifest["files"]
    if len(rows) != 261 or {r["source_name"] for r in rows} != set(contract["selected_source_names"]):
        raise RuntimeError("manifest does not match selected 261 files")
    scene_counts = {}
    pairs = {}
    expected_paths = set()
    size_mismatches = []
    for row in rows:
        path = DEST / row["relative_path"]
        if not path.resolve().is_relative_to(DEST.resolve()) or any(p.is_symlink() for p in (path, *path.parents)):
            raise RuntimeError("unsafe payload path")
        expected_paths.add(path)
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            raise RuntimeError("payload size mismatch")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            header = stream.read(12)
            digest.update(header)
            for block in iter(lambda: stream.read(1024**2), b""):
                digest.update(block)
        if digest.hexdigest() != row["sha256"]:
            raise RuntimeError("payload hash mismatch")
        if row["bytes"] != row["reported_bytes"]:
            size_mismatches.append(row["source_name"])
        if row["source_name"] == "README.txt":
            continue
        scene = row["source_name"].split("/")[0]
        counts = scene_counts.setdefault(scene, {"videos": 0, "annotations": 0, "bytes": 0})
        kind = "videos" if path.suffix == ".avi" else "annotations"
        counts[kind] += 1
        counts["bytes"] += row["bytes"]
        pairs.setdefault((scene, path.stem), set()).add(kind)
        if kind == "videos" and not (header[:4] == b"RIFF" and header[8:12] == b"AVI "):
            raise RuntimeError("not an AVI")
    actual_paths = {p for p in (DEST / "source").rglob("*") if p.is_file()}
    if expected_paths != actual_paths:
        raise RuntimeError("unexpected or missing source files")
    if len(pairs) != 130 or any(kinds != {"videos", "annotations"} for kinds in pairs.values()):
        raise RuntimeError("video/annotation pairing failed")
    expected_counts = {"Coffee_room_01": (47, 0, 1), "Coffee_room_02": (12, 8, 2),
                       "Home_01": (30, 0, 0), "Home_02": (7, 23, 0)}
    observed = {scene: [0, 0, 0] for scene in expected_counts}
    excluded = []
    for row in rows:
        if row["source_name"] == "README.txt" or not row["source_name"].endswith(".txt"):
            continue
        scene = row["source_name"].split("/")[0]
        path = DEST / row["relative_path"]
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        try:
            start, end = int(lines[0]), int(lines[1])
        except (ValueError, IndexError):
            observed[scene][2] += 1
            excluded.append(scene + "/" + path.name)
            continue
        if start == end == 0:
            observed[scene][1] += 1
        elif 0 < start <= end:
            observed[scene][0] += 1
        else:
            raise RuntimeError("invalid event header")
    if {k: tuple(v) for k, v in observed.items()} != expected_counts:
        raise RuntimeError("independent header audit differs from history")
    if sorted(excluded) != ["Coffee_room_01/video (26).txt", "Coffee_room_02/video (50).txt", "Coffee_room_02/video (52).txt"]:
        raise RuntimeError("excluded files differ from history")
    result = {"passed": True, "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "files": 261, "video_annotation_pairs": 130, "scene_counts": scene_counts,
              "valid_header_videos": 127, "fall": 96, "nonfall": 31, "excluded": sorted(excluded),
              "raw_bytes": sum(r["bytes"] for r in rows),
              "transport_bytes": sum(r["transport_bytes"] for r in rows),
              "zip_wrapped_files": sum(r["zip_wrapped"] for r in rows),
              "api_reported_size_mismatches": size_mismatches,
              "all_local_sha256_match": True, "full_decode_verified": False,
              "historical_byte_identity_verified": False,
              "free_bytes": shutil.disk_usage(DEST).free}
    output = DEST / "download_verification.json"
    if output.exists() or output.is_symlink():
        raise RuntimeError("verification output already exists; do not overwrite")
    with output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
