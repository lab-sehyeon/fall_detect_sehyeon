"""Acquire the historically used public Kaggle Le2i v2 subset, without credentials.

This only restores source assets; it does not remux, decode, train, or evaluate.
Run --phase annotations first, then --phase all. Completed files are hash-checked
on resume. Never overwrite an unverified pre-existing payload.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/source_archives/Le2i-FDD/kaggle_v2_20261002_r1"
REF = "tuyenldvn/falldataset-imvia"
API = "https://www.kaggle.com/api/v1/datasets"
VERSION = 2
GIB = 1024**3
RESERVE = 64 * GIB
BUDGET = 16 * GIB
FILE_LIMIT = GIB
SCENES = {"Coffee_room_01": (1, 48), "Coffee_room_02": (49, 70),
          "Home_01": (1, 30), "Home_02": (31, 60)}
EXCLUDED = {"Coffee_room_01/video (26).txt", "Coffee_room_02/video (50).txt",
            "Coffee_room_02/video (52).txt"}
EXPECTED_COUNTS = {"Coffee_room_01": (47, 0), "Coffee_room_02": (12, 8),
                   "Home_01": (30, 0), "Home_02": (7, 23)}


def safe_path(root: Path, relative: str) -> Path:
    part = PurePosixPath(relative)
    if not relative or part.is_absolute() or ".." in part.parts or "\\" in relative:
        raise ValueError("unsafe relative path")
    result = root.joinpath(*part.parts)
    for path in (result, *result.parents):
        if path.is_symlink():
            raise ValueError("symlink path refused")
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError("path escapes destination")
    return result


def save(path: Path, value: object) -> None:
    safe_path(DEST, str(path.relative_to(DEST)))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".metadata-", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def check_space(required: int = 0) -> None:
    if (DEST / "PAUSE").exists():
        raise RuntimeError("operator PAUSE requested")
    if shutil.disk_usage(DEST).free < RESERVE + required:
        raise RuntimeError("64 GiB free-space reserve reached")


def open_public(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": "FoundSkelModel-recovery/1.0"})
    for attempt in range(3):
        try:
            return urllib.request.urlopen(request, timeout=90)
        except urllib.error.HTTPError as exc:
            code = exc.code
            exc.close()
            if code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(10 * (attempt + 1))
                continue
            # No response bodies or signed redirect URLs in logs.
            raise RuntimeError(f"public download HTTP {code}; no credential fallback") from None
        except (urllib.error.URLError, TimeoutError):
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError("public download connection failed") from None


def get_json(url: str):
    with open_public(url) as response:
        return json.loads(response.read(8 * 1024**2))


def download_url(name: str) -> str:
    return f"{API}/download/{REF}/{urllib.parse.quote(name, safe='')}?datasetVersionNumber={VERSION}"


def selected_names() -> set[str]:
    names = {"README.txt"}
    for scene, (first, last) in SCENES.items():
        annotation_dir = "Annotations_files" if scene == "Coffee_room_02" else "Annotation_files"
        for number in range(first, last + 1):
            names.add(f"{scene}/{scene}/Videos/video ({number}).avi")
            names.add(f"{scene}/{scene}/{annotation_dir}/video ({number}).txt")
    return names


def inventory() -> dict:
    path = DEST / "source_inventory.json"
    if path.exists():
        value = json.loads(path.read_text())
    else:
        metadata = get_json(f"{API}/view/{REF}")
        rows, page_token, seen = [], None, set()
        while True:
            query = {"pageSize": 200, "datasetVersionNumber": VERSION}
            if page_token:
                query["pageToken"] = page_token
            page = get_json(f"{API}/list/{REF}?{urllib.parse.urlencode(query)}")
            rows.extend({"name": row["name"], "reported_bytes": int(row["totalBytes"]),
                         "creation_date": row.get("creationDate")} for row in page["datasetFiles"])
            page_token = page.get("nextPageToken")
            if not page_token:
                break
            if page_token in seen:
                raise RuntimeError("repeated inventory page")
            seen.add(page_token)
        value = {"ref": REF, "version": VERSION, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                 "metadata": {key: metadata.get(key) for key in
                              ("title", "ref", "licenseName", "currentVersionNumber", "lastUpdated",
                               "totalBytes", "isPrivate")}, "files": sorted(rows, key=lambda x: x["name"])}
        save(path, value)
    rows = value["files"]
    names = [row["name"] for row in rows]
    if value["ref"] != REF or value["version"] != VERSION or len(names) != 321 or len(set(names)) != 321:
        raise RuntimeError("unexpected pinned source inventory")
    if not selected_names().issubset(names):
        raise RuntimeError("historical subset missing from source")
    return value


def destination_name(name: str) -> str:
    if name == "README.txt":
        return "source/README.txt"
    parts = PurePosixPath(name).parts
    if len(parts) != 4 or parts[0] not in SCENES or parts[0] != parts[1]:
        raise ValueError("unexpected source name")
    return "source/" + "/".join((parts[0], *parts[2:]))


def zip_payload(archive: zipfile.ZipFile, expected_basename: str) -> zipfile.ZipInfo:
    files = [entry for entry in archive.infolist() if not entry.is_dir()]
    if len(files) != 1:
        raise ValueError("expected single-payload transport ZIP")
    entry = files[0]
    path = PurePosixPath(entry.filename)
    if path.is_absolute() or ".." in path.parts or "\\" in entry.filename:
        raise ValueError("unsafe ZIP member")
    if path.name != expected_basename or stat.S_ISLNK(entry.external_attr >> 16):
        raise ValueError("unexpected ZIP payload")
    if entry.file_size > FILE_LIMIT or entry.flag_bits & 1:
        raise ValueError("oversized or encrypted ZIP payload")
    return entry


def copy_bounded(source, target, limit: int, rate_limit: bool = False) -> int:
    total, start = 0, time.monotonic()
    while True:
        block = source.read(1024**2)
        if not block:
            break
        total += len(block)
        if total > limit:
            raise RuntimeError("per-file size limit exceeded")
        check_space(len(block))
        target.write(block)
        if rate_limit:
            delay = total / (16 * 1024**2) - (time.monotonic() - start)
            if delay > 0:
                time.sleep(min(delay, 1))
    return total


def validate_payload(path: Path, source_name: str) -> None:
    with path.open("rb") as stream:
        header = stream.read(12)
    if source_name.endswith(".avi"):
        if header[:4] != b"RIFF" or header[8:12] != b"AVI ":
            raise ValueError("download is not an AVI RIFF container")
    else:
        if path.stat().st_size > 1024**2 or not header or b"<html" in header.lower():
            raise ValueError("unexpected text payload")


def transfer(row: dict) -> dict:
    name = row["name"]
    target = safe_path(DEST, destination_name(name))
    receipt_path = safe_path(DEST, "receipts/" + hashlib.sha256(name.encode()).hexdigest() + ".json")
    if target.exists() or receipt_path.exists():
        if not target.is_file() or not receipt_path.is_file():
            raise RuntimeError(f"unpaired existing payload/receipt: {name}")
        receipt = json.loads(receipt_path.read_text())
        if (receipt["source_name"] != name or receipt["version"] != VERSION
                or receipt["bytes"] != target.stat().st_size or receipt["sha256"] != digest(target)):
            raise RuntimeError(f"existing payload verification failed: {name}")
        return receipt
    target.parent.mkdir(parents=True, exist_ok=True)
    check_space(2 * FILE_LIMIT)
    fd, temp_name = tempfile.mkstemp(prefix=".le2i-transport-", dir=DEST)
    transport = Path(temp_name)
    payload = None
    try:
        with os.fdopen(fd, "wb") as out, open_public(download_url(name)) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > FILE_LIMIT:
                raise RuntimeError("transport exceeds file limit")
            transport_bytes = copy_bounded(response, out, FILE_LIMIT, rate_limit=True)
        if content_length and transport_bytes != int(content_length):
            raise RuntimeError("truncated transport")
        wrapped, crc = zipfile.is_zipfile(transport), None
        if wrapped:
            with zipfile.ZipFile(transport) as archive:
                entry = zip_payload(archive, PurePosixPath(name).name)
                crc = f"{entry.CRC:08x}"
                fd, payload_name = tempfile.mkstemp(prefix=".le2i-payload-", dir=DEST)
                payload = Path(payload_name)
                with os.fdopen(fd, "wb") as out, archive.open(entry) as stream:
                    raw_bytes = copy_bounded(stream, out, FILE_LIMIT)
                if raw_bytes != entry.file_size:
                    raise RuntimeError("ZIP payload size mismatch")
        else:
            payload = transport
        validate_payload(payload, name)
        receipt = {"source_name": name, "ref": REF, "version": VERSION,
                   "relative_path": str(target.relative_to(DEST)), "bytes": payload.stat().st_size,
                   "sha256": digest(payload), "transport_bytes": transport_bytes,
                   "reported_bytes": row["reported_bytes"], "zip_wrapped": wrapped, "zip_crc32": crc,
                   "verification": "local SHA256; transport length; ZIP CRC when wrapped; AVI magic only",
                   "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        # link is an atomic, non-overwriting publish on the same filesystem.
        os.link(payload, target)
        save(receipt_path, receipt)
        print(json.dumps({"file": name, "bytes": receipt["bytes"], "status": "verified"}), flush=True)
        return receipt
    finally:
        # Only temporary files created by this call; never user datasets.
        if payload is not None and payload != transport:
            payload.unlink(missing_ok=True)
        transport.unlink(missing_ok=True)


def annotation_header(text: str) -> tuple[int, int] | None:
    lines = text.splitlines()
    if len(lines) < 2:
        return None
    try:
        start, end = int(lines[0].strip()), int(lines[1].strip())
    except ValueError:
        return None
    if start == end == 0:
        return (0, 0)
    if start <= 0 or end < start:
        raise ValueError("invalid fall header bounds")
    return start, end


def audit_annotations(rows: list[dict]) -> dict:
    counts = {scene: {"fall": 0, "nonfall": 0, "excluded": 0} for scene in SCENES}
    records, excluded = [], set()
    for row in rows:
        name = row["name"]
        if name == "README.txt" or not name.endswith(".txt"):
            continue
        scene, basename = name.split("/")[0], PurePosixPath(name).name
        path = safe_path(DEST, destination_name(name))
        event = annotation_header(path.read_text(encoding="utf-8-sig"))
        key = scene + "/" + basename
        if event is None:
            excluded.add(key)
            label = "excluded"
        else:
            label = "nonfall" if event == (0, 0) else "fall"
        counts[scene][label] += 1
        records.append({"source_name": name, "label": label, "raw_header": event})
    passed = (excluded == EXCLUDED and len(records) == 130 and
              all((counts[s]["fall"], counts[s]["nonfall"]) == EXPECTED_COUNTS[s] for s in SCENES))
    result = {"passed": passed, "counts": counts, "excluded": sorted(excluded), "records": records,
              "note": "Header-only audit; frame-index convention and full video decode not evaluated"}
    save(DEST / "annotation_audit.json", result)
    if not passed:
        raise RuntimeError("annotation counts/exclusions differ from historical contract; inspect audit")
    return result


def run(phase: str) -> None:
    inv = inventory()
    rows = [row for row in inv["files"] if row["name"] in selected_names()]
    if sum(row["reported_bytes"] for row in rows) > BUDGET:
        raise RuntimeError("inventory exceeds 16 GiB acquisition budget")
    contract = {"ref": REF, "version": VERSION, "selected_source_names": sorted(selected_names()),
                "reserve_gib": 64, "budget_gib": 16, "workers": 2,
                "max_network_mib_s_per_worker": 16,
                "script_sha256": digest(Path(__file__)), "no_training": True,
                "no_historical_byte_identity_claim": True, "no_authentication": True}
    contract_path = DEST / "acquisition_contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise RuntimeError("existing contract differs; use a new revision, do not overwrite")
    if not contract_path.exists():
        save(contract_path, contract)
    receipts = []
    for group in ("annotations", "videos"):
        if group == "videos" and phase == "annotations":
            break
        batch = [row for row in rows if row["name"].endswith(".avi") == (group == "videos")]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(transfer, row) for row in batch]
            try:
                for future in concurrent.futures.as_completed(futures):
                    receipts.append(future.result())
                    if sum(r["bytes"] for r in receipts) > BUDGET:
                        raise RuntimeError("raw payload budget exceeded")
                    save(DEST / "progress.json", {"status": "in_progress", "phase": group,
                         "verified_files": len(receipts), "total_files": len(rows),
                         "raw_bytes": sum(r["bytes"] for r in receipts)})
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        if group == "annotations":
            audit_annotations(rows)
    save(DEST / ("manifest.json" if phase == "all" else "annotation_manifest.json"),
         {"passed": True, "phase": phase, "files": sorted(receipts, key=lambda r: r["source_name"]),
          "full_decode_verified": False, "historical_bit_identity_verified": False})
    save(DEST / "progress.json", {"status": "completed" if phase == "all" else "annotations_completed",
         "verified_files": len(receipts), "total_files": len(rows),
         "raw_bytes": sum(r["bytes"] for r in receipts),
         "free_bytes": shutil.disk_usage(DEST).free})
    print("Acquisition phase completed: " + phase, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("annotations", "all"), required=True)
    args = parser.parse_args()
    safe_path(ROOT, str(DEST.relative_to(ROOT)))
    DEST.mkdir(parents=True, exist_ok=True)
    with (DEST / ".acquisition.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_space()
        try:
            run(args.phase)
        except Exception as exc:
            # Avoid dumping signed URLs in tracebacks; per-file receipts survive.
            save(DEST / "failure.json", {"type": type(exc).__name__,
                 "message": str(exc) if isinstance(exc, (ValueError, RuntimeError)) else "see error type",
                 "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
            print("Stopped safely: " + type(exc).__name__, flush=True)
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
