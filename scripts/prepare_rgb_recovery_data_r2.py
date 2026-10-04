"""Resume pinned SAFER OOD acquisition; audit OOPS media and annotation timelines.

Original acquisition contracts are immutable. This separate revision distinguishes
random-seek failure from sequential-decode failure and never modifies source media.
No model inference, event ground truth construction, frame conversion or training.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import fcntl
import math
import os
from pathlib import Path
import shutil
import sys
import time

from scripts import acquire_rgb_recovery_data as source

ROOT = source.ROOT
OUTPUT = ROOT / 'data/source_archives/rgb_preparation_20260928_r1'
require, read, save, sha = source.require, source.read, source.save, source.sha


def safety(extra=0):
    source.pause_space(extra)
    require(not (OUTPUT / 'PAUSE_REQUESTED').exists(), 'pause requested')


def status(task, stage, **fields):
    record = dict(task=task, stage=stage,
                  time=datetime.now(timezone.utc).isoformat(), **fields)
    save(OUTPUT / (task + '_status.json'), record)
    print(__import__('json').dumps(record, ensure_ascii=False), flush=True)


def sequential_probe(path, check=safety):
    import cv2
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(path))
    try:
        require(cap.isOpened(), 'video open failed')
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        require(count > 0 and math.isfinite(fps) and fps > 0 and width > 0 and height > 0,
                'invalid media metadata')
        decoded, last_check = 0, time.monotonic()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            require(frame is not None and frame.shape[:2] == (height, width),
                    'decoded frame shape changed')
            decoded += 1
            if decoded % 256 == 0 or time.monotonic() - last_check > 5:
                check()
                last_check = time.monotonic()
        require(decoded == count, 'sequential frame count differs from metadata')
        return dict(frames=count, decoded_frames=decoded, fps=fps, width=width,
                    height=height, full_frame_decode=True,
                    sequential_last_frame_decode=True, source_duration_seconds=count / fps,
                    duration_basis='frame_count/nominal_fps; not a PTS audit')
    finally:
        cap.release()


def robust_probe(path):
    try:
        result = source.video_probe(path)
        result['random_seek_last_frame'] = True
        return result
    except RuntimeError as error:
        if str(error) != 'video last frame decode failed':
            raise
        result = sequential_probe(path)
        result.update(random_seek_last_frame=False,
                      fallback_reason='last frame random seek failed; full sequential count matched')
        return result


def normalize_intervals(rows, label_map):
    """Decimal half-open intervals; specific labels take priority over other only."""
    parsed = []
    for row in rows:
        try:
            start, end = Decimal(str(row['start'])), Decimal(str(row['end']))
            label_id = int(row['label'])
        except (ValueError, TypeError, InvalidOperation) as error:
            raise RuntimeError('invalid annotation scalar') from error
        require(start.is_finite() and end.is_finite() and 0 <= start < end,
                'invalid annotation interval')
        require(label_id in label_map, 'unknown annotation label')
        parsed.append(dict(start=start, end=end, label=label_map[label_id],
                           label_id=label_id, source_row=row['source_row']))
    parsed.sort(key=lambda x: (x['start'], x['end'], x['source_row']))
    overlaps = []
    for i, a in enumerate(parsed):
        for b in parsed[i + 1:]:
            if b['start'] >= a['end']:
                break
            lo, hi = max(a['start'], b['start']), min(a['end'], b['end'])
            require((a['label'] == 'other') != (b['label'] == 'other'),
                    'overlap is not other versus specific label')
            overlaps.append(dict(source_rows=[a['source_row'], b['source_row']],
                                 labels=[a['label'], b['label']],
                                 start_seconds=str(lo), end_seconds=str(hi),
                                 removed_other_seconds=str(hi - lo)))
    normalized = []
    for row in parsed:
        pieces = [(row['start'], row['end'])]
        if row['label'] == 'other':
            for other in parsed:
                if other['label'] == 'other':
                    continue
                revised = []
                for lo, hi in pieces:
                    if other['end'] <= lo or other['start'] >= hi:
                        revised.append((lo, hi))
                        continue
                    if lo < other['start']:
                        revised.append((lo, other['start']))
                    if other['end'] < hi:
                        revised.append((other['end'], hi))
                pieces = revised
        normalized.extend({**row, 'start': lo, 'end': hi} for lo, hi in pieces)
    normalized.sort(key=lambda x: (x['start'], x['end'], x['source_row']))
    require(all(a['end'] <= b['start'] for a, b in zip(normalized, normalized[1:])),
            'overlap remains after normalization')
    # Independent conservation check: only other duration may change.
    def durations(items):
        totals = defaultdict(Decimal)
        for row in items:
            totals[row['label']] += row['end'] - row['start']
        return totals
    before, after = durations(parsed), durations(normalized)
    removed = sum((Decimal(r['removed_other_seconds']) for r in overlaps), Decimal(0))
    for label in before:
        require(before[label] - after[label] == (removed if label == 'other' else 0),
                'annotation duration conservation failed')
    output = [{**r, 'start': str(r['start']), 'end': str(r['end'])} for r in normalized]
    return output, dict(overlap_pairs=len(overlaps), removed_other_seconds=str(removed),
                       overlaps=overlaps, normalized_overlap_pairs=0)


def prepare_timeline():
    mapping_path = source.OOPS / 'metadata/data_files/oops_video_mapping.csv'
    label_path = source.OOPS / 'metadata/labels/OOPS.csv'
    names_path = source.OOPS / 'metadata/labels/label2id.csv'
    metadata = source.metadata('oops')
    for spec in metadata['files']:
        source.verify(source.safe_path(source.OOPS / 'metadata', spec['rfilename']), spec)
    with names_path.open(newline='') as stream:
        names = list(csv.DictReader(stream))
    label_map = {int(r['id']): r['label'] for r in names}
    require(len(label_map) == len(names) and len(set(label_map.values())) == len(names),
            'duplicate label vocabulary')
    with label_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    mapping = source.load_mapping(mapping_path)
    require(len(rows) == 4022 and {r['path'] + '.mp4' for r in rows} == set(mapping.values()),
            'annotation population changed')
    groups = defaultdict(list)
    for index, row in enumerate(rows, 2):
        require(row['dataset'] == 'OOPS', 'unexpected annotation dataset')
        groups[row['path'] + '.mp4'].append({**row, 'source_row': index})
    records = []
    for name, group in sorted(groups.items()):
        normalized, audit = normalize_intervals(group, label_map)
        records.append(dict(video=name, source_annotation_rows=len(group),
                            annotations=normalized, normalization=audit))
    report = dict(videos=len(records), original_annotation_rows=len(rows),
                  normalized_annotation_rows=sum(len(r['annotations']) for r in records),
                  label_counts=dict(Counter(label_map[int(r['label'])] for r in rows)),
                  overlap_pairs=sum(r['normalization']['overlap_pairs'] for r in records),
                  removed_other_seconds=str(sum((Decimal(r['normalization']['removed_other_seconds'])
                                                for r in records), Decimal(0))),
                  normalized_overlap_pairs=0, source_label_file_modified=False,
                  label_file_sha256=sha(label_path),
                  episode_ground_truth_constructed=False, historical_recovery_299_reproduced=False)
    save(OUTPUT / 'oops_annotation_report.json', report)
    return records, report


def safer_ood():
    from huggingface_hub import hf_hub_download
    manifest = source.metadata('safer_ood')
    path = OUTPUT / 'safer_ood_videos.json'
    journal = read(path) if path.exists() else {}
    require(set(journal) <= {s['rfilename'] for s in manifest['files']}, 'unknown journal file')
    remaining = sum(s['size'] for s in manifest['files']
                    if not source.safe_path(source.SAFER, s['rfilename']).exists())
    safety(remaining)
    for spec in manifest['files']:
        name = spec['rfilename']
        dest = source.safe_path(source.SAFER, name)
        safety(0 if dest.exists() else spec['size'])
        if not dest.exists():
            status('safer_ood', 'downloading', completed=len(journal), expected=30,
                   filename=Path(name).name)
            hf_hub_download(repo_id=manifest['repo'], filename=name, repo_type='dataset',
                            revision=manifest['revision'], local_dir=str(source.SAFER),
                            cache_dir=str(OUTPUT / 'hf_cache'), token=True, etag_timeout=60)
        verified = source.verify(dest, spec)
        if name in journal:
            require(all(journal[name][k] == v for k, v in verified.items()), 'completed video changed')
        else:
            verified['probe'] = robust_probe(dest)
            require(sha(dest) == verified['sha256'], 'source changed during probe')
            journal[name] = verified
            save(path, journal)
        status('safer_ood', 'validating', completed=len(journal), expected=30)
    report = dict(passed=len(journal) == 30, videos=len(journal),
                  bytes=sum(r['bytes'] for r in journal.values()), all_lfs_sha256_match=True,
                  fully_decoded_videos=sum(r['probe']['full_frame_decode'] for r in journal.values()),
                  seek_fallback_videos=[Path(k).name for k, v in journal.items()
                                        if not v['probe']['random_seek_last_frame']],
                  all_videos_full_frame_decode=all(r['probe']['full_frame_decode'] for r in journal.values()),
                  source_manifest_sha256=sha(source.CONTROL / 'safer_ood_source_manifest.json'),
                  videos_manifest_sha256=sha(path), transcoded=False,
                  model_input_alignment_validated=False)
    require(report['passed'], 'SAFER OOD incomplete')
    save(OUTPUT / 'safer_ood_final_report.json', report)
    status('safer_ood', 'completed', **report)


def oops_timeline():
    records, annotation_report = prepare_timeline()
    source_journal = read(source.CONTROL / 'oops_videos.json')
    require(len(source_journal) == 818 and read(source.CONTROL / 'oops_final_report.json')['passed'],
            'source OOPS download incomplete')
    path = OUTPUT / 'oops_decoded_videos.json'
    journal = read(path) if path.exists() else {}
    require(set(journal) <= set(source_journal), 'unknown decoded journal file')
    timeline_mismatches = []
    for record in records:
        name = record['video']
        expected = source_journal[name]
        dest = source.safe_path(source.OOPS, name)
        safety()
        require(dest.stat().st_size == expected['bytes'] and sha(dest) == expected['sha256'],
                'OOPS source changed')
        if name in journal:
            require(journal[name]['sha256'] == expected['sha256'], 'decoded journal source changed')
        else:
            probe = sequential_probe(dest)
            require(sha(dest) == expected['sha256'], 'OOPS source changed during decode')
            journal[name] = dict(sha256=expected['sha256'], bytes=expected['bytes'], probe=probe)
            save(path, journal)
        probe = journal[name]['probe']
        duration = probe['source_duration_seconds']
        latest = max(float(r['end']) for r in record['annotations'])
        tolerance = 1 / probe['fps'] + .001
        aligned = latest <= duration + tolerance
        record.update(source_sha256=expected['sha256'], media=probe,
                      annotation_max_end_seconds=latest,
                      annotation_end_within_source_duration=aligned,
                      source_end_tolerance_seconds=tolerance)
        if not aligned:
            timeline_mismatches.append(dict(video=name, duration_seconds=duration,
                                           annotation_end_seconds=latest, excess_seconds=latest - duration))
        if len(journal) % 25 == 0 or len(journal) == 818:
            status('oops_timeline', 'decoding', completed=len(journal), expected=818)
    require(len(journal) == 818, 'OOPS decode incomplete')
    label_path = source.OOPS / 'metadata/labels/OOPS.csv'
    require(sha(label_path) == annotation_report['label_file_sha256'], 'source annotation changed')
    save(OUTPUT / 'oops_input_manifest.json', dict(
        records=records, annotation_report=annotation_report, debug_subset=False,
        source_videos=818, source_modified=False, canonical_fps=25,
        canonical_frames_generated=False, event_ground_truth_ready=False,
        usable_for_model_evaluation=False))
    report = dict(passed=not timeline_mismatches, stage='input_audit_only', videos=818,
                  fully_decoded_videos=818,
                  decoded_frames=sum(r['probe']['decoded_frames'] for r in journal.values()),
                  source_duration_hours=sum(r['probe']['source_duration_seconds'] for r in journal.values()) / 3600,
                  annotation_report=annotation_report,
                  source_timeline_mismatches=timeline_mismatches,
                  source_timeline_ready=not timeline_mismatches,
                  event_ground_truth_ready=False, model_evaluation_run=False,
                  input_manifest_sha256=sha(OUTPUT / 'oops_input_manifest.json'),
                  decoded_videos_sha256=sha(path),
                  transcoded=False, source_label_file_modified=False)
    save(OUTPUT / 'oops_final_report.json', report)
    status('oops_timeline', 'completed' if report['passed'] else 'needs_timeline_review', **{
        k: v for k, v in report.items() if k not in ('stage', 'annotation_report', 'source_timeline_mismatches')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('safer_ood', 'oops_timeline'), required=True)
    args = parser.parse_args()
    require(Path(sys.prefix).name == 'fall_detect' and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
            'fall_detect CPU-only required')
    require(OUTPUT.resolve().is_relative_to(ROOT), 'output outside project')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / (args.task + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dataset = 'oops' if args.task == 'oops_timeline' else 'safer_ood'
        contract = dict(code_sha256=sha(Path(__file__)), helper_sha256=sha(Path(source.__file__)),
                        source_manifest_sha256=sha(source.CONTROL / (dataset + '_source_manifest.json')),
                        reserve_bytes=source.RESERVE, task=args.task)
        if dataset == 'oops':
            contract['source_videos_sha256'] = sha(source.CONTROL / 'oops_videos.json')
        contract_path = OUTPUT / (args.task + '_contract.json')
        if contract_path.exists():
            require(read(contract_path) == contract, 'preparation code or source contract changed')
        else:
            save(contract_path, contract)
        try:
            safety()
            status(args.task, 'preflight')
            safer_ood() if args.task == 'safer_ood' else oops_timeline()
        except Exception as error:
            paused = any((p / 'PAUSE_REQUESTED').exists() for p in (OUTPUT, source.CONTROL))
            status(args.task, 'paused' if paused else 'failed', error_type=type(error).__name__,
                   detail=str(error) if type(error) is RuntimeError else
                   'Transfer/validation failed; completed files preserved.')
            raise SystemExit(1)


if __name__ == '__main__':
    main()
