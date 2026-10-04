"""Derived alignment only: immutable official pixels/poses and bounded annotations."""
import fcntl
import os
from pathlib import Path
import sys
import numpy as np

from scripts import prepare_rgb_recovery_data_r2 as prep
from scripts.audit_oops_container_timing import movie_headers

ROOT = prep.ROOT
OUTPUT = ROOT / 'data/source_archives/rgb_alignment_20260928_r1'
REFS = {'extract_keypoints.py': '789b6e734549aaef74f1ecbe4d2b92735e904e3264fe721192b8ae2eebc77820',
        'yolov8_vitpose_extract_keypoints_single_video.py': 'e94b4850276cb33d156495a18ae5ae361d31baf39386e6fd0f159725e2f45594'}


def clip_intervals(intervals, duration):
    output, changes = [], []
    for row in intervals:
        start, end = float(row['start']), float(row['end'])
        prep.require(np.isfinite([start, end, duration]).all() and 0 <= start < end and duration > 0,
                     'invalid interval or duration')
        left, right = min(start, duration), min(end, duration)
        if right != end or left != start:
            changes.append(dict(source_row=row['source_row'], label=row['label'],
                                original_start=start, original_end=end, corrected_start=left,
                                corrected_end=right, removed_seconds=end-start-max(0., right-left),
                                entirely_outside=right <= left))
        if left < right:
            output.append({**row, 'start': left, 'end': right})
    return output, changes


def past_frame_indices(pts, duration, fps=25):
    pts = np.asarray(pts, np.float64)
    prep.require(len(pts) > 0 and np.isfinite(pts).all() and abs(pts[0]) < 1e-6
                 and (np.diff(pts) > 0).all() and pts[-1] < duration, 'invalid decoded timeline')
    times = np.arange(int(np.ceil(duration * fps)), dtype=np.float64) / fps
    times = times[times < duration]
    indices = np.searchsorted(pts, times, side='right') - 1
    prep.require((indices >= 0).all() and (indices < len(pts)).all()
                 and (pts[indices] <= times).all(), 'future/missing source frame')
    return times, indices.astype(np.int64)


def align():
    import cv2
    cv2.setNumThreads(1)
    prep.require(Path(sys.prefix).name == 'fall_detect' and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
                 'CPU-only fall_detect required')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, digest in REFS.items():
        prep.require(prep.sha(OUTPUT / 'official_reference' / name) == digest, 'official code evidence changed')
    contract = dict(code_sha256=prep.sha(Path(__file__)), references=REFS,
                    input_manifest_sha256=prep.sha(prep.OUTPUT / 'oops_input_manifest.json'),
                    source_audit_sha256=prep.sha(prep.OUTPUT / 'safer_global_source_audit/report.json'),
                    interval_policy='intersection with video track [0,duration)', fps=25,
                    sampling='latest actual decoded timestamp <= endpoint; no future',
                    original_modified=False, historical_exact=False, post_hoc_data_correction=True)
    path = OUTPUT / 'contract.json'
    if path.exists(): prep.require(prep.read(path) == contract, 'alignment contract changed')
    else: prep.save(path, contract)
    with (OUTPUT / 'alignment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = prep.read(prep.OUTPUT / 'oops_input_manifest.json')
        path = OUTPUT / 'oops_records.json'
        journal = prep.read(path) if path.exists() else {}
        for number, item in enumerate(source['records'], 1):
            prep.safety()
            prep.require(not (OUTPUT / 'PAUSE_REQUESTED').exists(), 'alignment pause requested')
            name = item['video']; video = prep.source.safe_path(prep.source.OOPS, name)
            prep.require(prep.sha(video) == item['source_sha256'], 'OOPS video changed')
            if name not in journal:
                headers = movie_headers(video)
                tracks = [r for r in headers['tracks'] if r['handler'] == 'vide']
                prep.require(len(tracks) == 1, 'video track count')
                duration = tracks[0]['media_duration']['seconds']
                prep.require(abs(duration - item['media']['source_duration_seconds']) < 1e-6,
                             'decoded/header duration mismatch')
                cap = cv2.VideoCapture(str(video)); times = []
                try:
                    while True:
                        ok, frame = cap.read()
                        if not ok: break
                        times.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
                finally: cap.release()
                prep.require(len(times) == item['media']['frames'], 'frame count changed')
                timeline, indices = past_frame_indices(times, duration)
                intervals, changes = clip_intervals(item['annotations'], duration)
                dest = OUTPUT / 'oops_timestamps' / f'{number:04d}.npz'
                dest.parent.mkdir(exist_ok=True)
                with dest.with_suffix('.tmp').open('wb') as stream:
                    np.savez_compressed(stream, source_timestamps=np.array(times),
                                        canonical_timestamps=timeline, source_indices=indices)
                os.replace(dest.with_suffix('.tmp'), dest)
                journal[name] = dict(source_sha256=item['source_sha256'], video_duration=duration,
                                     movie_duration=headers['movie_duration']['seconds'],
                                     annotations=intervals, changes=changes, canonical_frames=len(timeline),
                                     supports_64_frame_window=len(timeline) >= 64,
                                     mapping_file=str(dest.relative_to(ROOT)), mapping_sha256=prep.sha(dest))
                prep.save(path, journal)
            else:
                prep.require(journal[name]['source_sha256'] == item['source_sha256'] and
                             prep.sha(ROOT / journal[name]['mapping_file']) == journal[name]['mapping_sha256'],
                             'completed alignment changed')
            if number % 25 == 0 or number == 818:
                prep.save(OUTPUT / 'status.json', dict(stage='aligning_oops', completed=number, expected=818))
                print(f'OOPS alignment {number}/818', flush=True)
        audit = prep.read(prep.OUTPUT / 'safer_global_source_audit/report.json')
        safer = []
        for row in audit['ood_media_comparison_snapshot']:
            prep.require(row['frames_equal'] and row['pose_image_dimensions'] == [1920, 1080],
                         'SAFER source space changed')
            w, h = row['rgb_image_dimensions']; sx, sy = 1920 / w, 1080 / h
            points = np.array([[0.,0.],[w / 2,h / 2],[w,h]])
            mapped = points * [sx,sy]
            np.testing.assert_allclose(mapped / [1920,1080], points / [w,h], atol=1e-12, rtol=0)
            np.testing.assert_allclose(mapped / [sx,sy], points, atol=1e-12, rtol=0)
            safer.append({**row, 'action':'resize RGB to1920x1080; preserve source poses',
                          'rgb_to_pose_edge_coordinate_scale':[sx,sy], 'aligned':True})
        prep.require(len(safer) == 30 and len(journal) == 818, 'alignment population')
        changes = [c for r in journal.values() for c in r['changes']]
        report = dict(passed=True, oops_videos=818, safer_ood_sequences=30,
                      safer_resize_required=sum(not r['image_dimensions_equal'] for r in safer),
                      clipped_annotation_rows=len(changes),
                      fully_outside_annotations=sum(c['entirely_outside'] for c in changes),
                      removed_seconds=sum(c['removed_seconds'] for c in changes),
                      retained_fall_rows=sum(a['label']=='fall' for r in journal.values() for a in r['annotations']),
                      short_videos_retained=sum(not r['supports_64_frame_window'] for r in journal.values()),
                      all_endpoints_use_past_source_frame=True, raw_data_modified=False,
                      event_ground_truth_ready=False, model_evaluation_run=False,
                      oops_records_sha256=prep.sha(path), safer_alignment=safer, contract=contract)
        prep.save(OUTPUT / 'final_report.json', report)
        prep.save(OUTPUT / 'status.json', dict(stage='completed', report_sha256=prep.sha(OUTPUT / 'final_report.json')))
        print({k:v for k,v in report.items() if k not in ('safer_alignment','contract')}, flush=True)


if __name__ == '__main__': align()
