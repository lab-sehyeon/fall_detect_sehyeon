"""Pixel-verified FFV1 conversion and current-pipeline Le2i time alignment.

Raw AVI is decoded only by FFmpeg. OpenCV sees only derived FFV1, in an
isolated child for each clip. Source frames/annotations are never modified.
"""
import argparse
import fcntl
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fall_pipeline.external import rgb_document_io as io
from data_gen.rgb_alignment_reconstruction import past_frame_indices

SOURCE = ROOT / 'data/source_archives/Le2i-FDD/kaggle_v2_20261002_r1'
TOOLS = ROOT / 'data/source_archives/Le2i-FDD/ffmpeg_20261002_r1'
DEST = ROOT / 'data/source_archives/Le2i-FDD/ffv1_20261002_r1'


def safety():
    io.require(not (DEST/'PAUSE_REQUESTED').exists(), 'pause requested')
    io.require(shutil.disk_usage(ROOT).free > 64*1024**3, '64 GiB reserve')


def selected():
    source = io.read(SOURCE/'manifest.json')
    audit = io.read(SOURCE/'annotation_audit.json')
    io.require(source['passed'] and audit['passed'], 'acquisition incomplete')
    by_name = {r['source_name']: r for r in source['files']}
    rows = []
    for record in audit['records']:
        if record['label'] == 'excluded': continue
        annotation_name = record['source_name']
        scene = annotation_name.split('/')[0]
        number = int(re.search(r'video \((\d+)\)', annotation_name).group(1))
        video = by_name[f'{scene}/{scene}/Videos/video ({number}).avi']
        annotation = by_name[annotation_name]
        rows.append(dict(id=f'{scene}_{number:03d}', scene=scene, number=number,
                         video=video, annotation=annotation, raw_header=record['raw_header']))
    io.require(len(rows) == 127 and sum(r['raw_header'] != [0, 0] for r in rows) == 96, 'subset changed')
    return sorted(rows, key=lambda r: (r['scene'], r['number']))


def probe(path, frames=False):
    command = [str(TOOLS/'ffprobe'), '-v', 'error', '-select_streams', 'v:0', '-of', 'json']
    if frames:
        command += ['-show_frames', '-show_entries', 'frame=best_effort_timestamp_time']
    else:
        command += ['-show_entries', 'stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,pix_fmt']
    return json.loads(subprocess.check_output(command+[str(path)], stderr=subprocess.DEVNULL, timeout=120))


def raw_pixel_digest(source, logfile):
    command = [str(TOOLS/'ffmpeg'), '-nostdin', '-v', 'error', '-threads', '2', '-i', str(source),
               '-map', '0:v:0', '-an', '-sn', '-dn', '-pix_fmt', 'bgr24', '-f', 'rawvideo', '-threads', '2', 'pipe:1']
    digest = hashlib.sha256(); size = 0
    with logfile.open('ab') as errors, subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors) as process:
        while True:
            safety()
            block = process.stdout.read(1024**2)
            if not block: break
            digest.update(block); size += len(block)
        io.require(process.wait(timeout=120) == 0, 'source ffmpeg decode failed')
    return digest.hexdigest(), size


def one(row):
    import cv2
    cv2.setNumThreads(1)
    safety()
    sid = row['id']; target = DEST/(sid+'.mkv'); receipt = DEST/(sid+'.json')
    raw = SOURCE/row['video']['relative_path']; annotation = SOURCE/row['annotation']['relative_path']
    io.require(io.sha(raw) == row['video']['sha256'] and io.sha(annotation) == row['annotation']['sha256'], 'source changed')
    if receipt.exists():
        result = io.read(receipt)
        io.require(result['source_sha256'] == io.sha(raw) and result['video_sha256'] == io.sha(target)
                   and result['mapping_sha256'] == io.sha(ROOT/result['mapping']), 'completed preparation changed')
        return
    metadata = probe(raw)['streams'][0]
    fps = float(Fraction(metadata['avg_frame_rate']))
    width, height, count = metadata['width'], metadata['height'], int(metadata['nb_frames'])
    expected_fps = 25 if row['scene'].startswith('Coffee') else 24
    io.require(fps == expected_fps and width == 320 and height in (180, 240), 'unexpected source media metadata')
    log = DEST/(sid+'.ffmpeg.log')
    if not target.exists():
        pending = DEST/(sid+'.pending.mkv')
        io.require(not pending.exists(), 'unverified temporary remux exists; preserve and inspect')
        command = [str(TOOLS/'ffmpeg'), '-nostdin', '-n', '-v', 'error', '-threads', '2', '-i', str(raw),
                   '-map', '0:v:0', '-an', '-sn', '-dn', '-c:v', 'ffv1', '-level', '3', '-pix_fmt', 'bgr0',
                   '-threads', '2', '-f', 'matroska', str(pending)]
        with log.open('ab') as errors:
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=errors, check=True, timeout=300)
        pending.rename(target)
    original_hash, original_bytes = raw_pixel_digest(raw, log)
    io.require(original_bytes == width*height*3*count, 'source decoded frame count mismatch')
    cap = cv2.VideoCapture(str(target)); io.require(cap.isOpened(), 'FFV1 cannot open')
    digest = hashlib.sha256(); pts = []; decoded = 0
    try:
        while True:
            safety()
            ok, frame = cap.read()
            if not ok: break
            io.require(frame.shape == (height, width, 3), 'derived image shape changed')
            digest.update(frame.tobytes()); pts.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000); decoded += 1
    finally: cap.release()
    io.require(decoded == count and digest.hexdigest() == original_hash, 'lossless pixel equality failed')
    reference_pts = np.array([float(f['best_effort_timestamp_time']) for f in probe(target, True)['frames']])
    np.testing.assert_allclose(pts, reference_pts, atol=1e-6, rtol=0)
    duration = count/fps
    times, indices = past_frame_indices(pts, duration, 25)
    mapping = DEST/(sid+'_mapping.npz')
    io.npz(mapping, source_timestamps=np.array(pts), canonical_timestamps=times, source_indices=indices)
    result = dict(id=sid, scene=row['scene'], number=row['number'], source=str(raw.relative_to(ROOT)),
                  source_sha256=row['video']['sha256'], annotation=str(annotation.relative_to(ROOT)),
                  annotation_sha256=row['annotation']['sha256'], raw_header=row['raw_header'],
                  source_frames=count, source_fps=fps, source_width=width, source_height=height,
                  source_duration_seconds=duration, pixel_sha256=original_hash, pixel_identical=True,
                  video=str(target.relative_to(ROOT)), video_sha256=io.sha(target), bytes=target.stat().st_size,
                  mapping=str(mapping.relative_to(ROOT)), mapping_sha256=io.sha(mapping), frames=len(times),
                  alignment='current fixed actualPTS past-only25fps; different from historical nearest-frame')
    io.save(receipt, result)
    print(sid, count, 'frames pixel-identical; canonical', len(times), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--one'); a = p.parse_args()
    io.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and Path(sys.prefix).name == 'fall_detect', 'CPU fall_detect only')
    io.require(not any(path.is_symlink() for path in (DEST, *DEST.parents)), 'unsafe output path')
    DEST.mkdir(parents=True, exist_ok=True)
    rows = selected()
    if a.one:
        one(next(row for row in rows if row['id'] == a.one)); return
    tool_manifest = io.read(TOOLS/'manifest.json')
    for name in ('ffmpeg', 'ffprobe'):
        io.require(io.sha(TOOLS/name) == tool_manifest['files'][name]['sha256'], 'tool changed')
    contract = dict(code_sha256=io.sha(Path(__file__)), tools_manifest_sha256=io.sha(TOOLS/'manifest.json'),
                    acquisition_manifest_sha256=io.sha(SOURCE/'manifest.json'),
                    alignment_code_sha256=io.sha(ROOT/'data_gen/rgb_alignment_reconstruction.py'),
                    videos=[r['id'] for r in rows], expected=127, reserve_gib=64, budget_gib=16,
                    raw_avi_opencv_decode=False, timing='actualPTS past-only25fps; duration source_frames/source_fps')
    with (DEST/'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (DEST/'contract.json').exists(): io.require(io.read(DEST/'contract.json') == contract, 'preparation contract changed')
        else: io.save(DEST/'contract.json', contract)
        try:
            for number, row in enumerate(rows, 1):
                safety()
                subprocess.run([sys.executable, str(Path(__file__)), '--one', row['id']], cwd=ROOT, check=True, timeout=600)
                io.save(DEST/'status.json', dict(stage='preparing', completed=number, total=len(rows), id=row['id']))
            records = [io.read(DEST/(r['id']+'.json')) for r in rows]
            io.require(sum(r['bytes'] for r in records) < 16*1024**3, 'output budget exceeded')
            io.save(DEST/'manifest.json', dict(passed=True, records=records, pixel_identical=len(records),
                    source_duration_hours=sum(r['source_duration_seconds'] for r in records)/3600,
                    canonical_frames=sum(r['frames'] for r in records), bytes=sum(r['bytes'] for r in records),
                    contract_sha256=io.sha(DEST/'contract.json')))
            io.save(DEST/'status.json', dict(stage='completed', completed=len(records), total=len(rows)))
        except Exception as exc:
            io.save(DEST/'status.json', dict(stage='paused', error=f'{type(exc).__name__}: {exc}')); raise


if __name__ == '__main__': main()
