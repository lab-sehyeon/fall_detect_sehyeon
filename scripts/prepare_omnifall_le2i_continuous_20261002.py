"""Lossless full-video inputs for the fixed official Le2i-CS test list."""
from pathlib import Path
from fractions import Fraction
import argparse, fcntl, hashlib, os, shutil, subprocess, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fall_pipeline.external import rgb_document_io as io
from fall_pipeline.external.omnifall_le2i_20261002 import ASSET, official_rows
from data_gen.rgb_alignment_reconstruction import past_frame_indices
from scripts import prepare_le2i_current_evaluation_r2 as previous

DEST = ROOT/'data/source_archives/OmniFall/le2i_continuous_20261002_r1'
OLD = ROOT/'data/source_archives/Le2i-FDD/ffv1_20261002_r2'
TOOLS = previous.TOOLS


def identity(path):
    scene, basename = path.split('/')
    number = int(basename.removeprefix('video_'))
    return f'{scene}_{number:03d}', scene, number


def safety():
    io.require(not (DEST/'PAUSE_REQUESTED').exists(), 'pause requested')
    io.require(shutil.disk_usage(ROOT).free > 64*1024**3, '64GiB reserve')
    io.require(sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file()) < 8*1024**3, '8GiB input budget')


def selected():
    _, paths = official_rows()
    manifest = io.read(ASSET/'original_manifest.json')
    io.require(manifest['passed'], 'original assets incomplete')
    originals = {r['path']: r for r in manifest['files']}
    io.require(set(originals) == set(paths), 'original scope')
    return [originals[p] for p in paths]


def verify(row):
    for key in ['source', 'video', 'mapping']:
        io.require(io.sha(ROOT/row[key]) == row[key+'_sha256'], key+' changed')
    io.require(row['pixel_identical'], 'pixel equality missing')


def one(source):
    safety()
    sid, scene, number = identity(source['path'])
    receipt = DEST/(sid+'.json')
    if receipt.exists():
        row = io.read(receipt); verify(row)
        io.require(row['source_sha256'] == source['sha256'], 'resumed source changed')
        return
    prior = {r['id']: r for r in io.read(OLD/'manifest.json')['records']}
    if sid in prior:
        row = prior[sid]; verify(row)
        io.require(row['source_sha256'] == source['sha256'], 'prior original mismatch')
        io.save(receipt, dict(row, path=source['path'], reused_preparation=True,
            prior_manifest_sha256=io.sha(OLD/'manifest.json')))
        print(sid, 'reused pixel-verified full-video input', flush=True)
        return
    import cv2
    cv2.setNumThreads(1)
    raw = ROOT/source['local']
    io.require(io.sha(raw) == source['sha256'], 'raw changed')
    m = previous.probe(raw)['streams'][0]
    fps = float(Fraction(m['avg_frame_rate']))
    width, height, count = m['width'], m['height'], int(m['nb_frames'])
    io.require(abs(fps-25) < .001 and width == 320 and height == 240, 'new scene metadata outside contract')
    video = DEST/(sid+'.mkv'); log = DEST/(sid+'.ffmpeg.log')
    if not video.exists():
        pending = DEST/(sid+'.pending.mkv')
        io.require(not pending.exists(), 'preserve and inspect incomplete conversion')
        command = [str(TOOLS/'ffmpeg'), '-nostdin', '-n', '-v', 'error', '-threads', '2', '-i', str(raw),
            '-map', '0:v:0', '-an', '-sn', '-dn', '-c:v', 'ffv1', '-level', '3', '-pix_fmt', 'bgr0',
            '-threads', '2', '-f', 'matroska', str(pending)]
        with log.open('ab') as errors:
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=errors, check=True, timeout=300)
        pending.rename(video)
    previous.safety = safety
    original_hash, original_bytes = previous.raw_pixel_digest(raw, log)
    io.require(original_bytes == width*height*3*count, 'raw frame count')
    cap = cv2.VideoCapture(str(video)); io.require(cap.isOpened(), 'FFV1 open failed')
    digest = hashlib.sha256(); pts = []
    try:
        while True:
            safety(); ok, frame = cap.read()
            if not ok: break
            io.require(frame.shape == (height, width, 3), 'frame shape')
            digest.update(frame.tobytes()); pts.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
    finally:
        cap.release()
    io.require(len(pts) == count and digest.hexdigest() == original_hash, 'lossless equality failed')
    reference = [float(f['best_effort_timestamp_time']) for f in previous.probe(video, True)['frames']]
    np.testing.assert_allclose(pts, reference, atol=1e-6, rtol=0)
    duration = count/fps
    times, indices = past_frame_indices(pts, duration, 25)
    mapping = DEST/(sid+'_mapping.npz')
    io.npz(mapping, source_timestamps=np.array(pts), canonical_timestamps=times, source_indices=indices)
    io.save(receipt, dict(id=sid, path=source['path'], scene=scene, number=number,
        source=source['local'], source_sha256=source['sha256'], source_frames=count, source_fps=fps,
        source_fps_rational=m['avg_frame_rate'], source_width=width, source_height=height,
        source_duration_seconds=duration, pixel_sha256=original_hash, pixel_identical=True,
        video=str(video.relative_to(ROOT)), video_sha256=io.sha(video), bytes=video.stat().st_size,
        mapping=str(mapping.relative_to(ROOT)), mapping_sha256=io.sha(mapping), frames=len(times),
        alignment='current fixed actualPTS past-only25fps; different from historical nearest-frame',
        reused_preparation=False))
    print(sid, count, 'pixel-identical full-video frames', flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--one'); args = parser.parse_args()
    io.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and Path(sys.prefix).name == 'fall_detect', 'CPU fall_detect required')
    io.require(not any(p.is_symlink() for p in (DEST, *DEST.parents)), 'unsafe destination')
    DEST.mkdir(parents=True, exist_ok=True)
    rows = selected()
    if args.one:
        one(next(r for r in rows if identity(r['path'])[0] == args.one)); return
    for name, spec in io.read(TOOLS/'manifest.json')['files'].items():
        io.require(io.sha(TOOLS/name) == spec['sha256'], 'FFmpeg tool changed')
    contract = dict(code={p:io.sha(ROOT/p) for p in [
        'scripts/prepare_omnifall_le2i_continuous_20261002.py', 'scripts/prepare_le2i_current_evaluation_r2.py',
        'data_gen/rgb_alignment_reconstruction.py', 'fall_pipeline/external/omnifall_le2i_20261002.py']},
        original_manifest_sha256=io.sha(ASSET/'original_manifest.json'),
        prior_manifest_sha256=io.sha(OLD/'manifest.json'), tool_manifest_sha256=io.sha(TOOLS/'manifest.json'),
        videos=[r['path'] for r in rows], expected=38, reserve_gib=64, budget_gib=8)
    with (DEST/'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (DEST/'contract.json').exists(): io.require(io.read(DEST/'contract.json') == contract, 'preparation contract changed')
        else: io.save(DEST/'contract.json', contract)
        for n, row in enumerate(rows, 1):
            safety(); sid = identity(row['path'])[0]
            subprocess.run([sys.executable, str(Path(__file__)), '--one', sid], cwd=ROOT, check=True, timeout=600)
            io.save(DEST/'status.json', dict(stage='preparing', completed=n, total=38, id=sid))
        records = [io.read(DEST/(identity(r['path'])[0]+'.json')) for r in rows]
        io.require(sum(r['reused_preparation'] for r in records) == 22, 'expected22 inherited16 new')
        io.save(DEST/'manifest.json', dict(passed=True, records=records, pixel_identical=38,
            reused=22, fresh=16, contract_sha256=io.sha(DEST/'contract.json')))
        io.save(DEST/'status.json', dict(stage='completed', completed=38, total=38))


if __name__ == '__main__': main()
