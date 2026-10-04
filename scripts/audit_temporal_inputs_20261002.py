"""Read-only audit of saved temporal indices; no decoding, model run, or tuning."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/internal/2026-10-02_temporal_input_audit'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def same(a, b):
    np.testing.assert_array_equal(a, b)


def training():
    cache = ROOT / 'data/fall_processed/SAFER-Activities/v3_document_reconstruction_20260922_r3/sequence_cache'
    manifest = read(cache / 'manifest.json')
    splits = {}
    for split, spec in manifest['splits'].items():
        folder = cache / split
        for name, item in spec['payload'].items():
            assert sha(folder / name) == item['sha256']
        assert sha(folder / 'sequences.json') == spec['sequences_sha256']
        assert sha(folder / 'sample_names.json') == spec['sample_names_sha256']
        starts = np.load(folder / 'window_start.npy')
        ids = np.load(folder / 'sequence_index.npy')
        labels = np.load(folder / 'center_coarse_labels.npy')
        derived = np.load(folder / 'center_derived_labels.npy')
        same(derived, np.array([0] * 10 + [1, 2, 3] + [0] * 3)[labels])
        rows = read(folder / 'sequences.json')
        cursor = 0
        for sid, row in enumerate(rows):
            expected = np.arange(0, row['total_frames'] - 63, 8)
            begin, end = row['output_start'], row['output_stop']
            assert begin == cursor and end - begin == len(expected)
            same(starts[begin:end], expected)
            same(ids[begin:end], np.full(len(expected), sid))
            source = cache / row['uid'] / 'labels.npy'
            assert sha(source) == manifest['sequence_files'][row['uid']]['payload']['labels.npy']['sha256']
            same(labels[begin:end], np.load(source)[expected + 32])
            pose = np.load(cache / row['uid'] / 'ntu25.npy', mmap_mode='r')
            assert pose.shape == (row['total_frames'], 25, 3)
            cursor = end
        assert cursor == len(starts) == spec['count']
        splits[split] = dict(sequences=len(rows), windows=len(starts))
    return dict(passed=True, sequences=manifest['sequences'], frames=manifest['frames'],
                windows=sum(s['windows'] for s in splits.values()), splits=splits,
                all_starts_stride8=True, all_center_labels_offset32=True,
                scope='All stored index/label metadata and pose shapes; raw source video timestamps and all pose values not reverified')


def stages(folder, n, indices, times):
    front = read(folder / 'frontend.json')
    assert sha(folder / 'frontend.npz') == front['payload_sha256']
    with np.load(folder / 'frontend.npz') as values:
        same(values['source_indices'], indices)
        same(values['timestamps'], times)
        assert values['xy'].shape == (n, 17, 2)
    if not front['quality']['passed']:
        assert (folder / 'quality_rejection.json').exists()
        assert not (folder / 'inference.npz').exists()
        return dict(classified=False, windows=0)
    starts = np.arange(0, n - 63, 8)
    for name in ['lift', 'inference']:
        meta = read(folder / (name + '.json'))
        assert sha(folder / (name + '.npz')) == meta['payload_sha256']
    with np.load(folder / 'lift.npz') as values:
        same(values['window_starts'], starts)
        assert values['ntu25'].shape == (n, 25, 3)
        lifting = [0] if n <= 243 else list(range(0, n - 242, 121))
        if n > 243 and lifting[-1] != n - 243:
            lifting.append(n - 243)
        same(values['lifting_starts'], lifting)
        assert values['lifting_raw'].shape == (len(lifting), 243, 17, 3)
    with np.load(folder / 'inference.npz') as values:
        same(values['window_starts'], starts)
        same(values['window_endpoints'], starts + 63)
        assert values['G0'].shape == (len(starts), 4)
    return dict(classified=True, windows=len(starts), lifting_windows=len(lifting))


def continuous(run):
    folder = ROOT / 'data/fall_processed/RGB' / run
    rows = read(folder / 'plan.json')
    records = []
    for row in rows:
        if row.get('preprocessing_rejection'):
            dest = folder / row['id']
            assert not (dest / 'frontend.npz').exists()
            assert not (dest / 'inference.npz').exists()
            assert read(dest / 'frontend.json')['quality']['passed'] is False
            records.append(dict(id=row['id'], frames=0, classified=False, windows=0,
                                preprocessing_rejection=row['preprocessing_rejection'],
                                repeated_positions=0, precision_shifted_positions=0))
            continue
        path = ROOT / row['mapping']
        assert sha(path) == row['mapping_sha256']
        with np.load(path) as mapping:
            pts = mapping['source_timestamps']
            times = mapping['canonical_timestamps']
            indices = mapping['source_indices']
        assert len(times) == row['frames'] and np.all(np.diff(pts) > 0)
        same(times, np.arange(len(times)) / 25)
        same(indices, np.searchsorted(pts, times, side='right') - 1)
        assert indices.min() >= 0 and indices.max() < len(pts)
        assert np.all(np.diff(indices) >= 0) and np.all(pts[indices] <= times)
        # Diagnostic only: compare effectively coincident times as integer nanoseconds.
        # Never replace stored mappings or predictions with these indices.
        rounded = np.searchsorted(np.rint(pts * 1e9).astype(np.int64),
                                  np.rint(times * 1e9).astype(np.int64), side='right') - 1
        changed = np.flatnonzero(rounded != indices)
        diagnostics = dict(precision_shifted_positions=len(changed),
                           rounded_repeated_positions=int((np.diff(rounded) == 0).sum()))
        if len(changed):
            assert np.all(rounded[changed] == indices[changed] + 1)
            assert np.all(np.abs(pts[rounded[changed]] - times[changed]) < 1e-9)
            i = int(changed[0])
            diagnostics['precision_example'] = dict(position=i, selected=int(indices[i]),
                coincident=int(rounded[i]), canonical_time=float(times[i]),
                source_time=float(pts[rounded[i]]), selected_lag=float(times[i] - pts[indices[i]]))
        status = stages(folder / row['id'], len(times), indices, times)
        records.append(dict(id=row['id'], frames=len(times), source_fps=row.get('source_fps'),
                            repeated_positions=int((np.diff(indices) == 0).sum()),
                            max_past_lag_seconds=float((times - pts[indices]).max()), **diagnostics, **status))
    return dict(passed=True, plan_sha256=sha(folder / 'plan.json'), evaluation_sha256=sha(folder / 'evaluation.json'),
                videos=len(rows), frames=sum(r['frames'] for r in records),
                classified=sum(r['classified'] for r in records), windows=sum(r['windows'] for r in records),
                videos_with_repeats=sum(r['repeated_positions'] > 0 for r in records),
                repeated_positions=sum(r['repeated_positions'] for r in records),
                precision_affected_videos=sum(r['precision_shifted_positions'] > 0 for r in records),
                precision_shifted_positions=sum(r['precision_shifted_positions'] for r in records),
                mapped_videos=sum(r['frames'] > 0 for r in records), records=records)


def segments(run):
    folder = ROOT / 'data/fall_processed/RGB' / run
    rows = read(folder / 'plan.json')
    truth = {r['id']: r['fall'] for r in read(folder / 'ground_truth.json')}
    records, last_by_video = [], {}
    reversals = 0
    for row in rows:
        sample = row['sampling']
        idx = np.array(sample['source_indices'])
        pts = np.array(sample['source_pts'])
        actual = np.array(sample['actual_seconds'])
        requested = np.array(sample['requested_seconds'])
        assert len(idx) == len(actual) == 64
        assert np.all(np.diff(idx) >= 0) and np.all(np.diff(actual) >= 0) and np.all(np.diff(pts) >= 0)
        expected = np.array([row['start'] + i * (row['end'] - row['start']) / 63 for i in range(64)])
        same(requested, expected)
        assert len(np.unique(idx)) == sample['unique_frames']
        if row['path'] in last_by_video:
            reversals += int(row['start'] < last_by_video[row['path']])
        last_by_video[row['path']] = row['start']
        status = stages(folder / row['id'], 64, idx, actual)
        if status['classified']:
            assert status['windows'] == status['lifting_windows'] == 1
        records.append(dict(id=row['id'], video=row['path'], fall=bool(truth[row['id']]),
                            duration=row['end'] - row['start'], unique_frames=sample['unique_frames'],
                            repeated_positions=int((np.diff(idx) == 0).sum()),
                            skipped_transitions=int((np.diff(idx) > 1).sum()),
                            max_boundary_excursion_seconds=sample['max_boundary_excursion_seconds'], **status))
    falls = [r for r in records if r['fall']]
    return dict(passed=True, plan_sha256=sha(folder / 'plan.json'), evaluation_sha256=sha(folder / 'evaluation.json'),
                segments=len(rows), videos=len(last_by_video), frame_positions=64 * len(rows),
                classified=sum(r['classified'] for r in records),
                segments_with_repeats=sum(r['repeated_positions'] > 0 for r in records),
                segments_with_skips=sum(r['skipped_transitions'] > 0 for r in records),
                minimum_unique_frames=min(r['unique_frames'] for r in records),
                fall_segments=len(falls), fall_duration_median=float(np.median([r['duration'] for r in falls])),
                fall_segments_shorter_than_2_52s=sum(r['duration'] < 2.52 for r in falls),
                processing_order_reversals_within_video=reversals, records=records)


def main():
    report = dict(document_id='DOC-20261002-temporal-input-audit-R1', passed=False, training=training())
    report['continuous'] = {run: continuous(run) for run in [
        'le2i_current_evaluation_20261002_r1', 'omnifall_le2i_continuous_20261002_r1',
        'external_urfd_20261002_r1']}
    report['segments'] = {run: segments(run) for run in [
        'omnifall_gmdcsa_cs_20261002_r1', 'omnifall_le2i_cs_20261002_r1', 'omnifall_cauca_cs_20261002_r1']}
    report['scope'] = 'Saved metadata and stage payload hashes; no raw decoding, GPU inference, retraining, or output modification'
    report['input_issues'] = ['GT-segment uniform64 changes physical duration and sequence context',
        'Strict float past-only comparison selects preceding frame at effectively equal Le2i timestamps']
    report['passed_meaning'] = 'Audit assertions and saved artifact consistency passed; not absence of input issues'
    report['script_sha256'] = sha(Path(__file__))
    report['passed'] = True
    OUT.mkdir(exist_ok=True)
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['continuous', 'segments']}, indent=2))
    for group in ['continuous', 'segments']:
        for run, value in report[group].items():
            print(run, json.dumps({k: v for k, v in value.items() if k != 'records'}))


if __name__ == '__main__':
    main()
