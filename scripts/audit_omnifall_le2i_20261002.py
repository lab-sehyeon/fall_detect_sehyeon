"""Independent CPU scoring and checkpoint replay for the fixed 203-segment run."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score, recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ASSET = ROOT / 'data/source_archives/OmniFall/le2i_cs_20261002_r1'
OUT = ROOT / 'data/fall_processed/RGB/omnifall_le2i_cs_20261002_r1'
DEST = OUT / 'audit'
sys.path.insert(0, str(ROOT / 'data/source_archives/OmniFall/gmdcsa24_cs_20261002_r1/runtime'))
import pyarrow.parquet as pq


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def stage(directory, name):
    meta = read(directory / (name + '.json'))
    assert meta['passed']
    assert sha(directory / (name + '.npz')) == meta['payload_sha256']
    if 'models_before' in meta:
        assert meta['models_before'] == meta['models_after']
    if 'model_before' in meta:
        assert meta['model_before'] == meta['model_after']
    return meta


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    from fall_pipeline.external.omnifall_le2i_20261002 import setup, official_rows
    from fall_pipeline.external import rgb_document_io as io
    cfg = setup()
    io.locked(cfg)
    parquet = pq.read_table(ASSET / 'metadata/parquet/le2i/test-00000-of-00001.parquet').to_pylist()
    official, official_paths = official_rows()
    assert parquet == official
    for r in read(ASSET/'original_manifest.json')['files']:
        from scripts.prepare_omnifall_le2i_20261002 import crc
        assert sha(ROOT/r['local']) == r['sha256']
        assert (ROOT/r['local']).stat().st_size == r['bytes']
        assert crc(ROOT/r['local']) == r['official_member_crc32']
    plan = read(OUT / 'plan.json')
    evaluation = read(OUT / 'evaluation.json')
    assert len(parquet) == len(plan) == len(evaluation['rows']) == 203
    assert {r['subject'] for r in parquet} == {2,7}
    assert len({r['path'] for r in parquet}) == 38
    assert len({r['id'] for r in plan}) == 203

    global_cfg = read(ROOT / cfg['inference']['global_config'])
    state = torch.load(ROOT / global_cfg['joint_root'] / 'final/final_j1.pt', map_location='cpu', weights_only=True)
    selection = read(ROOT / global_cfg['output_dir'] / 'selection.json')['best']['G0']
    head = torch.load(ROOT / global_cfg['output_dir'] / 'G0' / f"epoch_{selection['epoch']:03d}.pt", map_location='cpu', weights_only=True)
    max_adapted = max_logits = 0.0
    rows = []
    trace_hashes = {}
    for index, (native, item, scored) in enumerate(zip(parquet, plan, evaluation['rows'])):
        assert item['id'] == scored['id'] == f'le2i_cs_{index:04d}'
        assert item['row_index'] == scored['row_index'] == index
        assert native['path'] == item['path']
        assert native['label'] == scored['label']
        assert native['start'] == item['start'] and native['end'] == item['end']
        assert scored['fall'] == int(native['label'] == 1)
        assert sha(ROOT / item['video']) == item['video_sha256']
        trace_path = OUT / 'source_traces' / (item['path'].replace('/', '_') + '.json')
        trace = read(trace_path)
        trace_hashes[str(trace_path.relative_to(ROOT))] = sha(trace_path)
        sample = item['sampling']
        assert len(sample['source_indices']) == len(sample['source_pts']) == len(sample['actual_seconds']) == 64
        assert len(set(sample['source_indices'])) == sample['unique_frames']
        for i, pts, rgb_hash in zip(sample['source_indices'], sample['source_pts'], sample['pixel_sha256']):
            assert trace['frames'][i] == {'pts': pts, 'rgb_sha256': rgb_hash}

        directory = OUT / item['id']
        front = stage(directory, 'frontend')
        assert front['decoded_rgb_sha256'] == item['decoded_rgb_sha256']
        assert front['labels_used'] is False and front['training'] is False
        with np.load(directory / 'frontend.npz', allow_pickle=False) as z:
            boxes, xy, scores = z['boxes'], z['xy'], z['scores']
            assert xy.shape == (64, 17, 2) and scores.shape == (64, 17)
            np.testing.assert_array_equal(z['source_indices'], sample['source_indices'])
            np.testing.assert_array_equal(z['timestamps'], sample['actual_seconds'])
            bbox_valid = np.isfinite(boxes).all(1) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 4] > 0)
            pose_valid = bbox_valid & np.isfinite(xy).all((1, 2)) & np.isfinite(scores).all(1) & (scores > 0).any(1)
            pelvis = np.where(pose_valid, scores[:, [11, 12]].min(1), 0)
            q = [float(bbox_valid.mean()), float(pose_valid.mean()), float(np.median(pelvis))]
            spec = cfg['quality']
            passed = bool(q[0] >= spec['bbox_coverage_min'] and q[1] >= spec['pose_coverage_min'] and q[2] >= spec['pelvis_median_min'])
        assert passed == front['quality']['passed'] == scored['quality_passed']
        np.testing.assert_array_equal(q, [front['quality'][k] for k in ['bbox_coverage', 'pose_coverage', 'pelvis_median']])
        klass, score, predicted = None, 0.0, 0
        if passed:
            lift = stage(directory, 'lift')
            infer = stage(directory, 'inference')
            assert lift['labels_used'] is False and infer['training'] is False
            assert lift['windows'] == infer['windows'] == 1
            with np.load(directory / 'lift.npz', allow_pickle=False) as z:
                assert z['ntu25'].shape == (64, 25, 3)
                np.testing.assert_array_equal(z['window_starts'], [0])
            with np.load(directory / 'inference.npz', allow_pickle=False) as z:
                assert z['pooled'].shape == z['adapted'].shape == (1, 2048)
                x = torch.from_numpy(z['pooled'])
                with torch.inference_mode():
                    h = F.layer_norm(x, (2048,), state['adapter.norm.weight'], state['adapter.norm.bias'], eps=1e-5)
                    h = F.gelu(F.linear(h, state['adapter.down.weight'], state['adapter.down.bias']), approximate='none')
                    adapted = x + F.linear(h, state['adapter.up.weight'], state['adapter.up.bias'])
                    logits = F.linear(adapted, head['weight'], head['bias']).numpy()
                np.testing.assert_allclose(adapted.numpy(), z['adapted'], atol=1e-4, rtol=1e-5)
                np.testing.assert_allclose(logits, z['G0'], atol=1e-4, rtol=1e-5)
                assert logits.argmax(1).tolist() == z['G0'].argmax(1).tolist()
                max_adapted = max(max_adapted, float(np.max(np.abs(adapted.numpy() - z['adapted']))))
                max_logits = max(max_logits, float(np.max(np.abs(logits - z['G0']))))
                klass = int(z['G0'][0].argmax())
                probabilities = torch.softmax(torch.from_numpy(z['G0'][0].astype(np.float64)), dim=0).numpy()
                score = float(probabilities[1])
                predicted = int(klass == 1)
        else:
            assert not (directory / 'inference.json').exists()
            assert read(directory / 'quality_rejection.json')['classifier_run'] is False
        assert klass == scored['predicted_class'] and predicted == scored['prediction']
        np.testing.assert_allclose(score, scored['fall_score'], atol=1e-12, rtol=1e-12)
        rows.append(dict(id=item['id'], path=item['path'], start=item['start'], end=item['end'],
                         gt_label=int(native['label']), gt_fall=int(native['label'] == 1),
                         prediction=predicted, predicted_class=klass, fall_score=score, quality_passed=passed,
                         unique_frames=sample['unique_frames'], boundary_excursion_seconds=sample['max_boundary_excursion_seconds']))

    y = np.array([r['gt_fall'] for r in rows])
    p = np.array([r['prediction'] for r in rows])
    s = np.array([r['fall_score'] for r in rows])
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel().tolist()
    metrics = dict(segments=len(rows), tp=tp, fp=fp, fn=fn, tn=tn,
                   recall=recall_score(y, p), specificity=recall_score(1-y, 1-p),
                   precision=precision_score(y, p, zero_division=0), f1=f1_score(y, p),
                   accuracy=float(np.mean(y == p)), ap=average_precision_score(y, s),
                   classified=sum(r['quality_passed'] for r in rows),
                   rejected_positive=sum(not r['quality_passed'] and r['gt_fall'] == 1 for r in rows),
                   rejected_negative=sum(not r['quality_passed'] and r['gt_fall'] == 0 for r in rows))
    for key, value in metrics.items():
        np.testing.assert_allclose(value, evaluation['metrics'][key], atol=1e-12, rtol=1e-12)
    per_class = []
    for label in sorted({r['gt_label'] for r in rows}):
        group = [r for r in rows if r['gt_label'] == label]
        per_class.append(dict(label=label, count=len(group), fall_predictions=sum(r['prediction'] for r in group),
                              quality_rejections=sum(not r['quality_passed'] for r in group)))
    report = dict(passed=True, segments=203, videos=38, subjects=[2,7], metrics=metrics, per_class=per_class,
                  checkpoint_replay=dict(classified_segments=metrics['classified'], device='CPU',
                                         max_adapted_abs=max_adapted, max_G0_logit_abs=max_logits,
                                         boundary='independent functional LayerNorm/GELU/residual and selected G0; cached DSTE features'),
                  sampling=dict(pixel_comparison_at_prepare='all 203 x 64 against independent sequential source trace',
                                repeated_frame_segments=sum(r['unique_frames'] < 64 for r in rows),
                                min_unique_frames=min(r['unique_frames'] for r in rows),
                                max_boundary_excursion_seconds=max(r['boundary_excursion_seconds'] for r in rows)),
                  provenance=dict(config_sha256=sha(io.CONFIG), contract_sha256=sha(OUT/'contract.json'),
                                  evaluation_sha256=sha(OUT/'evaluation.json'), script_sha256=sha(__file__),
                                  source_trace_sha256=trace_hashes),
                  independent_full_backbone_replay=False,
                  no_target_training_or_calibration=True)
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    with (DEST/'predictions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    io.locked(cfg)
    from fall_pipeline.external.omnifall_le2i_20261002 import status
    status('completed', metrics=metrics, audit_sha256=sha(DEST/'report.json'))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
