"""Independent read-back of Le2i inputs, feature/head traces and event counts."""
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fall_pipeline.external import rgb_document_io as io
from fall_pipeline.external import le2i_current_evaluation as run
from fall_pipeline.external.rgb_recovery_common import decode_recovery_endpoints, D1_DECOUPLED
from fall_pipeline.common.global_reconstruction import raw_channels, causal_arrays, features_from_causal, reference_feature
from fall_pipeline.joint.models import JointFallModel
from data_gen.safer_v2_geometry import normalize_ntu
from data_gen.safer_v3_geometry import fuse


def independent_counts(times, header, source_fps):
    start, end = header
    if start == end == 0: return 0, len(times), 0
    low, high = start/source_fps-.5, end/source_fps+3.
    tp = int(any(low <= time <= high for time in times))
    return tp, len(times)-tp, 1-tp


def validate_metrics(counts, published):
    tp, fp, fn = map(int, counts)
    values = dict(tp=tp, fp=fp, fn=fn, precision=tp/(tp+fp) if tp+fp else 0.,
                  recall=tp/(tp+fn) if tp+fn else 0., f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    for key, value in values.items():
        io.require(abs(published[key]-value) < 1e-12, 'independent metric mismatch '+key)
    return values


def main():
    torch.set_num_threads(2)
    config = run.setup(); output, plan = io.locked(config)
    published = io.read(output/'evaluation.json'); gt = {r['id']: r for r in io.read(output/'ground_truth.json')}
    glob = io.read(ROOT/config['inference']['global_config'])
    selected = io.read(ROOT/glob['output_dir']/'selection.json')['best']
    joint = JointFallModel().eval().requires_grad_(False)
    joint.load_state_dict(torch.load(ROOT/glob['joint_root']/'final/final_j1.pt', map_location='cpu', weights_only=True), strict=True)
    heads = {}
    for name, dim in [('G0', 2048), ('G1', 131), ('G2', 2179)]:
        head = torch.nn.Linear(dim, 4).eval().requires_grad_(False)
        head.load_state_dict(torch.load(ROOT/glob['output_dir']/name/f'epoch_{selected[name]["epoch"]:03d}.pt', map_location='cpu', weights_only=True), strict=True)
        heads[name] = head
    with np.load(ROOT/glob['output_dir']/'train_scaler.npz') as scale_file:
        mean, scale = scale_file['mean'], scale_file['scale']
    totals = {name: np.zeros(3, np.int64) for name in ('G2', 'G0')}
    quality_totals = {name: np.zeros(3, np.int64) for name in totals}
    rows = []; source_rows = {r['id']: r for r in io.read(run.PREP/'manifest.json')['records']}
    io.require(len(plan) == 127 and len(published['rows']) == 127, 'full denominator')
    maximum_head_error = 0.
    for number, (item, reported) in enumerate(zip(plan, published['rows']), 1):
        io.safety(config)
        sid = item['id']; dest = output/sid; truth = gt[sid]; prep = source_rows[sid]
        io.require(reported['id'] == sid, 'sample order changed')
        io.require(io.sha(ROOT/item['video']) == item['video_sha256'], 'derived video changed')
        io.require(io.sha(ROOT/prep['source']) == prep['source_sha256'], 'original AVI changed')
        annotation = ROOT/truth['annotation']
        io.require(io.sha(annotation) == truth['annotation_sha256'], 'annotation changed')
        header = list(map(int, annotation.read_text().splitlines()[:2]))
        io.require(header == truth['raw_header'], 'header mismatch')
        with np.load(ROOT/item['mapping']) as mapping, np.load(dest/'frontend.npz') as front:
            np.testing.assert_array_equal(front['source_indices'], mapping['source_indices'])
            np.testing.assert_array_equal(front['timestamps'], mapping['canonical_timestamps'])
            io.require(np.all(mapping['source_timestamps'][front['source_indices']] <= front['timestamps']), 'future input')
            boxes, xy, scores = front['boxes'], front['xy'], front['scores']
            bbox_valid = np.isfinite(boxes).all(1) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 4] > 0)
            pose_valid = bbox_valid & np.isfinite(xy).all((1, 2)) & np.isfinite(scores).all(1) & (scores > 0).any(1)
            pelvis = np.where(pose_valid, np.minimum(scores[:, 11], scores[:, 12]), 0)
            quality = bool(bbox_valid.mean() >= .8 and pose_valid.mean() >= .8 and np.median(pelvis) >= .3)
            a = io.annotation(front)
            previous = None
            for i in range(len(boxes)):
                candidates = front['candidates'][front['offsets'][i]:front['offsets'][i+1]]
                chosen = io.select_track(candidates, previous)
                np.testing.assert_array_equal(boxes[i], np.zeros(5, np.float32) if chosen is None else chosen)
                if chosen is not None: previous = chosen
                io.require(not front['rescue'][i] or len(candidates) <= 1, 'fallback not top1')
        front_meta = io.stage_done(dest, 'frontend')
        io.require(quality == front_meta['quality']['passed'] == reported['quality_passed'], 'quality mismatch')
        io.require(front_meta['models_before'] == front_meta['models_after'], 'frontend model mutation')
        times = {name: [] for name in totals}
        if quality:
            lift_meta = io.stage_done(dest, 'lift'); infer_meta = io.stage_done(dest, 'inference')
            io.require(lift_meta['model_before'] == lift_meta['model_after'], 'lifting model mutation')
            io.require(infer_meta['models_before'] == infer_meta['models_after'], 'classifier model mutation')
            with np.load(dest/'lift.npz') as lifted, np.load(dest/'inference.npz') as prediction:
                starts = np.arange(0, item['frames']-63, 8)
                np.testing.assert_array_equal(starts, lifted['window_starts'])
                np.testing.assert_array_equal(starts, prediction['window_starts'])
                np.testing.assert_array_equal(starts+63, prediction['window_endpoints'])
                h36m, coverage, _ = fuse(lifted['lifting_raw'], lifted['lifting_starts'], item['frames'], 'triangular', .05, True)
                np.testing.assert_array_equal(lifted['h36m'], h36m)
                np.testing.assert_array_equal(lifted['coverage'], coverage)
                ntu, _, _ = normalize_ntu(h36m, .5)
                np.testing.assert_array_equal(ntu, lifted['ntu25'])
                channels, mask = raw_channels(a); causal = causal_arrays(channels, mask)
                features = features_from_causal(causal, mask, starts)
                np.testing.assert_array_equal(features, lifted['global131'])
                for i, start in enumerate(starts):
                    np.testing.assert_array_equal(features[i], reference_feature(causal, mask, int(start)))
                standardized = ((features.astype(float)-mean)/scale).astype(np.float32)
                np.testing.assert_array_equal(standardized, prediction['global_standardized'])
                with torch.inference_mode():
                    adapted = joint.adapter(torch.from_numpy(prediction['pooled'])).numpy()
                    np.testing.assert_allclose(adapted, prediction['adapted'], atol=1e-4, rtol=1e-5)
                    for name, values in [('G0', prediction['adapted']), ('G1', standardized),
                                         ('G2', np.column_stack((prediction['adapted'], standardized)))]:
                        logits = heads[name](torch.from_numpy(values)).numpy()
                        np.testing.assert_allclose(logits, prediction[name], atol=1e-4, rtol=1e-5)
                        maximum_head_error = max(maximum_head_error, float(np.max(np.abs(logits-prediction[name]))))
                for name in totals:
                    decoded = decode_recovery_endpoints(starts+63, prediction[name], np.zeros(len(starts)), fall_alert_mode=D1_DECOUPLED)
                    events = [e for e in decoded['events'] if e['type'] == 'fall_trigger']
                    times[name] = [e['time'] for e in events]
                    io.require(times[name] == [e['time'] for e in reported['heads'][name]['predictions']], 'D1 event mismatch')
        else:
            io.require(not (dest/'lift.npz').exists() and not (dest/'inference.npz').exists(), 'quality bypass')
        row = dict(id=sid, scene=item['scene'], quality_passed=quality, heads={})
        for name in totals:
            counts = independent_counts(times[name], header, item['source_fps'])
            row['heads'][name] = validate_metrics(counts, reported['heads'][name])
            totals[name] += counts
            if quality: quality_totals[name] += counts
        rows.append(row)
        if number % 10 == 0: print('independent audit', number, 127, flush=True)
    result = dict(passed=True, videos=127, quality_pass=sum(r['quality_passed'] for r in rows), heads={}, rows=rows,
                  all_original_avi_hashes_unchanged=True, labels_reloaded_from_original_annotations=True,
                  full_feature_rebuild=True, independent_d1_replay=True, cpu_head_max_abs=maximum_head_error,
                  evaluation_sha256=io.sha(output/'evaluation.json'), audit_code_sha256=io.sha(Path(__file__)))
    for name in totals:
        result['heads'][name] = dict(primary=validate_metrics(totals[name], published['heads'][name]['primary']),
            quality_pass=validate_metrics(quality_totals[name], published['heads'][name]['quality_pass']))
        for scene, target in published['heads'][name]['per_scene'].items():
            counts = [sum(r['heads'][name][k] for r in rows if r['scene'] == scene) for k in ('tp', 'fp', 'fn')]
            validate_metrics(counts, target)
    io.require(result['quality_pass'] == published['quality_pass'], 'quality denominator mismatch')
    io.locked(config)
    io.save(output/'independent_audit.json', result)
    run.status('completed', videos=127, quality_pass=result['quality_pass'], heads=result['heads'])


if __name__ == '__main__': main()
