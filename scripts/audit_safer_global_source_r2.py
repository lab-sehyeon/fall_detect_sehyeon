"""Label-blind inventory/geometry audit of pinned SAFER global-motion sources.

New diagnostic definitions: confidence >= 0.2; strict in-image positive bbox.
Does not extract model features, modify sources, or access annotation labels.
"""
from collections import Counter
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from data_gen.inspect_safer_activities import load_pinned_pickle, PINNED_FILES
from scripts import prepare_rgb_recovery_data_r2 as prep

ROOT = prep.ROOT
OUTPUT = prep.OUTPUT / 'safer_global_source_audit'
CONFIG = ROOT / 'configs/safer_v2_document_reconstruction_v1.json'
SCORE_CUTOFF = .2


def geometry_counts(bbox, xy, score, width, height):
    """Return counts, not filtered frames: all omissions remain explicit."""
    bbox = np.asarray(bbox, np.float64)
    xy, score = np.asarray(xy), np.asarray(score)
    n = len(bbox)
    prep.require(bbox.shape == (n, 4) and xy.shape == (n, 17, 2) and score.shape == (n, 17),
                 'global source shape mismatch')
    prep.require(np.isfinite([width, height]).all() and min(width, height) > 0,
                 'global source image dimensions invalid')
    finite = np.isfinite(bbox).all(axis=1)
    lower = bbox[:, :2]
    upper_wh, upper_xy = lower + bbox[:, 2:], bbox[:, 2:]
    confident = np.isfinite(score) & (score >= SCORE_CUTOFF) & np.isfinite(xy).all(axis=2)
    result = dict(frames=n, bbox_nonfinite_frames=int((~finite).sum()),
                  bbox_zero_frames=int((bbox == 0).all(axis=1).sum()),
                  coordinate_nonfinite_frames=int((~np.isfinite(xy).all(axis=(1, 2))).sum()),
                  confidence_nonfinite_frames=int((~np.isfinite(score).all(axis=1)).sum()),
                  confident_joints=int(confident.sum()))
    for name, upper in [('xywh', upper_wh), ('xyxy', upper_xy)]:
        positive = finite & (upper > lower).all(axis=1)
        plausible = positive & (lower >= 0).all(axis=1) & (upper <= [width, height]).all(axis=1)
        inside = ((xy >= lower[:, None, :]).all(axis=2) &
                  (xy <= upper[:, None, :]).all(axis=2) & confident & positive[:, None])
        result[name + '_positive_frames'] = int(positive.sum())
        result[name + '_plausible_frames'] = int(plausible.sum())
        result[name + '_inside_confident_joints'] = int(inside.sum())
    return result


def main():
    prep.require(Path(sys.prefix).name == 'fall_detect' and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
                 'fall_detect CPU-only required')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / 'audit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = prep.read(CONFIG)
        contract = dict(code_sha256=prep.sha(Path(__file__)), config_sha256=prep.sha(CONFIG),
                        loader_sha256=prep.sha(ROOT / 'data_gen/inspect_safer_activities.py'),
                        helper_sha256=prep.sha(Path(prep.__file__)),
                        source=PINNED_FILES, score_cutoff=SCORE_CUTOFF,
                        labels_or_predictions_used=False, slack_pixels=0,
                        bbox_positive_size_required=True, original_media_modified=False)
        path = OUTPUT / 'contract.json'
        if path.exists():
            prep.require(prep.read(path) == contract, 'global audit contract changed')
        else:
            prep.save(path, contract)
        records, aggregate = [], Counter()
        try:
            prep.safety()
            for domain in ('normal', 'ood'):
                data = load_pinned_pickle(ROOT / config['source'][domain], domain)
                prep.require(len(data['annotations']) == PINNED_FILES[domain]['sequences'],
                             'source sequence count changed')
                count = 0
                for index, annotation in enumerate(data['annotations']):
                    prep.safety()
                    n = int(annotation['total_frames'])
                    bbox, xy, score = (np.asarray(annotation[k]) for k in
                                      ('bboxes', 'keypoint', 'keypoint_score'))
                    prep.require(xy.shape == (1, n, 17, 2) and score.shape == (1, n, 17)
                                 and bbox.shape == (n, 4), 'source timeline shape mismatch')
                    stats = geometry_counts(bbox, xy[0], score[0], annotation['width'], annotation['height'])
                    record = dict(uid=f'{domain}_{index:04d}', domain=domain,
                                  source_name=annotation['frame_dir'], width=int(annotation['width']),
                                  height=int(annotation['height']), metadata_bbox_format=annotation['bbox_format'],
                                  **stats)
                    records.append(record)
                    aggregate.update(stats)
                    count += n
                    if len(records) % 50 == 0:
                        print(f'global source audit {len(records)}/497', flush=True)
                prep.require(count == PINNED_FILES[domain]['frames'], 'source frame count changed')
                del data, annotation, bbox, xy, score
                gc.collect()
            prep.require(len(records) == 497 and aggregate['frames'] == 8091357,
                         'global source population mismatch')
            journal_path = prep.OUTPUT / 'safer_ood_videos.json'
            journal_bytes = journal_path.read_bytes() if journal_path.exists() else None
            journal = json.loads(journal_bytes) if journal_bytes else {}
            by_name = {Path(k).name: v['probe'] for k, v in journal.items()}
            media_comparisons = []
            for record in records:
                probe = by_name.get(Path(record['source_name']).name) if record['domain'] == 'ood' else None
                if probe is not None:
                    media_comparisons.append(dict(
                        uid=record['uid'], source_name=record['source_name'],
                        frames_equal=record['frames'] == probe['frames'],
                        pose_image_dimensions=[record['width'], record['height']],
                        rgb_image_dimensions=[probe['width'], probe['height']],
                        image_dimensions_equal=(record['width'], record['height']) ==
                                               (probe['width'], probe['height'])))
            report = dict(passed=True, audit_scope='source availability and diagnostic geometry only',
                          sequences=len(records), counts=dict(aggregate),
                          source_hashes={k: PINNED_FILES[k]['sha256'] for k in ('normal', 'ood')},
                          bbox_metadata_formats=dict(Counter(r['metadata_bbox_format'] for r in records)),
                          confidence_cutoff=SCORE_CUTOFF, records=records,
                          ood_media_comparison_snapshot=media_comparisons,
                          media_journal_snapshot_sha256=hashlib.sha256(journal_bytes).hexdigest() if journal_bytes else None,
                          labels_or_predictions_used=False, source_modified=False,
                          rgb_pose_pixel_mapping_validated=False,
                          exact_historical_audit=False, global_features_generated=False,
                          model_trained=False)
            for meaning in ('xywh', 'xyxy'):
                report[meaning + '_plausibility'] = aggregate[meaning + '_plausible_frames'] / aggregate['frames']
                report[meaning + '_confident_joint_inclusion'] = (
                    aggregate[meaning + '_inside_confident_joints'] / aggregate['confident_joints']
                    if aggregate['confident_joints'] else None)
            prep.save(OUTPUT / 'report.json', report)
            prep.save(OUTPUT / 'status.json', dict(stage='completed', report_sha256=prep.sha(OUTPUT / 'report.json')))
            print('global source audit completed: 497 sequences, 8091357 frames', flush=True)
        except Exception as error:
            prep.save(OUTPUT / 'status.json', dict(stage='failed', error_type=type(error).__name__,
                                                  detail=str(error) if type(error) is RuntimeError else 'audit failed'))
            raise


if __name__ == '__main__':
    main()
