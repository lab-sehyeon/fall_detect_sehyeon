"""Audit same-video disagreements from fixed results; no model execution."""
import csv
import json
from pathlib import Path

import make_fall_demo_20261004 as source
import make_paper_demo_20261004 as paper

ROOT = source.ROOT
EVIDENCE = ROOT / 'docs/internal/2026-10-04_ours_success_comparison_evidence'
SHARED = ROOT / 'docs/shared'
PREFIX = '2026-10-04_ours_success_comparison'
LABELS = {'own': 'Ours', 'usdrl_ntu60': 'USDRL_NTU60',
          'hfd_reproduction': 'HFD', 'privacy_x3d_uda_rgb': 'X3D_UDA', 'flash': 'FLASH'}


def main():
    source.remember(Path(__file__))
    source.remember(Path(source.__file__))
    source.remember(Path(paper.__file__))
    results, plan = source.load_evaluations()
    by = {m: {r['id']: r for r in rows} for m, rows in results.items()}
    models = list(by)
    others = models[1:]
    ids = set(by['own'])
    assert len(ids) == 230 and all(set(by[m]) == ids for m in models)
    grounds = []
    for folder in sorted(set(source.FOLDERS.values())):
        ground = {r['id']: r for r in source.read(source.BASE / folder / 'ground_truth.json')}
        assert set(ground) == ids
        grounds.append(ground)
        for raw in source.read(source.BASE / folder / 'evaluation.json')['rows']:
            if raw['model'] not in models:
                continue
            g = ground[raw['id']]
            if 'episodes' in raw:
                assert raw['episodes'] == g['episodes']
            if 'ground_truth' in raw:
                assert bool(raw['ground_truth']) == bool(g['episodes'])
    assert all(g == grounds[0] for g in grounds)
    originals = {}
    for scope in source.SCOPES:
        originals.update({r['id']: r for r in source.read(source.BASE / f'{scope}_20261003_r1' / 'plan.json')})
    assert set(originals) == ids
    # All RGB baselines reference the same source path/hash and timeline manifest.
    for folder in sorted(set(source.FOLDERS.values())):
        p = {r['id']: r for r in source.read(source.BASE / folder / 'plan.json')}
        assert set(p) == ids
        for ident, r in p.items():
            if folder == source.FOLDERS['usdrl_ntu60']:
                expected = source.BASE / f"{by['own'][ident]['scope']}_20261003_r1"
                assert ROOT / r['parent'] == expected
                assert by['usdrl_ntu60'][ident]['path'] == originals[ident]['path']
            else:
                for key in ('video', 'video_sha256', 'mapping_sha256', 'trace_sha256'):
                    assert r[key] == originals[ident][key], (folder, ident, key)

    rows = []
    for ident in sorted(ids):
        own = by['own'][ident]
        truth = bool(own['episodes'])
        assert all(by[m][ident]['scope'] == own['scope'] for m in models)
        assert all(bool(by[m][ident]['episodes']) == truth for m in models)
        missed = [m for m in others if truth and not by[m][ident]['video_prediction']]
        event_missed = [m for m in others if truth and 'event' in by[m][ident]
                        and by[m][ident]['event']['tp'] == 0 and by[m][ident]['event']['fn'] > 0]
        r = dict(id=ident, dataset=source.SCOPES[own['scope']], scope=own['scope'],
                 ground_truth='fall' if truth else 'non_fall',
                 gt_intervals=json.dumps(own['episodes'], separators=(',', ':')),
                 all_models_processed=all(by[m][ident]['processed'] for m in models),
                 ours_tp_vs_any_fn=truth and bool(own['video_prediction']) and bool(missed),
                 comparator_video_fn_count=len(missed),
                 comparator_video_fn=';'.join(LABELS[m] for m in missed),
                 comparator_event_miss=';'.join(LABELS[m] for m in event_missed))
        for m in models:
            item = by[m][ident]
            prefix = LABELS[m]
            r[prefix + '_video'] = paper.class_code(item)
            r[prefix + '_processed'] = item['processed']
            for key in ('tp', 'fp', 'fn'):
                r[prefix + '_event_' + key] = item.get('event', {}).get(key, '')
            r[prefix + '_alarm_times_s'] = (json.dumps([p['time'] for p in item['predictions']])
                                            if 'predictions' in item else '')
        rows.append(r)
    candidates = [r for r in rows if r['ours_tp_vs_any_fn']]
    report = {'common_videos': len(ids), 'identical_ground_truth': True,
              'same_source_video_manifests_and_usdrl_parent_verified': True,
              'selection_is_post_hoc': True, 'no_new_inference_or_tuning': True,
              'video_candidates': len(candidates), 'datasets': {}, 'recommended': {}}
    for scope in source.SCOPES:
        group = [r for r in rows if r['scope'] == scope]
        cand = [r for r in candidates if r['scope'] == scope]
        good = [r for r in cand if r['all_models_processed'] and r['Ours_event_tp'] > 0
                and r['Ours_event_fp'] == 0 and r['Ours_event_fn'] == 0]
        good.sort(key=lambda r: (-r['comparator_video_fn_count'], r['id']))
        report['recommended'][scope] = good[0]['id']
        report['datasets'][scope] = dict(
            common=len(group), all_processed=sum(r['all_models_processed'] for r in group),
            ours_tp=sum(r['Ours_video'] == 'TP' for r in group),
            candidates=len(cand), all_processed_candidates=sum(r['all_models_processed'] for r in cand),
            ours_tp_all_four_fn=sum(r['comparator_video_fn_count'] == 4 for r in cand),
            max_comparator_video_fn=max(r['comparator_video_fn_count'] for r in cand),
            maximum_disagreement_ids=[r['id'] for r in good if r['comparator_video_fn_count'] == good[0]['comparator_video_fn_count']],
            pairwise={LABELS[m]: dict(
                ours_tp_other_fn=sum(r['Ours_video'] == 'TP' and r[LABELS[m] + '_video'] == 'FN' for r in group),
                ours_fn_other_tp=sum(r['Ours_video'] == 'FN' and r[LABELS[m] + '_video'] == 'TP' for r in group),
                ours_tn_other_fp=sum(r['Ours_video'] == 'TN' and r[LABELS[m] + '_video'] == 'FP' for r in group),
                ours_event_match_other_miss=(sum(r['Ours_event_tp'] > 0 and r[LABELS[m] + '_event_tp'] == 0
                    and r[LABELS[m] + '_event_fn'] > 0 for r in group) if m != 'privacy_x3d_uda_rgb' else None)) for m in others})
    selected = list(report['recommended'].values())
    # Additional targeted examples: HFD classification miss and FLASH input/time failures.
    selected += ['cauca__side__FallLeftS2', 'cauca__forward__FallForwardS8', 'le2i__Coffee_room_01_003']
    report['verified_examples'] = []
    for ident in selected:
        source.remember(ROOT / plan[ident]['video'], plan[ident]['video_sha256'])
        for m in models:
            paper.load_series(by[m][ident], plan[ident])
        report['verified_examples'].append(next(r for r in rows if r['id'] == ident))
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, selected_rows in (('all_cases', rows), ('candidates', candidates)):
        p = SHARED / f'{PREFIX}_{suffix}.csv'
        with p.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(selected_rows)
        with p.open(newline='', encoding='utf-8') as f:
            assert len(list(csv.DictReader(f))) == len(selected_rows)
        outputs.append(p)
    for name, digest in source.SOURCES.items():
        assert source.sha(ROOT / name) == digest, name
    report['source_hashes'] = source.SOURCES
    report['output_hashes'] = {str(p.relative_to(ROOT)): source.sha(p) for p in outputs}
    report['passed'] = True
    (EVIDENCE / 'audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('source_hashes', 'verified_examples')}, indent=2))


if __name__ == '__main__':
    main()
