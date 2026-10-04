"""One figure per model and case, using the same two previously fixed videos."""
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile

import make_fall_demo_20261004 as source
import make_paper_demo_20261004 as paper
from audit_paper_demo_20261004 import verify_pdf, Links
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from PIL import Image

ROOT = source.ROOT
OUT = ROOT / 'docs/images/model_case_figures_20261004'
EVIDENCE = ROOT / 'docs/internal/2026-10-04_model_case_figures_evidence'
DELIVERY = ROOT.parent / 'model_case_images_20261004'
IDS = ['le2i__Coffee_room_01_014', 'cauca__forward__FallForwardS9']
MODELS = list(source.FOLDERS)
LABELS = ['Ours', 'USDRL + NTU60', 'HFD*', 'Privacy X3D-UDA', 'FLASH*']
FOLDERS = ['01_Ours', '02_USDRL_NTU60', '03_HFD', '04_Privacy_X3D_UDA', '05_FLASH']
PATHWAYS = [
    'RGB → 2D pose → 3D skeleton → frozen DSTE → residual adapter → four-class head',
    'RGB → 2D pose → 3D skeleton → frozen DSTE → NTU60 classification head',
    'RGB clips → C3D features → linear SVM',
    'RGB → five clips → X3D-UDA → mean logits → video classification',
    'RGB → MediaPipe pose → HyperMamba → pose-gated frame decisions',
]
RULES = ['Four-class argmax', 'NTU60 argmax; fall = A043', 'SVM margin > 0',
         'Five-clip mean logits', 'Logit > 0 and valid pose']
PLOTS = ['Window classification scores', 'Window classification scores', 'Clip decision margins',
         'Video classification scores', 'Frame logits and pose gating']


def score_plot(fig, row, data, limits):
    ax = paper.axis(fig, .62, 1.15, 4.55, 1.36)
    legend = []
    if not data['temporal']:
        p = data['probabilities']
        ax.barh([0, 1], p, color=[paper.GRAY, paper.BLUE], height=.48)
        ax.set_yticks([0, 1], ['Non-fall', 'Fall'])
        ax.set_xlim(0, 1)
        ax.set_xlabel('Video softmax score', labelpad=2)
        for j, value in enumerate(p):
            ax.text(.98, j, f'{value:.4f}', ha='right', va='center', fontsize=9,
                    bbox=dict(facecolor='white', edgecolor='none', pad=1))
        return legend
    for e in row['episodes']:
        ax.axvspan(e['fall_start'], e['fall_end'], facecolor='#ededed', edgecolor='#b7b7b7',
                   hatch='///', alpha=.55, linewidth=.5)
    legend.append(Patch(facecolor='#ededed', edgecolor='#b7b7b7', hatch='///', label='GT fall'))
    times, scores = data['times'], data['score']
    ax.plot(times, scores, color=paper.BLUE, lw=1)
    if 'competitor' in data:
        ax.plot(times, data['competitor'], color=paper.GRAY, ls='--', lw=.9)
        legend += [Line2D([0], [0], color=paper.BLUE, label='Fall score'),
                   Line2D([0], [0], color=paper.GRAY, ls='--', label='Max. non-fall score')]
        ax.set_ylabel('Softmax score', labelpad=4)
        ax.set_yticks([0, .5, 1])
    else:
        name = 'SVM margin' if row['model'] == 'hfd_reproduction' else 'Frame logit'
        legend += [Line2D([0], [0], color=paper.BLUE, label=name),
                   Line2D([0], [0], color=paper.GRAY, ls='--', label='Zero boundary')]
        ax.axhline(0, color=paper.GRAY, ls='--', lw=.8)
        ax.set_ylabel('Margin' if row['model'] == 'hfd_reproduction' else 'Logit', labelpad=4)
    ax.scatter(times[data['positive']], scores[data['positive']], s=9, color=paper.BLUE, zorder=4)
    for alarm in data['alarms']:
        ax.axvline(alarm, color=paper.ORANGE, lw=.8, alpha=.9)
    legend.append(Line2D([0], [0], color=paper.ORANGE, label='Alarm onset'))
    ax.set_ylim(*limits)
    ax.set_xlim(0, data['source_times'][-1])
    ax.set_xlabel('Time (s)', labelpad=2)
    ax.grid(axis='y', lw=.4, alpha=.25)
    return legend


def caption(row, number, label):
    text = (f'{label.rstrip("*")} on Case {number}: {source.SCOPES[row["scope"]]} '
            f'{paper.short_id(row["id"])} (video-level {paper.class_code(row)}). '
            'The same four source frames and timestamps are used for every model on this case. '
            'Both cases were selected post hoc for Ours success and misses by some comparators, not as '
            'representative performance estimates. No new predictions or target tuning were performed. ')
    m = row['model']
    if m == 'privacy_x3d_uda_rgb':
        text += ('Bars show the softmax of the five-clip averaged logits. This model has no temporal output; '
                 'event counts and alarm times are unavailable, not zero. ')
    else:
        text += 'Hatching marks GT fall and orange lines mark alarm onsets. '
        if m in ('own', 'usdrl_ntu60'):
            text += ('The blue curve is the fall softmax score and the dashed gray curve is the maximum '
                     'non-fall class score. Decisions use multiclass argmax, not a fixed 0.5 threshold. '
                     'Timestamps are window endpoints on the existing 25 Hz evaluation timeline. ')
        elif m == 'hfd_reproduction':
            text += 'The blue curve is the SVM decision margin at clip endpoints, with a fixed zero boundary. '
        else:
            text += ('The blue curve shows frame logits, not calibrated probabilities; a positive output '
                     'requires a logit above zero and valid pose. ')
        text += ('Blue points mark actual fall-positive outputs. Connecting segments are display aids, '
                 'not intermediate predictions. Event TP/FP/FN uses the existing matching window from '
                 'GT onset minus 0.5 s to GT end plus 3 s. Offline output times are not online latency. ')
    if m in ('hfd_reproduction', 'flash'):
        text += '*This is the documented project-retrained configuration, not the author\'s final model. '
    if m == 'usdrl_ntu60':
        text += 'The official backbone is paired with a project-trained NTU60 head; A043 is the fall class. '
    if m == 'flash':
        text += f'The fall is detected, with {row["event"]["fp"]} extra unmatched event alarms. '
    text += 'Score scales are shared between these two cases within a model, not across different model types.'
    return text


def render(row, case_number, model_index, data, frames, limits, book):
    times, indices, images = frames
    fig = plt.figure(figsize=(7.16, 4.65), dpi=160)
    label = LABELS[model_index]
    code = paper.class_code(row)
    dataset = source.SCOPES[row['scope']]
    paper.text(fig, .10, 4.48, label, 12, fontweight='bold')
    paper.text(fig, 7.04, 4.48, f'Case {case_number} · {dataset}', 10, ha='right')
    paper.text(fig, .10, 4.18, paper.short_id(row['id']), 9)
    for j, frame in enumerate(indices):
        x = .65 + j * 1.64
        paper.image_axis(fig, x, 3.13, 1.06, .80, images[frame])
        paper.text(fig, x + .53, 3.07, f'{times[frame]:.2f} s', 9, ha='center')
    paper.text(fig, .10, 2.76, PLOTS[model_index], 10)
    handles = score_plot(fig, row, data, limits)
    paper.text(fig, 5.52, 2.47, f'Video: {code}', 12, fontweight='bold',
               color=paper.BLUE if code == 'TP' else paper.ORANGE)
    paper.text(fig, 5.52, 2.16, 'Detected fall' if code == 'TP' else 'Missed fall', 9)
    paper.text(fig, 5.52, 1.91, 'Ground truth: Fall', 9)
    e = row['episodes'][0]
    paper.text(fig, 5.52, 1.68, f'{e["fall_start"]:.2f}–{e["fall_end"]:.2f} s', 9)
    if 'event' in row:
        q = row['event']
        paper.text(fig, 5.52, 1.40, 'Event TP / FP / FN', 8.5)
        paper.text(fig, 5.52, 1.17, f'{q["tp"]} / {q["fp"]} / {q["fn"]}', 11)
        paper.text(fig, 5.52, .88, f'Extra alarms: {q["fp"]}', 9)
    else:
        paper.text(fig, 5.52, 1.37, 'No temporal output', 9)
        paper.text(fig, 5.52, 1.10, 'Event metrics: N/A', 9)
    if handles:
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .52 / 4.65),
                   ncol=4, frameon=False, handlelength=1.3, columnspacing=1.0, fontsize=8.5)
    else:
        paper.text(fig, .62, .73, 'Video-level decision from five clips; no framewise prediction is available.', 9)
    paper.text(fig, .10, .40, 'Rule: ' + RULES[model_index], 8.5)
    paper.text(fig, .10, .20, PATHWAYS[model_index], 8.5)
    stem = f'{FOLDERS[model_index]}_Case{case_number}_{"Le2i" if case_number == 1 else "CAUCA"}_{code}'
    meta = dict(id=row['id'], model=label.rstrip('*'), model_key=row['model'], dataset=dataset,
                case_number=case_number, video_case=code, processed=row['processed'],
                frames=indices, frame_times=[float(times[i]) for i in indices], gt=row['episodes'],
                video_prediction=row['video_prediction'],
                event={k: row['event'][k] for k in ('tp', 'fp', 'fn')} if 'event' in row else None,
                alarm_times=data['alarms'].tolist() if data['temporal'] else None,
                score_y_limits=list(limits) if data['temporal'] else None,
                delivery_folder=FOLDERS[model_index])
    paper.save(fig, stem, caption(row, case_number, label), 'model_case', book, meta)


def main():
    source.cv2.setNumThreads(1)
    for path in (Path(__file__), Path(source.__file__), Path(paper.__file__),
                 ROOT / 'scripts/audit_paper_demo_20261004.py'):
        source.remember(path)
    previous_path = EVIDENCE / 'validation.json'
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    if previous:
        for name, digest in previous['artifact_hashes'].items():
            assert source.sha(OUT / name) == digest, f'Output changed: {name}'
    prior_path = ROOT / 'docs/internal/2026-10-04_ours_success_figures_evidence/validation.json'
    prior = source.read(prior_path)
    assert prior['passed'] and prior['cases'] == IDS
    prior_dir = ROOT / 'docs/images/ours_success_comparison_20261004'
    frozen = source.read(source.remember(prior_dir / 'figure_metadata.json', prior['artifact_hashes']['figure_metadata.json']))
    frozen_frames = {p['id']: p['frames'] for p in frozen[-1]['panels']}
    results, plan = source.load_evaluations()
    by = {m: {r['id']: r for r in rows} for m, rows in results.items()}
    frames = {ident: paper.verified_stills(by['own'][ident], plan[ident], 4) for ident in IDS}
    for ident in IDS:
        assert frames[ident][1] == frozen_frames[ident]
    series = {(m, ident): paper.load_series(by[m][ident], plan[ident]) for m in MODELS for ident in IDS}
    expected = [[1, 0, 0, 0, 1], [1, 0, 1, 0, 1]]
    for i, ident in enumerate(IDS):
        assert [int(by[m][ident]['video_prediction']) for m in MODELS] == expected[i]
        assert all(by[m][ident]['processed'] for m in MODELS)
    OUT.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    paper.OUT = OUT
    paper.CATALOG.clear()
    paper.RENDER_CHECKS.clear()
    with PdfPages(OUT / 'model_case_figures.pdf', metadata={'Title': 'Model-specific paired cases', 'Author': ''}) as book:
        for j, m in enumerate(MODELS):
            if m in ('own', 'usdrl_ntu60'):
                limits = (-.03, 1.03)
            elif m == 'privacy_x3d_uda_rgb':
                limits = None
            else:
                values = np.concatenate([series[m, ident]['score'] for ident in IDS])
                lo, hi = min(0., float(values.min())), max(0., float(values.max()))
                pad = max((hi - lo) * .08, .01)
                limits = (lo - pad, hi + pad)
            for case, ident in enumerate(IDS, 1):
                render(by[m][ident], case, j, series[m, ident], frames[ident], limits, book)
    assert len(paper.CATALOG) == 10
    (OUT / 'figure_metadata.json').write_text(json.dumps(paper.CATALOG, indent=2, ensure_ascii=False) + '\n')
    (OUT / 'captions.txt').write_text('\n\n'.join(r['stem'] + '\n' + r['caption'] for r in paper.CATALOG) + '\n')
    with (OUT / 'case_results.csv').open('w', newline='') as f:
        columns = ['model', 'case_number', 'dataset', 'id', 'video_case', 'event_tp', 'event_fp', 'event_fn']
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in paper.CATALOG:
            d = {k: r[k] for k in columns[:5]}
            d.update({'event_' + k: (r['event'] or {}).get(k, '') for k in ('tp', 'fp', 'fn')})
            writer.writerow(d)
    pages = ['<!doctype html><html lang="en"><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>Model-by-case fall recognition figures</title>',
             '<style>body{font:16px/1.5 system-ui;max-width:1100px;margin:30px auto;padding:0 20px}'
             'img{max-width:100%;height:auto}figure{margin:30px 0}</style>',
             '<h1>One model per case</h1><p>Five models × two identical videos. Case 1: Le2i Coffee_room_01_014; '
             'Case 2: CAUCAFall FallForwardS9. Actual frames and stored outputs; post-hoc illustrative selection.</p>',
             '<p><a href="model_case_figures.pdf">10-page PDF</a> · <a href="captions.txt">English captions</a> · '
             '<a href="case_results.csv">Results</a></p>']
    for r in paper.CATALOG:
        s = r['stem']
        pages.append(f'<figure><h2>{html.escape(r["model"])} — Case {r["case_number"]}</h2>'
                     f'<img src="{s}.png" alt="{html.escape(r["model"])} Case {r["case_number"]}">'
                     f'<figcaption>{html.escape(r["caption"])}<p><a href="{s}.pdf">PDF</a> · '
                     f'<a href="{s}.svg">SVG</a> · <a href="{s}.png">PNG</a></p></figcaption></figure>')
    pages.append('<p>RGB image-publication requirements and final venue formatting remain to be checked.</p></html>')
    (OUT / 'index.html').write_text(''.join(pages))

    for row, check in zip(paper.CATALOG, paper.RENDER_CHECKS):
        verify_pdf(OUT / f'{row["stem"]}.pdf', 1, 7.16, 4.65)
        with Image.open(OUT / f'{row["stem"]}.png') as image:
            assert image.size == (4296, 2790)
            image.verify()
        assert row['frames'] == frozen_frames[row['id']]
        r = by[row['model_key']][row['id']]
        assert row['video_case'] == paper.class_code(r)
        if 'event' in r:
            assert row['event'] == {k: r['event'][k] for k in ('tp', 'fp', 'fn')}
            assert row['alarm_times'] == [p['time'] for p in r['predictions']]
        else:
            assert row['event'] is None and row['alarm_times'] is None
    verify_pdf(OUT / 'model_case_figures.pdf', 10, 7.16, 4.65)
    for ident in IDS:
        entries = [r for r in paper.CATALOG if r['id'] == ident]
        assert len(entries) == 5
        assert all(r['frames'] == entries[0]['frames'] and r['frame_times'] == entries[0]['frame_times'] for r in entries)
    parser = Links()
    parser.feed((OUT / 'index.html').read_text())
    assert len(parser.images) == 10
    for target in parser.targets:
        assert (OUT / target).is_file()
    copies = {}
    for r in paper.CATALOG:
        name = f'Case{r["case_number"]}_{"Le2i" if r["case_number"] == 1 else "CAUCA"}_{r["video_case"]}.png'
        target = DELIVERY / r['delivery_folder'] / name
        original = OUT / f'{r["stem"]}.png'
        if target.exists():
            assert source.sha(target) in (source.sha(original), previous.get('delivery_hashes', {}).get(str(target.relative_to(DELIVERY))))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        assert source.sha(original) == source.sha(target)
        copies[str(target.relative_to(DELIVERY))] = source.sha(target)
    image_zip = DELIVERY.with_suffix('.zip')
    if image_zip.exists():
        assert source.sha(image_zip) == previous.get('image_zip_sha256')
    with zipfile.ZipFile(image_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in sorted(copies):
            z.write(DELIVERY / name, arcname=name)
    with zipfile.ZipFile(image_zip) as z:
        assert len(z.namelist()) == 10 and z.testzip() is None
        for name in z.namelist():
            assert hashlib.sha256(z.read(name)).hexdigest() == copies[name]
    for path, digest in source.SOURCES.items():
        assert source.sha(ROOT / path) == digest
    for name, digest in prior['artifact_hashes'].items():
        assert source.sha(prior_dir / name) == digest
    artifacts = {p.name: source.sha(p) for p in OUT.iterdir() if p.is_file()}
    report = dict(passed=True, images=10, models=5, cases=IDS, same_frames_per_case=True,
                  no_new_inference=True, no_gpu=True, previous_comparison_artifacts_unchanged=True,
                  source_hashes=source.SOURCES, artifact_hashes=artifacts,
                  render_checks=paper.RENDER_CHECKS, delivery_hashes=copies,
                  image_zip_sha256=source.sha(image_zip),
                  release_checks_pending=['Venue layout', 'Image publication requirements'])
    (EVIDENCE / 'validation.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(f'PASS: 10 model/case figures; {len(source.SOURCES)} source hashes; 10-page PDF; identical frames; model folders and ZIP verified.')


if __name__ == '__main__':
    main()
