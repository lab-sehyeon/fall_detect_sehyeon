"""Compose two fixed, paired examples from actual pixels and saved outputs."""
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
from matplotlib.lines import Line2D
from PIL import Image

ROOT = source.ROOT
OUT = ROOT / 'docs/images/ours_success_comparison_20261004'
EVIDENCE = ROOT / 'docs/internal/2026-10-04_ours_success_figures_evidence'
DELIVERY = ROOT.parent / 'ours_success_images_20261004'
IDS = ['le2i__Coffee_room_01_014', 'cauca__forward__FallForwardS9']
STEMS = ['01_le2i_Coffee_room_01_014', '02_cauca_FallForwardS9', '03_paired_comparison']
DATASETS = ['Le2i', 'CAUCAFall']
EXPECTED = [[1, 0, 0, 0, 1], [1, 0, 1, 0, 1]]


def panel(fig, ident, by, plan, base, letter, dataset):
    row = by['own'][ident]
    metadata = paper.comparison_panel(fig, row, plan[ident],
                                      {m: by[m][ident] for m in source.FOLDERS}, base, letter)
    original = f'({letter}) {paper.short_id(ident)}'
    title = next(t for t in fig.texts if t.get_text() == original)
    title.set_text(f'({letter}) {dataset}: {paper.short_id(ident)}')
    # Show the full alarm symbol even at t=0; do not shift any timestamp.
    for ax in fig.axes:
        for line in ax.lines:
            if line.get_marker() == '^':
                line.set_clip_on(False)
    metadata['dataset'] = dataset
    return metadata


def legend(fig):
    items = [Line2D([0], [0], color=paper.INK, lw=4, label='GT fall'),
             Line2D([0], [0], color=paper.BLUE, marker='|', ls='none', markersize=7,
                    label='Fall-positive output'),
             Line2D([0], [0], color=paper.ORANGE, marker='^', mfc='white', ls='none',
                    markersize=4, label='Alarm onset')]
    height = fig.get_size_inches()[1]
    fig.legend(handles=items, loc='lower center', bbox_to_anchor=(.5, .31 / height),
               ncol=3, frameon=False, handlelength=1.2, columnspacing=1.2)
    paper.text(fig, .10, .22, 'TP: detected fall; FN: missed fall. E-FP: unmatched event alarms.', 8.5)


COMMON = (
    'All five methods process the same source video and ground truth; all input pipelines completed. '
    'The four actual frames illustrate the annotated fall. The black bar denotes the ground-truth fall interval; '
    'vertical ticks denote stored fall-positive outputs, and open triangles denote alarm onsets. '
    'Outputs are located at window endpoints for Ours and USDRL+NTU60, clip endpoints for HFD, and valid-pose '
    'frame times for FLASH. Tick density reflects sampling frequency, not comparable confidence. '
    'Empty temporal rows contain no positive output. X3D-UDA provides only a video-level decision. '
    'Video TP/FN and E-FP (unmatched event alarms) are distinct; event matching uses GT onset minus 0.5 s '
    'through GT end plus 3 s. *HFD and FLASH are project-retrained configurations; USDRL uses the official '
    'backbone with a project-trained NTU60 head. Cases were selected post hoc for Ours success and comparator '
    'misses, not as representative performance estimates. Results are fixed offline outputs, without new tuning.'
)


def main():
    source.cv2.setNumThreads(1)
    previous_path = EVIDENCE / 'validation.json'
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    for name, digest in previous.get('artifact_hashes', {}).items():
        assert source.sha(OUT / name) == digest, f'Output edited since last render: {name}'
    for p in (Path(__file__), Path(source.__file__), Path(paper.__file__),
              ROOT / 'scripts/audit_paper_demo_20261004.py'):
        source.remember(p)
    audit = source.read(ROOT / 'docs/internal/2026-10-04_ours_success_comparison_evidence/audit.json')
    assert audit['passed'] and list(audit['recommended'].values()) == IDS
    results, plan = source.load_evaluations()
    by = {m: {r['id']: r for r in rows} for m, rows in results.items()}
    for i, ident in enumerate(IDS):
        for m in source.FOLDERS:
            assert by[m][ident]['processed']
        assert [int(by[m][ident]['video_prediction']) for m in source.FOLDERS] == EXPECTED[i]
        assert [by['own'][ident]['event'][k] for k in ('tp', 'fp', 'fn')] == [1, 0, 0]
        assert by['flash'][ident]['event']['fp'] == [2, 4][i]
    OUT.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    paper.OUT = OUT
    paper.CATALOG.clear()
    paper.RENDER_CHECKS.clear()
    captions = [
        'Paired qualitative comparison on Le2i Coffee_room_01_014. Ours detects the fall whereas '
        'USDRL+NTU60, HFD and X3D-UDA miss it. FLASH also detects the fall but has two extra event alarms. '
        'Ours has one matched event and no extra alarm. ' + COMMON,
        'Paired qualitative comparison on CAUCAFall forward/FallForwardS9. Ours detects the fall whereas '
        'USDRL+NTU60 and X3D-UDA miss it. HFD and FLASH also detect it; FLASH has four extra event alarms. '
        'Ours has one matched event and no extra alarm. ' + COMMON,
        'Paired qualitative comparisons on (a) Le2i Coffee_room_01_014 and '
        '(b) CAUCAFall forward/FallForwardS9. Ours detects both falls without extra alarms. '
        'USDRL+NTU60 and X3D-UDA miss both, whereas HFD misses only (a). FLASH detects both, '
        'with two and four extra event alarms, respectively. ' + COMMON,
    ]
    for i, ident in enumerate(IDS):
        fig = plt.figure(figsize=(7.16, 3.65), dpi=160)
        metadata = panel(fig, ident, by, plan, .65, 'ab'[i], DATASETS[i])
        legend(fig)
        paper.save(fig, STEMS[i], captions[i], 'individual', None, {'panels': [metadata]})
    fig = plt.figure(figsize=(7.16, 6.67), dpi=160)
    panels = [panel(fig, ident, by, plan, base, letter, dataset)
              for ident, base, letter, dataset in zip(IDS, [3.67, .65], 'ab', DATASETS)]
    legend(fig)
    paper.save(fig, STEMS[2], captions[2], 'combined', None, {'panels': panels})

    (OUT / 'captions.txt').write_text('\n\n'.join(f'{s}\n{c}' for s, c in zip(STEMS, captions)) + '\n')
    (OUT / 'figure_metadata.json').write_text(json.dumps(paper.CATALOG, indent=2, ensure_ascii=False) + '\n')
    columns = ['dataset', 'video', 'model', 'video_result', 'processed', 'event_tp', 'event_fp', 'event_fn']
    with (OUT / 'comparison_results.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for p in panels:
            for m in p['models']:
                event = m['event'] or {}
                w.writerow(dict(dataset=p['dataset'], video=paper.short_id(p['id']), model=m['model'],
                                video_result=m['case'], processed=m['processed'],
                                **{'event_' + k: event.get(k, '') for k in ('tp', 'fp', 'fn')}))
    content = ['<!doctype html><html lang="en"><meta charset="utf-8">',
               '<meta name="viewport" content="width=device-width,initial-scale=1">',
               '<title>Paired fall recognition examples</title>',
               '<style>body{font:16px/1.55 system-ui;max-width:1100px;margin:30px auto;padding:0 20px}'
               'img{max-width:100%;height:auto}figure{margin:35px 0}</style>',
               '<h1>Paired fall recognition examples</h1><p>Two cases, five models, identical source videos. '
               'Labels use Ours. PDF/SVG retain vector labels; PNG exports use 600 dpi. '
               'Illustrative post-hoc selection, not overall performance evidence.</p>',
               '<p><a href="captions.txt">English captions</a> · '
               '<a href="comparison_results.csv">Case results</a></p>']
    for stem, cap in zip(STEMS, captions):
        content.append(f'<figure><a href="{stem}.pdf"><img src="{stem}.png" alt="{html.escape(stem)}"></a>'
                       f'<figcaption>{html.escape(cap)}<p><a href="{stem}.pdf">PDF</a> · '
                       f'<a href="{stem}.svg">SVG</a> · <a href="{stem}.png">PNG</a></p></figcaption></figure>')
    content.append('<p>RGB frames remain subject to dataset terms and identifiable-image publication requirements.</p></html>')
    (OUT / 'index.html').write_text(''.join(content))

    # Independent artifact checks, including the preserved, source-based model decisions.
    for entry, check in zip(paper.CATALOG, paper.RENDER_CHECKS):
        stem = entry['stem']
        verify_pdf(OUT / f'{stem}.pdf', 1, check['width_inches'], check['height_inches'])
        with Image.open(OUT / f'{stem}.png') as im:
            assert im.size == (round(check['width_inches'] * 600), round(check['height_inches'] * 600))
            im.verify()
        for p in entry['panels']:
            assert len(p['frames']) == 4
            for m in p['models']:
                original = by[m['model_key']][p['id']]
                assert m['video_prediction'] == original['video_prediction']
                assert m['case'] == paper.class_code(original)
                if m['event'] is not None:
                    assert m['event'] == {k: original['event'][k] for k in ('tp', 'fp', 'fn')}
                    assert m['alarm_times'] == [q['time'] for q in original['predictions']]
                else:
                    assert m['positive_output_times'] is None and m['alarm_times'] is None
    links = Links()
    links.feed((OUT / 'index.html').read_text())
    for name in links.targets:
        assert (OUT / name).is_file(), name
    assert len(links.images) == 3
    DELIVERY.mkdir(exist_ok=True)
    for stem in STEMS:
        target = DELIVERY / f'{stem}.png'
        if target.exists():
            allowed = {source.sha(OUT / target.name), previous.get('artifact_hashes', {}).get(target.name)}
            assert source.sha(target) in allowed, f'Existing delivery differs: {target}'
        shutil.copy2(OUT / target.name, target)
        assert source.sha(target) == source.sha(OUT / target.name)
    image_zip = DELIVERY.with_suffix('.zip')
    if image_zip.exists():
        assert source.sha(image_zip) == previous.get('image_zip_sha256'), 'Existing image archive changed'
    with zipfile.ZipFile(image_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        for stem in STEMS:
            z.write(DELIVERY / f'{stem}.png', arcname=f'{stem}.png')
    artifact_hashes = {p.name: source.sha(p) for p in OUT.iterdir() if p.suffix != '.zip'}
    full_zip = OUT / 'paired_comparison_full.zip'
    with zipfile.ZipFile(full_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in sorted(artifact_hashes):
            z.write(OUT / name, arcname=name)
    for path, expected in ((image_zip, {f'{s}.png' for s in STEMS}), (full_zip, set(artifact_hashes))):
        with zipfile.ZipFile(path) as z:
            assert set(z.namelist()) == expected and z.testzip() is None
            for name in expected:
                assert hashlib.sha256(z.read(name)).hexdigest() == artifact_hashes[name]
    for path, digest in source.SOURCES.items():
        assert source.sha(ROOT / path) == digest, path
    legacy = source.read(ROOT / 'docs/internal/2026-10-04_paper_demo_evidence/validation.json')
    for name, digest in legacy['artifact_hashes'].items():
        assert source.sha(ROOT / 'docs/images/paper_demo_20261004' / name) == digest
    report = dict(passed=True, cases=IDS, individual_figures=2, combined_figures=1,
                  no_new_inference=True, no_gpu=True, unchanged_previous_paper_artifacts=True,
                  render_checks=paper.RENDER_CHECKS, source_hashes=source.SOURCES,
                  artifact_hashes=artifact_hashes, image_zip_sha256=source.sha(image_zip),
                  full_zip_sha256=source.sha(full_zip), delivery=str(DELIVERY),
                  release_checks_pending=['Venue formatting', 'Dataset image publication requirements'])
    (EVIDENCE / 'validation.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(f'PASS: 2 individual figures + 1 combined; {len(source.SOURCES)} source hashes; PNG/PDF/SVG; copies and ZIPs verified.')


if __name__ == '__main__':
    main()
