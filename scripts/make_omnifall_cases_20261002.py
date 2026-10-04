"""Export real benchmark examples; select the first official row per outcome."""
from pathlib import Path
import hashlib
import json
import os
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/omnifall-cases-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fall_pipeline.external.omnifall_sampling_20261002 import decode

BASE = ROOT / 'data/fall_processed/RGB/omnifall_gmdcsa_cs_20261002_r1'
AUDIT = ROOT / 'docs/internal/2026-10-02_omnifall_audit'
OUT = ROOT / 'docs/images/omnifall_gmdcsa_20261002'
GT_NAMES = ['walk', 'fall', 'fallen', 'sit_down', 'sitting', 'lie_down', 'lying', 'stand_up', 'standing', 'other']
PRED_NAMES = ['other', 'fall', 'lie_down', 'lying_down']
EDGES = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert read(AUDIT/'report.json')['passed']
    evaluation = read(BASE/'evaluation.json')
    plan = {r['id']: r for r in read(BASE/'plan.json')}
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 4, figsize=(19, 12), gridspec_kw={'width_ratios': [1,1,1,.9]})
    provenance = []
    for row_index, (case, gt, pred) in enumerate([('TP',1,1),('TN',0,0),('FP',0,1),('FN',1,0)]):
        matches = [r for r in evaluation['rows'] if r['fall'] == gt and r['prediction'] == pred]
        if not matches:
            for ax in axes[row_index]:
                ax.set_axis_off()
            axes[row_index,0].text(.05,.5, f'{case}: no observed case', fontsize=13)
            provenance.append(dict(case=case, observed=False))
            continue
        row = matches[0]
        item = plan[row['id']]
        directory = BASE/row['id']
        frames = decode(ROOT/item['video'], item['start'], item['end'], item['sampling'])
        with np.load(directory/'frontend.npz', allow_pickle=False) as z:
            xy, scores = z['xy'].copy(), z['scores'].copy()
        prediction = PRED_NAMES[row['predicted_class']] if row['predicted_class'] is not None else 'no alarm (quality rejection)'
        times = []
        for column, sample in enumerate([10, 32, 53]):
            ax = axes[row_index,column]
            ax.imshow(frames[sample])
            valid = scores[sample] >= .2
            for a,b in EDGES:
                if valid[a] and valid[b]:
                    ax.plot(xy[sample,[a,b],0], xy[sample,[a,b],1], color='#16e0b0', linewidth=1.6)
            ax.scatter(xy[sample,valid,0], xy[sample,valid,1], s=5, color='#ffdc63')
            actual = item['sampling']['actual_seconds'][sample]
            times.append(actual)
            title = f'sample {sample+1}/64 | actual t={actual:.3f}s'
            if column == 0:
                title = f'{case} | GT: {GT_NAMES[row["label"]]} | pred: {prediction}\n' + title
            ax.set_title(title, fontsize=9, loc='left')
            ax.set_axis_off()
        ax = axes[row_index,3]
        probabilities = None
        if row['quality_passed']:
            with np.load(directory/'inference.npz', allow_pickle=False) as z:
                values = z['G0'][0].astype(np.float64)
            exp = np.exp(values-values.max()); probabilities = (exp/exp.sum()).tolist()
            ax.barh(PRED_NAMES, probabilities, color=['#7893a6','#cb533e','#7893a6','#7893a6'])
            ax.invert_yaxis(); ax.set_xlim(0,1); ax.set_xlabel('4-class softmax score', fontsize=9)
            ax.tick_params(labelsize=8); ax.grid(axis='x', alpha=.2)
        else:
            ax.text(.05,.5, 'Classifier not run\nPose quality gate failed\nRetained in denominator', fontsize=11)
            ax.set_axis_off()
        ax.set_title(f'Official row {item["row_index"]}\nGT interval: {item["start"]:.2f}–{item["end"]:.2f}s', fontsize=10)
        provenance.append(dict(case=case, observed=True, id=row['id'], row_index=item['row_index'],
                               gt_label=row['label'], prediction=row['prediction'], predicted_class=row['predicted_class'],
                               quality_passed=row['quality_passed'], display_sample_indices=[10,32,53],
                               display_actual_seconds=times, softmax=probabilities,
                               video_sha256=item['video_sha256'], decoded_rgb_sha256=item['decoded_rgb_sha256'],
                               frontend_sha256=sha(directory/'frontend.npz'),
                               inference_sha256=sha(directory/'inference.npz') if row['quality_passed'] else None))
    fig.suptitle('OmniFall GMDCSA24-CS: observed segment classification cases', fontsize=19, y=.985)
    fig.text(.025,.016, 'Actual decoded RGB and inferred 2D pose. First official row per observed outcome; no confidence or appearance ranking.\nGT segment boundaries are supplied. Fall only is positive; fallen is negative. Uniform 64-sample input; one argmax decision per segment.', fontsize=10)
    fig.subplots_adjust(left=.025,right=.98,top=.93,bottom=.08,wspace=.17,hspace=.4)
    files = []
    for ext in ['png','pdf','svg']:
        path = OUT/f'gmdcsa_cases.{ext}'
        fig.savefig(path,dpi=170)
        files.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
    plt.close(fig)
    report = dict(passed=True,selection='first official Parquet row per actual TP/TN/FP/FN; fixed displayed samples10/32/53',
                  visual_confidence_cutoff=.2, visual_cutoff_changes_predictions=False,
                  posthoc_illustration=True, cases=provenance, files=files,
                  evaluation_sha256=sha(BASE/'evaluation.json'), script_sha256=sha(__file__))
    (AUDIT/'case_provenance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
