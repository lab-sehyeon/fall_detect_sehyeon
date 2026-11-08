"""Reproduce conditional paired CIs from fixed per-video predictions, CPU only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

def counts(draws, n):
    out = np.zeros((len(draws), n), dtype=np.int64)
    np.add.at(out, (np.arange(len(draws))[:, None], draws), 1)
    return out

def f1(pred, y, weights):
    tp = weights @ (pred * y).T
    fp = weights @ (pred * (1-y)).T
    fn = weights @ ((1-pred) * y).T
    den = 2*tp+fp+fn
    assert (den > 0).all()
    return 2*tp/den

def main():
    out = HERE/'generated'; out.mkdir(exist_ok=True)
    results = {}
    for ds in ['cauca100', 'le2i130']:
        q = pd.read_csv(HERE/'results'/f'{ds}_paired_inputs.csv')
        y = q.ground_truth.to_numpy(dtype=np.int64)
        raw = q[[f'raw_seed_{s}' for s in range(5)]].to_numpy(dtype=np.int64).T
        full = q[[f'full_seed_{s}' for s in range(5)]].to_numpy(dtype=np.int64).T
        groups = np.array([q.subject_id.to_numpy() == s for s in sorted(q.subject_id.unique())], dtype=np.int64)
        results[ds] = {}
        for mode in ['subject_cluster', 'class_stratified_video']:
            rng = np.random.default_rng(0)
            if mode == 'subject_cluster':
                draws = rng.choice(len(groups), (20000, len(groups)), replace=True)
                weights = counts(draws, len(groups)) @ groups
            else:
                pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
                draws = np.concatenate([rng.choice(pos, (20000, len(pos)), replace=True), rng.choice(neg, (20000, len(neg)), replace=True)], axis=1)
                weights = counts(draws, len(y))
            delta = 100*(f1(full, y, weights)-f1(raw, y, weights)).mean(axis=1)
            lo, hi = np.quantile(delta, [.025, .975], method='linear')
            results[ds][mode] = dict(low_pp=float(lo), high_pp=float(hi))
    (out/'paired_ci.json').write_text(json.dumps(results, indent=2)+'\n')
    expected = json.loads((HERE/'results/expected_paired_ci.json').read_text())
    for ds in results:
        for mode in results[ds]:
            np.testing.assert_allclose(list(results[ds][mode].values()), list(expected[ds][mode].values()), atol=1e-12, rtol=0)
    print('PASS: 20,000 paired resamples per dataset/method; five fitted pairs held fixed')

if __name__ == '__main__': main()
