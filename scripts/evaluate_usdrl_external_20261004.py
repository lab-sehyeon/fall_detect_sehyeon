"""Frozen USDRL backbone plus project-retrained NTU60 head, no J1/G0.

User-approved external transfer, NOT an author-trained fall-model reproduction.
Input skeletons/quality/GT are unchanged from the independently audited parents.
"""
import argparse
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fall_pipeline.external import rgb_document_io as io
from fall_pipeline.external.le2i_current_evaluation import match_events, metrics

CONFIG = ROOT / 'configs/usdrl_external_20261004_r1.json'
OUT = ROOT / 'data/fall_processed/RGB/usdrl_external_20261004_r1'


def status(stage, **fields):
    row = dict(stage=stage, time=datetime.now(timezone.utc).isoformat(), **fields)
    io.save(OUT / 'status.json', row)
    print(row, flush=True)


def edges(endpoints, logits):
    endpoints = np.asarray(endpoints)
    logits = np.asarray(logits)
    io.require(logits.shape == (len(endpoints), 60) and np.isfinite(logits).all(), 'invalid 60-class logits')
    io.require(np.all(np.diff(endpoints) > 0), 'unordered endpoints')
    fall = logits.argmax(1) == 42
    selected = np.flatnonzero(fall & ~np.r_[False, fall[:-1]]) if len(fall) else []
    return [dict(frame=int(endpoints[i]), time=float(endpoints[i] / 25)) for i in selected]


def summary(rows):
    y = np.array([bool(r['episodes']) for r in rows], bool)
    p = np.array([r['video_prediction'] for r in rows], bool)
    tp, fp, fn, tn = [int(x.sum()) for x in (y & p, ~y & p, y & ~p, ~y & ~p)]
    return dict(videos=len(rows), fall_events=sum(len(r['episodes']) for r in rows),
                processed=sum(r['processed'] for r in rows),
                rejected_positive=sum(bool(r['episodes']) and not r['processed'] for r in rows),
                positive_videos=int(y.sum()), predicted_fall_videos=int(p.sum()),
                event=metrics(*(sum(r['event'][k] for r in rows) for k in ('tp', 'fp', 'fn'))),
                video=dict(metrics(tp, fp, fn), tn=tn, accuracy=float((y == p).mean()),
                           specificity=tn / (tn + fp) if tn + fp else 0.))


def check_contract():
    frozen = io.read(OUT / 'contract.json')
    for path, digest in frozen['files'].items():
        io.require(io.sha(ROOT / path) == digest, 'frozen file changed: ' + path)
    return frozen


def prepare():
    cfg = io.read(CONFIG)
    OUT.mkdir(parents=True, exist_ok=True)
    io.safety(cfg)
    if (OUT / 'contract.json').exists():
        check_contract()
        return
    io.require(not (OUT / 'evaluation.json').exists(), 'refuse uncontracted previous results')
    files = {}

    def remember(path, wanted=None):
        path = Path(path)
        got = io.sha(path)
        io.require(wanted is None or got == wanted, 'unexpected input hash: ' + str(path))
        files[str(path.relative_to(ROOT))] = got

    remember(ROOT / cfg['encoder'], cfg['encoder_sha256'])
    remember(ROOT / cfg['head'], cfg['head_sha256'])
    remember((ROOT / cfg['head']).parent / 'run_config.json', cfg['head_config_sha256'])
    remember(ROOT / 'model/DSTE.py', cfg['dste_code_sha256'])
    for path in [CONFIG, Path(__file__), ROOT / 'scripts/audit_usdrl_external_20261004.py',
                 ROOT / 'tests/test_usdrl_external.py', ROOT / 'tools.py',
                 ROOT / 'fall_pipeline/external/rgb_document_io.py',
                 ROOT / 'fall_pipeline/external/le2i_current_evaluation.py',
                 ROOT / 'fall_pipeline/common/integrity.py',
                 ROOT / 'configs/safer_v2_controls_document_reconstruction_v1.json']:
        remember(path)
    plan, truth, own = [], [], []
    for scope, spec in cfg['datasets'].items():
        parent = ROOT / spec['parent']
        audit = io.read(parent / 'independent_audit.json')
        remember(parent / 'evaluation.json', spec['evaluation_sha256'])
        io.require(audit['passed'] and audit['evaluation_sha256'] == spec['evaluation_sha256'], 'parent not audited')
        for name in ('plan.json', 'ground_truth.json', 'contract.json', 'independent_audit.json'):
            remember(parent / name)
        parent_config = ROOT / f'configs/{scope}_20261003_r1.json'
        remember(parent_config, io.read(parent / 'contract.json')['config_sha256'])
        io.require(io.read(parent_config)['model']['encoder']['sha256'] == cfg['encoder_sha256'], 'different backbone')
        pp = io.read(parent / 'plan.json')
        gg = [r for r in io.read(parent / 'ground_truth.json') if r['scope'] == scope]
        ee = [dict(r, processed=r.get('processed', r.get('quality_passed')), model='own')
              for r in io.read(parent / 'evaluation.json')['rows']
              if r.get('model', 'own') == 'own' and r['scope'] == scope]
        io.require(len(pp) == len(gg) == len(ee) == spec['videos'], 'parent denominator')
        io.require({r['id'] for r in pp} == {r['id'] for r in gg} == {r['id'] for r in ee}, 'parent membership')
        io.require(sum(len(r['episodes']) for r in gg) == spec['events'], 'event count')
        erows = {r['id']: r for r in ee}
        count, nw = 0, 0
        for item in pp:
            dest = parent / item['id']
            front = io.stage_done(dest, 'frontend')
            remember(dest / 'frontend.json')
            passed = bool(front['quality']['passed'])
            io.require(passed == erows[item['id']]['processed'], 'quality changed')
            n = 0
            if passed:
                for stage in ('lift', 'inference'):
                    receipt = io.stage_done(dest, stage)
                    io.require(receipt['passed'], 'invalid parent stage')
                    remember(dest / f'{stage}.json')
                    remember(dest / f'{stage}.npz', receipt['payload_sha256'])
                with np.load(dest / 'lift.npz', allow_pickle=False) as z:
                    starts = z['window_starts']
                    np.testing.assert_array_equal(starts, np.arange(0, len(z['ntu25']) - 63, 8))
                    n = len(starts)
                io.require(n > 0, 'empty accepted video')
                count += 1
                nw += n
            else:
                io.require(not (dest / 'lift.npz').exists() and not (dest / 'inference.npz').exists(), 'quality bypass')
            plan.append(dict(id=item['id'], scope=scope, parent=spec['parent'], processed=passed,
                             windows=n, quality=front['quality']))
        io.require((count, nw) == (spec['processed'], spec['windows']), 'window/quality denominator')
        truth.extend(gg)
        own.extend(ee)
    io.require(len(plan) == len({r['id'] for r in plan}) == 230, 'duplicate identities')
    for name, value in [('plan.json', plan), ('ground_truth.json', truth), ('own_predictions.json', own)]:
        io.save(OUT / name, value)
        remember(OUT / name)
    io.save(OUT / 'contract.json', dict(files=files, created=datetime.now(timezone.utc).isoformat(),
                                      experiment_id=cfg['experiment_id'], no_training=True, no_target_tuning=True))
    status('prepared', videos=len(plan), windows=sum(r['windows'] for r in plan), frozen_files=len(files))


def infer():
    import torch
    from model.DSTE import Downstream
    from tools import remove_prefix
    from fall_pipeline.common.integrity import hash_named_tensors
    cfg = io.read(CONFIG)
    check_contract()
    dev = io.device(cfg)
    model = Downstream(150, 192, 1024, 1, 2, num_class=60, modality='joint', alpha=.5, gap=4, kernel_size=1)
    source = torch.load(ROOT / cfg['encoder'], map_location='cpu', weights_only=True)
    state = remove_prefix(source['state_dict'])
    loaded = model.load_state_dict(state, strict=False)
    io.require(set(loaded.missing_keys) == {'fc.weight', 'fc.bias'}, 'unexpected missing weights')
    io.require({k.split('.')[0] for k in loaded.unexpected_keys} == {'j_proj', 's_proj', 't_proj'}, 'unexpected weights')
    head = torch.load(ROOT / cfg['head'], map_location='cpu', weights_only=True)
    model.fc.load_state_dict(head, strict=True)
    io.require(set(dict(model.named_children())) == {'backbone', 'fc'}, 'unexpected active module')
    model.eval().requires_grad_(False).to(dev)
    io.require(all(not p.requires_grad for p in model.parameters()), 'trainable parameters')
    before = hash_named_tensors(model.state_dict().items())
    del source, state, head
    captured = []
    hook = model.fc.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach().cpu().numpy()))
    plan = io.read(OUT / 'plan.json')
    contract_sha = io.sha(OUT / 'contract.json')
    started = time.monotonic()
    for number, item in enumerate(plan, 1):
        io.safety(cfg)
        dest = OUT / item['id']
        dest.mkdir(exist_ok=True)
        if (dest / 'inference.json').exists():
            old = io.stage_done(dest, 'inference')
            io.require(old['contract_sha256'] == contract_sha, 'stale output')
            continue
        base = dict(id=item['id'], scope=item['scope'], processed=item['processed'], contract_sha256=contract_sha)
        if not item['processed']:
            io.save(dest / 'inference.json', dict(base, windows=0, reason='unchanged parent quality rejection'))
            continue
        parent = ROOT / item['parent'] / item['id']
        with np.load(parent / 'lift.npz', allow_pickle=False) as z:
            ntu, starts = z['ntu25'], z['window_starts']
        logits, features = [], []
        with torch.inference_mode():
            for first in range(0, len(starts), cfg['inference']['batch']):
                io.safety(cfg)
                batch = torch.from_numpy(io.windows(ntu, starts[first:first+32])).to(dev)
                jt = batch.permute(0, 2, 4, 3, 1).reshape(len(batch), 64, 150)
                js = batch.permute(0, 4, 3, 2, 1).reshape(len(batch), 50, 192)
                captured.clear()
                result = model(jt, js, None, None, None, None)
                io.require(len(captured) == 1, 'head input not captured exactly once')
                features.append(captured[0])
                logits.append(result.cpu().numpy())
        pooled, output = np.concatenate(features), np.concatenate(logits)
        io.require(pooled.shape == (item['windows'], 2048) and output.shape == (item['windows'], 60), 'output shape')
        io.require(np.isfinite(pooled).all() and np.isfinite(output).all(), 'nonfinite model output')
        with np.load(parent / 'inference.npz', allow_pickle=False) as prior:
            np.testing.assert_array_equal(starts, prior['window_starts'])
            np.testing.assert_array_equal(starts + 63, prior['window_endpoints'])
            np.testing.assert_allclose(pooled, prior['pooled'], atol=cfg['audit']['atol'], rtol=cfg['audit']['rtol'])
            delta = float(np.max(np.abs(pooled - prior['pooled'])))
        io.npz(dest / 'inference.npz', pooled=pooled, logits=output, window_starts=starts, window_endpoints=starts+63)
        io.save(dest / 'inference.json', dict(base, windows=len(starts), payload_sha256=io.sha(dest / 'inference.npz'),
                                            parent_pooled_max_abs_error=delta, training=False))
        if number % 10 == 0 or number == len(plan):
            status('inference', completed=number, total=230, elapsed_seconds=time.monotonic()-started)
    hook.remove()
    after = hash_named_tensors(model.state_dict().items())
    io.require(before == after, 'weights mutated')
    check_contract()
    io.save(OUT / 'model_audit.json', dict(passed=True, before=before, after=after, active_modules=['backbone', 'fc'],
                                        torch_version=torch.__version__, cuda_version=torch.version.cuda,
                                        device=torch.cuda.get_device_name(0), fresh_forward=True,
                                        peak_cuda_bytes=torch.cuda.max_memory_allocated(), seconds=time.monotonic()-started))
    status('inference_complete', videos=230, windows=6286)


def score():
    check_contract()
    io.require(io.read(OUT / 'model_audit.json')['passed'], 'no completed model audit')
    truth = {r['id']: r for r in io.read(OUT / 'ground_truth.json')}
    rows = []
    for item in io.read(OUT / 'plan.json'):
        dest = OUT / item['id']
        receipt = io.stage_done(dest, 'inference')
        io.require(receipt and receipt['processed'] == item['processed'], 'incomplete inference')
        predictions, fall_windows, max_score = [], 0, 0.
        hist = np.zeros(60, np.int64)
        if item['processed']:
            with np.load(dest / 'inference.npz', allow_pickle=False) as z:
                logits = z['logits'].astype(np.float64)
                predictions = edges(z['window_endpoints'], logits)
                labels = logits.argmax(1)
                hist = np.bincount(labels, minlength=60)
                fall_windows = int(hist[42])
                exp = np.exp(logits - logits.max(1, keepdims=True))
                max_score = float((exp[:, 42] / exp.sum(1)).max())
        gt = truth[item['id']]
        rows.append(dict(gt, model='usdrl_ntu60', processed=item['processed'], windows=item['windows'],
                         quality=item['quality'], predictions=predictions, video_prediction=int(fall_windows > 0),
                         max_fall_probability=max_score, fall_windows=fall_windows, class_histogram=hist.tolist(),
                         event=match_events(predictions, gt['episodes'])))
    summaries = {}
    for name, rr in [('usdrl_ntu60', rows), ('own', io.read(OUT / 'own_predictions.json'))]:
        summaries[name] = {}
        for scope in io.read(CONFIG)['datasets']:
            selected = [r for r in rr if r['scope'] == scope]
            s = summary(selected)
            if name == 'usdrl_ntu60':
                s['windows'] = sum(r['windows'] for r in selected)
                s['fall_windows'] = sum(r['fall_windows'] for r in selected)
                s['class_histogram'] = np.sum([r['class_histogram'] for r in selected], axis=0).tolist()
            summaries[name][scope] = s
    io.save(OUT / 'evaluation.json', dict(passed=True, pending_independent_audit=True, summaries=summaries, rows=rows,
                                         contract_sha256=io.sha(OUT / 'contract.json'), no_training=True, no_target_tuning=True))
    status('scored_pending_independent_audit', summaries=summaries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'infer', 'score'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        globals()[args.stage]()
