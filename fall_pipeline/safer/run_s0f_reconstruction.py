"""Frozen S0-A RAM features, two causal transition heads, locked evaluation."""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score

from fall_pipeline.safer import run_s0e_reconstruction as e
from fall_pipeline.safer import s0f_core as core

io, s, a = e.io, e.s, e.s.a
CONFIG = io.ROOT/'configs/s0f_document_reconstruction_v1.json'
CODE = e.CODE + ['fall_pipeline/safer/s0f_core.py', 'fall_pipeline/safer/run_s0f_reconstruction.py']


def publish(root, stage, **details):
    state = {'stage': stage, 'utc': datetime.now(timezone.utc).isoformat(), **details}
    io.save_json(root/'status.json', state)
    print(json.dumps(state, ensure_ascii=False), flush=True)
    if root.name.endswith('_smoke'):
        return
    status = 'completed' if stage == 'completed' else 'paused' if stage in ('failed', 'paused') else 'in_progress'
    header = f'- 문서 ID: `DOC-20260930-s0f-reconstruction-run-R1`\n- 기준일: 2026-09-30\n- 상태: `{status}`\n'
    internal = '# S0-F 자동 실행 기록\n\n'+header+'\n```json\n'+json.dumps(state, ensure_ascii=False, indent=2)+'\n```\n'
    internal += f'\nOutput: `{root}`\n'
    words = {'preflight':'입력·동결 검사', 'extract':'고정 표현 추출', 'train':'전이 보조 모델 학습',
             'val':'검증셋 비교', 'selection':'검증셋 선택 고정', 'evaluate':'고정 모델 평가',
             'audit':'최종 검산', 'completed':'사전 정의한 실행·검산 완료', 'paused':'일시 중단', 'failed':'실행 중단'}
    shared = '# S0-F 연구 진행\n\n'+header+f'\n현재 단계: {words[stage]}.\n'
    for key in ('epoch', 'epochs', 'windows', 'total', 'split', 'feasible_for_integration'):
        if key in details:
            shared += f'\n- {key}: {details[key]}\n'
    for kind, value in (('internal', internal), ('shared', shared)):
        value += f'\n[사전 조건과 한계](2026-09-30_s0f_reconstruction_{kind}.md). 전체 프로젝트 완료와 구분한다.\n'
        path = io.ROOT/f'docs/{kind}/2026-09-30_s0f_reconstruction_run_{kind}.md'
        tmp = path.with_suffix('.tmp'); tmp.write_text(value); os.replace(tmp, path)


def pause(root, cfg):
    e.pause(root, cfg)
    fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(fields['MemAvailable'].strip().split()[0])*1024
    if available < cfg['execution']['minimum_ram_available_gib']*1024**3:
        raise io.PauseRequested('RAM available reserve gate')


def immutable_json(path, value):
    if path.exists():
        io.require(io.read(path) == value, 'immutable artifact differs: '+str(path))
    else:
        io.save_json(path, value)


def target_records(data, cfg):
    rows = []
    for row, begin, end in data.sequences():
        _, y = data.arrays(row['sequence_index'])
        n = int(data.starts[end-1])+64
        truth = np.asarray(y[:n])
        rows.append({'uid': row['uid'], 'view': core.view_id(row['source_name']), 'begin': begin, 'end': end,
                     'truth': truth, 'target': core.targets(truth, cfg['targets']['positive_width'])})
    return rows


def source_summary(records):
    counts = sum((np.bincount(r['target'], minlength=4) for r in records), np.zeros(4, np.int64))
    points = [sum(len(core.events(r['truth'])[k]) for r in records) for k in (1, 2, 3)]
    return {'target_counts': counts.tolist(), 'event_counts': points, 'sequences': len(records),
            'frames': int(counts.sum()), 'views': sorted(set(r['view'] for r in records)),
            'target_digests': {r['uid']: io.digest(r['target']) for r in records}}


def saved_features(data, split, cfg, root):
    rows = target_records(data, cfg)
    for row in rows:
        pause(root, cfg)
        source = io.ROOT/cfg['s0a_root']/'evaluation'/split/row['uid']
        manifest = io.read(source/'manifest.json')
        for filename in ('sqrt_logits.npy', 'sqrt_pred.npy'):
            io.require(io.sha256_file(source/filename) == manifest['payload'][filename], 'saved A payload changed')
        logits = np.load(source/'sqrt_logits.npy', allow_pickle=False)
        prediction = np.load(source/'sqrt_pred.npy', allow_pickle=False)
        io.require(logits.shape == (len(row['truth']), 16), 'A covered support')
        np.testing.assert_array_equal(logits.argmax(1), prediction)
        row['feature'] = core.dynamics(logits)
    return rows


def extract_one(data, row, model, base, device, cfg, root, split):
    total = np.zeros((len(row['truth']), 16), np.float64)
    begin, end = row['begin'], row['end']; batch = cfg['input']['extraction_batch']
    with torch.no_grad():
        for lo in range(begin, end, batch):
            pause(root, cfg)
            hi = min(lo+batch, end)
            x, _ = data.batch(np.arange(lo, hi))
            feature = s.core.dense_features(model, torch.from_numpy(x).to(device))
            s.core.add_logits(total, data.starts[lo:hi], base(feature).cpu().numpy())
            if (lo-begin)//batch % 16 == 0:
                publish(root, 'extract', split=split, windows=hi, total=data.count)
    counts = s.core.coverage(len(row['truth']), data.starts[begin:end])
    io.require(np.all(counts > 0), 'training fusion covered prefix')
    return (total/counts[:, None]).astype(np.float32)


def frozen_model(ac, base_state, device):
    model = a.load_adl_model(io.ROOT/ac['encoder'], io.ROOT/ac['adl_head'], device)
    base = nn.Linear(2048, 16).to(device); base.load_state_dict(base_state, strict=True)
    base.eval().requires_grad_(False)
    return model, base


def make_heads(cfg, device):
    torch.manual_seed(cfg['training']['seed'])
    original = core.TransitionHead(cfg['model'])
    heads = {f'cap{cap}': copy.deepcopy(original).to(device) for cap in cfg['training']['caps']}
    tr = cfg['training']
    opts = {n: torch.optim.AdamW(h.parameters(), lr=tr['lr'], weight_decay=tr['weight_decay'],
                                betas=tuple(tr['betas']), eps=tr['eps']) for n, h in heads.items()}
    return heads, opts


def make_batch(records, windows, indices, device):
    x = np.empty((len(indices), 64, 35), np.float32)
    y = np.empty((len(indices), 64), np.int64)
    for i, (sid, start) in enumerate(windows[indices]):
        x[i] = records[sid]['feature'][start:start+64]
        y[i] = records[sid]['target'][start:start+64]
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def step(heads, opts, x, y, weights, cfg):
    losses = {}
    for name, head in heads.items():
        head.train(); opts[name].zero_grad(set_to_none=True)
        logits = head(x)
        numerator = F.cross_entropy(logits.reshape(-1, 4), y.reshape(-1), weight=weights[name], reduction='sum')
        denominator = weights[name][y].sum(); loss = numerator/denominator
        io.require(torch.isfinite(loss).item(), 'nonfinite training loss')
        loss.backward()
        norm = nn.utils.clip_grad_norm_(head.parameters(), cfg['training']['gradient_clip'])
        io.require(torch.isfinite(norm).item(), 'nonfinite gradients')
        opts[name].step()
        losses[name] = (float(numerator.detach()), float(denominator))
    return losses


def evaluate(records, heads, device, cfg, root, dest=None):
    metrics = {n: core.EventMetrics(cfg['evaluation']['matching_tolerance_frames'], cfg['input']['fps']) for n in heads}
    payload = {}
    for row in records:
        pause(root, cfg)
        for name, head in heads.items():
            logits = core.predict(head, row['feature'], device, cfg['evaluation']['chunk_frames'])
            pred = logits.argmax(1).astype(np.uint8)
            metrics[name].add(pred, row['truth'], row['view'])
            if dest is not None:
                for suffix, value in (('logits', logits), ('pred', pred)):
                    filename = row['uid']+'_'+name+'_'+suffix+'.npy'
                    path = dest/filename
                    if path.exists():
                        np.testing.assert_array_equal(np.load(path, allow_pickle=False), value)
                    else:
                        io.save_array(path, value)
                    payload[filename] = io.sha256_file(path)
    result = {n: m.result() for n, m in metrics.items()}
    if dest is not None:
        immutable_json(dest/'manifest.json', {'metrics': result, 'payload': payload,
                                             'heads': {n: s.hash_named_tensors(h.state_dict().items()) for n, h in heads.items()}})
    return result


def save_progress(root, state, heads, opts):
    state['cpu_rng'] = torch.get_rng_state()
    state['cuda_rng'] = torch.cuda.get_rng_state()
    a.save_state(root, state, heads, opts)


def train(root, records, windows, groups, val, heads, opts, device, cfg, locked, counts):
    state = a.restore_state(root, locked, heads, opts)
    if 'cpu_rng' in state:
        torch.set_rng_state(state['cpu_rng']); torch.cuda.set_rng_state(state['cuda_rng'])
    weights = {f'cap{cap}': torch.from_numpy(core.weights(counts, cap)).to(device) for cap in cfg['training']['caps']}
    tr = cfg['training']; batch = tr['batch_size']; names = list(heads)
    save_progress(root, state, heads, opts)
    try:
        while state['epoch'] <= tr['epochs']:
            epoch = state['epoch']; order = core.balanced_order(groups, len(windows), tr['seed']+epoch)
            immutable_json(root/'training'/f'order_{epoch:02d}.json', {'sha256': io.digest(order), 'windows': len(order)})
            publish(root, 'train', epoch=epoch, epochs=tr['epochs'], windows=state['next'], total=len(order))
            for lo in range(state['next'], len(order), batch):
                pause(root, cfg); tick = time.monotonic()
                hi = min(lo+batch, len(order))
                x, y = make_batch(records, windows, order[lo:hi], device)
                losses = step(heads, opts, x, y, weights, cfg)
                for n, (loss, den) in losses.items():
                    state['loss_sum'][n] += loss; state['loss_denom'][n] += den
                state['next'] = hi; state['train_seconds'] += time.monotonic()-tick
                if hi == len(order) or (lo//batch+1) % cfg['execution']['checkpoint_batches'] == 0:
                    save_progress(root, state, heads, opts)
                    publish(root, 'train', epoch=epoch, epochs=tr['epochs'], windows=hi, total=len(order))
            publish(root, 'val', epoch=epoch, epochs=tr['epochs'], split='val')
            metric = evaluate(val, heads, device, cfg, root)
            epoch_path = root/'training'/f'epoch_{epoch:02d}.pt'
            saved = {'heads': {n: io.cpu_state(h) for n, h in heads.items()}, 'epoch': epoch, 'contract': locked}
            if epoch_path.exists():
                previous = torch.load(epoch_path, map_location='cpu', weights_only=True)
                io.require(previous['contract'] == locked and previous['epoch'] == epoch, 'epoch lineage')
                for n in names:
                    io.require(s.hash_named_tensors(previous['heads'][n].items()) == s.hash_named_tensors(saved['heads'][n].items()), 'epoch resume exact')
            else:
                io.save_tensor(epoch_path, saved)
            state['history'].append({'epoch': epoch, 'metrics': metric, 'checkpoint_sha256': io.sha256_file(epoch_path),
                                     'mean_loss': {n: state['loss_sum'][n]/state['loss_denom'][n] for n in names}})
            io.save_json(root/'training/history.json', state['history'])
            state.update(epoch=epoch+1, next=0, loss_sum={n: 0. for n in names}, loss_denom={n: 0. for n in names})
            save_progress(root, state, heads, opts)
    except (io.PauseRequested, KeyboardInterrupt):
        save_progress(root, state, heads, opts)
        raise
    return state


def smoke(root, records, windows, groups, data, model, base, device, cfg, locked):
    indices = np.concatenate([group[:cfg['execution']['smoke_windows_per_group']] for group in groups])
    io.require(len(indices) == 4*cfg['execution']['smoke_windows_per_group'], 'smoke groups insufficient')
    original_ids = [records[sid]['begin']+int(start)//8 for sid, start in windows[indices]]
    x, _ = data.batch(np.asarray(original_ids))
    with torch.no_grad():
        values = base(s.core.dense_features(model, torch.from_numpy(x).to(device))).cpu().numpy()
    features = torch.from_numpy(np.stack([core.dynamics(v) for v in values])).to(device)
    targets = torch.from_numpy(np.stack([records[sid]['target'][start:start+64] for sid, start in windows[indices]])).to(device)
    heads, opts = make_heads(cfg, device)
    frozen = s.hash_named_tensors(model.state_dict().items()); base_hash = s.hash_named_tensors(base.state_dict().items())
    counts = source_summary(records)['target_counts']
    weights = {f'cap{cap}': torch.from_numpy(core.weights(counts, cap)).to(device) for cap in cfg['training']['caps']}
    state = a.restore_state(root, locked, heads, opts)
    io.require(state['epoch'] == 1 and state['next'] == 0, 'smoke uses fresh one-step state')
    step(heads, opts, features, targets, weights, cfg)
    save_progress(root, state, heads, opts)
    step(heads, opts, features, targets, weights, cfg)
    expected = {n: s.hash_named_tensors(h.state_dict().items()) for n, h in heads.items()}
    heads, opts = make_heads(cfg, device)
    restored = a.restore_state(root, locked, heads, opts)
    torch.set_rng_state(restored['cpu_rng']); torch.cuda.set_rng_state(restored['cuda_rng'])
    step(heads, opts, features, targets, weights, cfg)
    io.require(expected == {n: s.hash_named_tensors(h.state_dict().items()) for n, h in heads.items()}, 'optimizer/RNG resume exact')
    audit = core.prefix_audit(next(iter(heads.values())), device)
    io.require(frozen == s.hash_named_tensors(model.state_dict().items()), 'frozen encoder changed')
    io.require(base_hash == s.hash_named_tensors(base.state_dict().items()), 'frozen A changed')
    io.require(all(p.grad is None and not p.requires_grad for m in (model, base) for p in m.parameters()), 'frozen gradient leak')
    return {'passed': True, 'research_usable': False, 'holdouts_opened': False, 'windows': len(indices),
            'optimizer_rng_resume_exact': True, 'frozen_model_unchanged': True, 'audit': audit,
            'peak_gpu_gib': torch.cuda.max_memory_allocated()/1024**3}


def audit_saved(records, dest, name, cfg):
    manifest = io.read(dest/'manifest.json')
    metric = core.EventMetrics(cfg['evaluation']['matching_tolerance_frames'], cfg['input']['fps'])
    for row in records:
        for suffix in ('logits', 'pred'):
            filename = row['uid']+'_'+name+'_'+suffix+'.npy'
            io.require(io.sha256_file(dest/filename) == manifest['payload'][filename], 'final saved payload')
        logits = np.load(dest/(row['uid']+'_'+name+'_logits.npy'), allow_pickle=False)
        pred = np.load(dest/(row['uid']+'_'+name+'_pred.npy'), allow_pickle=False)
        np.testing.assert_array_equal(logits.argmax(1), pred)
        metric.add(pred, row['truth'], row['view'])
    io.require(metric.result() == manifest['metrics'][name], 'saved metric replay')
    return metric.result()


def ranking(records, dest, name):
    probability, truth = [], []
    for row in records:
        logits = np.load(dest/(row['uid']+'_'+name+'_logits.npy'), allow_pickle=False)
        probability.append(torch.softmax(torch.from_numpy(logits), -1).numpy())
        truth.append(row['target'])
    p, y = np.concatenate(probability), np.concatenate(truth)
    return {'split': 'val', 'threshold_fitted': False, 'selection_use': False,
            'per_class': [{'class': k, 'prevalence': float(np.mean(y == k)),
                           'average_precision': float(average_precision_score(y == k, p[:, k])) if np.any(y == k) else None}
                          for k in (1, 2, 3)]}


def run(args, cfg, root):
    ac, _, _, base_state = s.s0a_dependency(cfg)
    e.parent(cfg)
    locked = {'source': s.fingerprint(CODE), 'config_sha256': io.sha256_file(CONFIG),
              'torch': str(torch.__version__), 'numpy': str(np.__version__), 'smoke': args.smoke,
              'historical_exact_reproduction': False}
    s.lock_contract(root, locked, args.resume)
    if (root/'final_report.json').exists():
        io.require(io.read(root/'final_report.json')['passed'], 'existing run failed')
        io.require(io.sha256_file(root/'final_report.json') == io.read(root/'status.json')['final_report_sha256'], 'existing final changed')
        return
    pause(root, cfg)
    io.disk_gate(cfg, int(cfg['execution']['estimated_output_gib']*1024**3))
    if not args.smoke:
        smoke_root = Path(str(root)+'_smoke')
        io.require(io.read(smoke_root/'final_report.json')['passed'], 'GPU smoke required')
        io.require({**io.read(smoke_root/'run_contract.json'), 'smoke': False} == locked, 'smoke contract differs')
    publish(root, 'preflight', holdouts_opened=False)
    train_data, val_data = [s.core.verify_split(ac, split) for split in ('train', 'val')]
    io.require(not ({r['subject'] for r in train_data.rows} & {r['subject'] for r in val_data.rows}), 'subject split overlap')
    records = target_records(train_data, cfg)
    windows, groups = core.window_groups(records)
    io.require(len(windows) == train_data.count, 'window inventory differs')
    audit = {'train': source_summary(records), 'val': source_summary(target_records(val_data, cfg)),
             'sampling_group_windows': [len(g) for g in groups], 'train_windows': len(windows), 'holdouts_opened': False}
    immutable_json(root/'source_audit.json', audit)
    device = io.device_for(cfg)
    model, base = frozen_model(ac, base_state, device)
    frozen = {n: s.hash_named_tensors(m.state_dict().items()) for n, m in (('encoder', model), ('s0a', base))}
    if args.smoke:
        report = smoke(root, records, windows, groups, train_data, model, base, device, cfg, locked)
    else:
        val = saved_features(val_data, 'val', cfg, root)
        reference = extract_one(val_data, val[0], model, base, device, cfg, root, 'val_reference')
        original = np.load(io.ROOT/cfg['s0a_root']/'evaluation/val'/val[0]['uid']/'sqrt_logits.npy', allow_pickle=False)
        np.testing.assert_array_equal(reference, original)
        immutable_json(root/'frozen_reference.json', {'frozen': frozen, 'uid': val[0]['uid'],
                                                      'S0A_fused_reference_exact': True, 'logits_digest': io.digest(reference)})
        total_bytes = sum(r['feature'].nbytes for r in val)
        feature_audit = {}
        for number, row in enumerate(records, 1):
            row['feature'] = core.dynamics(extract_one(train_data, row, model, base, device, cfg, root, 'train'))
            total_bytes += row['feature'].nbytes
            io.require(total_bytes <= cfg['execution']['ram_feature_budget_gib']*1024**3, 'RAM feature budget')
            feature_audit[row['uid']] = io.digest(row['feature'])
            publish(root, 'extract', split='train', sequences=number, windows=row['end'], total=train_data.count)
        immutable_json(root/'RAM_feature_audit.json', {'digests': feature_audit, 'ram_bytes': total_bytes,
                                                      'disk_feature_cache': False, 'frozen': frozen})
        for name, module in (('encoder', model), ('s0a', base)):
            io.require(s.hash_named_tensors(module.state_dict().items()) == frozen[name], 'frozen extraction changed')
            io.require(all(p.grad is None and not p.requires_grad for p in module.parameters()), 'frozen gradient leak')
        del model, base
        torch.cuda.empty_cache()
        heads, opts = make_heads(cfg, device)
        state = train(root, records, windows, groups, val, heads, opts, device, cfg, locked, audit['train']['target_counts'])
        io.require(len(state['history']) == cfg['training']['epochs'], 'incomplete training')
        chosen = core.choose(state['history'], list(heads))
        selection = {**chosen, 'selected_using': 'val only', 'holdouts_opened_at_selection': False,
                     'contract_sha256': io.sha256_file(root/'run_contract.json'),
                     'history_sha256': io.sha256_file(root/'training/history.json')}
        immutable_json(root/'selection.json', selection); selected_sha = io.sha256_file(root/'selection.json')
        publish(root, 'selection', epoch=chosen['epoch'], candidate=chosen['candidate'])
        path = root/'training'/f'epoch_{chosen["epoch"]:02d}.pt'
        io.require(io.sha256_file(path) == chosen['checkpoint_sha256'], 'selected weights changed')
        best = torch.load(path, map_location='cpu', weights_only=True); name = chosen['candidate']
        heads[name].load_state_dict(best['heads'][name], strict=True); selected = {name: heads[name]}
        prefix = core.prefix_audit(heads[name], device)
        del records, windows, groups
        final_metrics = {}; final_support = {}
        for split in ('val', 'test', 'ood'):
            io.require(io.sha256_file(root/'selection.json') == selected_sha, 'selection changed before holdout')
            data = val_data if split == 'val' else s.core.verify_split(ac, split)
            rows = val if split == 'val' else saved_features(data, split, cfg, root)
            publish(root, 'evaluate', split=split)
            dest = root/'evaluation'/split
            final_metrics[split] = evaluate(rows, selected, device, cfg, root, dest)[name]
            io.require(audit_saved(rows, dest, name, cfg) == final_metrics[split], 'final metric replay')
            final_support[split] = source_summary(rows)
            if split == 'val':
                io.require(final_metrics[split] == chosen['metrics'], 'selected val exact replay differs')
                immutable_json(root/'validation_ranking.json', ranking(rows, dest, name))
            # Existing official C payloads must remain unchanged, including holdouts only after lock.
            c_dest = io.ROOT/cfg['s0c_root']/'evaluation'/split
            for filename, sha in io.read(c_dest/'manifest.json')['payload'].items():
                io.require(io.sha256_file(c_dest/filename) == sha, 'C timeline changed')
        publish(root, 'audit')
        for item in state['history']:
            path = root/'training'/f'epoch_{item["epoch"]:02d}.pt'
            io.require(io.sha256_file(path) == item['checkpoint_sha256'], 'epoch weights changed')
            weights = torch.load(path, map_location='cpu', weights_only=True)
            io.require(weights['contract'] == locked, 'epoch contract differs')
            io.require(all(torch.isfinite(v).all().item() for head in weights['heads'].values() for v in head.values()), 'nonfinite saved head')
        for split in ('train', 'val', 'test', 'ood'):
            s.core.verify_split(ac, split)
        report = {'passed': True, 'research_usable': True, 'historical_exact_reproduction': False,
                  'selected_epoch': chosen['epoch'], 'selected_candidate': name, 'epochs_per_candidate': cfg['training']['epochs'],
                  'selection_sha256': selected_sha, 'metrics': final_metrics, 'support': final_support,
                  'gate': core.gate(final_metrics['val'], cfg['evaluation']), 'frozen': frozen,
                  'frozen_models_unchanged': True, 'S0C_timelines_unchanged': True,
                  'selection_val_exact_replay': True, 'saved_payload_metric_replay': True,
                  'all_sources_reverified': True, 'training_seconds': state['train_seconds'], 'prefix_audit': prefix,
                  'automatic_deployment': False, 'end_to_end_causal': False}
        io.require(io.sha256_file(root/'selection.json') == selected_sha, 'final selection changed')
    io.require(s.fingerprint(CODE) == locked['source'] and io.sha256_file(CONFIG) == locked['config_sha256'], 'code/config changed')
    s.s0a_dependency(cfg); e.parent(cfg)
    io.save_json(root/'final_report.json', report)
    publish(root, 'completed', feasible_for_integration=report.get('gate', {}).get('feasible_for_integration', False),
            final_report_sha256=io.sha256_file(root/'final_report.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true'); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); cfg = io.read(CONFIG)
    root = io.guard(io.ROOT/(cfg['output']+('_smoke' if args.smoke else '')))
    io.require(root.parent == io.ROOT/'checkpoint/fall', 'output scope')
    root.mkdir(parents=True, exist_ok=True); s.install_signals()
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args, cfg, root)
        except io.PauseRequested as error:
            publish(root, 'paused', reason=str(error)); raise SystemExit(75)
        except Exception as error:
            publish(root, 'failed', error_type=type(error).__name__, reason=str(error)); raise


if __name__ == '__main__':
    main()
