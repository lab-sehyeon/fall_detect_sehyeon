"""New recovery-only IO; never rewrites the locked S0-A implementation."""
import json
import os
from pathlib import Path
import signal
from datetime import datetime, timezone

import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix

from fall_pipeline.safer import v2_controls_reconstruction as io
from fall_pipeline.safer import run_s0a_reconstruction as a
from fall_pipeline.safer import s0a_core as core
from fall_pipeline.common.integrity import hash_named_tensors

STOP = False


def signal_stop(signum, frame):
    global STOP
    STOP = True


def install_signals():
    signal.signal(signal.SIGTERM, signal_stop)
    signal.signal(signal.SIGINT, signal_stop)


def pause(root, config):
    parent = io.ROOT / 'checkpoint/fall/STATE_RECOVERY_QUEUE_20260929_R1/PAUSE_REQUESTED'
    if STOP or (root / 'PAUSE_REQUESTED').exists() or parent.exists():
        raise io.PauseRequested('state recovery pause requested')
    try:
        io.disk_gate(config)
    except RuntimeError as error:
        raise io.PauseRequested(str(error)) from error


def publish(root, topic, stage, **details):
    state = {'stage': stage, 'utc': datetime.now(timezone.utc).isoformat(), **details}
    io.save_json(root / 'status.json', state)
    print(json.dumps(state, ensure_ascii=False), flush=True)
    if root.name.endswith('_smoke'):
        return
    status = 'completed' if stage == 'completed' else 'paused' if stage in ('paused', 'failed') else 'in_progress'
    header = f'- 문서 ID: `DOC-20260929-{topic}-run-R1`\n- 기준일: 2026-09-29\n- 상태: `{status}`\n'
    internal = f'# {topic.upper()} 자동 실행 기록\n\n{header}\nOutput: `{root}`\n\n'
    internal += '```json\n' + json.dumps(state, ensure_ascii=False, indent=2) + '\n```\n'
    words = {'preflight':'입력·선행 실험 검산', 'initial':'초기 동등성 검증', 'train':'학습', 'val':'검증셋 평가',
             'selection':'선택 고정', 'evaluate':'고정 평가', 'audit':'최종 감사', 'decode':'고정 후보 비교',
             'targets':'문맥 타깃 생성', 'completed':'사전 정의한 실행·검산 완료', 'paused':'일시 중단', 'failed':'실행 중단'}
    shared = f'# {topic.upper()} 실험 진행\n\n{header}\n현재 단계: {words[stage]}.\n\n'
    labels = {'epoch':'현재 epoch', 'epochs':'전체 epoch', 'windows':'처리 window', 'total':'전체 window',
              'split':'대상 split', 'completed_candidates':'평가한 후보', 'adoption':'사전 채택 기준 충족'}
    for key, label in labels.items():
        if key in details:
            shared += f'- {label}: {details[key]}\n'
    for kind, text in (('internal', internal), ('shared', shared)):
        text += f'\n[방법과 한계](2026-09-29_{topic}_reconstruction_{kind}.md). 전체 프로젝트 완료와 구분한다.\n'
        path = io.ROOT / f'docs/{kind}/2026-09-29_{topic}_run_{kind}.md'
        tmp = path.with_suffix('.tmp'); tmp.write_text(text); os.replace(tmp, path)


def s0a_dependency(config):
    path = io.ROOT / config['s0a_config']
    io.require(io.sha256_file(path) == config['s0a_config_sha256'], 'S0-A config pin')
    ac = io.read(path); root = io.ROOT / config['s0a_root']
    for filename, key in [('final_report.json','s0a_report_sha256'), ('selection.json','s0a_selection_sha256')]:
        io.require(io.sha256_file(root / filename) == config[key], 'S0-A '+filename+' pin')
    report = io.read(root / 'final_report.json'); selected = io.read(root / 'selection.json')
    io.require(report['passed'] and report['research_usable'] and report['selection_val_exact_replay'], 'S0-A final gate')
    io.require(a.contract(ac, False) == io.read(root / 'run_contract.json'), 'S0-A source invariant')
    checkpoint = root / 'training' / f'epoch_{selected["epoch"]:02d}.pt'
    io.require(io.sha256_file(checkpoint) == selected['checkpoint_sha256'] == config['s0a_checkpoint_sha256'], 'S0-A selected checkpoint pin')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    return ac, report, selected, saved['heads'][selected['candidate']]


def lock_contract(root, contract, resume):
    path = root / 'run_contract.json'
    if path.exists():
        io.require(resume and io.read(path) == contract, 'resume contract differs')
    else:
        io.require(not any(p.name != 'run.lock' for p in root.iterdir()), 'orphan output; preserve and inspect')
        io.save_json(path, contract)


def fingerprint(paths):
    return {name: io.sha256_file(io.ROOT / name) for name in sorted(set(paths))}


def evaluate(root, dest, data, head, model, device, ac, config, save_logits=False, zero_reference=None):
    """S0-A-identical minibatches, fusion and metrics; source files stay read-only."""
    head.eval(); model.eval()
    head_hash = hash_named_tensors(head.state_dict().items())
    metrics = a.metric_instance(ac)
    truths, predictions = [], []
    tail = 0
    for number, (row, begin, end) in enumerate(data.sequences(), 1):
        pause(root, config)
        uid = row['uid']; out = dest / uid; manifest_path = out / 'manifest.json'
        _, y = data.arrays(row['sequence_index'])
        count = core.coverage(len(y), data.starts[begin:end]); mask = count > 0
        truth = y[mask]; tail += int(np.count_nonzero(~mask))
        io.require(np.array_equal(np.flatnonzero(mask), np.arange(len(truth))), 'coverage contiguous prefix')
        if manifest_path.exists():
            saved = io.read(manifest_path)
            io.require(saved['head_hash'] == head_hash and saved['begin'] == begin and saved['end'] == end and saved['save_logits'] == save_logits, 'eval resume lineage')
            for name, sha in saved['payload'].items():
                io.require(io.sha256_file(out / name) == sha, 'eval saved payload')
            if zero_reference is not None:
                io.require(saved.get('zero_reference_exact') is True, 'initial replay proof missing')
        else:
            total = np.zeros((len(y), 16), np.float64)
            with torch.no_grad():
                for lo in range(begin, end, config['evaluation']['batch_size']):
                    pause(root, config)
                    hi = min(lo + config['evaluation']['batch_size'], end)
                    x, _ = data.batch(np.arange(lo, hi))
                    feature = core.dense_features(model, torch.from_numpy(x).to(device))
                    values = head(feature)
                    if zero_reference is not None:
                        io.require(torch.equal(values, head.base(feature)), 'initial base window mismatch')
                    core.add_logits(total, data.starts[lo:hi], values.cpu().numpy())
            mean = (total[mask] / count[mask, None]).astype(np.float32)
            io.require(np.isfinite(mean).all(), 'fused logits finite')
            pred = mean.argmax(1).astype(np.uint8)
            if zero_reference is not None:
                original = zero_reference / uid
                ref_manifest = io.read(original / 'manifest.json')
                for name in ('sqrt_logits.npy','sqrt_pred.npy'):
                    io.require(io.sha256_file(original / name) == ref_manifest['payload'][name], 'S0-A initial reference payload')
                np.testing.assert_array_equal(mean, np.load(original / 'sqrt_logits.npy', allow_pickle=False))
                np.testing.assert_array_equal(pred, np.load(original / 'sqrt_pred.npy', allow_pickle=False))
            output = {'pred.npy': pred}
            if save_logits:
                output['logits.npy'] = mean
            for name, array in output.items():
                io.save_array(out / name, array)
            io.save_json(manifest_path, {'head_hash':head_hash, 'begin':begin, 'end':end, 'save_logits':save_logits,
                                        'zero_reference_exact':zero_reference is not None,
                                        'payload':{name:io.sha256_file(out / name) for name in output}})
        pred = np.load(out / 'pred.npy', allow_pickle=False)
        if save_logits:
            np.testing.assert_array_equal(np.load(out / 'logits.npy', mmap_mode='r', allow_pickle=False).argmax(1), pred)
        metrics.add(pred, truth); truths.append(truth); predictions.append(pred)
        if number == 1 or number % 5 == 0:
            publish(root, 's0b', 'evaluate' if save_logits else 'val', split=dest.name, windows=end, total=data.count)
    result = metrics.result(); truth, pred = np.concatenate(truths), np.concatenate(predictions)
    np.testing.assert_array_equal(confusion_matrix(truth, pred, labels=range(16)), result['confusion'])
    io.require(abs(f1_score(truth,pred,labels=range(16),average='macro',zero_division=0)-result['macro_f1']) < 1e-12, 'independent frame F1')
    io.save_json(dest / 'metrics.json', {'metrics':result, 'tail':tail, 'independent_frame_metrics_verified':True})
    return result
