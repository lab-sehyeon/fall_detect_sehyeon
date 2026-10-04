"""Explicit, audited R2 -> R3 continuation; source artifacts are never rewritten."""
import copy
import fcntl
import os
from pathlib import Path
import shutil
import sys

import torch

from fall_pipeline.safer import run_s0b_reconstruction_r2 as old
from fall_pipeline.safer import run_s0b_reconstruction_r3 as new

io = new.io


def validate_change(before, after):
    expected = copy.deepcopy(before)
    expected['experiment_id'] = 'S0B_DOCUMENT_RECONSTRUCTION_20260929_R3'
    expected['output'] = 'checkpoint/fall/' + expected['experiment_id']
    expected['training']['microbatch_size'] = 512
    expected['continuation'] = after['continuation']
    expected['parallel'] = after['parallel']
    expected['execution']['gpu_memory_fraction'] = .85
    io.require(after['parallel']['physical_devices'] == [0,1] and after['parallel']['global_batch'] == 1024 and after['parallel']['per_device_batch'] == 512, 'parallel batch contract')
    io.require(after['parallel']['secondary_dropout_seed_at_transition'] == 1, 'secondary RNG contract')
    io.require(expected == after, 'unauthorized change beyond microbatch/output/lineage')
    io.require(before['training']['microbatch_size'] == 512, 'parent microbatch differs')
    io.require(after['training']['batch_size'] == 1024, 'effective batch changed')


def equal_tree(first, second):
    if isinstance(first, torch.Tensor):
        return (isinstance(second, torch.Tensor) and first.dtype == second.dtype
                and first.shape == second.shape and torch.equal(first.cpu(), second.cpu()))
    if isinstance(first, dict):
        return isinstance(second, dict) and first.keys() == second.keys() and all(equal_tree(first[k], second[k]) for k in first)
    if isinstance(first, (list, tuple)):
        return type(first) is type(second) and len(first) == len(second) and all(equal_tree(a, b) for a, b in zip(first, second))
    return first == second


def main():
    io.require(Path(sys.prefix).name == 'fall_detect' and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'migration CPU-only fall_detect')
    config = io.read(new.CONFIG)
    lineage = config['continuation']
    parent = io.guard(io.ROOT / lineage['parent_root'])
    before_path = io.ROOT / lineage['parent_config']
    io.require(io.sha256_file(before_path) == lineage['parent_config_sha256'], 'parent config pin')
    before = io.read(before_path)
    validate_change(before, config)
    io.disk_gate(config)
    # A nonblocking lock proves the writer is no longer running, beyond its status file.
    with (parent / 'run.lock').open('a') as parent_lock:
        fcntl.flock(parent_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        io.require(io.read(parent / 'status.json')['stage'] == 'paused', 'parent is not paused')
        io.require(not (parent / 'final_report.json').exists(), 'completed parent is not this migration')
        pins = {'run_contract.json': lineage['parent_contract_sha256'],
                'training/progress.json': lineage['parent_progress_sha256']}
        pointer = io.read(parent / 'training/progress.json')
        checkpoint = f'training/resume_{pointer["slot"]}.pt'
        pins[checkpoint] = lineage['parent_checkpoint_sha256']
        for filename, sha in pins.items():
            io.require(io.sha256_file(parent / filename) == sha, 'parent pin ' + filename)
        old_contract = old.make_contract(before, False)
        io.require(old_contract == io.read(parent / 'run_contract.json'), 'parent code/config changed')
        state = torch.load(parent / checkpoint, map_location='cpu', weights_only=True)
        io.require(state['contract'] == old_contract and pointer['sha256'] == pins[checkpoint], 'parent checkpoint lineage')
        io.require(state['epoch'] == pointer['epoch'] == lineage['epoch'] == 1, 'migration epoch changed')
        io.require(state['next'] == pointer['next'] == lineage['next'] and state['next'] % 1024 == 0, 'not committed optimizer boundary')
        io.require(state['history'] == [], 'migration intentionally only supports first-epoch parent')
        io.require('cpu_rng' in state and 'cuda_rng' in state and state['optimizers']['tcn']['state'], 'optimizer/RNG missing')
        for tensor in state['heads']['tcn'].values():
            io.require(bool(torch.isfinite(tensor).all()), 'nonfinite parent head')
        initial = torch.load(parent / 'training/epoch_00.pt', map_location='cpu', weights_only=True)
        io.require(initial['contract'] == old_contract and initial['epoch'] == 0, 'initial checkpoint provenance')
        baseline = parent / 'validation/epoch_00'
        manifest_files = sorted(baseline.glob('*/manifest.json'))
        io.require(len(manifest_files) == 73, 'complete initial validation required')
        for path in manifest_files:
            manifest = io.read(path)
            io.require(manifest['zero_reference_exact'], 'initial equality proof absent')
            for name, sha in manifest['payload'].items():
                io.require(io.sha256_file(path.parent / name) == sha, 'initial payload changed')
        baseline_files = {str(p.relative_to(baseline)): io.sha256_file(p)
                          for p in sorted(baseline.rglob('*')) if p.is_file()}
        contract = new.make_contract(config, False)
        root = io.guard(io.ROOT / config['output'])
        io.require(not root.exists(), 'new output already exists; never overwrite migration')
        root.mkdir(parents=True)
        new.s.lock_contract(root, contract, False)
        migrated = copy.deepcopy(state)
        migrated['contract'] = contract
        migrated_initial = copy.deepcopy(initial)
        migrated_initial['contract'] = contract
        io.save_tensor(root / 'training/resume_0.pt', migrated)
        io.save_json(root / 'training/progress.json', {'slot': 0, 'sha256': io.sha256_file(root / 'training/resume_0.pt'),
                                                    'epoch': state['epoch'], 'next': state['next']})
        io.save_tensor(root / 'training/epoch_00.pt', migrated_initial)
        shutil.copytree(baseline, root / 'validation/epoch_00')
        for name, sha in baseline_files.items():
            io.require(io.sha256_file(root / 'validation/epoch_00' / name) == sha, 'initial-copy mismatch')
        saved = torch.load(root / 'training/resume_0.pt', map_location='cpu', weights_only=True)
        saved['contract'] = old_contract
        io.require(equal_tree(saved, state), 'continuation tensor/optimizer/RNG/cursor mismatch')
        saved_initial = torch.load(root / 'training/epoch_00.pt', map_location='cpu', weights_only=True)
        saved_initial['contract'] = old_contract
        io.require(equal_tree(saved_initial, initial), 'epoch0 tensor mismatch')
        for filename, sha in pins.items():
            io.require(io.sha256_file(parent / filename) == sha, 'parent modified during migration')
        report = {'passed': True, 'parent_pins': pins, 'parent_config_sha256': lineage['parent_config_sha256'],
                  'epoch': state['epoch'], 'next': state['next'], 'parent_train_seconds': state['train_seconds'],
                  'head_optimizer_rng_cursor_exact': True, 'initial_validation_copy_exact': True,
                  'initial_validation_files': baseline_files, 'parent_unchanged': True,
                  'only_checkpoint_metadata_contract_changed': True, 'bit_exact_future_continuation': False,
                  'new_contract_sha256': io.sha256_file(root / 'run_contract.json')}
        io.save_json(root / 'migration_report.json', report)
        print({k: v for k, v in report.items() if k != 'initial_validation_files'}, flush=True)


if __name__ == '__main__':
    main()
