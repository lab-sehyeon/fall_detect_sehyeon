"""Audit author weights on CPU; no dataset inference or performance scoring."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import contextlib
import hashlib
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'third_party/original_impact_stgcn_20261004'
EVIDENCE = ROOT / 'docs/internal/2026-10-04_fall_specialized_baselines_evidence'
sys.path.insert(0, str(ROOT / 'data/source_archives/OriginalBaselines_20261004_r1/runtime'))
sys.path.insert(0, str(REPO))
import h5py
import numpy as np
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(1)

def main():
    from GCN.stgcngru_bilstm import Stgcn_gru_biLstm
    path = REPO / 'models/trained_stgcn_Data_FALL_Kimore_ex5_SMV1_weights.h5'
    report = {'scope': 'Author asset compatibility only', 'tensorflow': tf.__version__,
              'checkpoint': str(path.relative_to(ROOT)),
              'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'external_evaluation_executed': False, 'training_executed': False}
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            wrapper = Stgcn_gru_biLstm(num_joints=33, num_channels=3, num_timesteps=100)
            model = wrapper.model
            model.load_weights(str(path), by_name=False, skip_mismatch=False)
        report['strict_load'] = True
        checks = []
        with h5py.File(path) as h:
            def decode(v):
                return v.decode() if isinstance(v, bytes) else str(v)
            for lname in h.attrs['layer_names']:
                lname = decode(lname)
                group = h[lname]
                names = [decode(v) for v in group.attrs['weight_names']]
                layer = model.get_layer(lname)
                assert len(names) == len(layer.weights), lname
                for name, variable in zip(names, layer.weights):
                    expected = group[name][()]
                    actual = variable.numpy()
                    assert np.array_equal(expected, actual), name
                    checks.append({'file_tensor': name, 'shape': list(expected.shape), 'exact': True})
            datasets = []
            h.visititems(lambda name, value: datasets.append(name) if isinstance(value, h5py.Dataset) else None)
            assert len(datasets) == len(checks) == len(model.weights)
        report['all_tensors_exact'] = True
        report['tensor_count'] = len(checks)
        report['tensor_checks'] = checks
        report['parameters'] = model.count_params()
        report['input_shape'] = list(model.input_shape)
        report['output_shape'] = list(model.output_shape)
        start = time.monotonic()
        output = model(np.zeros((1, 100, 33, 3), dtype=np.float32), training=False).numpy()
        assert output.shape == (1, 1) and np.isfinite(output).all()
        report['synthetic_shape_check'] = {'passed': True, 'shape': list(output.shape),
                                          'seconds': time.monotonic() - start}
        report['passed'] = True
    except Exception as exc:
        report['passed'] = False
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    EVIDENCE.mkdir(exist_ok=True)
    (EVIDENCE / 'stgcn_load_verification.log').write_text(log.getvalue())
    (EVIDENCE / 'stgcn_load_verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'tensor_checks'}, indent=2))

if __name__ == '__main__':
    main()
