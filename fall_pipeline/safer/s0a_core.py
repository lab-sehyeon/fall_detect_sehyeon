"""Separate S0-A reconstruction: compact input, dense fusion and explicit metrics."""
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from fall_pipeline.safer import v2_controls_reconstruction as io
from fall_pipeline.safer.f1_core import dense_backbone_features


def dense_features(model, x):
    temporal, context = dense_backbone_features(model, x)
    return torch.cat((temporal, context[:, None].expand(-1, 64, -1)), dim=-1)


class CompactSplit:
    """Read-only mmap storage, random windows without materialized dense cache."""
    def __init__(self, root, split, limit=None):
        self.root = Path(root) / 'sequence_cache'
        base = self.root / split
        self.rows = io.read(base / 'sequences.json')
        self.ids = np.load(base / 'sequence_index.npy', mmap_mode='r', allow_pickle=False)
        self.starts = np.load(base / 'window_start.npy', mmap_mode='r', allow_pickle=False)
        self.count = len(self.starts) if limit is None else min(limit, len(self.starts))
        self.cache = OrderedDict()

    def arrays(self, sid):
        uid = self.rows[int(sid)]['uid']
        if uid not in self.cache:
            self.cache[uid] = tuple(np.load(self.root / uid / name, mmap_mode='r', allow_pickle=False)
                                    for name in ('ntu25.npy', 'labels.npy'))
            # Full train/val fits under 512 descriptors*2; no writable mmap.
            if len(self.cache) > 512:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(uid)
        return self.cache[uid]

    def batch(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        io.require(indices.ndim == 1 and np.all((indices >= 0) & (indices < self.count)), 'window indices')
        ids, starts = self.ids[indices], self.starts[indices]
        x = np.zeros((len(indices), 3, 64, 25, 2), np.float32)
        y = np.empty((len(indices), 64), np.int64)
        for sid in np.unique(ids):
            chosen = np.flatnonzero(ids == sid)
            pose, labels = self.arrays(sid)
            frames = starts[chosen, None] + np.arange(64)
            io.require(np.all((frames >= 0) & (frames < len(labels))), 'frame indices')
            x[chosen, :, :, :, 0] = pose[frames].transpose(0, 3, 1, 2)
            y[chosen] = labels[frames]
        return x, y

    def sequences(self):
        for row in self.rows:
            begin, end = row['output_start'], min(row['output_stop'], self.count)
            if begin < end:
                yield row, begin, end

    def frequencies(self):
        counts = np.zeros(16, np.int64)
        frames = uncovered = 0
        for row, begin, end in self.sequences():
            _, y = self.arrays(row['sequence_index'])
            cover = coverage(len(y), self.starts[begin:end])
            counts += np.bincount(y[cover > 0], minlength=16)
            frames += int(np.count_nonzero(cover))
            uncovered += int(np.count_nonzero(cover == 0))
        return {'counts': counts.tolist(), 'covered_frames': frames, 'uncovered_frames': uncovered}


def coverage(n, starts):
    starts = np.asarray(starts, np.int64)
    io.require(np.all((starts >= 0) & (starts + 64 <= n)), 'coverage bounds')
    delta = np.zeros(n + 1, np.int64)
    np.add.at(delta, starts, 1)
    np.add.at(delta, starts + 64, -1)
    return np.cumsum(delta[:-1])


def add_logits(total, starts, logits):
    io.require(logits.shape == (len(starts), 64, 16) and np.isfinite(logits).all(), 'dense logits')
    # Distinct window starts permit vectorized accumulation for each temporal offset.
    io.require(len(np.unique(starts)) == len(starts), 'duplicate window starts')
    for offset in range(64):
        total[starts + offset] += logits[:, offset]


def segments(labels):
    labels = np.asarray(labels, np.int64)
    if len(labels) == 0:
        return (np.empty(0, np.int64),) * 3
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    return labels[starts], starts, np.r_[starts[1:], len(labels)]


def segment_counts(prediction, truth, threshold):
    pl, ps, pe = segments(prediction)
    gl, gs, ge = segments(truth)
    used = np.zeros(len(gl), bool)
    tp = 0
    for label, start, end in zip(pl, ps, pe):
        if not len(gl):
            continue
        intersection = np.maximum(0, np.minimum(end, ge) - np.maximum(start, gs))
        union = np.maximum(end, ge) - np.minimum(start, gs)
        overlap = intersection / union * (gl == label)
        best = int(np.argmax(overlap))
        if overlap[best] >= threshold and not used[best]:
            tp += 1
            used[best] = True
    return [tp, len(pl) - tp, len(gl) - tp]


def edit_distance(first, second):
    """Levenshtein with O(min(n,m)) memory; vectorized left-prefix recurrence."""
    a, b = np.asarray(first, np.int64), np.asarray(second, np.int64)
    if len(b) > len(a):
        a, b = b, a
    j = np.arange(len(b) + 1)
    previous = j.copy()
    for i, label in enumerate(a, 1):
        base = np.r_[i, np.minimum(previous[1:] + 1, previous[:-1] + (b != label))]
        previous = j + np.minimum.accumulate(base - j)
    return int(previous[-1])


def transition_counts(prediction, truth, pair, tolerance):
    def points(y):
        return np.flatnonzero((y[:-1] == pair[0]) & (y[1:] == pair[1])) + 1
    p, g = points(np.asarray(prediction)), points(np.asarray(truth))
    used = np.zeros(len(p), bool)
    delays = []
    for time in g:
        eligible = np.flatnonzero((np.abs(p - time) <= tolerance) & ~used)
        if len(eligible):
            index = eligible[0]
            used[index] = True
            delays.append(int(p[index] - time))
    return {'tp': len(delays), 'fp': len(p) - len(delays), 'fn': len(g) - len(delays), 'delays_frames': delays}


def prf(tp, fp, fn):
    return {'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
            'precision': float(tp / (tp + fp)) if tp + fp else 0.,
            'recall': float(tp / (tp + fn)) if tp + fn else 0.,
            'f1': float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.}


class Metrics:
    def __init__(self, pairs, tolerance, fps=25):
        self.confusion = np.zeros((16, 16), np.int64)
        self.overlap = {str(t): np.zeros(3, np.int64) for t in (10, 25, 50)}
        self.edits = []
        self.switches = self.frames = 0
        self.pairs, self.tolerance, self.fps = pairs, tolerance, fps
        self.transitions = {f'{a}->{b}': {'tp': 0, 'fp': 0, 'fn': 0, 'delays_frames': []} for a, b in pairs}

    def add(self, predicted, truth):
        p, y = np.asarray(predicted, np.int64), np.asarray(truth, np.int64)
        io.require(p.shape == y.shape and y.ndim == 1 and len(y) > 0, 'metric shape')
        io.require(np.all((p >= 0) & (p < 16) & (y >= 0) & (y < 16)), 'metric labels')
        self.confusion += np.bincount(16 * y + p, minlength=256).reshape(16, 16)
        for threshold, counts in self.overlap.items():
            counts += segment_counts(p, y, int(threshold) / 100)
        pl, yl = segments(p)[0], segments(y)[0]
        self.edits.append(1 - edit_distance(pl, yl) / max(len(pl), len(yl)))
        self.switches += int(np.count_nonzero(p[1:] != p[:-1]))
        self.frames += len(p)
        for pair in self.pairs:
            dest = self.transitions[f'{pair[0]}->{pair[1]}']
            for key, value in transition_counts(p, y, pair, self.tolerance).items():
                dest[key] += value

    def result(self):
        c = self.confusion
        per_class = [prf(c[i, i], c[:, i].sum() - c[i, i], c[i].sum() - c[i, i]) for i in range(16)]
        return {'frames': self.frames, 'sequences': len(self.edits), 'confusion': c.tolist(),
                'accuracy': float(c.trace() / max(1, c.sum())),
                'macro_f1': float(np.mean([x['f1'] for x in per_class])), 'per_class': per_class,
                'segments': {k: prf(*v) for k, v in self.overlap.items()},
                'segment_f1_50': prf(*self.overlap['50'])['f1'],
                'edit': float(np.mean(self.edits)) if self.edits else 0.,
                'switches': self.switches, 'switches_per_minute': self.switches * self.fps * 60 / max(1, self.frames),
                'transitions': {k: {**prf(v['tp'], v['fp'], v['fn']), 'delays_frames': v['delays_frames']}
                                for k, v in self.transitions.items()}}


def selection_key(metrics):
    return tuple(metrics[k] for k in ('macro_f1', 'segment_f1_50', 'edit'))


def verify_split(config, split):
    """No other split payloads are opened here (parent manifest is metadata only)."""
    root = io.ROOT / config['data_root']
    io.require(io.sha256_file(root / 'final_report.json') == config['data_report_sha256'], 'V3 parent pin')
    report = io.read(root / 'final_report.json')
    io.require(report['passed'] and report['research_usable'] and report['model_frozen'], 'V3 parent gate')
    base = root / 'sequence_cache' / split
    spec = report['splits'][split]
    io.require(spec['count'] == config['input']['split_counts'][split], 'split count')
    io.require(io.sha256_file(base / 'sequences.json') == spec['sequences_sha256'], 'sequence metadata')
    for name, wanted in spec['payload'].items():
        io.require(io.payload(base / name) == wanted, 'split payload ' + split + '/' + name)
    data = CompactSplit(root, split)
    last = 0
    for row in data.rows:
        sid = row['sequence_index']
        begin, end = row['output_start'], row['output_stop']
        io.require(begin == last and begin < end and np.all(data.ids[begin:end] == sid), 'sequence order')
        for name in ('ntu25.npy', 'labels.npy'):
            path = data.root / row['uid'] / name
            io.require(io.payload(path) == report['sequence_files'][row['uid']]['payload'][name], 'sequence payload ' + str(path))
        pose, labels = data.arrays(sid)
        io.require(pose.shape == (row['total_frames'], 25, 3) and labels.shape == (len(pose),), 'source shape')
        io.require(np.all((labels >= 0) & (labels < 16)), 'source class IDs')
        np.testing.assert_array_equal(data.starts[begin:end], np.arange(0, len(pose) - 63, 8))
        last = end
    io.require(last == data.count == spec['count'], 'total windows')
    return data
