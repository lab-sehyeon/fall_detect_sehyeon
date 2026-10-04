"""Project-specific S0-F reconstruction; targets, causal head and event metrics."""
from collections import defaultdict
import re

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fall_pipeline.safer import s0a_core as a
from fall_pipeline.safer import v2_controls_reconstruction as io

PAIRS = ((10, 12), (6, 4), (12, 7))


def view_id(source_name):
    match = re.search(r'_d(\d+)$', source_name)
    io.require(match is not None, 'camera view must be explicit in source_name')
    return int(match.group(1))


def events(labels):
    y = np.asarray(labels)
    io.require(y.ndim == 1 and np.all((y >= 0) & (y < 16)), '16-class labels required')
    return {k: np.flatnonzero((y[:-1] == p) & (y[1:] == q)) + 1
            for k, (p, q) in enumerate(PAIRS, 1)}


def targets(labels, width=5):
    io.require(width > 0 and width % 2 == 1, 'positive odd target width required')
    y = np.zeros(len(labels), np.int64)
    distance = np.full(len(labels), width + 1, np.int64)
    ordered = sorted((int(t), k) for k, times in events(labels).items() for t in times)
    for time, k in ordered:
        frame = np.arange(max(0, time-width//2), min(len(y), time+width//2+1))
        chosen = frame[np.abs(frame-time) < distance[frame]]
        y[chosen] = k
        distance[chosen] = np.abs(chosen-time)
    return y


def dynamics(logits):
    x = np.asarray(logits)
    io.require(x.ndim == 2 and x.shape[1] == 16 and len(x) > 0 and np.isfinite(x).all(), 'posterior input')
    p = torch.softmax(torch.from_numpy(np.array(x, dtype=np.float32, copy=True)), dim=-1)
    delta = torch.zeros_like(p)
    delta[1:] = p[1:] - p[:-1]
    entropy = -(p*p.clamp_min(1e-12).log()).sum(-1, keepdim=True)
    top = p.topk(2, dim=-1).values
    result = torch.cat((p, delta, entropy, top[:, :1], top[:, :1]-top[:, 1:2]), -1).numpy()
    io.require(result.shape == (len(x), 35) and np.isfinite(result).all(), '35D feature finite')
    return result


class CausalBlock(nn.Module):
    def __init__(self, hidden, dilation, dropout):
        super().__init__()
        self.left = 2*dilation
        self.conv = nn.Conv1d(hidden, hidden, 3, dilation=dilation)
        self.mix = nn.Conv1d(hidden, hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.mix(F.relu(self.conv(F.pad(x, (self.left, 0))))))


class TransitionHead(nn.Module):
    def __init__(self, model):
        super().__init__()
        io.require(model['kernel'] == 3 and model['classes'] == 4, 'head contract')
        self.context = 2*sum(model['dilations'])
        self.project = nn.Conv1d(35, model['hidden'], 1)
        self.blocks = nn.Sequential(*(CausalBlock(model['hidden'], d, model['dropout'])
                                      for d in model['dilations']))
        self.output = nn.Conv1d(model['hidden'], 4, 1)
        for layer in self.modules():
            if isinstance(layer, nn.Conv1d):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        io.require(x.ndim == 3 and x.shape[-1] == 35, 'head input B,T,35')
        return self.output(self.blocks(self.project(x.transpose(1, 2)))).transpose(1, 2)


def predict(head, feature, device, chunk=4096):
    head.eval()
    out = np.empty((len(feature), 4), np.float32)
    with torch.no_grad():
        for lo in range(0, len(feature), chunk):
            hi = min(lo+chunk, len(feature)); begin = max(0, lo-head.context)
            x = torch.from_numpy(np.array(feature[begin:hi], copy=True)).unsqueeze(0).to(device)
            out[lo:hi] = head(x)[0, lo-begin:].cpu().numpy()
    io.require(np.isfinite(out).all(), 'head prediction finite')
    return out


def prefix_audit(head, device):
    """Identical shapes give exact future-perturbation audit; chunks use tolerance."""
    head.eval()
    x = torch.linspace(-1, 1, 257*35, device=device).reshape(1, 257, 35)
    with torch.no_grad():
        base = head(x)
        for cut in (1, 17, 128):
            changed = x.clone(); changed[:, cut:] += 11
            io.require(torch.equal(base[:, :cut], head(changed)[:, :cut]), 'TCN future leakage')
    chunked = predict(head, x[0].cpu().numpy(), device, 31)
    np.testing.assert_allclose(chunked, base[0].cpu().numpy(), atol=2e-5, rtol=2e-5)
    return {'same_shape_future_perturbation_exact': True, 'chunk_full_atol_rtol': 2e-5,
            'end_to_end_causal': False}


def window_groups(records, window=64, stride=8):
    windows, members = [], [[], [], [], []]
    for index, row in enumerate(records):
        y = row['target']
        starts = np.arange(0, len(y)-window+1, stride, dtype=np.int64)
        present = np.zeros((len(starts), 3), bool)
        for k in (1, 2, 3):
            prefix = np.r_[0, np.cumsum(y == k)]
            present[:, k-1] = prefix[starts+window] > prefix[starts]
        offset = len(windows)
        windows.extend((index, int(start)) for start in starts)
        members[0].extend((offset+np.flatnonzero(~present.any(1))).tolist())
        for k in (1, 2, 3):
            members[k].extend((offset+np.flatnonzero(present[:, k-1])).tolist())
    io.require(all(len(v) > 0 for v in members), 'all four sampling groups required')
    return np.asarray(windows, np.int32), [np.asarray(v, np.int64) for v in members]


def balanced_order(groups, count, seed):
    io.require(count > 0 and len(groups) == 4 and all(len(v) for v in groups), 'balanced sampler groups')
    rng = np.random.default_rng(seed)
    ids = np.arange(count) % 4; rng.shuffle(ids)
    order = np.empty(count, np.int64)
    for k, group in enumerate(groups):
        positions = np.flatnonzero(ids == k)
        order[positions] = rng.choice(group, len(positions), replace=True)
    return order


def weights(counts, cap):
    counts = np.asarray(counts, np.float64)
    io.require(counts.shape == (4,) and np.all(counts > 0) and cap > 0, 'positive class counts')
    return np.r_[1., np.minimum(cap, np.sqrt(counts[0]/counts[1:]))].astype(np.float32)


def onsets(prediction, k):
    p = np.asarray(prediction)
    io.require(p.ndim == 1 and np.all((p >= 0) & (p < 4)), 'event prediction labels')
    active = p == k
    return np.flatnonzero(active & np.r_[True, ~active[:-1]]) if len(p) else np.empty(0, np.int64)


def match(predicted, truth, tolerance):
    p, g = np.asarray(predicted), np.asarray(truth)
    io.require(tolerance >= 0 and np.all(np.diff(p) >= 0) and np.all(np.diff(g) >= 0), 'ordered event points')
    used = np.zeros(len(p), bool); delays = []
    for point in g:
        possible = np.flatnonzero((np.abs(p-point) <= tolerance) & ~used)
        if len(possible):
            i = possible[0]; used[i] = True; delays.append(int(p[i]-point))
    return [len(delays), len(p)-len(delays), len(g)-len(delays)]


class EventMetrics:
    def __init__(self, tolerance=12, fps=25):
        self.tolerance, self.fps = tolerance, fps
        self.counts = np.zeros((3, 3), np.int64)
        self.views = defaultdict(lambda: np.zeros((3, 3), np.int64))
        self.frames = self.sequences = 0

    def add(self, pred, truth, view):
        io.require(len(pred) == len(truth) and len(truth) > 0, 'event metric support')
        for k, points in events(truth).items():
            counts = match(onsets(pred, k), points, self.tolerance)
            self.counts[k-1] += counts; self.views[int(view)][k-1] += counts
        self.frames += len(truth); self.sequences += 1

    def result(self):
        def summarize(counts):
            per_class = [a.prf(*row) for row in counts]
            return {'per_class': per_class, 'macro_f1': float(np.mean([r['f1'] for r in per_class])),
                    'micro': a.prf(*counts.sum(0))}
        out = summarize(self.counts)
        views = {str(k): summarize(v) for k, v in sorted(self.views.items())}
        return {**out, 'views': views, 'worst_view_macro_f1': min(v['macro_f1'] for v in views.values()),
                'false_events_per_minute': float(self.counts[:, 1].sum()*self.fps*60/max(self.frames, 1)),
                'frames': self.frames, 'sequences': self.sequences}


def gate(metric, config):
    checks = {'macro_f1': metric['macro_f1'] >= config['macro_f1_min'],
              'worst_view_macro_f1': metric['worst_view_macro_f1'] >= config['worst_view_macro_f1_min'],
              'each_recall': all(r['recall'] >= config['each_recall_min'] for r in metric['per_class']),
              'false_events': metric['false_events_per_minute'] <= config['false_events_per_minute_max']}
    return {'checks': checks, 'feasible_for_integration': all(checks.values())}


def choose(history, names):
    best = None
    for row in history:
        for name in names:
            metric = row['metrics'][name]
            key = (metric['worst_view_macro_f1'], metric['macro_f1'])
            if best is None or key > (best['metrics']['worst_view_macro_f1'], best['metrics']['macro_f1']):
                best = {'epoch': row['epoch'], 'candidate': name, 'metrics': metric,
                        'checkpoint_sha256': row['checkpoint_sha256']}
    io.require(best is not None, 'no validation candidates')
    return best
