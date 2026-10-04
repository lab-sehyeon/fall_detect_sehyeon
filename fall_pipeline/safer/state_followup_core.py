"""Document-grounded S0-B/C/GT0 definitions; no historical exactness claim."""
import copy
import numpy as np
import torch
from torch import nn
from fall_pipeline.safer import s0a_core as core


class DilatedBlock(nn.Module):
    def __init__(self, hidden, dilation, dropout):
        super().__init__()
        self.dilated = nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation)
        self.pointwise = nn.Conv1d(hidden, hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.pointwise(torch.relu(self.dilated(x))))


class ResidualStateHead(nn.Module):
    def __init__(self, base_state, spec):
        super().__init__()
        self.base = nn.Linear(2048, 16)
        self.base.load_state_dict(base_state, strict=True)
        self.base.requires_grad_(False)
        self.input = nn.Conv1d(2048, spec['hidden'], 1)
        self.blocks = nn.Sequential(*(DilatedBlock(spec['hidden'], d, spec['dropout']) for d in spec['dilations']))
        self.output = nn.Conv1d(spec['hidden'], 16, 1)
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight); nn.init.zeros_(module.bias)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def correction(self, x):
        return self.output(self.blocks(self.input(x.transpose(1,2)))).transpose(1,2)

    def forward(self, x):
        return self.base(x) + self.correction(x)


def adoption_b(candidate, baseline, rules):
    delta = {k:candidate[k]-baseline[k] for k in ('macro_f1','segment_f1_50','edit')}
    recall_delta = candidate['per_class'][10]['recall'] - baseline['per_class'][10]['recall']
    reduction = 1-candidate['switches_per_minute']/baseline['switches_per_minute'] if baseline['switches_per_minute'] else 0.
    gates = {'macro':delta['macro_f1'] >= -rules['macro_drop_max'],
             'segment':delta['segment_f1_50'] >= rules['segment_gain_min'],
             'edit':delta['edit'] >= rules['edit_gain_min'],
             'switches':reduction >= rules['switch_reduction_min'],
             'fall_recall':recall_delta >= -rules['fall_recall_drop_max']}
    return {'adoption':all(gates.values()), 'gates':gates, 'delta':delta,
            'fall_recall_delta':recall_delta, 'switch_reduction':reduction}


def decoder_grid(config):
    result = [{'name':'raw', 'alpha':None, 'duration':1, 'margin':0.}]
    grid = config['grid']
    for alpha in grid['ema_alpha']:
        result.append({'name':f'ema_{alpha:g}', 'alpha':alpha, 'duration':1, 'margin':0.})
    for alpha in grid['ema_alpha']:
        for duration in grid['minimum_duration']:
            for margin in grid['margin']:
                result.append({'name':f'ema_{alpha:g}_d{duration}_m{margin:g}', 'alpha':alpha, 'duration':duration, 'margin':margin})
    return result


def decode(logits, spec):
    logits = np.asarray(logits)
    if logits.ndim != 2 or logits.shape[1] != 16 or not np.isfinite(logits).all():
        raise ValueError('decoder needs finite Tx16 logits')
    if spec['alpha'] is None:
        return logits.argmax(1).astype(np.uint8)
    if not len(logits):
        return np.zeros(0, np.uint8)
    x = logits.astype(np.float64); x -= x.max(1, keepdims=True)
    p = np.exp(x); p /= p.sum(1, keepdims=True)
    output = np.empty(len(p), np.uint8)
    state = p[0].copy(); current = int(state.argmax()); pending, count = -1, 0
    output[0] = current
    for i in range(1,len(p)):
        state = spec['alpha']*state+(1-spec['alpha'])*p[i]
        proposed = int(state.argmax())
        if proposed == current or state[proposed]-state[current] < spec['margin']:
            pending, count = -1, 0
        else:
            count = count+1 if proposed == pending else 1
            pending = proposed
            if count >= spec['duration']:
                current = proposed; pending, count = -1, 0
        output[i] = current
    return output


def adoption_c(candidate, baseline, rules):
    reduction = 1-candidate['switches_per_minute']/baseline['switches_per_minute'] if baseline['switches_per_minute'] else 0.
    gates = {'macro':candidate['macro_f1']-baseline['macro_f1'] >= -rules['macro_drop_max'],
             'fall_recall':candidate['per_class'][10]['recall']-baseline['per_class'][10]['recall'] >= -rules['fall_recall_drop_max'],
             'switches':reduction >= rules['switch_reduction_min']}
    return {'eligible':all(gates.values()), 'gates':gates, 'switch_reduction':reduction}


def choose_decoder(rows):
    eligible = [r for r in rows if r['gate']['eligible']]
    pool = eligible or rows
    best = max(pool, key=lambda r:tuple(r['metrics'][k] for k in ('segment_f1_50','edit','macro_f1')))
    return {**copy.deepcopy(best), 'adoption':bool(eligible), 'eligible_count':len(eligible)}


def contextual_targets(labels):
    y = np.asarray(labels, np.int64)
    if y.ndim != 1 or not np.all((y>=0)&(y<16)):
        raise ValueError('coarse labels must be 1D and inside0..15')
    target = np.zeros(len(y), np.int8)
    values, starts, ends = core.segments(y)
    falls = np.flatnonzero(values == 10); episodes = []
    for index, seg in enumerate(falls):
        start, end = int(starts[seg]), int(ends[seg])
        limit = int(starts[falls[index+1]]) if index+1<len(falls) else len(y)
        getting = np.flatnonzero((values==7)&(starts>=end)&(starts<limit))
        target[start:end] = 1
        rs = re = None
        if len(getting):
            rs, re = int(starts[getting[0]]), int(ends[getting[0]])
            target[end:rs] = 2; target[rs:re] = 3
        else:
            target[end:limit] = -1
        episodes.append({'fall_start':start,'fall_end':end,'episode_end':limit,'recovery_start':rs,'recovery_end':re})
    return target, episodes


def reference_contextual(labels):
    """Independent framewise state machine for the label audit."""
    y = np.asarray(labels); out = np.zeros(len(y),np.int8)
    active = False; pending = []; recovering = False
    for i, label in enumerate(y):
        if label == 10:
            if i==0 or y[i-1]!=10:
                for frame in pending: out[frame] = -1
                pending = []; active = True
            out[i] = 1; recovering = False
        elif active and label == 7:
            for frame in pending: out[frame] = 2
            pending = []; out[i] = 3; recovering = True
        elif recovering:
            recovering = False; active = False
        elif active:
            pending.append(i)
    for frame in pending: out[frame] = -1
    return out


def latency_events(pred, truth, pairs, spec):
    pred, truth = np.asarray(pred), np.asarray(truth)
    output = []
    for before, after in pairs:
        boundaries = np.flatnonzero((truth[:-1]==before)&(truth[1:]==after))+1
        stable = (np.convolve((pred==after).astype(np.int8),np.ones(spec['successor_run_frames'],np.int8),mode='valid') == spec['successor_run_frames']) if len(pred)>=spec['successor_run_frames'] else np.zeros(0,bool)
        onsets = np.flatnonzero(stable)
        for boundary in boundaries:
            candidates = onsets[(onsets>=boundary)&(onsets<=boundary+spec['after_boundary_frames'])]
            previous = onsets[(onsets>=max(0,boundary-spec['before_boundary_frames'])) & (onsets+spec['successor_run_frames']<=boundary)]
            interval = pred[boundary:min(len(pred),boundary+spec['after_boundary_frames']+1)]
            output.append({'pair':f'{before}->{after}','boundary':int(boundary),
                           'detected':bool(len(candidates)), 'delay_frames':int(candidates[0]-boundary) if len(candidates) else None,
                           'premature':bool(len(previous)), 'boundary_correct':bool(pred[boundary]==after),
                           'miss_dominant_class':int(np.bincount(interval,minlength=16).argmax()) if not len(candidates) else None})
    return output
