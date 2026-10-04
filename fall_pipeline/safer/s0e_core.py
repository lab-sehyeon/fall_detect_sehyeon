"""Separate, preregistered hard-state reconstruction; never changes fall masks."""
from itertools import product
import numpy as np


def grid(config):
    keys = ('fall_settle', 'recovery_confirmation', 'recovery_hold', 'sit_confirmation')
    return [dict(zip(keys, values), name='fall%02d__rec%02d__hold%02d__sit%02d' % values)
            for values in product(*(config['grid'][key] for key in keys))]


def decode(prediction, spec, rules):
    x = np.asarray(prediction)
    if x.ndim != 1 or not np.all((x >= 0) & (x < 16)) or not np.equal(x, x.astype(np.int64)).all():
        raise ValueError('integer 16-state vector required')
    if any(spec[k] <= 0 for k in ('fall_settle', 'recovery_confirmation', 'recovery_hold')) or spec['sit_confirmation'] < 0:
        raise ValueError('invalid confirmation length')
    out = x.astype(np.uint8, copy=True)
    fall_active = grounded = False
    ground_count = rec_count = hold_left = sit_count = 0
    previous = -1
    for t, raw in enumerate(x):
        value = int(raw)
        if value == rules['fall']:
            fall_active = True; grounded = False
            ground_count = rec_count = hold_left = sit_count = 0
            previous = value
            continue
        if value == rules['sit'] and (previous == rules['sit_down'] or sit_count > 0):
            sit_count += 1
            if sit_count < spec['sit_confirmation']:
                out[t] = rules['sit_down']
        else:
            sit_count = 0
        if hold_left:
            out[t] = rules['recovery']; hold_left -= 1
            if not hold_left:
                fall_active = grounded = False; ground_count = rec_count = 0
        else:
            ground_evidence = (fall_active and value in rules['ground']) or value == rules['lying']
            ground_count = ground_count + 1 if ground_evidence else 0
            if ground_count >= spec['fall_settle']:
                grounded = True
            if grounded and value in rules['ground']:
                out[t] = rules['lying']; rec_count = 0
            elif grounded and value in rules['non_ground']:
                rec_count += 1
                if rec_count >= spec['recovery_confirmation']:
                    out[t] = rules['recovery']; hold_left = spec['recovery_hold'] - 1
                    if not hold_left:
                        fall_active = grounded = False; ground_count = rec_count = 0
            else:
                rec_count = 0
        previous = value
    if not np.array_equal(out == rules['fall'], x == rules['fall']):
        raise AssertionError('fall mask changed')
    return out


def gates(metric, baseline, latency, base_latency, mask_exact, rules):
    def recall(pair):
        a, b = latency[pair], base_latency[pair]
        return a['detected'] / a['events'] - b['detected'] / b['events'] if a['events'] and b['events'] else None
    fall_gain, rec_gain, sit_gain = (recall(k) for k in ('10->12', '12->7', '6->4'))
    p, q = latency['6->4'], base_latency['6->4']
    premature = q['premature']/q['events'] - p['premature']/p['events'] if p['events'] and q['events'] else None
    checks = {
        'macro': metric['macro_f1'] - baseline['macro_f1'] >= -rules['macro_drop_max'],
        'fall_mask_exact': bool(mask_exact),
        'switches': metric['switches_per_minute'] - baseline['switches_per_minute'] <= rules['switch_increase_max'],
        'lying_f1': metric['per_class'][12]['f1'] - baseline['per_class'][12]['f1'] >= -rules['lying_f1_drop_max'],
        'fall_successor': fall_gain is not None and fall_gain >= rules['fall_successor_gain_min'],
        'recovery_successor': rec_gain is not None and rec_gain >= rules['recovery_successor_gain_min'],
        'sit_premature': premature is not None and premature >= rules['sit_premature_reduction_min'],
        'sit_successor': sit_gain is not None and sit_gain >= -rules['sit_successor_drop_max'],
    }
    return {'eligible': all(checks.values()), 'passed_count': sum(checks.values()), 'checks': checks,
            'fall_successor_gain': fall_gain, 'recovery_successor_gain': rec_gain,
            'sit_successor_gain': sit_gain, 'sit_premature_reduction': premature}


def choose(rows):
    eligible = [r for r in rows if r['gate']['eligible']]
    if not rows:
        raise ValueError('no candidates')
    key = lambda r: ((0 if eligible else r['gate']['passed_count']),
                     *(r['metrics'][k] for k in ('macro_f1', 'segment_f1_50', 'edit')))
    best = max(eligible or rows, key=key)
    return {**best, 'adoption': bool(eligible), 'eligible_count': len(eligible)}
