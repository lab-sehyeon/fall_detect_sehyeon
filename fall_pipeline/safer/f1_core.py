"""Document-grounded F1 recovery components, NOT the lost historical source."""
from __future__ import annotations

import copy
import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score

from data_gen.safer_legacy_v1_gendata import derive_four_class
from fall_pipeline.common.metrics import precision_recall_f1
from fall_pipeline.safer.train_f0b_safer_heads import classification_metrics


def require(value, message):
    if not value:
        raise RuntimeError(message)


def dense_backbone_features(model, batch):
    require(batch.ndim == 5 and tuple(batch.shape[1:]) == (3, 64, 25, 2), "DSTE input shape")
    require(bool(torch.isfinite(batch).all()), "nonfinite input")
    b = len(batch)
    jt = batch.permute(0, 2, 4, 3, 1).reshape(b, 64, 150)
    js = batch.permute(0, 4, 3, 2, 1).reshape(b, 50, 192)
    temporal, spatial = model.backbone(jt, js)
    require(tuple(temporal.shape) == (b, 64, 1024), "temporal shape")
    require(tuple(spatial.shape) == (b, 50, 1024), "spatial shape")
    valid = js.ne(0).any(dim=-1)
    weight = valid.to(spatial.dtype).unsqueeze(-1)
    # Recovered common.features.dense_dste_feature uses this same convention.
    # Finite all-zero legacy windows stay in the dataset; no silent exclusions.
    context = (spatial * weight).sum(1) / weight.sum(1).clamp_min(1.0)
    require(bool(torch.isfinite(temporal).all() and torch.isfinite(context).all()), "nonfinite features")
    return temporal, context


class ResidualTemporalAdapter(nn.Module):
    """New explicit recovery architecture; retained hidden API supports later probes."""
    def __init__(self, input_dim, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim, eps=1e-5)
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.block_norm = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 4)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def hidden(self, feature):
        base = self.gelu(self.projection(self.input_norm(feature)))
        delta = self.block_norm(base).transpose(1, 2)
        delta = self.conv2(self.dropout(self.gelu(self.conv1(delta)))).transpose(1, 2)
        return base + delta

    def forward(self, feature):
        return self.classifier(self.hidden(feature))


def make_candidates(config, device):
    models = {}
    for index, (representation, dim) in enumerate((("temporal_only", 1024), ("temporal_spatial", 2048))):
        torch.manual_seed(config["training"]["seed"] + index)
        base = ResidualTemporalAdapter(dim, config["model"]["hidden_dim"], config["model"]["dropout"])
        for loss in ("plain", "sqrt"):
            models[f"{representation}__{loss}"] = copy.deepcopy(base).to(device)
    return models


def candidate_input(name, temporal, context):
    if name.startswith("temporal_only__"):
        return temporal
    require(name.startswith("temporal_spatial__"), "unknown candidate")
    return torch.cat((temporal, context[:, None, :].expand(-1, temporal.shape[1], -1)), dim=-1)


def sqrt_weights(counts):
    counts = np.asarray(counts, dtype=np.float64)
    require(counts.shape == (4,) and np.all(counts > 0), "all four train classes required")
    weights = 1.0 / np.sqrt(counts)
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def selection_key(metrics):
    secondary = metrics["fall_vs_unstable_auprc"]
    return (float(metrics["macro_f1"]), float(secondary) if secondary is not None else -1.0,
            float(metrics["lying_down_f1"]))


def runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def segment_counts(predicted, truth, threshold=0.1):
    """Deterministic descending-IoU greedy matching; intervals are half-open."""
    pairs = []
    first = 0
    for pi, (ps, pe) in enumerate(predicted):
        while first < len(truth) and truth[first][1] <= ps:
            first += 1
        for gi in range(first, len(truth)):
            gs, ge = truth[gi]
            if gs >= pe:
                break
            intersection = min(pe, ge) - max(ps, gs)
            iou = intersection / (pe - ps + ge - gs - intersection)
            if iou >= threshold:
                pairs.append((-iou, pi, gi))
    used_p, used_g = set(), set()
    for _, pi, gi in sorted(pairs):
        if pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
    return len(used_p), len(predicted) - len(used_p), len(truth) - len(used_g)


class Timeline:
    """Coverage and exact labels for overlapping windows; no padded-frame labels."""
    def __init__(self, sequences, sequence_index, starts, dense_coarse):
        self.sequences = sequences
        self.starts = np.asarray(starts, dtype=np.int64)
        sequence_index = np.asarray(sequence_index, dtype=np.int64)
        require(self.starts.shape == sequence_index.shape, "timeline index shape")
        require(dense_coarse.shape == (len(self.starts), 64), "dense labels shape")
        offsets = np.r_[0, np.cumsum([row["total_frames"] for row in sequences])].astype(np.int64)
        self.offsets = offsets
        self.coarse = np.full(int(offsets[-1]), -1, dtype=np.int64)
        self.counts = np.zeros(len(self.coarse), dtype=np.int64)
        self.window_offsets = np.empty(len(self.starts), dtype=np.int64)
        cursor = 0
        for i, row in enumerate(sequences):
            begin, end, frames = row["output_start"], row["output_stop"], row["total_frames"]
            require(row["sequence_index"] == i and begin == cursor and begin <= end <= len(starts), "sequence offsets")
            cursor = end
            require(np.all(sequence_index[begin:end] == i), "sequence index order")
            starts_i = self.starts[begin:end]
            require(np.all(starts_i >= 0) and np.all(starts_i + 64 <= frames), "timeline bounds")
            require(np.all(np.diff(starts_i) > 0), "window ordering")
            labels = np.asarray(dense_coarse[begin:end])
            require(np.all((labels >= 0) & (labels < 16)), "coarse labels outside 0..15")
            indices = (starts_i[:, None] + np.arange(64)).ravel()
            minimum = np.full(frames, 16, dtype=np.int64)
            maximum = np.full(frames, -1, dtype=np.int64)
            np.minimum.at(minimum, indices, labels.ravel())
            np.maximum.at(maximum, indices, labels.ravel())
            covered = maximum >= 0
            require(np.array_equal(minimum[covered], maximum[covered]), "overlap label inconsistency")
            lo, hi = offsets[i:i + 2]
            self.coarse[lo:hi] = maximum
            self.counts[lo:hi] = np.bincount(indices, minlength=frames)
            self.window_offsets[begin:end] = starts_i + lo
        require(cursor == len(starts), "missing sequence windows")
        self.covered = self.counts > 0
        require(np.array_equal(self.covered, self.coarse >= 0), "coverage/label mismatch")
        self.labels = derive_four_class(self.coarse[self.covered])

    def empty(self):
        return np.zeros((len(self.coarse), 4), dtype=np.float64)

    def add(self, totals, begin, logits):
        require(logits.shape == (len(logits), 64, 4) and np.isfinite(logits).all(), "window logits")
        require(0 <= begin <= begin + len(logits) <= len(self.starts), "logit slice bounds")
        indices = (self.window_offsets[begin:begin + len(logits), None] + np.arange(64)).ravel()
        np.add.at(totals, indices, np.asarray(logits).reshape(-1, 4))

    def mean(self, totals):
        require(totals.shape == (len(self.coarse), 4) and np.isfinite(totals).all(), "timeline logits")
        return (totals[self.covered] / self.counts[self.covered, None]).astype(np.float32)

    def metrics(self, logits):
        metrics = classification_metrics(self.labels, logits)
        coarse = self.coarse[self.covered]
        subset = np.isin(coarse, [9, 10])
        binary = coarse[subset] == 10
        score = torch.softmax(torch.from_numpy(logits[subset]), dim=1)[:, 1].numpy()
        metrics["fall_vs_unstable_auprc"] = (float(average_precision_score(binary, score))
                                               if binary.any() and (~binary).any() else None)
        metrics["lying_down_f1"] = metrics["per_class"]["lying_down"]["f1"]
        metrics["total_frames"] = len(self.coarse)
        metrics["covered_frames"] = int(self.covered.sum())
        metrics["uncovered_frames"] = int((~self.covered).sum())
        predicted = np.full(len(self.coarse), -1, dtype=np.int64)
        predicted[self.covered] = logits.argmax(1)
        counts = np.zeros(3, dtype=np.int64)
        for lo, hi in zip(self.offsets[:-1], self.offsets[1:]):
            counts += segment_counts(runs(predicted[lo:hi] == 1), runs(self.coarse[lo:hi] == 10))
        metrics["reconstructed_event_iou_0p1"] = precision_recall_f1(*map(int, counts))
        return metrics
