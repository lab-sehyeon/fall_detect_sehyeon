"""Dependency-light classification and onset matching metrics."""

from __future__ import annotations

from typing import Iterable


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def match_onsets(
    predictions: Iterable[float],
    episodes: Iterable[tuple[float, float]],
    early: float = 0.5,
    late: float = 3.0,
) -> dict:
    remaining = sorted(float(value) for value in predictions)
    episodes = list(episodes)
    tp = 0
    for start, end in sorted(episodes):
        low, high = float(start) - early, max(float(start), float(end)) + late
        match = next((index for index, value in enumerate(remaining) if low <= value <= high), None)
        if match is not None:
            tp += 1
            remaining.pop(match)
    return precision_recall_f1(tp, len(remaining), len(episodes) - tp)
