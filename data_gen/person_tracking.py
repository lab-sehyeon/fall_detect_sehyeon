"""Project-specific deterministic dominant-person association."""

from __future__ import annotations

from typing import Iterable, Mapping


def box_iou(left: Iterable[float], right: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, left)
    bx1, by1, bx2, by2 = map(float, right)
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = width * height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def select_person(
    candidates: Iterable[Mapping],
    previous_box: Iterable[float] | None = None,
    iou_weight: float = 0.7,
    confidence_weight: float = 0.3,
) -> Mapping | None:
    candidates = list(candidates)
    if not candidates:
        return None
    if previous_box is None:
        return max(candidates, key=lambda item: float(item["confidence"]))
    return max(
        candidates,
        key=lambda item: iou_weight * box_iou(previous_box, item["bbox_xyxy"])
        + confidence_weight * float(item["confidence"]),
    )


def dominant_track(frames: Iterable[Iterable[Mapping]]) -> list[Mapping | None]:
    output = []
    previous = None
    for candidates in frames:
        selected = select_person(candidates, previous)
        output.append(selected)
        if selected is not None:
            previous = selected["bbox_xyxy"]
    return output
