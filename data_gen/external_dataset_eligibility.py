#!/usr/bin/env python3
"""Model-blind external-dataset eligibility under the locked RGB contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class EndpointContract:
    fps: float = 25.0
    window_size: int = 64
    stride: int = 8
    early_tolerance_seconds: float = 0.5
    late_tolerance_seconds: float = 3.0


def prediction_frames(frame_count: int, contract: EndpointContract = EndpointContract()) -> list[int]:
    if frame_count < contract.window_size:
        return []
    frames = list(range(contract.window_size - 1, frame_count, contract.stride))
    final_start = frame_count - contract.window_size
    final_endpoint = final_start + contract.window_size - 1
    if frames[-1] != final_endpoint:
        frames.append(final_endpoint)
    return sorted(set(frames))


def assess_clip(
    frame_count: int,
    episodes: Iterable[Mapping[str, float]] = (),
    contract: EndpointContract = EndpointContract(),
) -> dict:
    endpoints = prediction_frames(frame_count, contract)
    times = [frame / contract.fps for frame in endpoints]
    assessed = []
    scorable = 0
    for episode in episodes:
        start = float(episode["fall_start"])
        end = float(episode.get("fall_end", start))
        lower = start - contract.early_tolerance_seconds
        upper = max(start, end) + contract.late_tolerance_seconds
        candidates = [index for index, time in enumerate(times) if lower <= time <= upper]
        item = dict(episode)
        item.update({
            "match_window": [lower, upper],
            "candidate_endpoints": len(candidates),
            "candidate_endpoint_indices": candidates,
            "event_scorable": bool(candidates),
        })
        assessed.append(item)
        scorable += bool(candidates)
    window_capable = bool(endpoints)
    negative_control = not assessed
    eligible = window_capable and (negative_control or scorable == len(assessed))
    reason = None
    if not window_capable:
        reason = "too_short_for_one_window"
    elif scorable != len(assessed):
        reason = "fall_event_has_no_candidate_endpoint"
    return {
        "canonical_frames": int(frame_count),
        "num_valid_windows": len(endpoints),
        "first_prediction_time": times[0] if times else None,
        "last_prediction_time": times[-1] if times else None,
        "window_capable": window_capable,
        "episodes": assessed,
        "scorable_episodes": scorable,
        "unscorable_episodes": len(assessed) - scorable,
        "negative_control_only": negative_control,
        "eligible": eligible,
        "unscorable_reason": reason,
        "contract": asdict(contract),
    }


__all__ = ["EndpointContract", "prediction_frames", "assess_clip"]
