"""Causal G2/S0-G0 event-state decoding recovered from the project record."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Iterable

import numpy as np


D0_LEGACY = "D0_LEGACY"
D1_DECOUPLED = "D1_DECOUPLED"


class DisplayState(IntEnum):
    NORMAL = 0
    FALLING = 1
    POST_FALL = 2
    RECOVERING = 3
    RECOVERED = 4


@dataclass(frozen=True)
class DecoderConfig:
    fps: float = 25.0
    background_endpoints_to_close: int = 3
    timeout_seconds: float = 25.0
    recovered_display_seconds: float = 2.0
    g2_fall_class: int = 1
    s0_background_class: int = 0
    s0_falling_class: int = 1
    s0_post_fall_class: int = 2
    s0_recovering_class: int = 3


def _class_ids(value: np.ndarray | Iterable[int]) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2:
        return array.argmax(axis=-1).astype(np.int64)
    if array.ndim != 1:
        raise ValueError("decoder input must be class ids or [endpoint,class] scores")
    return array.astype(np.int64, copy=False)


def run_length_encode(values: Iterable[int]) -> list[dict]:
    values = list(map(int, values))
    if not values:
        return []
    output = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            output.append({"start": start, "end": index - 1, "value": values[start]})
            start = index
    return output


def decode_recovery_endpoints(
    endpoint_frames: Iterable[int],
    g2: np.ndarray | Iterable[int],
    s0g0: np.ndarray | Iterable[int],
    *,
    frame_count: int | None = None,
    fall_alert_mode: str = D0_LEGACY,
    config: DecoderConfig = DecoderConfig(),
) -> dict:
    """Decode endpoint decisions without looking at future frames.

    D0 emits a fall alert only when the recovery episode latch opens.  D1 keeps
    that recovery state machine byte-for-byte independent and emits alerts on
    rising edges of the existing G2 argmax-fall decision.
    """
    if fall_alert_mode not in (D0_LEGACY, D1_DECOUPLED):
        raise ValueError("fall_alert_mode must be D0_LEGACY or D1_DECOUPLED")
    frames = np.asarray(list(endpoint_frames), dtype=np.int64)
    g2_prediction = _class_ids(g2)
    raw_s0 = _class_ids(s0g0)
    if not (len(frames) == len(g2_prediction) == len(raw_s0)):
        raise ValueError("endpoint frames, G2 and S0-G0 lengths differ")
    if len(frames) and (np.any(np.diff(frames) <= 0) or frames[0] < 0):
        raise ValueError("endpoint frames must be strictly increasing and non-negative")

    episode_open = False
    recovery_seen = False
    opened_frame = -1
    background_streak = 0
    previous_g2_fall = False
    recovered_until = -1
    episode_opens = normal_closes = timeout_closes = 0
    endpoint_state = np.full(len(frames), int(DisplayState.NORMAL), dtype=np.int8)
    episode_mask = np.zeros(len(frames), dtype=bool)
    events: list[dict] = []

    for index, frame in enumerate(frames.tolist()):
        g2_fall = int(g2_prediction[index]) == config.g2_fall_class
        rising_edge = g2_fall and not previous_g2_fall
        opened_here = False
        if g2_fall and not episode_open:
            episode_open = True
            recovery_seen = False
            opened_frame = frame
            background_streak = 0
            opened_here = True
            episode_opens += 1

        alert = opened_here if fall_alert_mode == D0_LEGACY else rising_edge
        if alert:
            events.append({"type": "fall_trigger", "frame": frame, "time": frame / config.fps})

        if episode_open:
            episode_mask[index] = True
            state = int(raw_s0[index])
            if opened_here or state == config.s0_falling_class:
                endpoint_state[index] = int(DisplayState.FALLING)
                background_streak = 0
            elif state == config.s0_post_fall_class:
                endpoint_state[index] = int(DisplayState.POST_FALL)
                background_streak = 0
            elif state == config.s0_recovering_class:
                endpoint_state[index] = int(DisplayState.RECOVERING)
                background_streak = 0
                if not recovery_seen:
                    recovery_seen = True
                    events.append({"type": "recovery_start", "frame": frame, "time": frame / config.fps})
            else:
                endpoint_state[index] = int(DisplayState.POST_FALL)
                background_streak += 1

            timeout = (frame - opened_frame) / config.fps >= config.timeout_seconds
            normal_close = recovery_seen and background_streak >= config.background_endpoints_to_close
            if normal_close or timeout:
                if normal_close:
                    normal_closes += 1
                    events.append({"type": "recovery_complete", "frame": frame, "time": frame / config.fps})
                    recovered_until = frame + round(config.recovered_display_seconds * config.fps)
                    endpoint_state[index] = int(DisplayState.RECOVERED)
                else:
                    timeout_closes += 1
                    events.append({"type": "timeout_close", "frame": frame, "time": frame / config.fps})
                episode_open = False
                recovery_seen = False
                background_streak = 0
        elif frame <= recovered_until:
            endpoint_state[index] = int(DisplayState.RECOVERED)

        previous_g2_fall = g2_fall

    if frame_count is None:
        frame_count = int(frames[-1] + 1) if len(frames) else 0
    frame_state = np.full(frame_count, int(DisplayState.NORMAL), dtype=np.int8)
    frame_episode_open = np.zeros(frame_count, dtype=bool)
    previous = 0
    for index, endpoint in enumerate(frames.tolist()):
        stop = min(endpoint + 1, frame_count)
        frame_state[previous:stop] = endpoint_state[index]
        frame_episode_open[previous:stop] = episode_mask[index]
        previous = stop
    if previous < frame_count and len(frames):
        frame_state[previous:] = endpoint_state[-1]
        frame_episode_open[previous:] = episode_mask[-1]

    return {
        "fall_alert_mode": fall_alert_mode,
        "contract": asdict(config),
        "endpoint_frames": frames,
        "endpoint_state": endpoint_state,
        "episode_open": episode_mask,
        "frame_state": frame_state,
        "frame_episode_open": frame_episode_open,
        "display_state": frame_state.copy(),
        "raw_s0g0_state": raw_s0,
        "g2_prediction": g2_prediction,
        "events": events,
        "diagnostics": {
            "episode_opens": episode_opens,
            "recovery_seen": sum(event["type"] == "recovery_start" for event in events),
            "normal_closes": normal_closes,
            "timeout_closes": timeout_closes,
            "fall_alerts": sum(event["type"] == "fall_trigger" for event in events),
        },
        "fall_detected": any(event["type"] == "fall_trigger" for event in events),
    }


__all__ = [
    "D0_LEGACY", "D1_DECOUPLED", "DisplayState", "DecoderConfig",
    "decode_recovery_endpoints", "run_length_encode",
]
