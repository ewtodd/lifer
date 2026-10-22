"""Burst -> encounter grouping.

Frames are ordered by capture time; a new encounter starts when the gap to the
previous frame exceeds the threshold (default 5 s). Within an encounter the
frame with the sharpest bird crop represents it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class FrameInfo:
    name: str
    timestamp: datetime
    n_dets: int
    best_conf: float
    sharpness: float  # Laplacian variance of the best bird crop; -1 if none


def best_frame_key(name: str, sharpness: float) -> tuple[float, str]:
    """Sort key for an encounter's best frame (max wins); the review app uses
    it too when a removed frame forces a new pick."""
    # Sharpness is a Laplacian variance of the bird crop, and the crop
    # depends on the detector box, which is not bit-stable across re-runs
    # on CUDA. Two frames within ~1% of each other differ in the box, not
    # in focus, so quantise log10(sharpness) to two decimals (one step is
    # ~1% relative) and break ties by filename: the best-frame choice is
    # then stable across re-detections (task 25).
    q = float("-inf") if sharpness <= 0 else round(math.log10(sharpness) * 100)
    return (q, name)


def _best_frame_key(f: FrameInfo) -> tuple[float, str]:
    return best_frame_key(f.name, f.sharpness)


@dataclass
class Encounter:
    frames: list[FrameInfo] = field(default_factory=list)

    @property
    def start(self) -> datetime:
        return self.frames[0].timestamp

    @property
    def end(self) -> datetime:
        return self.frames[-1].timestamp

    @property
    def best_frame(self) -> FrameInfo:
        # sharpness of the bird crop, not the whole frame
        return max(self.frames, key=_best_frame_key)


def group_into_encounters(frames: list[FrameInfo], gap_seconds: float = 5.0) -> list[Encounter]:
    # filename as secondary key: capture times are second-resolution, so a
    # burst can have duplicate timestamps and the order must stay stable
    frames = sorted(frames, key=lambda f: (f.timestamp, f.name))
    encounters: list[Encounter] = []
    current: Encounter | None = None
    for f in frames:
        if current is not None:
            gap = (f.timestamp - current.frames[-1].timestamp).total_seconds()
            if gap > gap_seconds:
                current = None
        if current is None:
            current = Encounter(frames=[f])
            encounters.append(current)
        else:
            current.frames.append(f)
    return encounters
