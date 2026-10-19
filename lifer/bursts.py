"""Burst -> encounter grouping.

Frames are ordered by capture time; a new encounter starts when the gap to the
previous frame exceeds the threshold (default 5 s). Within an encounter the
frame with the sharpest bird crop represents it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class FrameInfo:
    name: str
    timestamp: datetime
    n_dets: int
    best_conf: float
    sharpness: float  # Laplacian variance of the best bird crop; -1 if none


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
        return max(self.frames, key=lambda f: (f.sharpness, f.best_conf))


def group_into_encounters(frames: list[FrameInfo], gap_seconds: float = 5.0) -> list[Encounter]:
    frames = sorted(frames, key=lambda f: f.timestamp)
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
