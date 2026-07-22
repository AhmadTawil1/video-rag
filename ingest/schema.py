from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel, model_validator
from typing import Literal

class Word(BaseModel):
    start: float
    end: float
    text: str                        # whisper keeps a leading space (" word") — don't strip it
    probability: float | None = None # whisper's per-word confidence — NOT the LLM logprob from Day 1

    @model_validator(mode="after")
    def _ordered(self):
        if self.end < self.start:
            raise ValueError(f"word end {self.end} < start {self.start}")
        return self

class TranscriptSegment(BaseModel):
    start: float                     # segment bounds; can be slightly wider than the word bounds
    end: float
    text: str
    words: list[Word]                # required — you run whisper with word_timestamps=True

    @model_validator(mode="after")
    def _ordered(self):
        if self.end < self.start:
            raise ValueError(f"segment end {self.end} < start {self.start}")
        return self

# Option A — minimal: scene bounds just go None for interval frames
class Keyframe(BaseModel):
    ts: float
    path: Path
    method: Literal["scene", "interval"]
    scene_start: float | None = None
    scene_end: float | None = None


class VideoManifest(BaseModel):
    video_id: str
    source: str            # original path or URL
    duration: float
    segments: list[TranscriptSegment]
    keyframes: list[Keyframe]

    @model_validator(mode="after")
    def _within_duration(self):
        for kf in self.keyframes:
            if kf.ts > self.duration:
                raise ValueError(f"keyframe ts {kf.ts} beyond duration {self.duration}")
        return self