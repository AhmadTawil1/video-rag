import pytest
from pydantic import ValidationError

from ingest.schema import Word, TranscriptSegment, Keyframe, VideoManifest


def test_manifest_roundtrips():
    m = VideoManifest(
        video_id="podcast01",
        source="data/raw/podcast01/podcast01.mp4",
        duration=2320.0,
        segments=[
            TranscriptSegment(
                start=40.0, end=44.5, text=" so as you can see here",
                words=[
                    Word(start=40.3, end=40.6, text=" so", probability=0.98),
                    Word(start=40.6, end=41.0, text=" as", probability=0.95),
                ],
            )
        ],
        keyframes=[
            Keyframe(ts=285.0, path="frames/00-04-45.jpg",
                     method="scene", scene_start=240.0, scene_end=330.0),
            Keyframe(ts=600.0, path="frames/00-10-00.jpg",
                     method="interval"),
        ],
    )

    dumped = m.model_dump_json()
    restored = VideoManifest.model_validate_json(dumped)

    assert restored == m


def test_word_end_before_start_rejected():
    with pytest.raises(ValidationError):
        Word(start=5.0, end=3.0, text=" bad")


def test_keyframe_past_duration_rejected():
    with pytest.raises(ValidationError):
        VideoManifest(
            video_id="x", source="x", duration=100.0,
            segments=[],
            keyframes=[Keyframe(ts=5000.0, path="f.jpg", method="interval")],
        )