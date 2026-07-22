import subprocess
from pathlib import Path

from faster_whisper import WhisperModel
from ingest.schema import Word, TranscriptSegment

import json



def extract_audio(video_path: Path, out_wav: Path) -> Path:
    """Job 1: pull 16 kHz mono audio out of the video with ffmpeg."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path),
         "-ar", "16000",   # 16 kHz — the quality whisper expects
         "-ac", "1",       # mono — one channel, it's speech not music
         str(out_wav)],
        check=True,
    )
    return out_wav


def transcribe(audio_path: Path, model_size: str = "large-v3") -> list[TranscriptSegment]:
    """Job 2: whisper listens, then we repackage its output into our schema."""
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    segments, info = model.transcribe(str(audio_path), word_timestamps=True)

    result: list[TranscriptSegment] = []
    for seg in segments:
        words = [
            Word(start=w.start, end=w.end, text=w.word, probability=w.probability)
            for w in seg.words
        ]
        result.append(
            TranscriptSegment(start=seg.start, end=seg.end, text=seg.text, words=words)
        )
    return result


def transcribe_cached(audio_path: Path, cache_path: Path) -> list[TranscriptSegment]:
    # 1. Does the saved file already exist?
    if cache_path.exists():
        # yes → load it, skip whisper entirely
        raw = json.loads(cache_path.read_text())
        return [TranscriptSegment.model_validate(seg) for seg in raw]

    # 2. No → do the slow work once
    segments = transcribe(audio_path)          # the function from before

    # 3. Save it so we never do that again
    data = [seg.model_dump(mode="json") for seg in segments]
    cache_path.write_text(json.dumps(data, indent=2))

    return segments