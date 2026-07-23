# ingest/keyframes.py
import json
import sys
from pathlib import Path

import cv2
import imagehash
from PIL import Image
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector


# ---------- 1. scenes (cached — decodes the video, do this once) ----------

def detect_scenes(
    video_path: str,
    threshold: float = 27.0,
    cache: Path | None = None,
) -> list[tuple[float, float]]:
    if cache and cache.exists():
        return [tuple(p) for p in json.loads(cache.read_text())]

    video = open_video(video_path)
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold))
    manager.detect_scenes(video, show_progress=True)

    scenes = [(s.seconds, e.seconds) for s, e in manager.get_scene_list()]

    if cache:
        cache.write_text(json.dumps(scenes))
    return scenes


# ---------- 2. which timestamps to sample ----------

def mid_scene_timestamps(scenes: list[tuple[float, float]]) -> list[float]:
    """Midpoint of each scene — never the cut frame, which is motion-blurred mush."""
    return [(s + e) / 2 for s, e in scenes]


def interval_timestamps(duration: float, sample_fps: float = 0.2) -> list[float]:
    """Fixed-interval fallback. 0.2 fps = one frame every 5s.

    Scene detection undersamples a slide lecture: bullet builds and animated
    reveals don't clear ContentDetector's threshold, so whole slides never get
    a frame. This catches them.
    """
    step = 1.0 / sample_fps
    out, t = [], 0.0
    while t < duration:
        out.append(t)
        t += step
    return out


# ---------- 3. extract (the only other thing that touches the video) ----------

def video_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(str(video_path))
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frames / fps if fps else 0.0


def ts_filename(seconds: float) -> str:
    """93.5 -> '00-01-33'. Display convenience; the exact float lives in the manifest."""
    total = int(seconds)
    return f"{total // 3600:02d}-{(total % 3600) // 60:02d}-{total % 60:02d}"


def extract_frames(
    video_path: str,
    timestamps: list[float],
    out_dir: Path,
) -> list[tuple[float, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dupes = out_dir / "_dupes"
    cap = cv2.VideoCapture(str(video_path))

    saved: list[tuple[float, Path]] = []
    for ts in sorted(timestamps):
        name = f"{ts_filename(ts)}.jpg"
        path = out_dir / name
        # already extracted — either kept, or already judged a dupe on a past run
        if path.exists() or (dupes / name).exists():
            continue

        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ok, frame = cap.read()
        if not ok:
            print(f"  skipped {ts:.2f}s — unreadable")
            continue

        cv2.imwrite(str(path), frame)
        saved.append((ts, path))

    cap.release()
    return saved


# ---------- 4. dedupe (disk only — retune this freely, costs seconds) ----------

def dedupe_frames(frame_dir: Path, hamming: int = 6) -> tuple[list[Path], list[Path]]:
    """Sequential perceptual-hash dedupe in time order.

    A static slide held for 60s yields 12 near-identical interval frames.
    Keeps the earliest of each run, returns (kept, dropped).
    """
    paths = sorted(frame_dir.glob("*.jpg"))
    kept: list[Path] = []
    dropped: list[Path] = []
    last_hash = None

    for p in paths:
        h = imagehash.phash(Image.open(p))
        if last_hash is not None and (h - last_hash) <= hamming:
            dropped.append(p)
            continue
        kept.append(p)
        last_hash = h

    return kept, dropped


def apply_dedupe(frame_dir: Path, dropped: list[Path]) -> None:
    """Move, don't delete. A wrong threshold is recoverable; a deleted slide isn't."""
    dupes = frame_dir / "_dupes"
    dupes.mkdir(exist_ok=True)
    for p in dropped:
        p.rename(dupes / p.name)


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "lecture01"
    BASE = Path("data/raw") / name
    VIDEO = str(BASE / f"{name}.mp4")
    FRAMES = BASE / "frames"

    scenes = detect_scenes(VIDEO, cache=BASE / "scenes.json")
    duration = video_duration(VIDEO)
    print(f"{name}: {len(scenes)} scenes · {duration/60:.1f} min")

    mids = mid_scene_timestamps(scenes)
    intervals = interval_timestamps(duration, sample_fps=0.2)
    extract_frames(VIDEO, mids + intervals, FRAMES)

    total = len(list(FRAMES.glob("*.jpg")))
    kept, dropped = dedupe_frames(FRAMES, hamming=4)
    print(f"{len(mids)} scene + {len(intervals)} interval -> {total} on disk "
          f"-> {len(kept)} kept, {len(dropped)} near-dupes")

    apply_dedupe(FRAMES, dropped)