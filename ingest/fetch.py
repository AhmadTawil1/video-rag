"""
ingest/fetch.py — get source video onto disk, with provenance.

The first step of the pipeline and the first half of the reproducibility story:
every file under data/raw/ carries a provenance.json saying where it came from,
under what licence, and exactly what command produced it.

    python -m ingest.fetch --url "https://..." --slug lecture01
    python -m ingest.fetch --url "https://..." --slug lecture01 --trim-duration 2400
    python -m ingest.fetch --config data/sources.json
    python -m ingest.fetch --config data/sources.json --force

Requires ffmpeg and ffprobe on PATH.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from yt_dlp import YoutubeDL

# 720p is deliberate: enough pixels to read slide text, roughly a quarter the
# disk and decode cost of 1080p, and CLIP resizes to 224x224 regardless.
FORMAT_SELECTOR = (
    "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b"
)

DEFAULT_RAW_DIR = Path("data/raw")


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #

@dataclass
class Provenance:
    """What data/README.md needs to be able to say about every source file."""
    slug: str
    url: str | None
    site_video_id: str | None
    title: str | None
    uploader: str | None
    upload_date: str | None
    licence: str | None          # None means the uploader declared nothing
    duration_s: float | None
    resolution: str | None
    fps: float | None
    format_selector: str
    fetched_at: str
    trim_offset_s: float = 0.0   # seconds into the ORIGINAL that this file starts
    trim_duration_s: float | None = None
    file: str | None = None


def write_provenance(prov: Provenance, dest: Path) -> None:
    (dest / "provenance.json").write_text(json.dumps(asdict(prov), indent=2))


# --------------------------------------------------------------------------- #
# external tools
# --------------------------------------------------------------------------- #

def require_ffmpeg() -> None:
    missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing:
        sys.exit(
            f"missing on PATH: {', '.join(missing)}\n"
            "  macOS:   brew install ffmpeg\n"
            "  Ubuntu:  sudo apt install ffmpeg\n"
            "  Windows: winget install Gyan.FFmpeg  (then reopen the terminal)"
        )


def probe(path: Path) -> dict:
    """Ground truth about the file on disk — not what you asked for, what you got."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries",
            "format=duration:stream=codec_type,codec_name,width,height,"
            "r_frame_rate,channels,sample_rate",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def summarise(path: Path) -> str:
    info = probe(path)
    dur = float(info.get("format", {}).get("duration", 0.0))
    video = next((s for s in info["streams"] if s["codec_type"] == "video"), {})
    audio = next((s for s in info["streams"] if s["codec_type"] == "audio"), {})
    num, _, den = (video.get("r_frame_rate") or "0/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    return (
        f"{path.name} · {dur / 60:6.1f} min · "
        f"{video.get('width')}x{video.get('height')} @ {fps:.2f} fps · "
        f"{video.get('codec_name')} / {audio.get('codec_name')} "
        f"{audio.get('sample_rate')} Hz {audio.get('channels')}ch · "
        f"{path.stat().st_size / 1e6:.0f} MB"
    )


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #

def fetch(
    url: str,
    slug: str,
    raw_dir: Path = DEFAULT_RAW_DIR,
    cookies_from_browser: str | None = None,
    force: bool = False,
) -> Path:
    """Download one video into raw_dir/slug/. Returns the mp4 path."""
    dest = raw_dir / slug
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / f"{slug}.mp4"

    if target.exists() and not force:
        print(f"[skip] {target} already exists — pass --force to refetch")
        return target

    opts = {
        "format": FORMAT_SELECTOR,
        "merge_output_format": "mp4",
        "outtmpl": str(dest / f"{slug}.%(ext)s"),
        "writeinfojson": True,      # full raw metadata, kept and never regenerated
        "writesubtitles": False,    # you want Whisper's word timings, not YouTube's
        "writeautomaticsub": False,
        "noplaylist": True,
        "retries": 5,
        "concurrent_fragment_downloads": 4,
        "overwrites": force,
    }
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)

    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    prov = Provenance(
        slug=slug,
        url=info.get("webpage_url"),
        site_video_id=info.get("id"),
        title=info.get("title"),
        uploader=info.get("uploader"),
        upload_date=info.get("upload_date"),
        licence=info.get("license"),
        duration_s=info.get("duration"),
        resolution=f'{info.get("width")}x{info.get("height")}',
        fps=info.get("fps"),
        format_selector=FORMAT_SELECTOR,
        fetched_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        file=target.name,
    )
    write_provenance(prov, dest)

    print(f"\n[ok]   {summarise(target)}")
    print(f"       licence: {prov.licence or 'NOT DECLARED — read the source page yourself'}")
    return target


# --------------------------------------------------------------------------- #
# trim
# --------------------------------------------------------------------------- #

def trim(src: Path, start_s: float, duration_s: float, force: bool = False) -> Path:
    """
    Cut [start_s, start_s + duration_s) out of src.

    Re-encodes on purpose. `-c copy` cuts at the nearest keyframe, which shifts
    every timestamp downstream by up to several seconds — and timestamps are the
    identifier this whole project is built on.
    """
    dst = src.with_name(f"{src.stem}_t{int(start_s)}-{int(start_s + duration_s)}.mp4")
    if dst.exists() and not force:
        print(f"[skip] {dst} already exists")
        return dst

    subprocess.run(
        [
            "ffmpeg", "-y", "-ss", str(start_s), "-i", str(src),
            "-t", str(duration_s),
            "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
            "-c:a", "aac", "-b:a", "128k",
            str(dst),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # The offset is not cosmetic: an answer at 12:30 in this file is at
    # 12:30 + start_s in the original. Record it or lose it.
    prov_path = src.parent / "provenance.json"
    if prov_path.exists():
        prov = json.loads(prov_path.read_text())
        prov["trim_offset_s"] = start_s
        prov["trim_duration_s"] = duration_s
        prov["file"] = dst.name
        prov_path.write_text(json.dumps(prov, indent=2))

    print(f"[ok]   {summarise(dst)}")
    print(f"       offset {start_s}s relative to the original — recorded in provenance.json")
    return dst


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch source video with provenance.")
    ap.add_argument("--url")
    ap.add_argument("--slug", help="short id, e.g. lecture01 / podcast01 / demo01")
    ap.add_argument("--config", type=Path,
                    help='JSON: [{"slug":..,"url":..,"trim_start":0,"trim_duration":2400}]')
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    ap.add_argument("--trim-start", type=float, default=0.0)
    ap.add_argument("--trim-duration", type=float, default=None)
    ap.add_argument("--cookies-from-browser", default=None,
                    help="chrome | firefox | edge — only if you hit a bot check")
    ap.add_argument("--force", action="store_true", help="redo work already done")
    args = ap.parse_args()

    require_ffmpeg()

    if args.config:
        entries = json.loads(args.config.read_text())
    elif args.url and args.slug:
        entries = [{
            "slug": args.slug,
            "url": args.url,
            "trim_start": args.trim_start,
            "trim_duration": args.trim_duration,
        }]
    else:
        ap.error("give either --config, or both --url and --slug")

    t0 = time.time()
    for e in entries:
        path = fetch(
            e["url"], e["slug"],
            raw_dir=args.raw_dir,
            cookies_from_browser=args.cookies_from_browser,
            force=args.force,
        )
        if e.get("trim_duration"):
            trim(path, e.get("trim_start", 0.0), e["trim_duration"], force=args.force)

    print(f"\n{len(entries)} source(s) ready in {args.raw_dir} · {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
