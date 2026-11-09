#!/usr/bin/env python3

import argparse
import csv
import re
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

MERGE_WINDOW = 15.0 # detections closer than this collapse into one event
VIDEO_EXTS = {".mov", ".mp4", ".m4v"}

# Loudness detection: sustained energy well above the game's own baseline.
LOUD_HOP = 0.4 # seconds per analysis hop
LOUD_SMOOTH = 2.0 # seconds of smoothing
LOUD_MIN_DURATION = 1.5 # seconds the spike must sustain


def extract_audio(video: Path, wav_path: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video),
         "-map", "0:a:0", "-ac", "1", "-ar", "16000", str(wav_path)],
        check=True,
    )


def transcribe_keywords(wav_path: Path, keyword: re.Pattern,
                        model_name: str) -> list[tuple[float, str]]:
    """Return (timestamp, transcribed_text) for every keyword hit."""
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(
        str(wav_path),
        language="en",
        word_timestamps=True,
        condition_on_previous_text=False,
        vad_filter=False,
    )
    hits = []
    for seg in segments:
        if not keyword.search(seg.text):
            continue
        for word in seg.words or []:
            if keyword.search(word.word):
                hits.append((word.start, seg.text.strip()))
                break
        else:
            hits.append((seg.start, seg.text.strip()))
    return hits


def loudness_events(wav_path: Path, percentile: float,
                    max_events: int) -> list[tuple[float, str]]:
    """Return (timestamp, description) for sustained loudness eruptions."""
    with wave.open(str(wav_path), "rb") as w:
        rate = w.getframerate()
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    samples = samples.astype(np.float32) / 32768.0

    hop = int(rate * LOUD_HOP)
    n_hops = len(samples) // hop
    rms = np.sqrt(np.mean(
        samples[: n_hops * hop].reshape(n_hops, hop) ** 2, axis=1))
    db = 20 * np.log10(rms + 1e-8)

    kernel = np.ones(max(1, int(LOUD_SMOOTH / LOUD_HOP)))
    smooth = np.convolve(db, kernel / len(kernel), mode="same")

    threshold = np.percentile(smooth, percentile)
    above = smooth >= threshold

    events = []
    start = None
    for i, flag in enumerate(above):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            duration = (i - start) * LOUD_HOP
            if duration >= LOUD_MIN_DURATION:
                peak_idx = start + int(np.argmax(smooth[start:i]))
                peak_db = smooth[peak_idx] - np.median(smooth)
                events.append((peak_idx * LOUD_HOP, peak_db, duration))
            start = None

    if start is not None:
        duration = (len(above) - start) * LOUD_HOP
        if duration >= LOUD_MIN_DURATION:
            peak_idx = start + int(np.argmax(smooth[start:]))
            events.append((peak_idx * LOUD_HOP,
                           smooth[peak_idx] - np.median(smooth), duration))

    events.sort(key=lambda e: e[1], reverse=True)
    return [(t, f"eruption +{db:.1f}dB over baseline for {dur:.1f}s")
            for t, db, dur in events[:max_events]]


def merge_events(events: list[tuple[float, str, str]]) -> list[tuple[float, str, str]]:
    """Collapse events within MERGE_WINDOW; keyword events win over loudness."""
    merged = []
    for t, kind, detail in sorted(events):
        if merged and t - merged[-1][0] <= MERGE_WINDOW:
            pt, pkind, pdetail = merged[-1]
            if pkind == "loud" and kind == "keyword":
                merged[-1] = (t, kind, detail)
            continue
        merged.append((t, kind, detail))
    return merged


def cut_clip(video: Path, event_time: float, out_path: Path,
             pre: float, post: float) -> None:
    start = max(0.0, event_time - pre)
    end = event_time + post
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.2f}", "-i", str(video),
         "-t", f"{end - start:.2f}", "-map", "0:v:0", "-map", "0:a:0",
         "-c", "copy", str(out_path)],
        check=True,
    )


def fmt_time(t: float) -> str:
    return f"{int(t) // 60}m{int(t) % 60:02d}s"


def process(video: Path, manifest: csv.writer, args) -> int:
    print(f"\n=== {video.name} ===")
    keyword = re.compile(args.keyword, re.IGNORECASE)
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        print("  extracting audio...")
        extract_audio(video, wav)

        print("  transcribing  ")
        keyword_hits = transcribe_keywords(wav, keyword, args.model)
        print(f"  {len(keyword_hits)} keyword hit(s)")

        loud_hits = ([] if args.no_loud else
                     loudness_events(wav, args.loud_percentile, args.max_loud))
        print(f"  {len(loud_hits)} loudness eruption(s)")

    events = ([(t, "keyword", text) for t, text in keyword_hits]
              + [(t, "loud", desc) for t, desc in loud_hits])
    events = merge_events(events)

    if not events:
        print("  no events detected in this video")
        return 0

    for i, (t, kind, detail) in enumerate(events, 1):
        suffix = "" if kind == "keyword" else "_loud"
        out = args.out / f"{video.stem}_clip{i}_t{fmt_time(t)}{suffix}.mp4"
        cut_clip(video, t, out, args.pre, args.post)
        manifest.writerow([video.name, i, f"{t:.1f}", fmt_time(t), kind, detail, out.name])
        print(f"  [{i}] {fmt_time(t)} ({kind}) {detail!r} -> {out.name}")
    return len(events)


def collect_videos(paths: list[Path]) -> list[Path]:
    videos = []
    for p in paths:
        if p.is_dir():
            videos.extend(sorted(
                f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTS))
        elif p.is_file():
            videos.append(p)
        else:
            raise SystemExit(f"not found: {p}")
    return videos


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Auto-clip game footage on 'clip that' yells and crowd eruptions.")
    ap.add_argument("paths", nargs="+", type=Path,
                    help="video files and/or folders of footage")
    ap.add_argument("--out", type=Path, default=Path("clips"),
                    help="output folder (default: clips/)")
    ap.add_argument("--pre", type=float, default=20.0,
                    help="seconds of footage before the trigger (default: 20)")
    ap.add_argument("--post", type=float, default=5.0,
                    help="seconds after the trigger (default: 5)")
    ap.add_argument("--keyword", default="clip",
                    help="regex to match in the transcript (default: clip)")
    ap.add_argument("--model", default="small",
                    help="faster-whisper model size (default: small)")
    ap.add_argument("--loud-percentile", type=float, default=97.5,
                    help="loudness threshold percentile (default: 97.5)")
    ap.add_argument("--max-loud", type=int, default=8,
                    help="max loudness clips per video (default: 8)")
    ap.add_argument("--no-loud", action="store_true",
                    help="disable loudness eruption detection")
    args = ap.parse_args()

    videos = collect_videos(args.paths)
    if not videos:
        raise SystemExit("no videos found")
    print(f"{len(videos)} video(s) to process")
    args.out.mkdir(parents=True, exist_ok=True)

    manifest_path = args.out / "manifest.csv"
    new_file = not manifest_path.exists()
    with open(manifest_path, "a", newline="") as f:
        manifest = csv.writer(f)
        if new_file:
            manifest.writerow(["source", "clip_num", "time_s", "time",
                               "trigger", "detail", "file"])
        total = sum(process(v, manifest, args) for v in videos)
    print(f"\nDone: {total} clip(s) in {args.out}")


if __name__ == "__main__":
    main()
