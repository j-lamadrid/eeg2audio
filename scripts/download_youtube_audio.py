from __future__ import annotations

import argparse
import concurrent.futures
from pathlib import Path

import pandas as pd
import torchaudio
import yt_dlp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and trim YouTube audio referenced by a CSV.")
    parser.add_argument("csv", help="CSV with YouTube ID, start seconds, and end seconds columns.")
    parser.add_argument("output_dir")
    parser.add_argument("--timeout", type=int, default=20)
    return parser.parse_args()


def download_audio(video_id: str, output_dir: Path) -> bool:
    ydl_opts = {
        "verbose": False,
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio/best",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "outtmpl": str(output_dir / f"{video_id}.%(ext)s"),
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        return True
    except Exception as exc:
        print(f"https://www.youtube.com/watch?v={video_id} failed: {exc}")
        return False


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos = pd.read_csv(csv_path)
    failed_ids: list[str] = []

    for _, row in videos.iterrows():
        video_id = str(row["YouTube ID"])
        start_seconds = int(row["start seconds"]) // 1000
        end_seconds = int(row["end seconds"]) // 1000
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(download_audio, video_id, output_dir)
            try:
                ok = future.result(timeout=args.timeout)
            except concurrent.futures.TimeoutError:
                print(f"https://www.youtube.com/watch?v={video_id} timed out after {args.timeout} seconds")
                ok = False

        wav_path = output_dir / f"{video_id}.wav"
        if not ok or not wav_path.exists():
            failed_ids.append(video_id)
            continue

        waveform, sample_rate = torchaudio.load(str(wav_path))
        trimmed = waveform[:, sample_rate * start_seconds : sample_rate * end_seconds]
        torchaudio.save(str(wav_path), trimmed, sample_rate)

    output_csv = csv_path.with_name(csv_path.stem + "_download_report.csv")
    videos["download_failed"] = videos["YouTube ID"].astype(str).isin(failed_ids)
    videos.to_csv(output_csv, index=False)
    print(f"{len(failed_ids)} videos unavailable")
    print(f"wrote report to {output_csv}")


if __name__ == "__main__":
    main()

