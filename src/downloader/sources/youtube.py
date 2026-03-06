import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

import yt_dlp

logger = logging.getLogger(__name__)

AUDIO_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "audio"
VIDEO_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "video"


@dataclass
class DownloadResult:
    audio_path: str
    video_id: str
    title: str


def get_playlist_urls(url: str) -> List[str]:
    """Return list of video URLs from a playlist (or single video URL)."""
    probe_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "ignoreerrors": True,
    }
    with yt_dlp.YoutubeDL(probe_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    if info is None:
        return []

    if info.get("_type") == "playlist":
        entries = info.get("entries") or []
        return [
            e["url"] if e.get("url", "").startswith("http") else f"https://www.youtube.com/watch?v={e['id']}"
            for e in entries
            if e and e.get("id")
        ]
    return [url]


def download_audio(url: str, output_dir: str | None = None) -> DownloadResult:
    """Download audio track from YouTube URL.

    Args:
        url: YouTube video URL.
        output_dir: Directory to save the audio file.
                    Defaults to a temp directory.

    Returns:
        DownloadResult with path to the mp3 file, video_id and title.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(Path(output_dir) / "%(title)s.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "windowsfilenames": True,
    }

    logger.info("Downloading audio: %s", url)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        audio_path = str(Path(ydl.prepare_filename(info)).with_suffix(".mp3"))

    video_id = info["id"]
    title = info.get("title", video_id)

    logger.info("Downloaded: [%s] %s → %s", video_id, title, audio_path)
    return DownloadResult(audio_path=audio_path, video_id=video_id, title=title)


def download_video(url: str) -> str:
    """Download full video from YouTube URL to data/video/.

    Returns:
        Path to the saved mp4 file.
    """
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": str(VIDEO_DIR / "%(title)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "windowsfilenames": True,
    }

    logger.info("Downloading video: %s", url)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_path = str(Path(ydl.prepare_filename(info)).with_suffix(".mp4"))

    logger.info("Video saved: %s → %s", info.get("title"), video_path)
    return video_path
