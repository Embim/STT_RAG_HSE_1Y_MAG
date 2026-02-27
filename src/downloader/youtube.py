import tempfile
from dataclasses import dataclass
from pathlib import Path

import yt_dlp


@dataclass
class DownloadResult:
    audio_path: str
    video_id: str
    title: str


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
        "outtmpl": str(Path(output_dir) / "%(id)s.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    video_id = info["id"]
    title = info.get("title", video_id)
    audio_path = str(Path(output_dir) / f"{video_id}.mp3")

    return DownloadResult(audio_path=audio_path, video_id=video_id, title=title)
