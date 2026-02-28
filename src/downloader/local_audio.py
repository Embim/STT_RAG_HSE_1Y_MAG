import hashlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LocalAudioResult:
    audio_path: str
    file_id: str
    title: str


def extract_audio_from_video(video_path: str, output_dir: str | None = None) -> LocalAudioResult:
    """Extract audio from a local video file using FFmpeg.

    Args:
        video_path: Path to the local video file.
        output_dir: Directory to save the extracted audio.
                    Defaults to a temp directory.

    Returns:
        LocalAudioResult with path to the mp3 file, file_id and title.

    Raises:
        RuntimeError: If FFmpeg fails.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg не найден в PATH. "
            "Скачайте: https://ffmpeg.org/download.html и добавьте в PATH."
        )

    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    src = Path(video_path)
    title = src.stem
    file_id = hashlib.sha256(src.name.encode()).hexdigest()[:16]

    dst = Path(output_dir) / f"{file_id}.mp3"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        "-q:a", "2",
        str(dst),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{proc.stderr}")

    return LocalAudioResult(audio_path=str(dst), file_id=file_id, title=title)
