"""
YouTube source plugin.

Скачивает видео с YouTube и конвертирует в аудио.
"""

from typing import Iterator, Dict, Any, List, Optional
from pathlib import Path
import logging
import time

import yt_dlp
from tqdm import tqdm

from ..core.interfaces import BaseSource
from ..core.models import ContentItem, ContentType, SourceType, ProcessingStage
from ..core.registry import register_source

logger = logging.getLogger(__name__)


@register_source
class YouTubeSource(BaseSource):
    """
    Источник данных YouTube.

    Скачивает видео с YouTube каналов и отдельных URL,
    конвертирует в аудио формат для последующей транскрипции.
    """

    source_name = "youtube"
    supported_content_types = [ContentType.AUDIO, ContentType.VIDEO]

    def __init__(
        self,
        output_dir: str = "E:/video_pipeline/audio",
        audio_format: str = "wav",
        max_retries: int = 3,
        retry_delay: int = 5,
    ):
        """
        Инициализация источника YouTube.

        Args:
            output_dir: Директория для сохранения аудио файлов
            audio_format: Формат аудио (wav, mp3 и т.д.)
            max_retries: Максимальное количество попыток при ошибке
            retry_delay: Задержка между попытками в секундах
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audio_format = audio_format
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.total_count = 0  # Для отображения в progress bar

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Проверяет валидность конфигурации."""
        return "urls" in config or "url" in config

    def fetch(
        self,
        urls: List[str] = None,
        skip_ids: List[str] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> Iterator[ContentItem]:
        """
        Получает видео с YouTube.

        Args:
            urls: Список URL (видео, каналы, плейлисты)
            skip_ids: Список video_id для пропуска (уже обработанные)
            show_progress: Показывать прогресс

        Yields:
            ContentItem объекты со скачанным аудио
        """
        urls = urls or kwargs.get("url", [])
        if isinstance(urls, str):
            urls = [urls]

        skip_ids = set(skip_ids or [])

        # Сброс счетчика
        self.total_count = 0

        for url in urls:
            # Определяем тип URL
            if "/videos" in url or "/@" in url or "/playlist" in url:
                # Канал или плейлист - сначала быстро получаем список ID
                logger.info(f"Fetching video list from {url}")
                all_videos = list(self._get_video_urls(url))
                total_videos = len(all_videos)

                # Фильтруем уже обработанные
                video_list = [(vurl, vid) for vurl, vid in all_videos if vid not in skip_ids]

                # Устанавливаем total для progress bar
                self.total_count += len(video_list)

                skipped = total_videos - len(video_list)
                logger.info(f"Found {len(video_list)} videos to process (skipped {skipped} already processed)")

                # Теперь скачиваем и обрабатываем по одному
                for video_url, video_id in video_list:
                    item = self._download_video(video_url)
                    if item:
                        yield item
            else:
                # Одиночное видео
                video_id = self._extract_video_id(url)
                if video_id and video_id in skip_ids:
                    logger.debug(f"Skipping already processed: {video_id}")
                    continue

                item = self._download_video(url)
                if item:
                    yield item

    def _get_video_urls(self, url: str) -> Iterator[tuple]:
        """
        Извлекает URL видео из канала/плейлиста по одному.

        Yields:
            Кортежи (video_url, video_id)
        """
        ydl_opts = {
            "extract_flat": True,
            "quiet": True,
            "no_warnings": True,
            "yesplaylist": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

                if info is None:
                    return

                entries = info.get("entries", [])
                for entry in entries:
                    if entry is None:
                        continue

                    video_id = entry.get("id")
                    if video_id:
                        video_url = f"https://www.youtube.com/watch?v={video_id}"
                        yield (video_url, video_id)

        except Exception as e:
            logger.error(f"Error extracting URLs from {url}: {e}")

    def _extract_video_id(self, url: str) -> Optional[str]:
        """Извлекает video_id из URL."""
        if "v=" in url:
            return url.split("v=")[-1].split("&")[0]
        if "youtu.be/" in url:
            return url.split("youtu.be/")[-1].split("?")[0]
        return None

    def _download_video(self, url: str) -> Optional[ContentItem]:
        """
        Скачивает одно видео и конвертирует в аудио.

        Args:
            url: URL видео

        Returns:
            ContentItem или None при ошибке
        """
        ydl_opts = self._get_ydl_opts()

        for attempt in range(self.max_retries):
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)

                    if info is None:
                        logger.error(f"Failed to extract info for {url}")
                        return None

                    video_id = info["id"]
                    audio_file = self.output_dir / f"{video_id}.{self.audio_format}"

                    if not audio_file.exists():
                        logger.error(f"Audio file not created for {video_id}")
                        return None

                    item = ContentItem(
                        source_id=video_id,
                        content_type=ContentType.AUDIO,
                        source_type=SourceType.YOUTUBE,
                        title=info.get("title", "Unknown"),
                        author=info.get("uploader", "Unknown"),
                        url=url,
                        source_path=audio_file,
                        duration=info.get("duration"),
                        processing_stage=ProcessingStage.DOWNLOADED,
                        metadata={
                            "upload_date": info.get("upload_date"),
                            "view_count": info.get("view_count"),
                            "like_count": info.get("like_count"),
                            "description": (info.get("description", "") or "")[:500],
                            "channel_id": info.get("channel_id"),
                            "channel_url": info.get("channel_url"),
                        },
                    )

                    logger.debug(f"Downloaded: {item.title} ({video_id})")
                    return item

            except yt_dlp.utils.DownloadError as e:
                logger.warning(
                    f"Download error for {url} (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"Failed to download {url} after {self.max_retries} attempts")

            except Exception as e:
                logger.error(f"Unexpected error downloading {url}: {e}")
                return None

        return None

    def _get_ydl_opts(self) -> dict:
        """Возвращает опции для yt-dlp."""
        return {
            "format": "bestaudio/best",
            "outtmpl": str(self.output_dir / "%(id)s.%(ext)s"),
            "ignoreerrors": True,
            "no_warnings": True,
            "quiet": True,
            "no_progress": True,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": self.audio_format,
                    "preferredquality": "0",
                }
            ],
            "keepvideo": True,  # Сохраняем оригинальное видео после конвертации
            "writethumbnail": False,  # Не скачиваем миниатюры
            "retries": self.max_retries,
            "fragment_retries": self.max_retries,
            "socket_timeout": 30,
            "extractor_retries": self.max_retries,
            "extractor_args": {
                "youtube": {
                    "player_client": ["android", "web"],
                    "skip": ["dash", "hls"],
                }
            },
        }
