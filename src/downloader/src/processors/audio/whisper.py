"""
Whisper audio transcription processor.

Транскрибирует аудио файлы с помощью Transformers Whisper.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
import time
import warnings

# Подавляем DeprecationWarning от audioread (используется в librosa)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="audioread")

import torch
import librosa
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq

from ...core.interfaces import BaseProcessor
from ...core.models import ContentItem, ContentType, TextSegment, ProcessingStage
from ...core.registry import register_processor

logger = logging.getLogger(__name__)


@register_processor
class WhisperProcessor(BaseProcessor):
    """
    Процессор транскрипции аудио через Whisper.

    Использует Transformers Whisper для преобразования
    аудио/видео контента в текстовые сегменты с timestamps.
    """

    processor_name = "whisper"
    input_types = [ContentType.AUDIO, ContentType.VIDEO]
    output_type = ContentType.TEXT

    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = None,
        beam_size: int = 5,
    ):
        """
        Инициализация процессора Whisper.

        Args:
            model_name: HuggingFace модель Whisper
            device: Устройство (cuda, cpu)
            compute_type: Тип данных (float16, float32)
            language: Язык аудио (None для автоопределения)
            beam_size: Размер beam для декодирования
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.pipe = None
        self._is_setup = False

    def setup(self, config: Dict[str, Any]) -> None:
        """Загружает модель Whisper."""
        if self._is_setup:
            return

        logger.info(f"Loading Whisper model: {self.model_name} on {self.device}")
        start_time = time.time()

        # Определяем torch_dtype
        if self.compute_type == "float16":
            torch_dtype = torch.float16
        elif self.compute_type == "float32":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Загружаем модель
        device_map = self.device if self.device == "cpu" else f"{self.device}:0"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(device_map)

        processor = AutoProcessor.from_pretrained(self.model_name)

        # Создаём pipeline
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=device_map,
            )

        load_time = time.time() - start_time
        logger.info(f"Whisper model loaded in {load_time:.2f} seconds")
        self._is_setup = True

    def teardown(self) -> None:
        """Освобождает ресурсы."""
        if self.pipe:
            del self.pipe
            self.pipe = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
        self._is_setup = False
        logger.debug("Whisper processor teardown complete")

    def can_process(self, item: ContentItem) -> bool:
        """Проверяет, может ли процессор обработать item."""
        return item.content_type in self.input_types

    def process(self, item: ContentItem) -> ContentItem:
        """
        Транскрибирует аудио в текстовые сегменты.

        Args:
            item: ContentItem с аудио файлом

        Returns:
            ContentItem с заполненными сегментами
        """
        if not self._is_setup:
            self.setup({})

        if not item.source_path or not item.source_path.exists():
            raise FileNotFoundError(f"Audio file not found: {item.source_path}")

        logger.debug(f"Transcribing: {item.source_path}")
        start_time = time.time()

        # Загружаем аудио для определения длительности (подавляем audioread warnings)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            audio_array, sample_rate = librosa.load(str(item.source_path), sr=16000)
        duration = len(audio_array) / sample_rate

        # Обновляем длительность если не была установлена
        if item.duration is None:
            item.duration = duration

        # Настраиваем параметры генерации
        generate_kwargs = {
            "num_beams": self.beam_size,
            "return_timestamps": True,
        }

        if self.language:
            generate_kwargs["language"] = self.language

        # Транскрибируем
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*chunk_length_s.*")
            warnings.filterwarnings("ignore", message=".*ending timestamp.*")
            warnings.filterwarnings("ignore", message=".*WhisperTimeStampLogitsProcessor.*")
            warnings.filterwarnings("ignore", message=".*Whisper did not predict.*")
            warnings.simplefilter("ignore")

            # Временно повышаем уровень логирования transformers
            import logging as py_logging
            transformers_logger = py_logging.getLogger("transformers")
            old_level = transformers_logger.level
            transformers_logger.setLevel(py_logging.ERROR)

            result = self.pipe(
                str(item.source_path),
                generate_kwargs=generate_kwargs,
                chunk_length_s=30,
                stride_length_s=5,
                return_timestamps=True,
            )

            transformers_logger.setLevel(old_level)

        # Парсим результат в сегменты
        segments = []

        if "chunks" in result:
            for idx, chunk in enumerate(result["chunks"]):
                timestamp = chunk.get("timestamp", (0.0, duration))
                start = timestamp[0] if isinstance(timestamp, tuple) and timestamp[0] is not None else 0.0
                end = timestamp[1] if isinstance(timestamp, tuple) and timestamp[1] is not None else duration

                seg = TextSegment(
                    id=idx,
                    text=chunk["text"].strip(),
                    start=start,
                    end=end,
                )
                if seg.text:  # Пропускаем пустые сегменты
                    segments.append(seg)
        else:
            # Простой результат без timestamps
            text = result.get("text", "")
            if text:
                segments.append(TextSegment(
                    id=0,
                    text=text.strip(),
                    start=0.0,
                    end=duration,
                ))

        transcribe_time = time.time() - start_time

        # Обновляем item
        item.segments = segments
        item.raw_text = " ".join(seg.text for seg in segments)
        item.language = self.language or "auto"
        item.processing_stage = ProcessingStage.PROCESSED
        item.metadata.update({
            "transcribe_time": transcribe_time,
            "segments_count": len(segments),
            "rtf": transcribe_time / duration if duration > 0 else 0,
        })

        logger.debug(
            f"Transcription complete: {len(segments)} segments, "
            f"{transcribe_time:.2f}s, RTF: {item.metadata['rtf']:.2f}x"
        )

        return item
