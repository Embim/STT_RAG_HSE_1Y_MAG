from typing import Optional
from urllib.parse import urlparse
from pydantic import BaseModel, Field, field_validator


class ForwardRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Вопрос пользователя")
    top_k: Optional[int] = Field(None, ge=1, le=10, description="Количество возвращаемых результатов")
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Порог схожести")
    use_rewrite: bool = Field(True, description="Переформулировать вопрос перед поиском")
    source_file_name: Optional[str] = Field(None, description="Искать только в выбранном исходном файле")
    source_title: Optional[str] = Field(None, description="Искать только в выбранном видео по названию")


_ALLOWED_INGEST_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be",
    "vimeo.com", "www.vimeo.com",
}


class IngestRequest(BaseModel):
    url: str = Field(..., description="YouTube video or playlist URL")
    export_txt: bool = Field(False, description="Save plain transcript .txt to data/transcripts/ and include in response")
    export_json: bool = Field(False, description="Save full transcript .json with segments/timestamps to data/transcripts/")
    keep_audio: bool = Field(False, description="Save extracted audio to data/audio/")
    use_ocr: bool = Field(False, description="Extract text from video using EasyOCR")

    @field_validator("url")
    @classmethod
    def _validate_url(cls, v: str) -> str:
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("url must be http(s)")
        host = (parsed.hostname or "").lower()
        if host not in _ALLOWED_INGEST_HOSTS:
            raise ValueError("url host not allowed")
        return v


