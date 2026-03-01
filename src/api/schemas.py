from typing import Optional
from pydantic import BaseModel, Field


class ForwardRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Вопрос пользователя")
    top_k: Optional[int] = Field(None, ge=1, le=10, description="Количество возвращаемых результатов")
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Порог схожести")


class IngestRequest(BaseModel):
    url: str = Field(..., description="YouTube video or playlist URL")
    export_txt: bool = Field(False, description="Include transcript text in response")
    keep_video: bool = Field(False, description="Download and save full video to data/video/")


