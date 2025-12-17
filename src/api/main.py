from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
from src.system.rag.pipeline import run

app = FastAPI(title="DS Navigator API")


class ForwardRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Вопрос пользователя")
    top_k: Optional[int] = Field(None, ge=1, le=10, description="Количество возвращаемых результатов")
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Порог схожести")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"detail": "bad request"},
    )


@app.post("/forward")
async def forward(req: ForwardRequest):
    try:
        result = await run(
            question=req.question,
            top_k=req.top_k,
            similarity_threshold=req.similarity_threshold
        )

        return result
    except Exception:
        raise HTTPException(
            status_code=403,
            detail="модель не смогла обработать данные"
        )