from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
from system.rag.pipeline import run
from system.llm.llm_services import chat_vector_store_manager

app = FastAPI(title="DS Navigator API")


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to DS Navigator API",
        "documentation": "/docs"
    }


@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health"])
async def healthcheck():
    return {
        "status": "ok"
    }

@app.get("/check-vdb", tags=["Health"])
async def check_vdb():
    try:
        collection = chat_vector_store_manager.vector_store._collection
        count = collection.count()

        return {
            "status": "ok",
            "documents_in_vdb": count
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="vector db is not available"
        )


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


@app.post("/forward", tags=["Usage"])
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