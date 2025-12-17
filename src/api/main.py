from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.system.rag.pipeline import run

app = FastAPI(title="DS Navigator API")


class ForwardRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    similarity_threshold: Optional[float] = None


@app.post("/forward")
async def forward(req: ForwardRequest):
    # Проверка базового формата
    if not req.question:
        raise HTTPException(status_code=400, detail="bad request")

    try:
        # Передаем параметры в run
        result = await run(
            question=req.question,
            top_k=req.top_k,
            similarity_threshold=req.similarity_threshold
        )
        if not result:
            raise HTTPException(
                status_code=403,
                detail="модель не смогла обработать данные"
            )

        return result

    except Exception as e:
        # Любые неожиданные ошибки тоже возвращаем 403
        raise HTTPException(status_code=403, detail=str(e))
