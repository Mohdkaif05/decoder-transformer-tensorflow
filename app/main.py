from fastapi import FastAPI, HTTPException

from app.model_service import (
    load_model_once,
    generate_text,
)

from app.schemas import GenerateRequest

app = FastAPI(
    title="MiniGPT API",
    version="1.0.0",
)


@app.on_event("startup")
def startup():
    load_model_once()


@app.get("/")
def home():
    return {
        "message": "MiniGPT API is running."
    }


@app.post("/generate")
def generate(request: GenerateRequest):

    try:
        output = generate_text(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
        )

        return {
            "prompt": request.prompt,
            "generated_text": output,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )