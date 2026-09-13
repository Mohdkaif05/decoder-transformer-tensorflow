from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        description="Input prompt",
    )

    max_new_tokens: int = Field(
        default=50,
        ge=1,
        le=200,
    )

    temperature: float = Field(
        default=1.0,
        gt=0,
        le=2,
    )

    top_k: int = Field(
        default=50,
        ge=1,
        le=100,
    )