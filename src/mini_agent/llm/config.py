from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(min_length=1)
    chat_max_output_tokens: int = Field(default=2_048, ge=1)
    compaction_max_output_tokens: int = Field(default=4_096, ge=1)
    max_attempts: int = Field(default=3, ge=1)
