from pydantic import BaseModel, ConfigDict, Field


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_steps: int = Field(default=8, ge=1)
    max_chat_usage: int = Field(default=50_000, ge=1)
    max_compaction_usage: int = Field(default=800_000, ge=1)
