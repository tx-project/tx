"""Database models for the Tinker API."""
from datetime import datetime
from pathlib import Path
from sqlmodel import SQLModel, Field

# SQLite database path
DB_PATH = Path(__file__).parent / "tinker.db"


# SQLModel table definitions
class ModelDB(SQLModel, table=True):
    __tablename__ = "models"

    model_id: str = Field(primary_key=True)
    base_model: str
    lora_config: str | None = None  # JSON string
    status: str
    request_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class FutureDB(SQLModel, table=True):
    __tablename__ = "futures"

    request_id: str = Field(primary_key=True, index=True)
    request_type: str  # "create_model", "forward_backward", "optim_step"
    model_id: str | None = None
    request_data: str  # JSON string with request details
    result_data: str | None = None  # JSON string with result, None if not yet processed
    status: str = Field(default="pending", index=True)  # "pending", "completed", "failed"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
