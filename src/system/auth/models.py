from datetime import datetime, timezone
from sqlmodel import SQLModel, Field


class User(SQLModel, table=True):
    username: str = Field(primary_key=True)
    password_hash: str
    role: str = Field(default="user")  # "user" | "admin"
    is_active: bool = Field(default=True)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
