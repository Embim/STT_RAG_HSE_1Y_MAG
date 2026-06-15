from pathlib import Path
from sqlmodel import SQLModel, Session, create_engine
from settings import settings

_engine = None


def _db_path() -> str:
    if settings.AUTH_DB_PATH:
        return settings.AUTH_DB_PATH
    return str(Path(__file__).resolve().parents[3] / "auth" / "users.db")


def get_engine():
    global _engine
    if _engine is None:
        path = _db_path()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
    return _engine


def init_db() -> None:
    import system.auth.models  # noqa: F401 — register the table
    SQLModel.metadata.create_all(get_engine())


def get_session() -> Session:
    return Session(get_engine())
