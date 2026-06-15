from pathlib import Path
from sqlmodel import SQLModel, Session, create_engine
from settings import settings

_engine = None


def _db_url() -> str:
    if settings.AUTH_DATABASE_URL:
        return settings.AUTH_DATABASE_URL
    path = settings.AUTH_DB_PATH or str(Path(__file__).resolve().parents[3] / "auth" / "users.db")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


def get_engine():
    global _engine
    if _engine is None:
        url = _db_url()
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True)
    return _engine


def init_db() -> None:
    import system.auth.models  # noqa: F401 — register the table
    SQLModel.metadata.create_all(get_engine())


def get_session() -> Session:
    return Session(get_engine())
