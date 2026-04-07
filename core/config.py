import os
import tempfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def _load_env() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env")
    load_dotenv()


def _num(value: Optional[str], default: int | float) -> int | float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _as_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


_load_env()


class EnvConfig:
    def __init__(self) -> None:
        default_db = Path(tempfile.gettempdir()) / "mindtrail.db"
        self.db_url = (
            os.getenv("MINDTRAIL_DB_URL")
            or os.getenv("OM_DB_URL")
            or f"sqlite:///{default_db.as_posix()}"
        )
        self.vec_dim = int(_num(os.getenv("MINDTRAIL_VEC_DIM") or os.getenv("OM_VEC_DIM"), 768))
        self.seg_size = int(_num(os.getenv("MINDTRAIL_SEGMENT_SIZE") or os.getenv("OM_SEG_SIZE"), 5000))
        self.use_summary_only = _as_bool(
            os.getenv("MINDTRAIL_USE_SUMMARY_ONLY") or os.getenv("OM_USE_SUMMARY_ONLY"),
            default=False,
        )
        self.summary_max_length = int(
            _num(os.getenv("MINDTRAIL_SUMMARY_MAX_LENGTH") or os.getenv("OM_SUMMARY_MAX_LENGTH"), 600)
        )
        self.keyword_min_length = int(
            _num(os.getenv("MINDTRAIL_KEYWORD_MIN_LENGTH") or os.getenv("OM_KEYWORD_MIN_LENGTH"), 3)
        )
        self.large_document_threshold = int(
            _num(os.getenv("MINDTRAIL_ROOT_THRESHOLD") or os.getenv("OM_ROOT_THRESHOLD"), 8000)
        )
        self.section_size_chars = int(
            _num(os.getenv("MINDTRAIL_SECTION_SIZE") or os.getenv("OM_SECTION_SIZE"), 3000)
        )

    @property
    def database_url(self) -> str:
        return self.db_url

    @database_url.setter
    def database_url(self, value: str) -> None:
        self.db_url = value


env = EnvConfig()
