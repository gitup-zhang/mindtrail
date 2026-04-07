import logging
import sqlite3
import time
from importlib import resources
from pathlib import Path
from typing import Optional

from .config import env

logger = logging.getLogger("mindtrail.db")


class DB:
    def __init__(self) -> None:
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        if self.conn is not None:
            return

        url = env.database_url
        if not url.startswith("sqlite:///"):
            raise ValueError(f"Unsupported database URL: {url}")

        raw_path = url.replace("sqlite:///", "", 1)
        path = Path(raw_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        try:
            self.conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            self.conn.execute("PRAGMA journal_mode=DELETE")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")

        logger.info("Connected to %s", path)
        self.run_migrations()

    def run_migrations(self) -> None:
        assert self.conn is not None
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS _migrations (name TEXT PRIMARY KEY, applied_at INTEGER NOT NULL)"
        )
        try:
            files = [
                item.name
                for item in resources.files("mindtrail.migrations").iterdir()
                if item.name.endswith(".sql")
            ]
        except Exception:
            migration_dir = Path(__file__).resolve().parents[1] / "migrations"
            files = [item.name for item in migration_dir.iterdir() if item.suffix == ".sql"]

        for name in sorted(files):
            already_applied = self.fetchone("SELECT 1 FROM _migrations WHERE name=?", (name,))
            if already_applied:
                continue

            try:
                sql = resources.files("mindtrail.migrations").joinpath(name).read_text(encoding="utf-8")
            except Exception:
                sql = ((Path(__file__).resolve().parents[1] / "migrations") / name).read_text(encoding="utf-8")

            self.conn.executescript(sql)
            self.conn.execute(
                "INSERT INTO _migrations (name, applied_at) VALUES (?, ?)",
                (name, int(time.time())),
            )
            self.conn.commit()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        self.connect()
        assert self.conn is not None
        return self.conn.execute(sql, params)

    def fetchall(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        self.connect()
        assert self.conn is not None
        return self.conn.execute(sql, params).fetchall()

    def fetchone(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        self.connect()
        assert self.conn is not None
        return self.conn.execute(sql, params).fetchone()

    def commit(self) -> None:
        if self.conn is not None:
            self.conn.commit()


db = DB()


class Queries:
    def ins_mem(self, **kwargs: object) -> None:
        sql = """
        INSERT INTO memories (
            id, user_id, segment, content, simhash, primary_sector, tags, meta,
            created_at, updated_at, last_seen_at, salience, decay_lambda,
            version, mean_dim, mean_vec, compressed_vec, feedback_score
        )
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET
            user_id=excluded.user_id,
            segment=excluded.segment,
            content=excluded.content,
            simhash=excluded.simhash,
            primary_sector=excluded.primary_sector,
            tags=excluded.tags,
            meta=excluded.meta,
            created_at=excluded.created_at,
            updated_at=excluded.updated_at,
            last_seen_at=excluded.last_seen_at,
            salience=excluded.salience,
            decay_lambda=excluded.decay_lambda,
            version=excluded.version,
            mean_dim=excluded.mean_dim,
            mean_vec=excluded.mean_vec,
            compressed_vec=excluded.compressed_vec,
            feedback_score=excluded.feedback_score
        """
        values = (
            kwargs.get("id"),
            kwargs.get("user_id"),
            kwargs.get("segment", 0),
            kwargs.get("content"),
            kwargs.get("simhash"),
            kwargs.get("primary_sector"),
            kwargs.get("tags"),
            kwargs.get("meta"),
            kwargs.get("created_at"),
            kwargs.get("updated_at"),
            kwargs.get("last_seen_at"),
            kwargs.get("salience", 0.5),
            kwargs.get("decay_lambda", 0.02),
            kwargs.get("version", 1),
            kwargs.get("mean_dim"),
            kwargs.get("mean_vec"),
            kwargs.get("compressed_vec"),
            kwargs.get("feedback_score", 0),
        )
        db.execute(sql, values)
        db.commit()

    def get_mem(self, memory_id: str) -> Optional[sqlite3.Row]:
        return db.fetchone("SELECT * FROM memories WHERE id=?", (memory_id,))

    def all_mem(self, limit: int = 10, offset: int = 0) -> list[sqlite3.Row]:
        return db.fetchall(
            "SELECT * FROM memories ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )

    def all_mem_by_user(self, user_id: str, limit: int = 10, offset: int = 0) -> list[sqlite3.Row]:
        return db.fetchall(
            "SELECT * FROM memories WHERE user_id=? ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (user_id, limit, offset),
        )

    def ins_log(self, entry_id: str, model: str, status: str, ts: int, err: Optional[str] = None) -> None:
        db.execute(
            "INSERT INTO embed_logs (id, model, status, ts, err) VALUES (?, ?, ?, ?, ?)",
            (entry_id, model, status, ts, err),
        )
        db.commit()

    def upd_log(self, entry_id: str, status: str, err: Optional[str] = None) -> None:
        db.execute(
            "UPDATE embed_logs SET status=?, err=? WHERE id=?",
            (status, err, entry_id),
        )
        db.commit()

    def del_mem(self, memory_id: str) -> None:
        db.execute("DELETE FROM waypoints WHERE src_id=? OR dst_id=?", (memory_id, memory_id))
        db.execute("DELETE FROM vectors WHERE id=?", (memory_id,))
        db.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        db.commit()

    def del_mem_by_user(self, user_id: str) -> None:
        db.execute("DELETE FROM vectors WHERE user_id=?", (user_id,))
        db.execute(
            """
            DELETE FROM waypoints
            WHERE src_id IN (SELECT id FROM memories WHERE user_id=?)
               OR dst_id IN (SELECT id FROM memories WHERE user_id=?)
            """,
            (user_id, user_id),
        )
        db.execute("DELETE FROM memories WHERE user_id=?", (user_id,))
        db.commit()


q = Queries()


def transaction() -> Optional[sqlite3.Connection]:
    db.connect()
    return db.conn
