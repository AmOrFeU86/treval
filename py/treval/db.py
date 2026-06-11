"""Local SQLite storage for treval spans."""

from __future__ import annotations

import json
from typing import Callable
import sqlite3
import threading
from pathlib import Path

DB_PATH = Path.home() / ".treval" / "spans.db"

# Hooks post-save para streaming (OTEL, webhooks, etc.)
_post_save_hooks: list[Callable[[dict], None]] = []


def add_post_save_hook(hook: Callable[[dict], None]) -> None:
    """Registers a callback that runs after each span is saved."""
    _post_save_hooks.append(hook)


class SpanStore:
    """Span store backed by local SQLite."""

    _local = threading.local()
    _lock = threading.Lock()

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = Path(db_path) if db_path else DB_PATH
        self._init_db()

    @property
    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._local.conn = sqlite3.connect(str(self._db_path))
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        with self._lock:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._db_path))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS spans (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    name        TEXT    NOT NULL,
                    type        TEXT    NOT NULL DEFAULT 'TOOL',
                    status      TEXT    NOT NULL DEFAULT 'ok',
                    parent_id   INTEGER,
                    input       TEXT,
                    output      TEXT,
                    duration_ms REAL,
                    metadata    TEXT,
                    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
                )
            """)
            # Add parent_id if table already existed without it
            try:
                conn.execute("ALTER TABLE spans ADD COLUMN parent_id INTEGER")
            except sqlite3.OperationalError:
                pass  # ya existe
            conn.commit()
            conn.close()

    def save(self, *, name: str, type: str = "TOOL", status: str = "ok",
             parent_id: int | None = None,
             input: str | None = None, output: str | None = None,
             duration_ms: float | None = None,
             metadata: dict | None = None) -> int:
        """Saves a span and returns its ID."""
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO spans (name, type, status, parent_id, input, output, duration_ms, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (name, type, status, parent_id, input, output, duration_ms,
                 json.dumps(metadata) if metadata else None),
            )
            self._conn.commit()
            span_id = cur.lastrowid
            # Notify post-save hooks (OTEL streaming, etc.)
            if _post_save_hooks:
                span = self.get(span_id)
                if span:
                    for hook in _post_save_hooks:
                        try:
                            hook(span)
                        except Exception:
                            pass
            return span_id

    def update(self, span_id: int, **fields) -> None:
        """Updates fields of an existing span.

        fields can include: status, output, duration_ms, metadata.
        A dict `metadata` is JSON-encoded (same as save() does), so
        callers don't have to pre-encode and don't have to special-case
        the difference between `save` and `update`.
        """
        allowed = {"status", "output", "duration_ms", "metadata", "name", "input"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        if "metadata" in updates:
            md = updates["metadata"]
            if md is not None and not isinstance(md, str):
                updates["metadata"] = json.dumps(md)
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values())
        values.append(span_id)
        with self._lock:
            self._conn.execute(
                f"UPDATE spans SET {set_clause} WHERE id = ?", values
            )
            self._conn.commit()

    def list_spans(self, limit: int = 100, type: str | None = None) -> list[dict]:
        """Lists the most recent spans, optionally filtered by type."""
        if type:
            rows = self._conn.execute(
                "SELECT * FROM spans WHERE type = ? ORDER BY id DESC LIMIT ?",
                (type, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM spans ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get(self, span_id: int) -> dict | None:
        """Gets a span by ID."""
        row = self._conn.execute(
            "SELECT * FROM spans WHERE id = ?", (span_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_children(self, parent_id: int) -> list[dict]:
        """Gets the child spans of a given span."""
        rows = self._conn.execute(
            "SELECT * FROM spans WHERE parent_id = ? ORDER BY id ASC",
            (parent_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def clear(self) -> None:
        """Clears all spans (useful in tests)."""
        with self._lock:
            self._conn.execute("DELETE FROM spans")
            self._conn.commit()

    def count(self) -> int:
        """Total number of spans."""
        row = self._conn.execute("SELECT COUNT(*) as cnt FROM spans").fetchone()
        return row["cnt"]