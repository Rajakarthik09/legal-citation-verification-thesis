"""SQLite cache for CourtListener citation-lookup responses."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path


class CitationCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS citation_lookup (
                text_hash TEXT PRIMARY KEY,
                response_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.commit()

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> list[dict] | None:
        key = self._hash(text)
        row = self._conn.execute(
            "SELECT response_json FROM citation_lookup WHERE text_hash = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def set(self, text: str, response: list[dict]) -> None:
        key = self._hash(text)
        self._conn.execute(
            "INSERT OR REPLACE INTO citation_lookup (text_hash, response_json) VALUES (?, ?)",
            (key, json.dumps(response)),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
