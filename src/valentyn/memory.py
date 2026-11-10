"""Append-only experiment archive; only accepted proofs enter theorem memory."""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3


class Memory:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with closing(sqlite3.connect(path)) as db, db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY, created TEXT NOT NULL,
                    task_id TEXT NOT NULL, split TEXT NOT NULL,
                    formula TEXT NOT NULL, strategy TEXT NOT NULL,
                    status TEXT NOT NULL, nodes INTEGER NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS theorems (
                    source_hash TEXT PRIMARY KEY, formula TEXT NOT NULL,
                    source TEXT NOT NULL, lean_version TEXT NOT NULL,
                    attempt_id INTEGER NOT NULL REFERENCES attempts(id)
                );
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY, created TEXT NOT NULL,
                    kind TEXT NOT NULL, payload TEXT NOT NULL
                );
            """)

    def record(self, attempt: dict) -> int:
        verification = attempt.get("verification") or {}
        if (attempt["status"] == "verified") != bool(verification.get("accepted")):
            raise ValueError("Attempt status and verification disagree")
        with closing(sqlite3.connect(self.path)) as db, db:
            cursor = db.execute(
                "INSERT INTO attempts(created,task_id,split,formula,strategy,status,nodes,payload) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), attempt["task_id"], attempt["split"],
                 attempt["formula"], attempt["strategy"], attempt["status"], attempt["nodes"],
                 json.dumps(attempt)))
            identifier = cursor.lastrowid
            if verification.get("accepted"):
                db.execute("INSERT OR IGNORE INTO theorems VALUES(?,?,?,?,?)",
                           (verification["source_hash"], attempt["formula"], verification["source"],
                            verification["lean_version"], identifier))
            return int(identifier)

    def experiment(self, kind: str, payload: dict) -> None:
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("INSERT INTO experiments(created,kind,payload) VALUES(?,?,?)",
                       (datetime.now(timezone.utc).isoformat(), kind, json.dumps(payload)))

    def history(self, limit: int = 20) -> list[dict]:
        if not 1 <= limit <= 1000:
            raise ValueError("History limit must be 1..1000")
        with closing(sqlite3.connect(self.path)) as db:
            rows = db.execute("SELECT id,payload FROM attempts ORDER BY id DESC LIMIT ?", (limit,))
            return [{"id": row[0], **json.loads(row[1])} for row in rows]

    def contains(self, formula: str) -> bool:
        with closing(sqlite3.connect(self.path)) as db:
            return db.execute("SELECT 1 FROM theorems WHERE formula=? LIMIT 1", (formula,)).fetchone() is not None

    def summary(self) -> dict:
        with closing(sqlite3.connect(self.path)) as db:
            return {"attempts": db.execute("SELECT count(*) FROM attempts").fetchone()[0],
                    "verified_proof_artifacts": db.execute("SELECT count(*) FROM theorems").fetchone()[0],
                    "experiments": db.execute("SELECT count(*) FROM experiments").fetchone()[0]}
