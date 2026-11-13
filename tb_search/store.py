"""SQLite store so discovery runs accumulate: papers, per-run candidates, trace, assessment cache."""

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime

from tb_search.sources import Paper

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    key TEXT PRIMARY KEY, title TEXT, year TEXT, doi TEXT, abstract TEXT, cited_by INTEGER
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, started TEXT, query TEXT, seed_key TEXT, config TEXT
);
CREATE TABLE IF NOT EXISTS candidates (
    run_id INTEGER, paper_key TEXT, round INTEGER, routes TEXT, score REAL, relevance INTEGER,
    PRIMARY KEY (run_id, paper_key)
);
CREATE TABLE IF NOT EXISTS assessments (
    paper_key TEXT, topic_key TEXT, model TEXT, assessment TEXT,
    PRIMARY KEY (paper_key, topic_key, model)
);
CREATE TABLE IF NOT EXISTS events (
    run_id INTEGER, at TEXT, round INTEGER, kind TEXT, detail TEXT
);
"""


class Store:
    """Pass ":memory:" for a throwaway store."""

    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)

    def start_run(self, query: str, seed_key: str, config: dict) -> int:
        cursor = self.db.execute(
            "INSERT INTO runs (started, query, seed_key, config) VALUES (?, ?, ?, ?)",
            (
                datetime.now().isoformat(timespec="seconds"),
                query,
                seed_key,
                json.dumps(config),
            ),
        )
        self.db.commit()
        return cursor.lastrowid

    def save_paper(self, paper: Paper) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO papers VALUES (:key, :title, :year, :doi, :abstract, :cited_by)",
            asdict(paper),
        )

    def save_candidate(
        self,
        run_id: int,
        paper_key: str,
        round_number: int,
        routes: list[str],
        score: float,
        relevance: int | None,
    ) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO candidates VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, paper_key, round_number, json.dumps(routes), score, relevance),
        )

    def cached_assessment(
        self, paper_key: str, topic_key: str, model: str
    ) -> dict | None:
        row = self.db.execute(
            "SELECT assessment FROM assessments WHERE paper_key = ? AND topic_key = ? AND model = ?",
            (paper_key, topic_key, model),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def save_assessment(
        self, paper_key: str, topic_key: str, model: str, assessment: dict
    ) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO assessments VALUES (?, ?, ?, ?)",
            (paper_key, topic_key, model, json.dumps(assessment)),
        )
        self.db.commit()

    def event(self, run_id: int, round_number: int, kind: str, detail: dict) -> None:
        """One trace step. The trace is what a process-level rubric would grade."""
        self.db.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
            (
                run_id,
                datetime.now().isoformat(timespec="seconds"),
                round_number,
                kind,
                json.dumps(detail),
            ),
        )
        self.db.commit()

    def events(self, run_id: int) -> list[tuple[int, str, dict]]:
        rows = self.db.execute(
            "SELECT round, kind, detail FROM events WHERE run_id = ? ORDER BY rowid",
            (run_id,),
        )
        return [
            (round_number, kind, json.loads(detail))
            for round_number, kind, detail in rows
        ]

    def close(self) -> None:
        self.db.commit()
        self.db.close()
