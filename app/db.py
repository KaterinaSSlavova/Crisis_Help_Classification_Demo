import sqlite3
from flask import current_app, g
import json
from datetime import datetime, timedelta

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  user_text TEXT NOT NULL,
  probs_json TEXT NOT NULL,
  preds_json TEXT NOT NULL,
  xai_json TEXT NOT NULL
);
"""

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DB_PATH"])
        g.db.row_factory = sqlite3.Row
    return g.db

def close_db(_e=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def init_db(app):
    import os
    os.makedirs(app.instance_path, exist_ok=True)

    @app.before_request
    def _ensure_db():
        db = get_db()
        db.executescript(SCHEMA)
        db.commit()

    app.teardown_appcontext(close_db)

def insert_request(user_text: str, probs_json: str, preds_json: str, xai_json: str) -> int:
    db = get_db()
    cur = db.execute(
        "INSERT INTO requests(created_at, user_text, probs_json, preds_json, xai_json) VALUES (?, ?, ?, ?, ?)",
        (datetime.utcnow().isoformat() + "Z", user_text, probs_json, preds_json, xai_json),
    )
    db.commit()
    return int(cur.lastrowid)

def list_requests(limit: int = 50, offset: int = 0):
    db = get_db()
    return db.execute(
        """
        SELECT id, created_at,
               substr(user_text, 1, 220) AS snippet
        FROM requests
        ORDER BY id DESC
        LIMIT ? OFFSET ?
        """,
        (limit, offset),
    ).fetchall()

def get_request(req_id: int):
    db = get_db()
    row = db.execute(
        "SELECT id, created_at, user_text, probs_json, preds_json, xai_json FROM requests WHERE id = ?",
        (req_id,),
    ).fetchone()
    return row

def label_counts():
    import json
    db = get_db()
    rows = db.execute("SELECT preds_json FROM requests").fetchall()
    counts = None
    for r in rows:
        preds = json.loads(r["preds_json"])
        if counts is None:
            counts = {k: 0 for k in preds.keys()}
        for k, v in preds.items():
            counts[k] += int(v)
    return counts or {}

def list_requests_by_label(label: str, limit: int = 200):
    import json
    db = get_db()
    rows = db.execute(
        "SELECT id, created_at, user_text, preds_json FROM requests ORDER BY id DESC"
    ).fetchall()

    out = []
    for r in rows:
        preds = json.loads(r["preds_json"])
        if int(preds.get(label, 0)) == 1:
            out.append({
                "id": r["id"],
                "created_at": r["created_at"],
                "snippet": (r["user_text"][:220]),
            })
            if len(out) >= limit:
                break
    return out

def today_label_counts_all(labels: list[str]) -> dict:
    """Counts per label for requests created today (UTC), including zeros."""
    today = datetime.utcnow().date().isoformat()
    db = get_db()
    rows = db.execute(
        "SELECT preds_json FROM requests WHERE substr(created_at,1,10)=?",
        (today,)
    ).fetchall()

    counts = {lab: 0 for lab in labels}
    for r in rows:
        preds = json.loads(r["preds_json"])
        for lab in labels:
            if int(preds.get(lab, 0)) == 1:
                counts[lab] += 1
    return counts


def requests_per_day_last_n(n_days: int = 7) -> dict:
    end = datetime.utcnow().date()
    start = end - timedelta(days=n_days - 1)

    db = get_db()
    rows = db.execute(
        """
        SELECT substr(created_at,1,10) as day, COUNT(*) as c
        FROM requests
        WHERE day BETWEEN ? AND ?
        GROUP BY day
        ORDER BY day
        """,
        (start.isoformat(), end.isoformat())
    ).fetchall()

    out = {}
    d = start
    while d <= end:
        out[d.isoformat()] = 0
        d += timedelta(days=1)

    for r in rows:
        out[r["day"]] = r["c"]

    return out