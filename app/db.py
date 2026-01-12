import sqlite3
from flask import current_app, g
from datetime import datetime

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  user_text TEXT NOT NULL,
  probs_json TEXT NOT NULL,
  preds_json TEXT NOT NULL
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
    # ensure instance dir exists
    import os
    os.makedirs(app.instance_path, exist_ok=True)

    @app.before_request
    def _ensure_db():
        db = get_db()
        db.executescript(SCHEMA)
        db.commit()

    app.teardown_appcontext(close_db)

def insert_request(user_text: str, probs_json: str, preds_json: str) -> int:
    db = get_db()
    cur = db.execute(
        "INSERT INTO requests(created_at, user_text, probs_json, preds_json) VALUES (?, ?, ?, ?)",
        (datetime.utcnow().isoformat() + "Z", user_text, probs_json, preds_json),
    )
    db.commit()
    return int(cur.lastrowid)

def list_requests(limit: int = 50):
    db = get_db()
    rows = db.execute(
        "SELECT id, created_at, substr(user_text,1,220) AS snippet FROM requests ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return rows

def get_request(req_id: int):
    db = get_db()
    row = db.execute(
        "SELECT id, created_at, user_text, probs_json, preds_json FROM requests WHERE id = ?",
        (req_id,),
    ).fetchone()
    return row

def label_counts():
    """
    Counts predicted positives per label across all stored requests.
    """
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
