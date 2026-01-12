import json
from datetime import datetime, timezone
from flask import Blueprint, render_template, abort, request
from .db import list_requests, get_request, label_counts, today_label_counts_all, requests_per_day_last_n, list_requests_by_label
from .labels import LABELS, LABEL_DESCRIPTIONS

bp_manager = Blueprint("manager", __name__, url_prefix="/manager")


def fmt_utc_iso(ts: str) -> str:
    if not ts:
        return ""
    dt = datetime.fromisoformat(ts.replace("Z", ""))
    dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%d %b %Y · %H:%M UTC")


@bp_manager.get("/")
def manager_home():
    rows = list_requests(limit=50)
    rows = [dict(r, created_at_pretty=fmt_utc_iso(r["created_at"])) for r in rows]

    today_counts = today_label_counts_all(LABELS)
    per_day = requests_per_day_last_n(7)

    return render_template(
        "manager_home.html",
        rows=rows,
        counts=label_counts(),
        # chart payloads
        chart_today_labels=json.dumps(today_counts),
        chart_days=json.dumps(list(per_day.keys())),
        chart_day_counts=json.dumps(list(per_day.values())),
    )

@bp_manager.get("/request/<int:req_id>")
def manager_request(req_id: int):
    row = get_request(req_id)
    if not row:
        abort(404)

    row = dict(row)

    probs = json.loads(row["probs_json"])
    preds = json.loads(row["preds_json"])
    xai = json.loads(row["xai_json"])

    thr = {}
    if "thr_json" in row and row["thr_json"]:
        thr = json.loads(row["thr_json"])
    else:
        thr = {k: 0.0 for k in probs.keys()}

    return render_template(
        "user_result.html",
        req_id=row["id"],
        text=row.get("user_text", ""),
        probs=probs,
        preds=preds,
        xai=xai,
        thr=thr,
    )

@bp_manager.get("/label/<label>")
def manager_label(label: str):
    if label not in LABEL_DESCRIPTIONS:
        abort(404)

    rows = list_requests_by_label(label, limit=15)
    rows = [dict(r, created_at_pretty=fmt_utc_iso(r["created_at"])) for r in rows]

    return render_template(
        "manager_label.html",
        label=label,
        label_desc=LABEL_DESCRIPTIONS[label],
        rows=rows,
    )

@bp_manager.get("/requests")
def manager_requests_all():
    page = int(request.args.get("page", 1))
    per_page = 25
    offset = (page - 1) * per_page

    rows = list_requests(limit=per_page, offset=offset)
    rows = [dict(r, created_at_pretty=fmt_utc_iso(r["created_at"])) for r in rows]

    has_next = len(rows) == per_page

    return render_template(
        "manager_request.html",
        rows=rows,
        page=page,
        has_next=has_next,
    )
