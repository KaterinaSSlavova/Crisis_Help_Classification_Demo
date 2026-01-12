import json
from flask import Blueprint, render_template, abort
from .db import list_requests, get_request, label_counts

bp_manager = Blueprint("manager", __name__, url_prefix="/manager")

@bp_manager.get("/")
def manager_home():
    rows = list_requests(limit=50)
    counts = label_counts()
    return render_template("manager_home.html", rows=rows, counts=counts)

@bp_manager.get("/request/<int:req_id>")
def manager_request(req_id: int):
    row = get_request(req_id)
    if not row:
        abort(404)

    probs = json.loads(row["probs_json"])
    preds = json.loads(row["preds_json"])
    return render_template(
        "manager_request.html",
        row=row,
        probs=probs,
        preds=preds,
    )
