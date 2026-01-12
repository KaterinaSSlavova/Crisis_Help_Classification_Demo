import json
from flask import Blueprint, current_app, render_template, request, redirect, url_for, flash

bp_user = Blueprint("user", __name__)

@bp_user.before_app_request
def _load_once():
    from .model import load_model

    load_model(
        current_app.config["MODEL_DIR"],
        current_app.config["THRESHOLDS_PATH"],
    )

@bp_user.get("/")
def home():
    from .labels import LABELS, LABEL_DESCRIPTIONS
    return render_template("user_home.html", labels=LABELS, label_desc=LABEL_DESCRIPTIONS)

@bp_user.post("/classify")
def classify():
    from .model import predict
    from .labels import LABELS
    from .xai import explain_positive_labels
    from .db import insert_request

    text = (request.form.get("text") or "").strip()
    if not text:
        flash("Please paste a message first.")
        return redirect(url_for("user.home"))

    max_chars = int(current_app.config.get("MAX_TEXT_CHARS", 4000))
    if len(text) > max_chars:
        flash(f"Text too long. Limit is {max_chars} characters.")
        return redirect(url_for("user.home"))

    probs, preds, raw, thr = predict(text)

    positives = [lab for lab in LABELS if int(preds.get(lab, 0)) == 1]
    if positives:
        xai = explain_positive_labels(
            text=text,
            positives=positives,
            top_k=int(current_app.config.get("TOP_TOKENS", 8)),
            max_length=int(current_app.config.get("MAX_LENGTH", 128)),
        )
    else:
        best = max(LABELS, key=lambda l: float(probs.get(l, 0.0)))
        xai = explain_positive_labels(
            text=text,
            positives=[best],
            top_k=int(current_app.config.get("TOP_TOKENS", 8)),
            max_length=int(current_app.config.get("MAX_LENGTH", 128)),
        )

    insert_request(
        user_text=text,
        probs_json=json.dumps(probs),
        preds_json=json.dumps(preds),
        xai_json=json.dumps(xai),
    )
    return render_template("user_submitted.html")