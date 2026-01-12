import json
from flask import Blueprint, current_app, render_template, request, redirect, url_for, flash
from .model import load_model, predict
from .xai import token_importance
from .db import insert_request
from .labels import LABELS

bp_user = Blueprint("user", __name__)

@bp_user.before_app_request
def _load_once():
    # safe: will no-op if already loaded
    load_model(
        current_app.config["MODEL_DIR"],
        current_app.config["THRESHOLDS_PATH"],
    )

@bp_user.get("/")
def home():
    return render_template("user_home.html", labels=LABELS)

@bp_user.post("/classify")
def classify():
    text = (request.form.get("text") or "").strip()
    if not text:
        flash("Please paste a message first.")
        return redirect(url_for("user.home"))

    if len(text) > current_app.config["MAX_TEXT_CHARS"]:
        flash(f"Text too long. Limit is {current_app.config['MAX_TEXT_CHARS']} characters.")
        return redirect(url_for("user.home"))

    probs, preds, raw, thr = predict(text)

    # XAI only for labels predicted positive (or top prob if none)
    positives = [l for l in LABELS if preds[l] == 1]
    xai = {}

    if positives:
        for lab in positives:
            xai[lab] = token_importance(
                text, lab,
                top_k=current_app.config["TOP_TOKENS"]
            )
    else:
        # show explanation for best label by probability
        best = max(LABELS, key=lambda l: probs[l])
        xai[best] = token_importance(text, best, top_k=current_app.config["TOP_TOKENS"])

    req_id = insert_request(
        user_text=text,
        probs_json=json.dumps(probs),
        preds_json=json.dumps(preds),
    )

    return render_template(
        "user_result.html",
        req_id=req_id,
        text=text,
        probs=probs,
        preds=preds,
        thr=thr,
        xai=xai,
    )
