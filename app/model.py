import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import expit
from .labels import LABELS

_MODEL = None
_TOKENIZER = None
_THRESHOLDS = None

def load_model(model_dir: str, thresholds_path: str):
    global _MODEL, _TOKENIZER, _THRESHOLDS

    if _MODEL is not None:
        return

    _TOKENIZER = AutoTokenizer.from_pretrained(model_dir)
    _MODEL = AutoModelForSequenceClassification.from_pretrained(model_dir)
    _MODEL.eval()

    thr = np.load(thresholds_path)
    if thr.shape[0] != len(LABELS):
        raise ValueError(f"Thresholds length {thr.shape[0]} != labels {len(LABELS)}")
    _THRESHOLDS = thr.astype(float)

def predict(text: str, max_length: int = 256):
    """
    Returns:
      probs: dict label->prob
      preds: dict label->0/1 using per-label thresholds
      raw:   dict label->logit (optional debugging)
    """
    if _MODEL is None or _TOKENIZER is None or _THRESHOLDS is None:
        raise RuntimeError("Model not loaded. Call load_model() at startup.")

    with torch.no_grad():
        enc = _TOKENIZER(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        out = _MODEL(**enc)
        logits = out.logits.squeeze(0).cpu().numpy()

    probs_arr = expit(logits)
    preds_arr = (probs_arr >= _THRESHOLDS).astype(int)

    probs = {lab: float(p) for lab, p in zip(LABELS, probs_arr)}
    preds = {lab: int(y) for lab, y in zip(LABELS, preds_arr)}
    raw = {lab: float(l) for lab, l in zip(LABELS, logits)}
    thr = {lab: float(t) for lab, t in zip(LABELS, _THRESHOLDS)}
    return probs, preds, raw, thr

def get_tokenizer_and_model():
    return _TOKENIZER, _MODEL
