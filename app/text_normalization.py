# app/text_normalization.py
import re
from typing import Optional

try:
    import spacy
except Exception:
    spacy = None

_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        if spacy is None:
            raise RuntimeError("spaCy is not installed. Install spacy + en_core_web_sm or set lemmatize=False.")
        _nlp = spacy.load("en_core_web_sm")
    return _nlp

def preprocess_text(text: str, lemmatize: bool = True) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"([!?.,])\1{1,}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    if text == "":
        return ""

    if not lemmatize:
        return text

    extra = {"however", "also", "therefore", "moreover", "thus"}
    keep = {"no", "not", "never", "n't"}

    doc = get_nlp()(text)
    lemmas = []
    for token in doc:
        if token.ent_type_ in ("GPE", "LOC", "FAC"):
            continue
        if token.is_space or token.is_punct:
            continue
        if (token.is_stop and token.text.lower() not in keep) or (token.text.lower() in extra):
            continue
        lemmas.append(token.lemma_.lower())

    return " ".join(lemmas)