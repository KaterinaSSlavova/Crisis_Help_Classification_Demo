import torch
import numpy as np
from .model import get_tokenizer_and_model
from .labels import LABELS

def token_importance(text: str, label: str, top_k: int = 12, max_length: int = 256):
    """
    Returns list of (token_str, score) for top_k tokens.
    score is normalized absolute gradient*embedding magnitude proxy.
    """
    if label not in LABELS:
        raise ValueError("Unknown label")

    tokenizer, model = get_tokenizer_and_model()
    model.eval()

    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # Get embeddings and enable grad
    emb_layer = model.get_input_embeddings()
    embeds = emb_layer(input_ids)
    embeds.retain_grad()
    embeds.requires_grad_(True)

    # Forward using inputs_embeds
    out = model(inputs_embeds=embeds, attention_mask=attention_mask)
    logits = out.logits  # [1, num_labels]
    idx = LABELS.index(label)
    target_logit = logits[0, idx]

    model.zero_grad(set_to_none=True)
    if embeds.grad is not None:
        embeds.grad.zero_()

    target_logit.backward()

    # gradient × embedding magnitude proxy
    grad = embeds.grad[0]             # [seq, dim]
    emb = embeds.detach()[0]          # [seq, dim]
    scores = torch.sum(torch.abs(grad * emb), dim=-1)  # [seq]

    scores = scores.cpu().numpy()
    ids = input_ids[0].cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(ids)
    # remove special tokens
    filtered = []
    for tok, sc, m in zip(tokens, scores, attention_mask[0].cpu().numpy()):
        if m == 0:
            continue
        if tok in (tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token):
            continue
        filtered.append((tok, float(sc)))

    if not filtered:
        return []

    # normalize to 0..1
    arr = np.array([s for _, s in filtered], dtype=float)
    arr = arr / (arr.max() + 1e-12)
    filtered = [(t, float(v)) for (t, _), v in zip(filtered, arr)]

    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered[:top_k]
