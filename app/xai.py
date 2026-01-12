import torch
import numpy as np
from captum.attr import IntegratedGradients

def explain_positive_labels(text: str, positives: list[str], top_k=10, max_length=128):
    from .model import get_tokenizer_and_model
    from .text_normalization import preprocess_text
    from .labels import LABELS

    tokenizer, model = get_tokenizer_and_model()
    model.eval()

    text_norm = preprocess_text(text, lemmatize=False)

    enc = tokenizer(
        text_norm,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True,
    )

    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]
    offsets = enc["offset_mapping"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    embed = model.get_input_embeddings()

    def forward_logits(inputs_embeds, attention_mask):
        out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return out.logits  # [1, num_labels]

    with torch.no_grad():
        inp_emb = embed(input_ids)
        base_emb = torch.zeros_like(inp_emb)

    ig = IntegratedGradients(forward_logits)

    def aggregate_to_words(attr_scores):
        words, scores = [], []
        cur_word, cur_score = "", 0.0

        for tok, (s, e), sc in zip(tokens, offsets, attr_scores):
            if s == 0 and e == 0:
                continue

            piece = tok
            if piece.startswith("##"):
                cur_word += piece[2:]
                cur_score += float(sc)
                continue

            if cur_word:
                words.append(cur_word)
                scores.append(cur_score)

            cur_word = piece
            cur_score = float(sc)

        if cur_word:
            words.append(cur_word)
            scores.append(cur_score)

        return words, scores

    out = {}
    pos_set = set(positives or [])

    for j, lab in enumerate(LABELS):
        if lab not in pos_set:
            continue

        attributions = ig.attribute(
            inputs=inp_emb,
            baselines=base_emb,
            additional_forward_args=(attn,),
            target=j,     # ✅ label-specific
            n_steps=24,
        )

        tok_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()

        # ✅ why YES: only positive contributions
        tok_scores = np.clip(tok_scores, 0, None)

        mx = float(tok_scores.max()) if tok_scores.size else 0.0
        if mx > 0:
            tok_scores = tok_scores / mx

        words, scores = aggregate_to_words(tok_scores)

        pairs = [(w, float(s)) for w, s in zip(words, scores) if s > 0]
        pairs.sort(key=lambda x: x[1], reverse=True)
        top = pairs[:top_k]

        out[lab] = {
            "words": [w for w, _ in top],
            "scores": [s for _, s in top],
            "all_words": words,
            "all_scores": [float(s) for s in scores],
        }

    return out