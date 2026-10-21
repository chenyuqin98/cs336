"""Autoregressive sampling with temperature and nucleus (top-p) filtering."""

import torch

from .my_torch import softmax


@torch.no_grad()
def generate(model, token_ids, max_new_tokens, context_length, temperature=1.0, top_p=1.0, eos_token_id=None):
    if temperature < 0 or not 0 < top_p <= 1 or max_new_tokens < 0 or context_length <= 0:
        raise ValueError("Invalid sampling parameters")
    if token_ids.ndim != 2 or token_ids.shape[1] == 0:
        raise ValueError("Expected a nonempty (batch, sequence) prompt")
    was_training = model.training
    model.eval()
    finished = torch.zeros(token_ids.shape[0], dtype=torch.bool, device=token_ids.device)
    try:
        for _ in range(max_new_tokens):
            logits = model(token_ids[:, -context_length:])[:, -1, :]
            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = softmax()(logits / temperature, dim=-1)
                sorted_probs, indices = probs.sort(descending=True, dim=-1)
                # Retain the first token crossing the probability threshold.
                remove = sorted_probs.cumsum(-1) - sorted_probs >= top_p
                sorted_probs.masked_fill_(remove, 0)
                sorted_probs /= sorted_probs.sum(-1, keepdim=True)
                sampled = torch.multinomial(sorted_probs, 1)
                next_token = indices.gather(-1, sampled)
            if eos_token_id is not None:
                next_token[finished] = eos_token_id
                finished |= next_token.squeeze(-1) == eos_token_id
            token_ids = torch.cat((token_ids, next_token), dim=-1)
            if finished.all():
                break
    finally:
        model.train(was_training)
    return token_ids
