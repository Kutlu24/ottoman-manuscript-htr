from __future__ import annotations

import torch
import torch.nn.functional as F


def recognition_loss(
    log_probs: torch.Tensor,  # (B, T, vocab) log-probabilities
    targets: torch.Tensor,  # (sum(target_lengths),)
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
) -> torch.Tensor:
    """CTC loss over the fused character-membership sequence."""
    log_probs = log_probs.transpose(0, 1)  # CTCLoss wants (T, B, vocab)
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, zero_infinity=True)


def fuzzy_agreement_loss(source_memberships: list[torch.Tensor]) -> torch.Tensor:
    """Penalizes disagreement between evidence sources at timesteps where at least one source is
    confident -- discourages the fused layer from silently overriding a strongly-committed
    source instead of resolving the disagreement through training signal."""
    losses = []
    for i in range(len(source_memberships)):
        for j in range(i + 1, len(source_memberships)):
            a, b = source_memberships[i], source_memberships[j]
            confidence = torch.maximum(a.max(dim=-1).values, b.max(dim=-1).values)
            disagreement = F.kl_div(a.clamp_min(1e-8).log(), b, reduction="none").sum(dim=-1)
            losses.append((confidence * disagreement).mean())
    return torch.stack(losses).mean() if losses else torch.tensor(0.0)


def geometric_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """InfoNCE-style loss pulling geometric embeddings of the same ground-truth character label
    together (and pushing different labels apart) across scribes/samples -- tests the "same
    letter, different hand, nearby in embedding space" idea directly. Needs per-character
    embeddings + labels, i.e. char-level annotation (see docs/architecture.md)."""
    embeddings = F.normalize(embeddings, dim=-1)
    sim = embeddings @ embeddings.T / temperature
    same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
    same_label.fill_diagonal_(False)
    log_prob = F.log_softmax(sim, dim=-1)
    pos_count = same_label.sum(dim=-1).clamp_min(1)
    loss = -(log_prob * same_label).sum(dim=-1) / pos_count
    valid = same_label.sum(dim=-1) > 0
    return loss[valid].mean() if valid.any() else torch.tensor(0.0)
