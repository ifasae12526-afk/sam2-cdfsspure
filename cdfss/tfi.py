# cdfss/tfi.py
from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

TFILayers = Literal["low", "mid", "high", "all"]


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def default_tfi_lr(benchmark: str) -> float:
    """
    PATNet paper (Sec 6.3) uses:
      - 1e-3 for Deepglobe & ISIC
      - 5e-5 for Chest X-ray & FSS-1000
    We'll map:
      deepglobe/isic -> 1e-3
      fss/lung       -> 5e-5
      others         -> 1e-3
    """
    if benchmark in {"deepglobe", "isic"}:
        return 1e-3
    if benchmark in {"fss", "lung"}:
        return 5e-5
    if benchmark in {"chick"}:
        return 5e-4
    return 1e-3


def resize_mask(mask: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    """(B,H,W) or (B,1,H,W) -> (B,1,Hf,Wf)"""
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.float()
    if mask.max() > 1.5:  # 0/255 style
        mask = (mask > 127).float()
    return F.interpolate(mask, size=size_hw, mode="nearest")


def masked_avg_pool(feat: torch.Tensor, mask_rs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """feat: (B,C,H,W), mask_rs: (B,1,H,W) -> (B,C)"""
    masked = feat * mask_rs
    denom = mask_rs.sum(dim=(2, 3)) + eps  # (B,1)
    return masked.sum(dim=(2, 3)) / denom  # (B,C)


def soft_masked_avg_pool_from_prob(
    feat: torch.Tensor,
    prob_rs: torch.Tensor,
    tau: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Query prototype from predicted prob map:
      p_hat = sum(F * prob * 1[prob>=tau]) / sum(prob)
    feat: (B,C,H,W), prob_rs: (B,1,H,W) in [0,1]
    """
    hard = (prob_rs >= tau).float()

    # Fallback kalau tidak ada pixel yg lolos threshold (biar tidak "mati total")
    hard_sum = hard.sum(dim=(2, 3), keepdim=True)  # (B,1,1,1)
    hard = torch.where(hard_sum > 0, hard, torch.ones_like(hard))

    num = (feat * prob_rs * hard).sum(dim=(2, 3))  # (B,C)
    den = prob_rs.sum(dim=(2, 3)) + eps            # (B,1)
    return num / den


@torch.no_grad()
def precompute_tfi_protos(
    model: nn.Module,
    query_img: torch.Tensor,
    support_imgs: torch.Tensor,
    support_masks: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Precompute fixed (no-grad) features & support prototypes from *pre-PAT* projected features.

    Returns:
      q_feats_lvls: [q2,q3,q4], each (B,C,Hl,Wl) detached
      s_proto_lvls: [p_s2,p_s3,p_s4], each (B,C) detached (averaged across shots)
    """
    m = unwrap_model(model)

    # Ensure K-shot shape
    if support_imgs.dim() == 4:
        support_imgs = support_imgs.unsqueeze(1)
    if support_masks.dim() == 3:
        support_masks = support_masks.unsqueeze(1)

    # Query features (pre-PAT)
    _, q2, q3, q4 = m._project(m._encode(query_img))
    q_feats_lvls = [q2, q3, q4]

    b, k = support_imgs.shape[:2]
    dims = [q2.size(1), q3.size(1), q4.size(1)]
    proto_accum = [
        torch.zeros((b, d), device=query_img.device, dtype=q2.dtype) for d in dims
    ]

    for si in range(k):
        _, s2, s3, s4 = m._project(m._encode(support_imgs[:, si]))
        s_feats = [s2, s3, s4]
        s_mask = support_masks[:, si]

        for li, s_feat in enumerate(s_feats):
            m_rs = resize_mask(s_mask, s_feat.shape[-2:])
            proto_accum[li] += masked_avg_pool(s_feat, m_rs)

    s_proto_lvls = [p / float(k) for p in proto_accum]
    return [x.detach() for x in q_feats_lvls], [x.detach() for x in s_proto_lvls]


def tfi_kl_loss(
    prob_fg: torch.Tensor,
    q_feats_lvls: Sequence[torch.Tensor],
    s_proto_lvls: Sequence[torch.Tensor],
    tau: float = 0.5,
) -> torch.Tensor:
    """
    Compute L_kl = sum_l D_KL(p_s || p_q_hat) with prototype softmax over channel dim.
    prob_fg: (B,1,H,W)
    """
    assert len(q_feats_lvls) == len(s_proto_lvls)
    loss = prob_fg.new_tensor(0.0)

    for feat_l, proto_s_l in zip(q_feats_lvls, s_proto_lvls):
        prob_l = F.interpolate(prob_fg, size=feat_l.shape[-2:], mode="bilinear", align_corners=False)
        proto_q_l = soft_masked_avg_pool_from_prob(feat_l, prob_l, tau=tau)

        # D_KL(p_s || p_q)  -> kl_div(log p_q, p_s)
        loss_l = F.kl_div(
            F.log_softmax(proto_q_l, dim=-1),
            F.softmax(proto_s_l, dim=-1),
            reduction="batchmean",
        )
        loss = loss + loss_l

    return loss


def _select_pat_layer_indices(layers: TFILayers, num_levels: int) -> List[int]:
    if layers == "all":
        return list(range(num_levels))
    if layers == "low":
        return [0]
    if layers == "mid":
        return [1] if num_levels > 1 else [0]
    if layers == "high":
        return [num_levels - 1]
    raise ValueError(f"Unknown layers={layers}")


def tfi_adapt_episode(
    model: nn.Module,
    query_img: torch.Tensor,
    support_imgs: torch.Tensor,
    support_masks: torch.Tensor,
    *,
    benchmark: str,
    steps: int = 50,
    lr: Optional[float] = None,
    tau: float = 0.5,
    layers: TFILayers = "low",
) -> Dict[str, float]:
    """
    Run TFI adaptation on a single episode (B=1 recommended).
    Update only model.pat.reference_layers (selected by `layers`).
    """
    m = unwrap_model(model)
    if not hasattr(m, "pat"):
        raise AttributeError("Model has no attribute `pat` (PATAnchorTransform).")

    layer_ids = _select_pat_layer_indices(layers, num_levels=len(m.pat.reference_layers))

    # Freeze all params, enable only selected anchor params
    for p in m.parameters():
        p.requires_grad = False

    optim_params: List[torch.nn.Parameter] = []
    for li in layer_ids:
        ref_layer = m.pat.reference_layers[li]
        for name, p in ref_layer.named_parameters():
            if "weight" in name:
                p.requires_grad = True
                optim_params.append(p)
            else:
                p.requires_grad = False

    if lr is None:
        lr = default_tfi_lr(benchmark)

    optimizer = torch.optim.Adam(optim_params, lr=lr)

    # Precompute fixed support prototypes & query feats (no graph)
    q_feats_lvls, s_proto_lvls = precompute_tfi_protos(m, query_img, support_imgs, support_masks)

    m.eval()  # keep BN frozen
    loss_last = 0.0
    loss_sum = 0.0

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)

        logits = model(query_img, support_imgs, support_masks)  # (B,2,H,W)
        prob_fg = torch.softmax(logits, dim=1)[:, 1:2]          # (B,1,H,W)

        loss = tfi_kl_loss(prob_fg, q_feats_lvls, s_proto_lvls, tau=tau)
        loss.backward()
        optimizer.step()

        loss_last = float(loss.detach().item())
        loss_sum += loss_last

    return {
        "loss_last": loss_last,
        "loss_avg": loss_sum / max(1, steps),
        "lr": float(lr),
        "steps": float(steps),
    }