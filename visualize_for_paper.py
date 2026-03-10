#!/usr/bin/env python3
"""
Generate paper-quality visualizations for SAM2-UNet CD-FSS.

Output per query image (saved to gambarforpaper/):
  Row 1: (a) Query image | (b) Ground truth mask | (d) 1-shot prediction | (f) 5-shot prediction
  Row 2: Support image(s) + Support mask(s)

Usage:
  python visualize_for_paper.py --load <checkpoint.pt> --sam2_ckpt <sam2_hiera_large.pt>
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from cdfss.sam2unet_cdfss_aggressive import SAM2CDFSSConfig, SAM2UNetCDFSSAggressive
from data.dataset import FSSDataset
from common import utils

# ImageNet normalization used by the dataloader
IMG_MEAN = np.array([0.485, 0.456, 0.406])
IMG_STD = np.array([0.229, 0.224, 0.225])


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized (C,H,W) tensor back to (H,W,3) uint8 image."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)  # (H,W,3)
    img = img * IMG_STD + IMG_MEAN
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def logits_to_pred(logits: torch.Tensor, thr: float = 0.5) -> np.ndarray:
    """Convert model logits to binary mask (H,W) numpy."""
    c = logits.shape[1]
    if c == 1:
        mask = (logits.sigmoid() > thr).long().squeeze(1)
    elif c == 2:
        mask = logits.argmax(dim=1).long()
    else:
        mask = logits.argmax(dim=1).long()
    return mask[0].cpu().numpy().astype(np.uint8)


@torch.no_grad()
def predict(model: nn.Module, batch: dict) -> np.ndarray:
    """Run model forward and return predicted binary mask (H,W)."""
    logits = model(batch["query_img"], batch["support_imgs"], batch["support_masks"])
    return logits_to_pred(logits)


def build_figure(
    query_img: np.ndarray,
    gt_mask: np.ndarray,
    pred_1shot: np.ndarray,
    pred_5shot: np.ndarray,
    support_imgs_1shot: list[np.ndarray],
    support_masks_1shot: list[np.ndarray],
    support_imgs_5shot: list[np.ndarray],
    support_masks_5shot: list[np.ndarray],
) -> plt.Figure:
    """Build a paper-quality figure with query, GT, 1-shot, 5-shot, and support."""

    n_sup_1 = len(support_imgs_1shot)
    n_sup_5 = len(support_imgs_5shot)
    # Row 1: 4 panels (query, GT, 1-shot pred, 5-shot pred)
    # Row 2: 1-shot support images + masks
    # Row 3: 5-shot support images + masks
    n_cols = max(4, n_sup_5 * 2)

    fig, axes = plt.subplots(3, n_cols, figsize=(5 * n_cols, 5 * 3))

    # Make sure axes is 2D
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    # Turn off all axes
    for row in axes:
        for ax in row:
            ax.axis("off")

    # --- Row 1: Query, GT, 1-shot pred, 5-shot pred ---
    axes[0, 0].imshow(query_img, interpolation="lanczos")
    axes[0, 0].set_title("(a) Query Image", fontsize=14, fontweight="bold")

    axes[0, 1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0, 1].set_title("(b) Ground Truth", fontsize=14, fontweight="bold")

    axes[0, 2].imshow(pred_1shot, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0, 2].set_title("(d) 1-Shot Prediction", fontsize=14, fontweight="bold")

    axes[0, 3].imshow(pred_5shot, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0, 3].set_title("(f) 5-Shot Prediction", fontsize=14, fontweight="bold")

    # --- Row 2: 1-shot support images + masks ---
    col = 0
    for i in range(n_sup_1):
        axes[1, col].imshow(support_imgs_1shot[i], interpolation="lanczos")
        axes[1, col].set_title(f"Support Img {i+1}", fontsize=12)
        col += 1
        axes[1, col].imshow(support_masks_1shot[i], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[1, col].set_title(f"Support Mask {i+1}", fontsize=12)
        col += 1
    # Label for the row
    axes[1, 0].set_ylabel("1-Shot Support", fontsize=10, fontweight="bold", rotation=90, labelpad=15)
    axes[1, 0].axis("on")
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])

    # --- Row 3: 5-shot support images + masks ---
    col = 0
    for i in range(n_sup_5):
        axes[2, col].imshow(support_imgs_5shot[i], interpolation="lanczos")
        axes[2, col].set_title(f"Support Img {i+1}", fontsize=12)
        col += 1
        axes[2, col].imshow(support_masks_5shot[i], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[2, col].set_title(f"Support Mask {i+1}", fontsize=12)
        col += 1
    axes[2, 0].set_ylabel("5-Shot Support", fontsize=10, fontweight="bold", rotation=90, labelpad=15)
    axes[2, 0].axis("on")
    axes[2, 0].set_xticks([])
    axes[2, 0].set_yticks([])

    fig.tight_layout(pad=1.0)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser("Visualize SAM2 U-Net CD-FSS for paper figures")
    parser.add_argument("--load", type=str, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--sam2_cfg", type=str, default="sam2_hiera_l.yaml")
    parser.add_argument("--sam2_ckpt", type=str,
                        default=r"F:\CCTV\SAM-2CDFSS\model\sam2_hiera_large.pt")
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--attn_heads", type=int, default=8)
    parser.add_argument("--num_fg_tokens", type=int, default=32)
    parser.add_argument("--benchmark", type=str, default="chick",
                        choices=["pascal", "fss", "deepglobe", "isic", "lung", "chick"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--datapath", type=str, default="./dataset")
    parser.add_argument("--outdir", type=str, default="gambarforpaper")
    parser.add_argument("--max_images", type=int, default=-1,
                        help="Max number of query images to visualize. -1 = all.")
    parser.add_argument("--dpi", type=int, default=300, help="Output image DPI")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    utils.fix_randseed(0)

    # --- Build model ---
    cfg = SAM2CDFSSConfig(
        sam2_model_cfg=args.sam2_cfg,
        sam2_checkpoint=None if args.sam2_ckpt == "" else args.sam2_ckpt,
        embed_dim=args.embed_dim,
        attn_heads=args.attn_heads,
        num_fg_tokens=args.num_fg_tokens,
    )
    model = SAM2UNetCDFSSAggressive(cfg)
    if torch.cuda.is_available():
        model = model.cuda()

    state = torch.load(args.load, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[INFO] Model loaded from {args.load}")

    # --- Build dataloaders for 1-shot and 5-shot ---
    FSSDataset.initialize(img_size=args.img_size, datapath=args.datapath)

    dl_1shot = FSSDataset.build_dataloader(
        args.benchmark, bsz=1, nworker=0, fold=args.fold, split="test", shot=1
    )
    dl_5shot = FSSDataset.build_dataloader(
        args.benchmark, bsz=1, nworker=0, fold=args.fold, split="test", shot=5
    )

    print(f"[INFO] 1-shot episodes: {len(dl_1shot)}, 5-shot episodes: {len(dl_5shot)}")

    count = 0
    for idx, (batch_1, batch_5) in enumerate(zip(dl_1shot, dl_5shot)):
        if 0 < args.max_images <= count:
            break

        if torch.cuda.is_available():
            batch_1 = utils.to_cuda(batch_1)
            batch_5 = utils.to_cuda(batch_5)

        # --- Predictions ---
        pred_1 = predict(model, batch_1)
        pred_5 = predict(model, batch_5)

        # --- Denormalize query image (same for both shots) ---
        query_img = denormalize(batch_1["query_img"][0])
        gt_mask = batch_1["query_mask"][0].cpu().numpy().astype(np.uint8)

        # --- Support images/masks for 1-shot ---
        s_imgs_1 = batch_1["support_imgs"][0]  # (K,3,H,W)
        s_masks_1 = batch_1["support_masks"][0]  # (K,H,W)
        if s_imgs_1.dim() == 3:  # single support -> (3,H,W)
            s_imgs_1 = s_imgs_1.unsqueeze(0)
            s_masks_1 = s_masks_1.unsqueeze(0)

        sup_imgs_1 = [denormalize(s_imgs_1[k]) for k in range(s_imgs_1.shape[0])]
        sup_masks_1 = [s_masks_1[k].cpu().numpy().astype(np.uint8) for k in range(s_masks_1.shape[0])]

        # --- Support images/masks for 5-shot ---
        s_imgs_5 = batch_5["support_imgs"][0]  # (K,3,H,W)
        s_masks_5 = batch_5["support_masks"][0]  # (K,H,W)
        if s_imgs_5.dim() == 3:
            s_imgs_5 = s_imgs_5.unsqueeze(0)
            s_masks_5 = s_masks_5.unsqueeze(0)

        sup_imgs_5 = [denormalize(s_imgs_5[k]) for k in range(s_imgs_5.shape[0])]
        sup_masks_5 = [s_masks_5[k].cpu().numpy().astype(np.uint8) for k in range(s_masks_5.shape[0])]

        # --- Build and save figure ---
        fig = build_figure(
            query_img, gt_mask, pred_1, pred_5,
            sup_imgs_1, sup_masks_1,
            sup_imgs_5, sup_masks_5,
        )

        outpath_png = os.path.join(args.outdir, f"query_{idx:03d}.png")
        outpath_pdf = os.path.join(args.outdir, f"query_{idx:03d}.pdf")
        fig.savefig(outpath_png, dpi=args.dpi, bbox_inches="tight", pad_inches=0.1)
        fig.savefig(outpath_pdf, dpi=args.dpi, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        print(f"[{idx+1}] Saved: {outpath_png} + {outpath_pdf}")
        count += 1

    print(f"\n[DONE] {count} figures saved to '{args.outdir}/'")


if __name__ == "__main__":
    main()
