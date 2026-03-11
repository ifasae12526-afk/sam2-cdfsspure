#!/usr/bin/env python3
"""
Generate paper-quality visualizations for SAM2-UNet CD-FSS.

Each query creates a subfolder in gambarforpaper/ with individual images:
  a_query_image.png         - (a) Query image
  b_ground_truth.png        - (b) Ground truth mask
  d_pred_1shot.png          - (d) SAM2 U-Net 1-shot prediction
  f_pred_5shot.png          - (f) SAM2 U-Net 5-shot prediction
  support_1shot_img_N.png   - 1-shot support image(s)
  support_1shot_mask_N.png  - 1-shot support mask(s)
  support_5shot_img_N.png   - 5-shot support image(s)
  support_5shot_mask_N.png  - 5-shot support mask(s)

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


def save_single_image(img: np.ndarray, path: str, dpi: int, is_mask: bool = False) -> None:
    """Save a single image/mask as a standalone high-res file (no axes, no borders)."""
    h, w = img.shape[:2]
    fig, ax = plt.subplots(1, 1, figsize=(w / 100, h / 100), dpi=dpi)
    ax.axis("off")
    if is_mask:
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    else:
        ax.imshow(img, interpolation="lanczos")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


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

        # --- Save each image individually into subfolder ---
        query_dir = os.path.join(args.outdir, f"query_{idx:03d}")
        os.makedirs(query_dir, exist_ok=True)

        # (a) Query image
        save_single_image(query_img, os.path.join(query_dir, "a_query_image.png"), args.dpi)
        # (b) Ground truth mask
        save_single_image(gt_mask, os.path.join(query_dir, "b_ground_truth.png"), args.dpi, is_mask=True)
        # (d) 1-shot prediction
        save_single_image(pred_1, os.path.join(query_dir, "d_pred_1shot.png"), args.dpi, is_mask=True)
        # (f) 5-shot prediction
        save_single_image(pred_5, os.path.join(query_dir, "f_pred_5shot.png"), args.dpi, is_mask=True)

        # Support images & masks (1-shot)
        for k, (simg, smsk) in enumerate(zip(sup_imgs_1, sup_masks_1)):
            save_single_image(simg, os.path.join(query_dir, f"support_1shot_img_{k+1}.png"), args.dpi)
            save_single_image(smsk, os.path.join(query_dir, f"support_1shot_mask_{k+1}.png"), args.dpi, is_mask=True)

        # Support images & masks (5-shot)
        for k, (simg, smsk) in enumerate(zip(sup_imgs_5, sup_masks_5)):
            save_single_image(simg, os.path.join(query_dir, f"support_5shot_img_{k+1}.png"), args.dpi)
            save_single_image(smsk, os.path.join(query_dir, f"support_5shot_mask_{k+1}.png"), args.dpi, is_mask=True)

        print(f"[{idx+1}] Saved to: {query_dir}/")
        count += 1

    print(f"\n[DONE] {count} figures saved to '{args.outdir}/'")


if __name__ == "__main__":
    main()
