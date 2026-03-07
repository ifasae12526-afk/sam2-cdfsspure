#!/usr/bin/env python3
"""
Train SAM2UNetCDFSSAggressive — Pure Pascal training.

Workflow:
  Meta-train on PASCAL-5i (episodic few-shot segmentation).
  Setelah training selesai, gunakan test.py untuk evaluasi 1-way 5-shot
  pada dataset chick (CCTV).

Notes
-----
- Model expects episode batches with:
    query_img:    (B,3,H,W)
    support_imgs: (B,K,3,H,W)
    support_masks:(B,K,H,W)
    query_mask:   (B,H,W)  (0/1)
- Output is 2-channel logits (bg, fg).

"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from cdfss.sam2unet_cdfss_aggressive import SAM2CDFSSConfig, SAM2UNetCDFSSAggressive

from data.dataset import FSSDataset
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils


def build_model(args: argparse.Namespace) -> nn.Module:
    cfg = SAM2CDFSSConfig(
        sam2_model_cfg=args.sam2_cfg,
        sam2_checkpoint=None if args.sam2_ckpt == "" else args.sam2_ckpt,
        embed_dim=args.embed_dim,
        attn_heads=args.attn_heads,
        num_fg_tokens=args.num_fg_tokens,
    )
    model = SAM2UNetCDFSSAggressive(cfg)

    # Log a concise config dump (helps later)
    Logger.info("Model cfg: " + str(asdict(cfg)))

    if torch.cuda.is_available():
        model = model.cuda()

    if args.dp and torch.cuda.device_count() > 1:
        Logger.info(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    return model


def dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0,
              ignore_index: int = 255) -> torch.Tensor:
    """Soft Dice loss for 2-class segmentation (bg/fg).

    Dice is inherently balanced across classes and does not suffer from
    the class-imbalance problem that causes all-background collapse.

    Args:
        logits: (B,2,H,W) raw model output
        target: (B,H,W) long with values {0,1,255}
        smooth: Laplace smoothing to avoid division by zero
        ignore_index: label value to ignore (e.g. 255 for boundaries)

    Returns:
        Scalar dice loss (1 - mean_dice).
    """
    probs = F.softmax(logits, dim=1)            # (B,2,H,W)
    fg_prob = probs[:, 1]                        # (B,H,W) – foreground probability
    fg_gt = (target == 1).float()                # (B,H,W)
    valid = (target != ignore_index).float()     # (B,H,W)

    # Only compute dice over valid (non-ignored) pixels
    fg_prob = fg_prob * valid
    fg_gt = fg_gt * valid

    intersection = (fg_prob * fg_gt).sum(dim=(1, 2))          # (B,)
    cardinality = fg_prob.sum(dim=(1, 2)) + fg_gt.sum(dim=(1, 2))  # (B,)
    dice = (2.0 * intersection + smooth) / (cardinality + smooth)  # (B,)
    return 1.0 - dice.mean()


def focal_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha_fg: float = 0.75,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Focal cross-entropy loss – down-weights easy (background) pixels.

    Args:
        logits: (B,2,H,W)
        target: (B,H,W) long {0,1,255}
        gamma:  focusing parameter (higher -> more focus on hard examples)
        alpha_fg: weight for the foreground class (>0.5 = boost fg)
    """
    # Per-pixel CE (unreduced)
    ce = F.cross_entropy(logits, target, ignore_index=ignore_index, reduction="none")  # (B,H,W)

    # Per-pixel pt (probability of the correct class)
    probs = F.softmax(logits, dim=1)  # (B,2,H,W)
    num_classes = logits.shape[1]
    # Clamp to valid class range to avoid gather OOB (ignore pixels have target=255)
    target_safe = target.clamp(0, num_classes - 1)
    pt = probs.gather(1, target_safe.unsqueeze(1)).squeeze(1)  # (B,H,W)

    # Focal modulation
    focal_weight = (1.0 - pt) ** gamma

    # Alpha weighting: give foreground pixels more weight
    alpha = torch.where(target == 1, alpha_fg, 1.0 - alpha_fg)  # (B,H,W)

    # Mask out ignored pixels
    valid = (target != ignore_index).float()
    loss = (alpha * focal_weight * ce * valid).sum() / valid.sum().clamp(min=1.0)
    return loss


def compute_ce_loss(logits: torch.Tensor, batch: dict) -> torch.Tensor:
    """Combined loss: Focal-CE + Dice.

    Dice loss handles extreme class imbalance (e.g. small foreground objects)
    by treating each class equally regardless of pixel count.
    Focal-CE down-weights easy background pixels so the gradient signal is
    dominated by hard foreground pixels.

    logits: (B,2,H,W)
    query_mask: (B,H,W) float {0,1} or long
    query_ignore_idx: (B,H,W) float {0,1} (optional) -> ignored as 255
    """
    target = batch["query_mask"]
    if not torch.is_floating_point(target):
        target = target.float()
    target = target.long()

    ignore = batch.get("query_ignore_idx", None)
    if ignore is not None:
        ignore = ignore.bool()
        target = target.clone()
        target[ignore] = 255

    # Focal CE (handles class imbalance via alpha + focal modulation)
    loss_focal = focal_ce_loss(logits, target, gamma=2.0, alpha_fg=0.75)

    # Dice loss (inherently balanced, prevents all-background collapse)
    loss_dice = dice_loss(logits, target)

    # Combined: both terms are roughly in [0, 1] range
    return loss_focal + loss_dice


def run_epoch(
    epoch: int,
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer | None,
    *,
    training: bool,
    amp: bool,
    scaler: GradScaler | None,
    aux_weight: float,
    write_batch_idx: int,
    grad_clip: float,
    label: str = "",
) -> tuple[float, float, float]:
    """
    Returns:
        avg_loss, miou, fb_iou
    """
    utils.fix_randseed(None if training else 0, deterministic=not training)

    if training:
        model.train()
    else:
        model.eval()

    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch)

        support_imgs = batch["support_imgs"]
        support_masks = batch["support_masks"]

        with autocast(enabled=amp):
            # Return aux for optional deep supervision
            out = model(batch["query_img"], support_imgs, support_masks, return_aux=(aux_weight > 0))
            if aux_weight > 0:
                logits, aux = out
            else:
                logits = out
                aux = None

            loss = compute_ce_loss(logits, batch)

            if aux is not None:
                loss_a = compute_ce_loss(aux["logit_a"], batch)
                loss_b = compute_ce_loss(aux["logit_b"], batch)
                loss = loss + aux_weight * (loss_a + loss_b)

        if training:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

            # Guard: skip update if loss non-finite
            if not torch.isfinite(loss):
                Logger.info(f"[WARN] non-finite loss at epoch={epoch} batch={idx}, skip step")
                continue
            if amp:
                assert scaler is not None
                scaler.scale(loss).backward()
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        pred_mask = logits.argmax(dim=1)  # (B,H,W)

        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        average_meter.update(area_inter, area_union, batch["class_id"], loss.detach().clone())

        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=write_batch_idx)

    tag = label if label else ("Training" if training else "Validation")
    average_meter.write_result(tag, epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()
    return float(avg_loss), float(miou), float(fb_iou)


def main() -> None:
    parser = argparse.ArgumentParser("SAM2UNet CD-FSS — Pure Pascal Training")

    # Model / SAM2
    parser.add_argument("--sam2_cfg", type=str, default="sam2_hiera_l.yaml")
    parser.add_argument("--sam2_ckpt", type=str, default="", help="Path to SAM2 checkpoint (.pt). If empty, build_sam2 will init randomly (not recommended).")
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--attn_heads", type=int, default=8)
    parser.add_argument("--num_fg_tokens", type=int, default=32)

    # Data
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--fold", type=int, default=4, choices=[0, 1, 2, 3, 4], help="PASCAL-5i fold. Use 4 to train on all 20 classes.")
    parser.add_argument("--val_fold", type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--benchmark_val", type=str, default="pascal", choices=["pascal", "fss", "deepglobe", "isic", "lung", "chick"])
    parser.add_argument("--split_val", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--train_shot", type=int, default=1)
    parser.add_argument("--val_shot", type=int, default=1)
    parser.add_argument("--datapath_src", type=str, default="./dataset", help="Path to VOCdevkit (contains VOC2012/).")
    parser.add_argument("--datapath_tgt", type=str, default="./dataset", help="Path containing target datasets (chick/, FSS-1000/, etc.).")

    # Optimization
    parser.add_argument("--bsz", type=int, default=2)
    parser.add_argument("--bsz_val", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--niter", type=int, default=2000)
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp mixed precision")
    parser.add_argument("--aux_weight", type=float, default=0.5, help="Deep supervision weight for branch-A/branch-B logits.")
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--warmup_epochs", type=int, default=20, help="Linear warmup epochs")
    parser.add_argument("--cosine_T0", type=int, default=100, help="CosineAnnealingWarmRestarts T_0")
    parser.add_argument("--cosine_Tmult", type=int, default=2, help="CosineAnnealingWarmRestarts T_mult")
    parser.add_argument("--lr_min", type=float, default=1e-6, help="Minimum LR for cosine schedule")

    # Runtime
    parser.add_argument("--nworker", type=int, default=4)
    parser.add_argument("--logpath", type=str, default="sam2unet_cdfss")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--dp", action="store_true", help="Enable nn.DataParallel if multiple GPUs")
    parser.add_argument("--write_batch_idx", type=int, default=50)

    args = parser.parse_args()

    # Fixed: training is always on Pascal
    args.benchmark_train = "pascal"

    # Logger init
    Logger.initialize(args, training=True)
    Evaluator.initialize()

    # Build model
    model = build_model(args)
    Logger.log_params(model)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    Logger.info(f"Trainable params: {sum(p.numel() for p in params):,}")
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scaler = GradScaler(enabled=args.amp)

    # ══════════════════════════════════════════════════════════════════
    # Build dataloaders — Pure Pascal training (NO mixed-domain)
    # ══════════════════════════════════════════════════════════════════

    # ── Train dataset: Pascal only ──
    FSSDataset.initialize(img_size=args.img_size, datapath=args.datapath_src)
    dataloader_trn = FSSDataset.build_dataloader(
        "pascal", args.bsz, args.nworker, args.fold, "trn", shot=args.train_shot
    )

    # ── Val dataset ──
    datapath_val = args.datapath_src if args.benchmark_val == "pascal" else args.datapath_tgt
    FSSDataset.initialize(img_size=args.img_size, datapath=datapath_val)
    dataloader_val = FSSDataset.build_dataloader(
        args.benchmark_val, args.bsz_val, args.nworker, args.val_fold, args.split_val, shot=args.val_shot
    )

    best_val_miou = float("-inf")
    start_time = time.time()

    # LR Scheduler: cosine with warm restarts
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.cosine_T0,
                                            T_mult=args.cosine_Tmult,
                                            eta_min=args.lr_min)

    # ══════════════════════════════════════════════════════════════════
    # Training loop — Pure Pascal
    # ══════════════════════════════════════════════════════════════════
    for epoch in range(args.niter):
        # Linear warmup
        if epoch < args.warmup_epochs:
            warmup_lr = args.lr * (epoch + 1) / args.warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        # ── Train on Pascal ──
        trn_loss, trn_miou, trn_fb_iou = run_epoch(
            epoch, model, dataloader_trn, optimizer,
            training=True, amp=args.amp, scaler=scaler,
            aux_weight=args.aux_weight, write_batch_idx=args.write_batch_idx,
            grad_clip=args.grad_clip,
            label="Train-pascal",
        )

        # Step scheduler after warmup
        if epoch >= args.warmup_epochs:
            scheduler.step()

        # ── Validation ──
        torch.cuda.empty_cache()
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = run_epoch(
                epoch, model, dataloader_val, optimizer=None,
                training=False, amp=args.amp, scaler=None,
                aux_weight=0.0, write_batch_idx=max(1, args.write_batch_idx // 5),
                grad_clip=0.0,
                label=f"Val-{args.benchmark_val}",
            )

        # Save best by mIoU
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            Logger.save_model_miou(model, epoch, val_miou)

        # Periodic snapshot
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(Logger.logpath, f"epoch_{epoch+1:04d}.pt")
            torch.save(model.state_dict(), ckpt_path)
            Logger.info(f"Saved snapshot: {ckpt_path}")

        # Tensorboard
        Logger.tbd_writer.add_scalars("loss", {"trn": trn_loss, "val": val_loss}, epoch)
        Logger.tbd_writer.add_scalars("miou", {"trn": trn_miou, "val": val_miou}, epoch)
        Logger.tbd_writer.add_scalars("fb_iou", {"trn": trn_fb_iou, "val": val_fb_iou}, epoch)
        Logger.tbd_writer.flush()

        elapsed = time.time() - start_time
        if (epoch + 1) % 5 == 0:
            Logger.info(f"[time] elapsed {elapsed/3600:.2f}h | best val mIoU {best_val_miou:.2f}")

    Logger.tbd_writer.close()
    Logger.info("==================== Finished Training ====================")


if __name__ == "__main__":
    main()
