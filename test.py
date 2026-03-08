#!/usr/bin/env python3
"""
Test SAM2UNetCDFSSAggressive — 1-way 5-shot evaluation on chick (CCTV) dataset.

Workflow (default):
  - Model di-train pada Pascal (pure), lalu di-test pada dataset chick.
  - Support Set : 5 gambar (beserta mask) dari train split sebagai contoh
                  visual objek (1-way 5-shot).
  - Query Set   : Sisa gambar di test split sebagai data uji.
  - Alasan 5-shot: gambar CCTV memiliki variasi pencahayaan/sudut pandang,
    5 contoh membantu Cross-Attention menangkap ciri visual objek lebih stabil.

Tambahan:
  --tfi: Task-adaptive Fine-tuning Inference (TFI) untuk fine-tune anchor layers
         (PATM) per-episode/task sebelum prediksi final.

Patch penting:
  - pred_mask aman untuk logits (B,1,H,W) atau (B,2,H,W)
  - debug episode pertama untuk cek data/pred
"""

from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

from cdfss.sam2unet_cdfss_aggressive import SAM2CDFSSConfig, SAM2UNetCDFSSAggressive
from cdfss.tfi import tfi_adapt_episode
from data.dataset import FSSDataset
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def _dummy_loss(device: torch.device) -> torch.Tensor:
    # AverageMeter.write_result() di repo kamu butuh loss_buf tidak kosong.
    return torch.tensor(0.0, device=device)


def logits_to_pred_mask(logits: torch.Tensor, thr: float = 0.5,
                        use_prob_pred: bool = False, fg_thr: float = 0.7) -> torch.Tensor:
    """
    Return pred_mask shape (B,H,W) in {0,1}.

    Supports:
      - logits (B,1,H,W): sigmoid > thr
      - logits (B,2,H,W): either argmax or softmax-threshold on fg prob (class=1)
    """
    if logits.dim() != 4:
        raise ValueError(f"logits must be 4D (B,C,H,W), got {tuple(logits.shape)}")

    c = logits.shape[1]
    if c == 1:
        return (logits.sigmoid() > thr).long().squeeze(1)  # (B,H,W)

    if c == 2:
        if use_prob_pred:
            probs = torch.softmax(logits, dim=1)
            fg_prob = probs[:, 1]  # assume class 1 = foreground
            return (fg_prob > fg_thr).long()
        else:
            return logits.argmax(dim=1).long()

    return logits.argmax(dim=1).long()


def logits_to_prob(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert raw logits to foreground probability map (B,H,W) in [0,1].
    """
    c = logits.shape[1]
    if c == 1:
        return logits.sigmoid().squeeze(1)
    # c >= 2: softmax, take fg channel (class=1)
    return torch.softmax(logits, dim=1)[:, 1]


def compute_mae(prob: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Mean Absolute Error antara probability map dan ground truth.
    prob: (B,H,W) float [0,1]
    gt:   (B,H,W) int {0,1}
    """
    return (prob - gt.float()).abs().mean().item()


def compute_ap_per_image(prob: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Average Precision (AP) per batch, rata-rata per image.
    prob: (B,H,W) float [0,1]
    gt:   (B,H,W) int {0,1}
    Returns mean AP across images in the batch.
    """
    B = prob.shape[0]
    aps = []
    for i in range(B):
        y_true = gt[i].cpu().numpy().ravel().astype(np.int32)
        y_score = prob[i].cpu().numpy().ravel()
        # AP undefined jika GT hanya satu kelas; skip
        if y_true.sum() == 0 or y_true.sum() == y_true.size:
            continue
        aps.append(average_precision_score(y_true, y_score))
    return float(np.mean(aps)) if aps else 0.0


def _debug_first_episode(args: argparse.Namespace, batch: dict, logits: torch.Tensor, pred_mask: torch.Tensor) -> None:
    """
    Print debug info for the first episode only.
    This helps detect: GT empty, support empty, pred always background, logits channel mismatch.
    """
    # Shapes
    q_img = batch.get("query_img")
    s_imgs = batch.get("support_imgs")
    s_masks = batch.get("support_masks")

    Logger.info(f"[DEBUG] query_img shape: {tuple(q_img.shape) if q_img is not None else None}")
    Logger.info(f"[DEBUG] support_imgs shape: {tuple(s_imgs.shape) if s_imgs is not None else None}")
    Logger.info(f"[DEBUG] support_masks shape: {tuple(s_masks.shape) if s_masks is not None else None}")
    Logger.info(f"[DEBUG] logits shape: {tuple(logits.shape)}")

    # Extra: inspect fg prob for 2-channel logits (for prob-threshold pred)
    if logits is not None and logits.dim() == 4 and logits.shape[1] == 2:
        probs = torch.softmax(logits, dim=1)
        fg_prob = probs[:, 1]
        Logger.info(f"[DEBUG] fg_prob mean={fg_prob.mean().item():.4f} "
                    f"min={fg_prob.min().item():.4f} max={fg_prob.max().item():.4f}")

    # GT pixels
    q_mask = batch.get("query_mask", None)
    if q_mask is not None:
        q_fg = int((q_mask > 0).sum().item())
        Logger.info(f"[DEBUG] query_mask fg_pixels: {q_fg}")
    else:
        Logger.info("[DEBUG] query_mask not found in batch keys (check dataset output keys).")

    if s_masks is not None:
        s_fg = int((s_masks > 0).sum().item())
        Logger.info(f"[DEBUG] support_masks total fg_pixels: {s_fg}")

    # Pred pixels
    pred_fg = int((pred_mask > 0).sum().item())
    Logger.info(f"[DEBUG] pred fg_pixels: {pred_fg}")

    # Quick heuristic warnings
    if q_mask is not None and int((q_mask > 0).sum().item()) == 0:
        Logger.info("[DEBUG][WARN] query_mask appears EMPTY (all background). Check mask loading/format.")
    if s_masks is not None and int((s_masks > 0).sum().item()) == 0:
        Logger.info("[DEBUG][WARN] support_masks appear EMPTY. Support sampling / mask paths may be wrong.")
    if pred_fg == 0:
        Logger.info("[DEBUG][WARN] prediction is ALL background. Could be: wrong logits->mask conversion, "
                    "model collapse, or domain shift.")
    if pred_fg == int(pred_mask.numel()):
        Logger.info("[DEBUG][WARN] prediction is ALL foreground. Threshold too low / channel mismatch / collapse.")


@torch.no_grad()
def test_no_tfi(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    args: argparse.Namespace,
) -> Tuple[float, float, float, float]:
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)
    model.eval()

    mae_sum = 0.0
    ap_sum = 0.0
    n_episodes = 0

    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch)

        logits = model(batch["query_img"], batch["support_imgs"], batch["support_masks"])
        pred_mask = logits_to_pred_mask(
            logits,
            thr=args.pred_thr,
            use_prob_pred=args.use_prob_pred,
            fg_thr=args.fg_thr
        )

        # Probability map for mAP / MAE
        prob = logits_to_prob(logits)
        gt = batch["query_mask"]
        if gt.dim() == 2:
            gt = gt.unsqueeze(0)
        mae_sum += compute_mae(prob, gt)
        ap_sum += compute_ap_per_image(prob, gt)
        n_episodes += 1

        if args.debug_episode and idx == 0:
            _debug_first_episode(args, batch, logits, pred_mask)

        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch["class_id"], loss=_dummy_loss(batch["query_img"].device))
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

    average_meter.write_result("Test", 0)
    miou, fb_iou = average_meter.compute_iou()
    mean_mae = mae_sum / max(n_episodes, 1)
    mean_ap = ap_sum / max(n_episodes, 1)
    return float(miou), float(fb_iou), mean_ap, mean_mae


def test_with_tfi(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    args: argparse.Namespace,
) -> Tuple[float, float, float, float]:
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)
    model.eval()

    m = _unwrap(model)

    mae_sum = 0.0
    ap_sum = 0.0
    n_episodes = 0

    # Snapshot anchor weights (PATM) untuk reset tiap episode
    base_pat_state = {k: v.detach().clone() for k, v in m.pat.state_dict().items()}

    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch)

        bsz = batch["query_img"].shape[0]
        pred_masks = []
        prob_maps = []

        # TFI harus per-task; kalau bsz>1, loop per item
        for bi in range(bsz):
            q_img = batch["query_img"][bi : bi + 1]
            s_imgs = batch["support_imgs"][bi : bi + 1]
            s_masks = batch["support_masks"][bi : bi + 1]

            # Reset anchor ke kondisi sebelum adapt (per-episode)
            m.pat.load_state_dict(base_pat_state, strict=True)

            # Adapt
            tfi_adapt_episode(
                model,
                q_img,
                s_imgs,
                s_masks,
                benchmark=args.benchmark,
                steps=args.tfi_iter,
                lr=None if args.tfi_lr < 0 else args.tfi_lr,
                tau=args.tfi_tau,
                layers=args.tfi_layers,
            )

            # Final prediction (no-grad)
            with torch.no_grad():
                logits = model(q_img, s_imgs, s_masks)
                pred = logits_to_pred_mask(
                    logits,
                    thr=args.pred_thr,
                    use_prob_pred=args.use_prob_pred,
                    fg_thr=args.fg_thr
                )
                pred_masks.append(pred)
                prob_maps.append(logits_to_prob(logits))

                if args.debug_episode and idx == 0 and bi == 0:
                    # build a small batch dict for debug (use original batch where possible)
                    dbg_batch = {
                        "query_img": q_img,
                        "support_imgs": s_imgs,
                        "support_masks": s_masks,
                    }
                    # if original batch has query_mask/class_id, attach for fg checks
                    if "query_mask" in batch:
                        dbg_batch["query_mask"] = batch["query_mask"][bi : bi + 1]
                    if "class_id" in batch:
                        dbg_batch["class_id"] = batch["class_id"][bi : bi + 1]
                    _debug_first_episode(args, dbg_batch, logits, pred)

        pred_mask = torch.cat(pred_masks, dim=0)  # (B,H,W)
        prob = torch.cat(prob_maps, dim=0)         # (B,H,W)

        gt = batch["query_mask"]
        if gt.dim() == 2:
            gt = gt.unsqueeze(0)
        mae_sum += compute_mae(prob, gt)
        ap_sum += compute_ap_per_image(prob, gt)
        n_episodes += 1

        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch["class_id"], loss=_dummy_loss(batch["query_img"].device))
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

    average_meter.write_result("Test (TFI)", 0)
    miou, fb_iou = average_meter.compute_iou()
    mean_mae = mae_sum / max(n_episodes, 1)
    mean_ap = ap_sum / max(n_episodes, 1)
    return float(miou), float(fb_iou), mean_ap, mean_mae


def main() -> None:
    parser = argparse.ArgumentParser("SAM2UNet CD-FSS (aggressive) test")
    parser.add_argument("--load", type=str, required=True, help="Path to trained model state_dict (.pt).")
    parser.add_argument("--sam2_cfg", type=str, default="sam2_hiera_l.yaml")
    parser.add_argument(
        "--sam2_ckpt",
        type=str,
        default=r"F:\CCTV\SAM-2CDFSS\model\sam2_hiera_large.pt",
        help="SAM2 checkpoint used to build the backbone. Needed only if you did NOT save the full model weights.",
    )
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--attn_heads", type=int, default=8)
    parser.add_argument("--num_fg_tokens", type=int, default=32)

    parser.add_argument(
        "--benchmark",
        type=str,
        default="chick",
        choices=["pascal", "fss", "deepglobe", "isic", "lung", "chick"],
    )
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--fold", type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--nshot", type=int, default=5, choices=[1, 5],
                        help="Number of support shots: 1 for 1-way 1-shot, 5 for 1-way 5-shot (default: 5).")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--nworker", type=int, default=0)

    parser.add_argument("--datapath_src", type=str, default="./dataset")
    parser.add_argument("--datapath_tgt", type=str, default="./dataset")

    parser.add_argument("--dp", action="store_true", help="Enable DataParallel if multiple GPUs")
    parser.add_argument("--logpath", type=str, default="")

    # ---- Pred / Debug args ----
    parser.add_argument("--pred_thr", type=float, default=0.5, help="Threshold for 1-channel logits (sigmoid).")
    parser.add_argument("--debug_episode", action="store_true", help="Print debug info for the first episode only.")

    # ---- TFI args ----
    parser.add_argument("--tfi", action="store_true", help="Enable Task-adaptive Fine-tuning Inference (TFI)")
    parser.add_argument("--tfi_iter", type=int, default=50, help="TFI steps per episode (paper uses 50)")
    parser.add_argument(
        "--tfi_lr",
        type=float,
        default=-1.0,
        help="TFI learning rate. If <0, use default mapping (deepglobe/isic=1e-3, fss/lung=5e-5).",
    )
    parser.add_argument("--tfi_tau", type=float, default=0.5, help="TFI threshold tau (paper uses 0.5)")
    parser.add_argument(
        "--tfi_layers",
        type=str,
        default="low",
        choices=["low", "mid", "high", "all"],
        help="Which PAT anchor layer(s) to fine-tune. Paper recommends low.",
    )
    parser.add_argument("--fg_thr", type=float, default=0.7,
                    help="Foreground threshold for 2-channel softmax prob (class=1). Used when --use_prob_pred.")
    parser.add_argument("--use_prob_pred", action="store_true",
                        help="Use softmax probability thresholding instead of argmax for 2-channel logits.")

    args = parser.parse_args()

    # Compatibility for common/logger.py (expects benchmark_train)
    if not hasattr(args, "benchmark_train"):
        args.benchmark_train = args.benchmark

    Logger.initialize(args, training=False)
    Evaluator.initialize()

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

    if args.dp and torch.cuda.device_count() > 1:
        Logger.info(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    state = torch.load(args.load, map_location="cpu")
    model.load_state_dict(state, strict=True)
    Logger.info("Loaded checkpoint successfully")

    # Dataset
    datapath = args.datapath_src if args.benchmark == "pascal" else args.datapath_tgt
    FSSDataset.initialize(img_size=args.img_size, datapath=datapath)
    dataloader = FSSDataset.build_dataloader(
        args.benchmark, args.bsz, args.nworker, args.fold, args.split, shot=args.nshot
    )
    Logger.info(f"Testing mode: 1-way {args.nshot}-shot")
    Logger.info(f"Num test episodes (batches): {len(dataloader)}")
    if len(dataloader) == 0:
        Logger.info("WARNING: dataloader is empty. Check datapath/benchmark/split/fold.")
        return

    if args.tfi:
        miou, fb_iou, mean_ap, mean_mae = test_with_tfi(model, dataloader, args)
    else:
        miou, fb_iou, mean_ap, mean_mae = test_no_tfi(model, dataloader, args)

    Logger.info("mIoU: %5.2f \t FB-IoU: %5.2f" % (miou, fb_iou))
    Logger.info("mAP:  %5.4f \t MAE:   %5.4f" % (mean_ap, mean_mae))
    Logger.info("==================== Finished Testing ====================")


if __name__ == "__main__":
    main()