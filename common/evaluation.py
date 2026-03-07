import torch

class Evaluator:
    @classmethod
    def initialize(cls):
        pass

    @staticmethod
    @torch.no_grad()
    def classify_prediction(pred_mask: torch.Tensor, batch: dict):
        """
        pred_mask: (B,H,W) int/long, values {0,1}
        batch["query_mask"]: (B,H,W) int/long, values {0,1}
        optional batch["query_ignore_idx"]: (B,H,W) bool/0-1 (ignore pixels)
        returns:
            area_inter: (2,B) [bg, fg]
            area_union: (2,B) [bg, fg]
        """

        gt_mask = batch["query_mask"]

        # Make sure shapes are (B,H,W)
        if pred_mask.dim() == 2:
            pred_mask = pred_mask.unsqueeze(0)
        if gt_mask.dim() == 2:
            gt_mask = gt_mask.unsqueeze(0)

        pred_mask = pred_mask.long()
        gt_mask = gt_mask.long()

        B, H, W = gt_mask.shape

        # Handle ignore pixels if provided
        ignore = batch.get("query_ignore_idx", None)
        if ignore is not None:
            if ignore.dim() == 2:
                ignore = ignore.unsqueeze(0)
            ignore = ignore.bool()
            valid = ~ignore
        else:
            valid = torch.ones_like(gt_mask, dtype=torch.bool)

        # Flatten for easy sum
        pred = pred_mask.view(B, -1)
        gt = gt_mask.view(B, -1)
        valid = valid.view(B, -1)

        # Only count valid pixels
        pred = pred[valid].view(B, -1) if valid.all() else pred
        gt = gt[valid].view(B, -1) if valid.all() else gt
        # If not all valid, do per-batch masking:
        # We'll compute with valid mask directly (more robust)
        # So override back:
        pred = pred_mask.view(B, -1)
        gt = gt_mask.view(B, -1)
        valid = valid.view(B, -1)

        # Foreground (class=1)
        pred_fg = (pred == 1) & valid
        gt_fg   = (gt == 1) & valid
        inter_fg = (pred_fg & gt_fg).sum(dim=1)              # (B,)
        union_fg = (pred_fg | gt_fg).sum(dim=1)              # (B,)

        # Background (class=0)
        pred_bg = (pred == 0) & valid
        gt_bg   = (gt == 0) & valid
        inter_bg = (pred_bg & gt_bg).sum(dim=1)              # (B,)
        union_bg = (pred_bg | gt_bg).sum(dim=1)              # (B,)

        # Stack as (2,B)
        area_inter = torch.stack([inter_bg, inter_fg], dim=0).float().cuda()
        area_union = torch.stack([union_bg, union_fg], dim=0).float().cuda()

        return area_inter, area_union