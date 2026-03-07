# data/chick.py
import os
from glob import glob

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


class DatasetChick(Dataset):
    """Meta-testing dataset for chick (CCTV) images.

    Single folder: dataset/chick/images/ + dataset/chick/segmentations/
    First `shot` images (sorted) → fixed support set.
    Remaining images → query set (evaluated).
    """

    def __init__(self, datapath, fold, transform, split, shot=1,
                 episodes_per_epoch=200):

        self.root = os.path.join(datapath, "chick")

        self.transform = transform
        self.split = split
        self.shot = int(shot)

        self.benchmark = "chick"
        self.nclass = 1
        self.class_ids = [0]

        # Single folder structure
        self.img_dir = os.path.join(self.root, "images")
        self.msk_dir = os.path.join(self.root, "segmentations")

        all_ids = self._collect_ids(self.img_dir, self.msk_dir)

        # First `shot` images are fixed support, ALL images are query
        self.support_ids = all_ids[:self.shot]
        self.episodes = all_ids  # test all 18 images

    def __len__(self):
        return len(self.episodes)

    def _collect_ids(self, img_dir, mask_dir):

        imgs = glob(os.path.join(img_dir, "*.jpg")) + glob(os.path.join(img_dir, "*.png"))

        ids = []

        for p in imgs:

            stem = os.path.splitext(os.path.basename(p))[0]

            mask_path = os.path.join(mask_dir, stem + ".png")

            if os.path.exists(mask_path):
                ids.append(stem)

        return sorted(ids)

    def _load_img(self, img_dir, stem):

        path_jpg = os.path.join(img_dir, stem + ".jpg")
        path_png = os.path.join(img_dir, stem + ".png")

        path = path_jpg if os.path.exists(path_jpg) else path_png

        return Image.open(path).convert("RGB")

    def _load_mask(self, mask_dir, stem):

        path = os.path.join(mask_dir, stem + ".png")

        m = Image.open(path).convert("L")

        m = np.array(m)

        m = (m > 0).astype(np.uint8)

        return m

    def _resize_mask(self, mask, size):

        return np.array(
            Image.fromarray(mask).resize(size, resample=Image.NEAREST)
        )

    def __getitem__(self, idx):

        q_stem = self.episodes[idx]

        q_img = self._load_img(self.img_dir, q_stem)
        q_mask = self._load_mask(self.msk_dir, q_stem)

        # Fixed support set; exclude query if it's one of the support images
        sup = [s for s in self.support_ids if s != q_stem]
        if len(sup) < len(self.support_ids):
            # query was a support image, borrow next available non-support image
            all_ids = self.support_ids + [s for s in self.episodes if s not in self.support_ids]
            for candidate in all_ids:
                if candidate != q_stem and candidate not in sup:
                    sup.append(candidate)
                    break

        s_imgs = [self._load_img(self.img_dir, s) for s in sup]
        s_masks = [self._load_mask(self.msk_dir, s) for s in sup]

        q_img_t = self.transform(q_img)

        s_imgs_t = torch.stack([self.transform(x) for x in s_imgs], dim=0)

        _, H, W = q_img_t.shape

        q_mask = self._resize_mask(q_mask, (W, H))
        q_mask = torch.from_numpy(q_mask).long()

        s_masks_resized = []

        for m in s_masks:
            m = self._resize_mask(m, (W, H))
            s_masks_resized.append(torch.from_numpy(m).long())

        s_masks_t = torch.stack(s_masks_resized)

        return {
            "query_img": q_img_t,
            "query_mask": q_mask,
            "support_imgs": s_imgs_t,
            "support_masks": s_masks_t,
            "class_id": torch.tensor([0], dtype=torch.long),
        }