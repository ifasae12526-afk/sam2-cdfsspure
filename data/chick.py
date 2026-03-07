# data/chick.py
import os
import random
from glob import glob

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class DatasetChick(Dataset):
    """Few-shot segmentation dataset for chick images.

    Key improvements for small datasets:
    - Episode multiplication: generates many more query-support combinations
    - Joint spatial augmentation: same random transform applied to image+mask
    - Color/photometric augmentation on images only
    """

    def __init__(self, datapath, fold, transform, split, shot=1,
                 episodes_per_epoch=200):

        self.root = os.path.join(datapath, "chick")

        self.transform = transform
        self.split = split
        self.shot = int(shot)
        self.episodes_per_epoch = episodes_per_epoch if split == "trn" else 0

        self.benchmark = "chick"
        self.nclass = 1
        self.class_ids = [0]

        self.train_img_dir = os.path.join(self.root, "train/images")
        self.train_msk_dir = os.path.join(self.root, "train/segmentations")

        self.test_img_dir = os.path.join(self.root, "test/images")
        self.test_msk_dir = os.path.join(self.root, "test/segmentations")

        self.train_ids = self._collect_ids(self.train_img_dir, self.train_msk_dir)
        self.test_ids = self._collect_ids(self.test_img_dir, self.test_msk_dir)

        if split == "trn":
            self.episodes = self.train_ids
        else:
            self.episodes = self.test_ids

    def __len__(self):
        if self.split == "trn" and self.episodes_per_epoch > 0:
            return self.episodes_per_epoch
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

    # ---- Joint spatial augmentation (image + mask together) ----

    def _augment_pair(self, img_pil, mask_np, target_size=512):
        """Apply random spatial + color augmentation to an image-mask pair.
        Spatial transforms are applied identically to both.
        Color transforms are applied only to the image.
        """
        mask_pil = Image.fromarray(mask_np * 255, mode="L")

        # 1) Random horizontal flip
        if random.random() > 0.5:
            img_pil = TF.hflip(img_pil)
            mask_pil = TF.hflip(mask_pil)

        # 2) Random vertical flip
        if random.random() > 0.5:
            img_pil = TF.vflip(img_pil)
            mask_pil = TF.vflip(mask_pil)

        # 3) Random rotation (0-360)
        angle = random.uniform(0, 360)
        img_pil = TF.rotate(img_pil, angle, fill=0)
        mask_pil = TF.rotate(mask_pil, angle, fill=0)

        # 4) Random resized crop (scale 0.5-1.0)
        w, h = img_pil.size
        scale = random.uniform(0.5, 1.0)
        new_h, new_w = int(h * scale), int(w * scale)
        top = random.randint(0, max(0, h - new_h))
        left = random.randint(0, max(0, w - new_w))
        img_pil = TF.crop(img_pil, top, left, new_h, new_w)
        mask_pil = TF.crop(mask_pil, top, left, new_h, new_w)

        # 5) Color jitter (image only)
        if random.random() > 0.3:
            img_pil = ImageEnhance.Brightness(img_pil).enhance(random.uniform(0.7, 1.3))
            img_pil = ImageEnhance.Contrast(img_pil).enhance(random.uniform(0.7, 1.3))
            img_pil = ImageEnhance.Color(img_pil).enhance(random.uniform(0.7, 1.3))

        # 6) Random Gaussian blur (image only)
        if random.random() > 0.7:
            img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        # Convert mask back to numpy binary
        mask_np = np.array(mask_pil)
        mask_np = (mask_np > 127).astype(np.uint8)

        return img_pil, mask_np

    def __getitem__(self, idx):

        if self.split == "trn":
            # Random episode sampling for training (episode multiplication)
            q_stem = random.choice(self.train_ids)

            q_img = self._load_img(self.train_img_dir, q_stem)
            q_mask = self._load_mask(self.train_msk_dir, q_stem)

            # Apply augmentation to query
            q_img, q_mask = self._augment_pair(q_img, q_mask)

            pool = [s for s in self.train_ids if s != q_stem]

            if len(pool) == 0:
                pool = self.train_ids

            s_stems = random.choices(pool, k=self.shot)

            s_imgs = []
            s_masks = []
            for s in s_stems:
                s_img = self._load_img(self.train_img_dir, s)
                s_mask = self._load_mask(self.train_msk_dir, s)
                # Apply independent augmentation to each support
                s_img, s_mask = self._augment_pair(s_img, s_mask)
                s_imgs.append(s_img)
                s_masks.append(s_mask)

        else:
            q_stem = self.episodes[idx]

            q_img = self._load_img(self.test_img_dir, q_stem)
            q_mask = self._load_mask(self.test_msk_dir, q_stem)

            pool = self.train_ids

            s_stems = random.sample(pool, min(self.shot, len(pool)))

            s_imgs = [self._load_img(self.train_img_dir, s) for s in s_stems]
            s_masks = [self._load_mask(self.train_msk_dir, s) for s in s_stems]

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