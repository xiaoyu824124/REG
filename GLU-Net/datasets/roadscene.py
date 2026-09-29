"""Paired visible/warped-infrared RoadScene frames and target-to-source flow."""

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch


def read_flo(path):
    with open(path, "rb") as file:
        if file.read(4) != b"PIEH":
            raise ValueError(f"Invalid .flo header: {path}")
        width, height = np.fromfile(file, dtype="<i4", count=2)
        flow = np.fromfile(file, dtype="<f4", count=int(width * height * 2))
    if flow.size != width * height * 2:
        raise ValueError(f"Incomplete .flo file: {path}")
    return flow.reshape(int(height), int(width), 2)


class RoadScenePairs(Dataset):
    """Visible is target; warped IR is source; .flo maps visible to IR."""

    def __init__(self, root, split):
        self.root = Path(root)
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        split_dir = self.root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(split_dir)
        self.samples = []
        seen_ids = set()
        for flow_path in sorted(split_dir.glob("*/truth_flow/*.flo")):
            pair_dir = flow_path.parent.parent / "image_pair"
            stem = flow_path.stem
            if stem in seen_ids:
                raise ValueError(f"Duplicate RoadScene pair ID in {split}: {stem}")
            seen_ids.add(stem)
            visible = pair_dir / f"{stem}_visible.tif"
            infrared = pair_dir / f"{stem}_infrared_warped.tif"
            if not visible.is_file() or not infrared.is_file():
                raise FileNotFoundError(f"Incomplete RoadScene pair: {stem} in {pair_dir}")
            self.samples.append((visible, infrared, flow_path))
        if not self.samples:
            raise FileNotFoundError(f"No complete RoadScene pairs in {split_dir}")
        list_path = self.root / f"{split}_list.txt"
        if list_path.is_file():
            listed_ids = [line.strip() for line in list_path.read_text().splitlines()
                          if line.strip()]
            if len(listed_ids) != len(seen_ids) or set(listed_ids) != seen_ids:
                raise ValueError(f"{list_path} does not match the image/flow files")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        visible_path, infrared_path, flow_path = self.samples[index]
        with Image.open(visible_path) as image:
            target = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        with Image.open(infrared_path) as image:
            source = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        flow = read_flo(flow_path)
        height, width = target.shape[:2]
        if source.shape[:2] != (height, width) or flow.shape[:2] != (height, width):
            raise ValueError(f"Pair and flow sizes differ: {flow_path}")
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        mapped_x = x + flow[..., 0]
        mapped_y = y + flow[..., 1]
        valid = (np.isfinite(flow).all(axis=2) &
                 (mapped_x >= 0) & (mapped_x <= width - 1) &
                 (mapped_y >= 0) & (mapped_y <= height - 1))
        flow = np.where(valid[..., None], flow, 0).astype(np.float32)
        return {
            "target_image": torch.from_numpy(target.transpose(2, 0, 1).copy()),
            "source_image": torch.from_numpy(source.transpose(2, 0, 1).copy()),
            "flow_map": torch.from_numpy(flow.transpose(2, 0, 1).copy()),
            "correspondence_mask": torch.from_numpy(valid),
            "name": flow_path.stem,
        }
