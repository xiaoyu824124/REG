"""VTMOT sequential frames with GT isolated from registration decisions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from vtmot_geometry import grid


def resize_affine(source_hw, target_hw=(512, 512)):
    """Original XY -> centered-crop/resized XY, including PIL half pixels."""
    sh, sw = source_hw
    th, tw = target_hw
    if sw / sh > tw / th:
        ch, cw = sh, round(sh * tw / th)
        top, left = 0, (sw - cw) // 2
    elif sw / sh < tw / th:
        ch, cw = round(sw * th / tw), sw
        top, left = (sh - ch) // 2, 0
    else:
        ch, cw, top, left = sh, sw, 0, 0
    sx, sy = tw / cw, th / ch
    crop = np.array([[1., 0., -left], [0., 1., -top], [0., 0., 1.]])
    scale = np.array([[sx, 0., .5 * sx - .5],
                      [0., sy, .5 * sy - .5], [0., 0., 1.]])
    return scale @ crop, (top, left, ch, cw)


def read_image(path, mode='RGB', target_hw=(512, 512)):
    with Image.open(path) as opened:
        original_hw = (opened.height, opened.width)
        _, (top, left, height, width) = resize_affine(original_hw, target_hw)
        resized = opened.convert(mode).crop((left, top, left + width, top + height))
        resized = resized.resize((target_hw[1], target_hw[0]), Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=np.uint8).copy(), original_hw


class VTMOTVideos:
    """Only curated CSV frames; frame order preserved and reset per sequence."""

    def __init__(self, root, split_file, split='eval', target_hw=(512, 512),
                 max_sequences=0, max_frames=0):
        self.root = Path(root)
        self.split_file = Path(split_file)
        if split not in ('train', 'eval'):
            raise ValueError('test split is locked; this program only accepts train/eval')
        self.split = split
        self.target_hw = tuple(target_hw)
        if self.target_hw != (512, 512):
            raise ValueError('A1 and this evaluation protocol require 512x512')
        splits = json.loads(self.split_file.read_text(encoding='utf-8'))
        self.sequences = {}
        for name in splits[split][:max_sequences or None]:
            manifest = self.split_file.parent / f'{name}.csv'
            stems = []
            if manifest.is_file():
                with manifest.open(newline='', encoding='utf-8-sig') as stream:
                    rows = list(csv.DictReader(stream))
                for row in rows:
                    ir = Path(row['ir'])
                    visible = Path(row['rgb'])
                    if (ir.parent.as_posix() != 'infrared' or
                            visible != Path('visible_mis') / f'{ir.stem}.png'):
                        raise ValueError(f'invalid VTMOT manifest pair: {name}/{row}')
                    stems.append(ir.stem)
            else:
                # The released split JSON is sufficient on the training host:
                # all 48 local sequence CSVs were audited against the sorted
                # infrared stems and were identical (200 frames each).
                stems = sorted(p.stem for p in (self.root / name / 'infrared').glob('*.jpg'))
                if stems != [f'{index:06d}' for index in range(200)]:
                    raise ValueError(f'{name} does not match audited 000000..000199 '
                                     'frame manifest; provide per-sequence CSVs')
            for stem in stems[:max_frames or None]:
                for relative in (f'infrared/{stem}.jpg', f'visible_mis/{stem}.png',
                                 f'gt_h/{stem}.npy'):
                    if not (self.root / name / relative).is_file():
                        raise FileNotFoundError(self.root / name / relative)
            stems = stems[:max_frames or None]
            if not stems:
                raise ValueError(f'empty sequence {name}')
            self.sequences[name] = stems

    def input_frame(self, sequence, stem):
        base = self.root / sequence
        ir, ir_hw = read_image(base / 'infrared' / f'{stem}.jpg', 'RGB', self.target_hw)
        vi, vi_hw = read_image(base / 'visible_mis' / f'{stem}.png', 'RGB', self.target_hw)
        if ir_hw != vi_hw:
            raise ValueError(f'IR/VI original raster mismatch: {sequence}/{stem}')
        return {'ir': ir, 'vi': vi, 'sequence': sequence, 'stem': stem,
                'original_hw': ir_hw}

    def flow_truth(self, sequence, stem, original_hw):
        """Ground-truth VI-to-IR flow; never used by registration decisions."""
        base = self.root / sequence
        h_original = np.load(base / 'gt_h' / f'{stem}.npy').astype(np.float64)
        affine, _ = resize_affine(original_hw, self.target_hw)
        matrix = affine @ h_original @ np.linalg.inv(affine)
        xy = grid(self.target_hw)
        xy1 = np.concatenate((xy, np.ones((*self.target_hw, 1), np.float32)), axis=-1)
        projected = xy1 @ matrix.T
        mapped = projected[..., :2] / projected[..., 2:3]
        flow = (mapped - xy).astype(np.float32)
        valid = (np.isfinite(flow).all(axis=-1) &
                 (mapped[..., 0] >= 0) & (mapped[..., 0] <= self.target_hw[1] - 1) &
                 (mapped[..., 1] >= 0) & (mapped[..., 1] <= self.target_hw[0] - 1))
        return flow, valid

    def evaluation_truth(self, sequence, stem, original_hw):
        """Called strictly after a method has emitted its flow and trigger."""
        base = self.root / sequence
        flow, valid = self.flow_truth(sequence, stem, original_hw)
        aligned_visible, gt_hw = read_image(base / 'visible_gt' / f'{stem}.png',
                                             'RGB', self.target_hw)
        if gt_hw != original_hw:
            raise ValueError(f'visible_gt shape mismatch: {sequence}/{stem}')
        return flow, valid, aligned_visible


def direction_check(dataset, sequence, stem):
    """GT should warp aligned visible_gt back to visible_mis."""
    from vtmot_geometry import backward_warp
    frame = dataset.input_frame(sequence, stem)
    flow, valid, aligned = dataset.evaluation_truth(sequence, stem, frame['original_hw'])
    warped, _ = backward_warp(aligned, flow)
    zero_mse = np.mean((aligned[valid].astype(np.float32) - frame['vi'][valid]) ** 2)
    gt_mse = np.mean((warped[valid].astype(np.float32) - frame['vi'][valid]) ** 2)
    return {'sequence': sequence, 'stem': stem, 'valid_pixels': int(valid.sum()),
            'gt_warp_mse_255': float(gt_mse), 'zero_warp_mse_255': float(zero_mse),
            'direction': 'VI target -> IR/aligned-VI source, XY'}
