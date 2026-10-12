"""Matched A (single frame) versus B (three aligned frames) VTMOT fusion.

Requires a verified, frozen VTMOT fine-registration checkpoint. No keyframe
logic, registration training, temporal loss or test-split access is present.
The paper inspiration is UniVF's multi-frame feature warping, not a UniVF
reproduction. A/B share the same image core, loss, data order and update count.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from datasets.vtmot_video import VTMOTVideos, read_image
from models.our_models.video_fusion_minimal import (MinimalVideoFusion,
                                                    SpatialFusionLoss, luminance)
from roadscene_coarse import sha256_file
from vtmot_a1_fine_adapt import load_start
from vtmot_a1_adapt import install_mutual_guard, seed_all
from vtmot_geometry import backward_warp, grid


FEATURE_HW = (128, 128)
EPOCHS = 20
TRAIN_STRIDE = 5
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
OCCLUSION_FB_THRESHOLD = 1.5  # 128-grid pixels


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False), encoding='utf-8')


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verified_audit(args):
    audit = json.loads(args.audit_report.read_text(encoding='utf-8'))
    if (audit.get('verified') is not True or
            audit.get('test_accessed') is not False or
            audit.get('best_checkpoint_sha256') != sha256_file(args.fine_checkpoint) or
            audit.get('source_coarse_checkpoint_sha256') !=
            sha256_file(args.coarse_adapt_checkpoint) or
            audit['all']['frames'] != 800 or
            audit['all']['valid_pixels'] != 204309555):
        raise ValueError('fine registration must pass the locked read-only audit')
    return audit


def frozen_registration(args, device):
    recipe = SimpleNamespace(pretrained=args.pretrained,
                             coarse_checkpoint=args.coarse_checkpoint,
                             a1_checkpoint=args.a1_checkpoint,
                             split_file=args.split_file,
                             coarse_adapt_checkpoint=args.coarse_adapt_checkpoint)
    model, _ = load_start(recipe, device)
    payload = torch.load(args.fine_checkpoint, map_location='cpu',
                         weights_only=False)
    expected = {'recipe': 'vtmot_a1_fine_only_v1', 'stage': 'fine',
                'source_adapted_sha256': sha256_file(args.coarse_adapt_checkpoint),
                'source_a1_sha256': sha256_file(args.a1_checkpoint),
                'coarse_sha256': sha256_file(args.coarse_checkpoint),
                'pretrained_sha256': sha256_file(args.pretrained),
                'split_sha256': sha256_file(args.split_file),
                'test_accessed': False}
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f'fine registration provenance mismatch: {key}')
    model.load_state_dict(payload['model_state_dict'], strict=True)
    model.base.train_coarse_encoder = False
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def synchronized_time(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return time.perf_counter()


@torch.inference_mode()
def register_frame(model, frame, device):
    source = torch.from_numpy(frame['ir'].transpose(2, 0, 1).copy())[None].to(device)
    target = torch.from_numpy(frame['vi'].transpose(2, 0, 1).copy())[None].to(device)
    start = synchronized_time(device)
    flow = model({'source_image': source, 'target_image': target},
                 device)['final512'][0].permute(1, 2, 0).float().cpu().numpy()
    warped, valid = backward_warp(frame['ir'], flow)
    ir_gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
    reg_ms = 1000 * (synchronized_time(device) - start)
    if flow.shape != (512, 512, 2) or not np.isfinite(flow).all():
        raise RuntimeError(f'nonfinite or wrong-grid final registration flow: {flow.shape}')
    return ir_gray, valid, reg_ms


def fb_temporal(current, neighbour, current_valid, neighbour_valid):
    """Current 128 grid -> neighbour 128 grid with forward/backward mask."""
    forward = cv2.calcOpticalFlowFarneback(current, neighbour, None,
                                            .5, 3, 15, 3, 5, 1.2, 0)
    reverse = cv2.calcOpticalFlowFarneback(neighbour, current, None,
                                            .5, 3, 15, 3, 5, 1.2, 0)
    xy = grid(FEATURE_HW) + forward
    reverse_here = cv2.remap(reverse, xy[..., 0], xy[..., 1], cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)
    neighbour_here = cv2.remap(neighbour_valid.astype(np.uint8),
                               xy[..., 0], xy[..., 1], cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT) > 0
    valid = (current_valid & neighbour_here &
             (xy[..., 0] >= 0) & (xy[..., 0] <= FEATURE_HW[1]-1) &
             (xy[..., 1] >= 0) & (xy[..., 1] <= FEATURE_HW[0]-1) &
             (np.linalg.norm(forward + reverse_here, axis=-1) <=
              OCCLUSION_FB_THRESHOLD))
    if not np.isfinite(forward).all():
        raise RuntimeError('nonfinite temporal optical flow')
    return forward.astype(np.float16), valid.astype(np.uint8)


def feature_image(rgb_or_gray):
    gray = (cv2.cvtColor(rgb_or_gray, cv2.COLOR_RGB2GRAY)
            if rgb_or_gray.ndim == 3 else rgb_or_gray)
    return cv2.resize(gray, FEATURE_HW[::-1], interpolation=cv2.INTER_AREA)


def prepare_sequence(dataset, name, stems, model, device, cache_root):
    folder = cache_root / dataset.split / name
    folder.mkdir(parents=True, exist_ok=True)
    small = {}
    timings = {}
    for stem in stems:
        frame = dataset.input_frame(name, stem)
        reg_ir, valid, reg_ms = register_frame(model, frame, device)
        np.savez(folder / f'{stem}_reg.npz', ir=reg_ir,
                 valid=valid.astype(np.uint8))
        small[stem] = {'ir': feature_image(reg_ir),
                       'vi': feature_image(frame['vi']),
                       'ir_valid': cv2.resize(valid.astype(np.uint8),
                                              FEATURE_HW[::-1],
                                              interpolation=cv2.INTER_NEAREST).astype(bool)}
        timings[stem] = {'registration_ms': reg_ms}
    all_valid = np.ones(FEATURE_HW, bool)
    zero_flow = np.zeros((*FEATURE_HW, 2), np.float16)
    zero_mask = np.zeros(FEATURE_HW, np.uint8)
    for index, stem in enumerate(stems):
        start = time.perf_counter()
        fields = {}
        for side, neighbour_index in (('before', index-1), ('after', index+1)):
            for modality in ('ir', 'vi'):
                if neighbour_index < 0 or neighbour_index >= len(stems):
                    flow, mask = zero_flow, zero_mask
                else:
                    here, there = small[stem], small[stems[neighbour_index]]
                    current_valid = here['ir_valid'] if modality == 'ir' else all_valid
                    neighbour_valid = there['ir_valid'] if modality == 'ir' else all_valid
                    flow, mask = fb_temporal(here[modality], there[modality],
                                             current_valid, neighbour_valid)
                fields[f'flow_{modality}_{side}'] = flow
                fields[f'mask_{modality}_{side}'] = mask
        timings[stem]['motion_ms'] = 1000 * (time.perf_counter() - start)
        np.savez(folder / f'{stem}_motion.npz', **fields)
    return timings


def prepare(args, device):
    if args.cache_root.exists() and any(args.cache_root.iterdir()):
        raise ValueError('cache root must be a new, empty directory')
    args.cache_root.mkdir(parents=True, exist_ok=True)
    model = frozen_registration(args, device)
    durations = {}
    for split in ('train', 'eval'):
        videos = VTMOTVideos(args.data_root, args.split_file, split)
        for name, stems in videos.sequences.items():
            durations[name] = prepare_sequence(videos, name, stems,
                                                model, device, args.cache_root)
            print(f'prepared {split}/{name}: {len(stems)} frames', flush=True)
    metadata = {'recipe': 'vtmot_fusion_cache_v1',
                'frozen_registration_sha256': sha256_file(args.fine_checkpoint),
                'split_sha256': sha256_file(args.split_file),
                'feature_hw': FEATURE_HW,
                'flow_direction': 'current feature grid -> neighbour feature grid',
                'occlusion_fb_threshold_feature_px': OCCLUSION_FB_THRESHOLD,
                'sequences': {'train': 38, 'eval': 4},
                'frames': {'train': 7600, 'eval': 800},
                'timings': durations, 'test_accessed': False}
    write_json(args.cache_root / 'cache_meta.json', metadata)
    return metadata


def cached_metadata(args):
    meta = json.loads((args.cache_root / 'cache_meta.json').read_text(encoding='utf-8'))
    if (meta.get('recipe') != 'vtmot_fusion_cache_v1' or
            meta.get('frozen_registration_sha256') != sha256_file(args.fine_checkpoint) or
            meta.get('split_sha256') != sha256_file(args.split_file) or
            tuple(meta['feature_hw']) != FEATURE_HW or
            meta['frames'] != {'train': 7600, 'eval': 800} or
            meta.get('test_accessed') is not False):
        raise ValueError('prepared cache does not match frozen registration/split')
    return meta


class CachedTriplets(Dataset):
    def __init__(self, videos, cache_root, phase=None):
        self.videos = videos
        self.cache_root = cache_root
        self.items = [(name, stems, index)
                      for name, stems in videos.sequences.items()
                      for index in range(len(stems))
                      if phase is None or index % TRAIN_STRIDE == phase]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, position):
        name, stems, index = self.items[position]
        stem = stems[index]
        root = self.cache_root / self.videos.split / name
        with np.load(root / f'{stem}_reg.npz') as current:
            current_ir = current['ir'].copy()
            current_valid = current['valid'].copy()
        vi, _ = read_image(self.videos.root / name / 'visible_mis' /
                           f'{stem}.png', 'RGB')
        reg_ir = current_ir.astype(np.float32) / 255.
        valid = current_valid.astype(np.float32)
        vi_float = vi.astype(np.float32) / 255.
        vi_y = .299*vi_float[..., 0] + .587*vi_float[..., 1] + .114*vi_float[..., 2]
        reg_ir = np.where(valid > .5, reg_ir, vi_y)
        neighbours = {}
        with np.load(root / f'{stem}_motion.npz') as motion:
            for side, offset in (('before', -1), ('after', 1)):
                other = stems[min(max(index+offset, 0), len(stems)-1)]
                with np.load(root / f'{other}_reg.npz') as other_reg:
                    other_ir_image = other_reg['ir'].copy()
                    other_valid = other_reg['valid'].astype(bool)
                other_vi, _ = read_image(self.videos.root / name / 'visible_mis' /
                                         f'{other}.png', 'RGB')
                other_vi = other_vi.astype(np.float32) / 255.
                other_y = .299*other_vi[..., 0] + .587*other_vi[..., 1] + .114*other_vi[..., 2]
                other_ir = np.where(other_valid, other_ir_image/255., other_y)
                neighbours[f'ir_{side}'] = torch.from_numpy(
                    other_ir.astype(np.float32)[None].copy())
                neighbours[f'vi_{side}'] = torch.from_numpy(other_vi.transpose(2, 0, 1).copy())
                for modality in ('ir', 'vi'):
                    neighbours[f'flow_{modality}_{side}'] = torch.from_numpy(
                        motion[f'flow_{modality}_{side}'].astype(np.float32)
                        .transpose(2, 0, 1).copy())
                    neighbours[f'mask_{modality}_{side}'] = torch.from_numpy(
                        motion[f'mask_{modality}_{side}'].astype(np.float32)[None].copy())
        return {'sequence': name, 'stem': stem,
                'ir': torch.from_numpy(reg_ir[None].astype(np.float32).copy()),
                'vi': torch.from_numpy(vi_float.transpose(2, 0, 1).copy()),
                'valid': torch.from_numpy(valid[None].copy()),
                'neighbours': neighbours}


def device_batch(batch, device):
    return (batch['ir'].to(device), batch['vi'].to(device),
            batch['valid'].to(device),
            {key: value.to(device) for key, value in batch['neighbours'].items()})


def model_fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def geometry_self_test():
    from models.our_models.video_fusion_minimal import warp_feature
    marker = torch.zeros(1, 1, 16, 16)
    marker[0, 0, 5, 8] = 1.
    flow = torch.zeros(1, 2, 16, 16)
    flow[:, 0] = 3.
    flow[:, 1] = -2.
    warped, mask = warp_feature(marker, flow, torch.ones(1, 1, 16, 16))
    if abs(float(warped[0, 0, 7, 5]) - 1.) > 1e-5 or mask[0, 0, 7, 5] != 1:
        raise AssertionError('current-to-neighbour feature warp direction failed')
    if float(mask[0, 0, 0, 15]) != 0:
        raise AssertionError('temporal warp out-of-bounds mask failed')


def code_check(args, device):
    geometry_self_test()
    train = VTMOTVideos(args.data_root, args.split_file, 'train',
                        max_sequences=1, max_frames=3)
    name, stems = next(iter(train.sequences.items()))
    model_reg = frozen_registration(args, device)
    scratch = args.output / 'check_cache'
    tiny_timing = prepare_sequence(train, name, stems, model_reg, device, scratch)
    batch = next(iter(DataLoader(CachedTriplets(train, scratch), batch_size=1)))
    ir, vi, valid, neighbours = device_batch(batch, device)
    seed_all()
    a = MinimalVideoFusion(False).to(device)
    seed_all()
    b = MinimalVideoFusion(True).to(device)
    if model_fingerprint(a) == model_fingerprint(b):
        raise AssertionError('temporal arm must add its own parameters')
    a_core = {key: value for key, value in a.state_dict().items()}
    b_core = {key: b.state_dict()[key] for key in a_core}
    if any(not torch.equal(value, b_core[key]) for key, value in a_core.items()):
        raise AssertionError('A/B image core initialization differs')
    a_y, _ = a(ir, vi)
    b_y, _ = b(ir, vi, neighbours)
    same = float((a_y - b_y).abs().max())
    if same != 0:
        raise AssertionError(f'zero temporal update did not reproduce A: {same}')
    criterion = SpatialFusionLoss().to(device)
    loss_a, _ = criterion(a_y, ir, vi, valid)
    loss_b, _ = criterion(b_y, ir, vi, valid)
    if not torch.isfinite(loss_a) or not torch.isfinite(loss_b):
        raise AssertionError('fusion loss is nonfinite')
    loss_b.backward()
    gradient = float(b.temporal_ir.net[-1].weight.grad.norm())
    if gradient <= 0 or not np.isfinite(gradient):
        raise AssertionError('temporal branch has no trainable gradient')
    loss_a.backward()
    optimizer_a = torch.optim.AdamW(a.parameters(), lr=LEARNING_RATE)
    optimizer_b = torch.optim.AdamW(b.parameters(), lr=LEARNING_RATE)
    optimizer_a.step()
    optimizer_b.step()
    with torch.inference_mode():
        updated_a, _ = a(ir, vi)
        updated_b, _ = b(ir, vi, neighbours)
    if not torch.isfinite(updated_a).all() or not torch.isfinite(updated_b).all():
        raise AssertionError('one-step A/B fusion output is nonfinite')
    current_y = updated_b[0, 0].cpu().numpy()
    current_ir = ir[0, 0].cpu().numpy()
    current_vi = luminance(vi)[0, 0].cpu().numpy()
    current_mask = valid[0, 0].cpu().numpy() > .5
    metrics = cv2_metrics(current_y, current_ir, current_vi, current_mask)
    if not all(np.isfinite(value) for value in metrics.values()
               if value is not None):
        raise AssertionError('fusion spatial evaluation is nonfinite')
    identity_l1, identity_ssim, _ = temporal_metrics(
        current_y, current_y, np.zeros((*FEATURE_HW, 2), np.float32),
        np.ones(FEATURE_HW, np.uint8), current_mask, current_mask)
    if abs(identity_l1) > 1e-7 or abs(identity_ssim - 1) > 1e-5:
        raise AssertionError('identity temporal metric failed')
    global EPOCHS
    full_budget = EPOCHS
    EPOCHS = 1
    try:
        miniature = SimpleNamespace(cache_root=scratch, output=args.output,
                                    fine_checkpoint=args.fine_checkpoint,
                                    split_file=args.split_file)
        meta = {'timings': {name: tiny_timing}}
        for arm in ('A', 'B'):
            history, checkpoint = train_one(miniature, arm, train, train, device)
            if history[-1]['updates'] != 1:
                raise AssertionError('miniature train budget was not one update')
            rows, _ = evaluate_arm(miniature, arm, checkpoint, train, meta, device)
            if len(rows) != 3:
                raise AssertionError('miniature evaluation did not cover three frames')
    finally:
        EPOCHS = full_budget
    report = {'passed': True, 'frozen_registration_sha256': sha256_file(args.fine_checkpoint),
              'A_B_initial_fused_y_max_difference': same,
              'spatial_loss_A': float(loss_a), 'spatial_loss_B': float(loss_b),
              'temporal_gradient_norm': gradient,
              'one_step_A_B_finite': True,
              'spatial_metrics_finite': True,
              'identity_temporal_l1': identity_l1,
              'identity_temporal_ssim': identity_ssim,
              'three_frame_train_evaluate_smoke': True,
              'geometry': 'current 128-grid -> neighbour 128-grid backward XY passed',
              'mask': 'forward/backward occlusion and registration bounds',
              'test_accessed': False}
    write_json(args.output / 'code_check.json', report)
    return report


def train_one(args, arm, train_videos, val_videos, device):
    temporal = arm == 'B'
    seed_all()
    model = MinimalVideoFusion(temporal).to(device)
    initial_core = {name: value.detach().cpu().clone() for name, value in
                    model.state_dict().items() if not name.startswith('temporal_')}
    criterion = SpatialFusionLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5)
    history, best = [], float('inf')
    checkpoint = args.output / f'best_{arm}.pth'
    steps = 0
    for epoch in range(1, EPOCHS+1):
        model.train()
        dataset = CachedTriplets(train_videos, args.cache_root,
                                 phase=(epoch-1) % TRAIN_STRIDE)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            generator=torch.Generator().manual_seed(2026+epoch),
                            num_workers=0, pin_memory=device.type == 'cuda')
        order = hashlib.sha256()
        accumulated = 0.
        for batch in loader:
            for name, stem in zip(batch['sequence'], batch['stem']):
                order.update(f'{name}/{stem}\n'.encode())
            ir, vi, valid, neighbours = device_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            fused, _ = model(ir, vi, neighbours if temporal else None)
            loss, _ = criterion(fused, ir, vi, valid)
            if not torch.isfinite(loss):
                raise RuntimeError(f'nonfinite fusion loss: {arm} epoch {epoch}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise RuntimeError(f'nonfinite fusion gradient: {arm} epoch {epoch}')
            optimizer.step()
            accumulated += float(loss.detach()) * ir.shape[0]
            steps += 1
        train_loss = accumulated / len(dataset)
        score = validation_loss(model, criterion, val_videos, args.cache_root,
                                device, temporal)
        scheduler.step()
        record = {'epoch': epoch, 'updates': steps, 'train_spatial_loss': train_loss,
                  'val_spatial_loss': score, 'image_order_sha256': order.hexdigest(),
                  'lr_next': optimizer.param_groups[0]['lr']}
        history.append(record)
        if score < best:
            best = score
            payload = {'recipe': 'vtmot_fusion_ab_v1', 'arm': arm,
                       'registration_sha256': sha256_file(args.fine_checkpoint),
                       'split_sha256': sha256_file(args.split_file),
                       'core_initial_sha256': fingerprint_tensors(initial_core),
                       'epoch': epoch, 'updates': steps, 'val_spatial_loss': score,
                       'model_state_dict': model.state_dict(),
                       'optimizer_state_dict': optimizer.state_dict(),
                       'scheduler_state_dict': scheduler.state_dict(),
                       'torch_rng_state': torch.get_rng_state(),
                       'numpy_rng_state': np.random.get_state(),
                       'python_rng_state': random.getstate(),
                       'cuda_rng_state': (torch.cuda.get_rng_state()
                                          if device.type == 'cuda' else None),
                       'test_accessed': False}
            temporary = checkpoint.with_suffix('.tmp')
            torch.save(payload, temporary)
            os.replace(temporary, checkpoint)
        write_json(args.output / f'history_{arm}.json', history)
        print(f'{arm} epoch {epoch}/{EPOCHS}: train={train_loss:.5f}, '
              f'val spatial={score:.5f}, best={best:.5f}, updates={steps}',
              flush=True)
    return history, checkpoint


def fingerprint_tensors(tensors):
    digest = hashlib.sha256()
    for key, value in tensors.items():
        digest.update(key.encode())
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def validation_loss(model, criterion, videos, cache_root, device, temporal):
    model.eval()
    dataset = CachedTriplets(videos, cache_root)
    total = 0.
    for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
        ir, vi, valid, neighbours = device_batch(batch, device)
        fused, _ = model(ir, vi, neighbours if temporal else None)
        loss, _ = criterion(fused, ir, vi, valid)
        total += float(loss)
    return total / len(dataset)


def cv2_metrics(fused, ir, vi, valid):
    """Predeclared no-reference spatial checks at 512 pixels."""
    y = np.clip(fused * 255, 0, 255).astype(np.uint8)
    i = np.clip(ir * 255, 0, 255).astype(np.uint8)
    v = np.clip(vi * 255, 0, 255).astype(np.uint8)
    edge_f = (cv2.Canny(y, 75, 150) > 0) & valid
    edge_i = (cv2.Canny(i, 75, 150) > 0) & valid
    edge_v = (cv2.Canny(v, 75, 150) > 0) & valid
    edge_union = edge_i | edge_v
    near_union = cv2.dilate(edge_union.astype(np.uint8),
                            np.ones((5, 5), np.uint8)) > 0
    near_f = cv2.dilate(edge_f.astype(np.uint8),
                        np.ones((5, 5), np.uint8)) > 0
    fused_edges = edge_f & valid
    source_edges = edge_union & valid
    ghost = float((fused_edges & ~near_union).sum()/max(1, fused_edges.sum()))
    retention = float((source_edges & near_f).sum()/max(1, source_edges.sum()))
    hot = (ir > .6) & (ir > vi + .1) & valid
    target = float((fused[hot] >= .9 * ir[hot]).mean()) if hot.any() else None
    dx_f = cv2.Sobel(y.astype(np.float32)/255., cv2.CV_32F, 1, 0)
    dy_f = cv2.Sobel(y.astype(np.float32)/255., cv2.CV_32F, 0, 1)
    dx_i = cv2.Sobel(i.astype(np.float32)/255., cv2.CV_32F, 1, 0)
    dy_i = cv2.Sobel(i.astype(np.float32)/255., cv2.CV_32F, 0, 1)
    dx_v = cv2.Sobel(v.astype(np.float32)/255., cv2.CV_32F, 1, 0)
    dy_v = cv2.Sobel(v.astype(np.float32)/255., cv2.CV_32F, 0, 1)
    mag_f = np.hypot(dx_f, dy_f)
    mag_input = np.maximum(np.hypot(dx_i, dy_i), np.hypot(dx_v, dy_v))
    sharpness = float(mag_f[valid].mean()/max(1e-8, mag_input[valid].mean()))
    return {'ghost_edge_fraction': ghost, 'input_edge_retention': retention,
            'hot_target_retention': target, 'hot_target_pixels': int(hot.sum()),
            'gradient_ratio': sharpness, 'fused_edge_pixels': int(fused_edges.sum())}


def moving_ghost_metric(fused, ir, vi, valid, flow128, mask128):
    """Unsupported fused edges specifically where visible motion is appreciable."""
    moving128 = ((np.linalg.norm(flow128.astype(np.float32), axis=-1) >= 1.5) &
                 (mask128 > .5))
    moving = cv2.resize(moving128.astype(np.uint8), (512, 512),
                        interpolation=cv2.INTER_NEAREST).astype(bool) & valid
    if not moving.any():
        return None
    def canny(image):
        return cv2.Canny(np.clip(image*255, 0, 255).astype(np.uint8),
                         75, 150) > 0
    fused_edge = canny(fused) & moving
    if not fused_edge.any():
        return None
    support = (canny(ir) | canny(vi)) & valid
    near_support = cv2.dilate(support.astype(np.uint8),
                              np.ones((5, 5), np.uint8)) > 0
    return float((fused_edge & ~near_support).sum()/fused_edge.sum())


def temporal_metrics(current, previous, flow128, mask128,
                     current_valid, previous_valid):
    """Warp previous fused output onto current VI coordinates for both arms."""
    flow512 = cv2.resize(flow128.astype(np.float32), (512, 512),
                         interpolation=cv2.INTER_LINEAR) * 4.
    xy = grid((512, 512)) + flow512
    old = cv2.remap(previous, xy[..., 0], xy[..., 1], cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT)
    old_valid = cv2.remap(previous_valid.astype(np.uint8),
                          xy[..., 0], xy[..., 1], cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT) > 0
    mask = (cv2.resize(mask128.astype(np.uint8), (512, 512),
                       interpolation=cv2.INTER_NEAREST) > 0)
    mask &= current_valid & old_valid
    if mask.sum() < 100:
        return None, None, int(mask.sum())
    mu_c = cv2.GaussianBlur(current, (11, 11), 1.5)
    mu_o = cv2.GaussianBlur(old, (11, 11), 1.5)
    sigma_c = cv2.GaussianBlur(current * current, (11, 11), 1.5) - mu_c*mu_c
    sigma_o = cv2.GaussianBlur(old * old, (11, 11), 1.5) - mu_o*mu_o
    covariance = cv2.GaussianBlur(current * old, (11, 11), 1.5) - mu_c*mu_o
    ssim = ((2*mu_c*mu_o+.01**2) * (2*covariance+.03**2) /
            ((mu_c**2+mu_o**2+.01**2)*(sigma_c+sigma_o+.03**2)))
    return float(np.abs(current-old)[mask].mean()), float(ssim[mask].mean()), int(mask.sum())


@torch.inference_mode()
def evaluate_arm(args, arm, checkpoint, val_videos, meta, device):
    temporal = arm == 'B'
    model = MinimalVideoFusion(temporal).to(device)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (payload['registration_sha256'] != sha256_file(args.fine_checkpoint) or
            payload['split_sha256'] != sha256_file(args.split_file) or
            payload['arm'] != arm or payload.get('test_accessed') is not False):
        raise ValueError('fusion checkpoint provenance mismatch')
    model.load_state_dict(payload['model_state_dict'], strict=True)
    model.eval()
    criterion = SpatialFusionLoss().to(device)
    rows = []
    prior = {}
    dataset = CachedTriplets(val_videos, args.cache_root)
    examples = {('photo-0319-18', '000196'), ('photo-0306-02', '000000'),
                ('wurenji-0302-02', '000100')}
    for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
        name, stem = batch['sequence'][0], batch['stem'][0]
        ir, vi, valid, neighbours = device_batch(batch, device)
        start = synchronized_time(device)
        fused_y, fused_rgb = model(ir, vi, neighbours if temporal else None)
        fusion_ms = 1000 * (synchronized_time(device) - start)
        loss, _ = criterion(fused_y, ir, vi, valid)
        fy = fused_y[0, 0].float().cpu().numpy()
        ii = ir[0, 0].float().cpu().numpy()
        vv = luminance(vi)[0, 0].float().cpu().numpy()
        ok = valid[0, 0].cpu().numpy() > .5
        spatial = cv2_metrics(fy, ii, vv, ok)
        motion_flow = batch['neighbours']['flow_vi_before'][0].permute(1, 2, 0).numpy()
        motion_mask = batch['neighbours']['mask_vi_before'][0, 0].numpy()
        motion_ghost = moving_ghost_metric(fy, ii, vv, ok,
                                           motion_flow, motion_mask)
        prev = prior.get(name)
        if prev is None:
            temp_l1, temp_ssim, temp_valid = None, None, 0
        else:
            temp_l1, temp_ssim, temp_valid = temporal_metrics(
                fy, prev['fused_y'], motion_flow, motion_mask, ok, prev['valid'])
        prior[name] = {'fused_y': fy, 'valid': ok}
        times = meta['timings'][name][stem]
        total_ms = times['registration_ms'] + fusion_ms + (
            times['motion_ms'] if temporal else 0.)
        row = {'sequence': name, 'stem': stem, 'valid_pixels': int(ok.sum()),
               'spatial_loss': float(loss), **spatial,
               'moving_ghost_edge_fraction': motion_ghost,
               'temporal_warp_l1': temp_l1, 'temporal_warp_ssim': temp_ssim,
               'temporal_valid_pixels': temp_valid,
               'registration_ms': times['registration_ms'],
               'motion_ms_counted': times['motion_ms'] if temporal else 0.,
               'fusion_ms': fusion_ms, 'total_compute_ms': total_ms}
        rows.append(row)
        if (name, stem) in examples:
            image = np.concatenate([
                cv2.cvtColor(np.clip(vv*255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB),
                cv2.cvtColor(np.clip(ii*255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB),
                np.clip(fused_rgb[0].permute(1, 2, 0).cpu().numpy()*255,
                        0, 255).astype(np.uint8)], axis=1)
            folder = args.output / 'visuals'
            folder.mkdir(exist_ok=True)
            cv2.imwrite(str(folder / f'{arm}_{name}_{stem}.png'),
                        cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return rows, {'checkpoint_sha256': sha256_file(checkpoint),
                  'selected_epoch': payload['epoch'], 'updates_at_selection': payload['updates'],
                  'parameter_count': sum(p.numel() for p in model.parameters()),
                  'all': summarize(rows),
                  'per_sequence': {name: summarize([r for r in rows if r['sequence'] == name])
                                   for name in val_videos.sequences}}


def summarize(rows):
    if not rows:
        raise ValueError('empty validation group')
    metrics = ('spatial_loss', 'ghost_edge_fraction',
               'moving_ghost_edge_fraction', 'input_edge_retention',
               'hot_target_retention', 'gradient_ratio', 'temporal_warp_l1',
               'temporal_warp_ssim', 'registration_ms', 'motion_ms_counted',
               'fusion_ms', 'total_compute_ms')
    values = {}
    for metric in metrics:
        available = [float(r[metric]) for r in rows if r[metric] is not None]
        values[metric] = sum(available)/len(available) if available else None
    values['total_compute_seconds'] = sum(r['total_compute_ms'] for r in rows)/1000.
    values['frames'] = len(rows)
    values['valid_pixels'] = sum(r['valid_pixels'] for r in rows)
    values['hot_target_pixels'] = sum(r['hot_target_pixels'] for r in rows)
    values['temporal_valid_pixels'] = sum(r['temporal_valid_pixels'] for r in rows)
    return values


def paired_decision(a_rows, b_rows):
    a = {(r['sequence'], r['stem']): r for r in a_rows}
    b = {(r['sequence'], r['stem']): r for r in b_rows}
    if set(a) != set(b) or len(a) != 800:
        raise ValueError('A/B must have identical complete validation frames')
    pairs = []
    for key in a:
        left, right = a[key], b[key]
        if left['valid_pixels'] != right['valid_pixels']:
            raise ValueError(f'A/B spatial mask differs: {key}')
        pairs.append({'sequence': key[0], 'stem': key[1],
                      'A_spatial_loss': left['spatial_loss'],
                      'B_spatial_loss': right['spatial_loss'],
                      'B_minus_A_spatial_loss': right['spatial_loss']-left['spatial_loss'],
                      'A_temporal_warp_l1': left['temporal_warp_l1'],
                      'B_temporal_warp_l1': right['temporal_warp_l1'],
                      'A_ghost_fraction': left['ghost_edge_fraction'],
                      'B_ghost_fraction': right['ghost_edge_fraction'],
                      'A_moving_ghost_fraction': left['moving_ghost_edge_fraction'],
                      'B_moving_ghost_fraction': right['moving_ghost_edge_fraction'],
                      'A_hot_retention': left['hot_target_retention'],
                      'B_hot_retention': right['hot_target_retention'],
                      'A_total_ms': left['total_compute_ms'],
                      'B_total_ms': right['total_compute_ms']})
    return pairs


def run_train(args, device):
    meta = cached_metadata(args)
    train = VTMOTVideos(args.data_root, args.split_file, 'train')
    val = VTMOTVideos(args.data_root, args.split_file, 'eval')
    if len(train.sequences) != 38 or len(val.sequences) != 4:
        raise ValueError('locked VTMOT split changed')
    histories, checkpoints = {}, {}
    for arm in ('A', 'B'):
        histories[arm], checkpoints[arm] = train_one(args, arm, train, val, device)
    for a, b in zip(histories['A'], histories['B']):
        if a['epoch'] != b['epoch'] or a['updates'] != b['updates'] or a['image_order_sha256'] != b['image_order_sha256']:
            raise RuntimeError('A/B update budgets or sample ordering differ')
    payloads = [torch.load(checkpoints[arm], map_location='cpu',
                           weights_only=False) for arm in ('A', 'B')]
    if payloads[0]['core_initial_sha256'] != payloads[1]['core_initial_sha256']:
        raise RuntimeError('A/B shared core initial weights differ')
    data, results = {}, {}
    for arm in ('A', 'B'):
        data[arm], results[arm] = evaluate_arm(args, arm, checkpoints[arm],
                                             val, meta, device)
        save_csv(args.output / f'per_frame_{arm}.csv', data[arm])
    paired = paired_decision(data['A'], data['B'])
    save_csv(args.output / 'per_frame_paired.csv', paired)
    # Selection of fusion structure is based on all spatial/temporal safeguards,
    # never on one temporal score alone. No new loss is enabled here.
    a, b = results['A']['all'], results['B']['all']
    quality = (b['spatial_loss'] < a['spatial_loss'] and
               b['input_edge_retention'] >= a['input_edge_retention'] - .01 and
               b['gradient_ratio'] >= a['gradient_ratio'] - .02 and
               b['ghost_edge_fraction'] <= a['ghost_edge_fraction'] + .01 and
               b['moving_ghost_edge_fraction'] is not None and
               a['moving_ghost_edge_fraction'] is not None and
               b['moving_ghost_edge_fraction'] <=
               a['moving_ghost_edge_fraction'] + .01 and
               b['hot_target_retention'] is not None and
               a['hot_target_retention'] is not None and
               b['hot_target_retention'] >= a['hot_target_retention'] - .01 and
               b['temporal_warp_l1'] < a['temporal_warp_l1'] and
               sum(r['B_minus_A_spatial_loss'] < 0 for r in paired) >= 500)
    report = {'recipe': 'VTMOT matched single-frame A vs aligned three-frame B',
              'frozen_registration_sha256': sha256_file(args.fine_checkpoint),
              'split_sha256': sha256_file(args.split_file),
              'train_sequences': 38, 'validation_sequences': 4,
              'validation_frames': 800, 'test_accessed': False,
              'source': 'UniVF-inspired feature warping; not a UniVF reproduction',
              'training': {'epochs': EPOCHS, 'updates_per_arm': histories['A'][-1]['updates'],
                           'batch_size': BATCH_SIZE, 'lr': LEARNING_RATE,
                           'phase_stride': TRAIN_STRIDE,
                           'loss': 'same intensity + 2 edge + 0.5 hot, no temporal loss',
                           'best_checkpoint': 'lowest full validation spatial fusion loss'},
              'motion': {'flow_grid': FEATURE_HW,
                         'flow_direction': 'current -> neighbour, per modality',
                         'occlusion': 'forward/backward difference <= 1.5 feature pixels',
                         'boundary': 'first/last neighbour invalid'},
              'timing': 'per frame registration network+warp, B four-direction temporal Farneback, fused forward; disk IO and metric calculation excluded',
              'A': results['A'], 'B': results['B'],
              'paired_spatial_improved_frames': sum(r['B_minus_A_spatial_loss'] < 0 for r in paired),
              'adopt_B': quality,
              'adoption_rule': 'B lower spatial loss and temporal warp L1, >=500/800 spatially improved, edge retention/gradient/hot loss <=.01/.02/.01 tolerance, global and moving ghost increases <=.01'}
    write_json(args.output / 'report_val.json', report)
    print(json.dumps({'adopt_B': quality, 'A': results['A']['all'],
                      'B': results['B']['all']}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('check', 'prepare', 'train'), required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--split-file', type=Path,
                        default=Path('experiments/vtmot_a1_adapt/split.json'))
    parser.add_argument('--pretrained', type=Path,
                        default=Path('pre_trained_models/GLUNet_DPED_CityScape_ADE.pth'))
    parser.add_argument('--coarse-checkpoint', type=Path,
                        default=Path('roadscene_runs/no_dcn_v1/best_coarse.pth'))
    parser.add_argument('--a1-checkpoint', type=Path,
                        default=Path('roadscene_runs/fusion_hierarchical_2x2/best_A1.pth'))
    parser.add_argument('--coarse-adapt-checkpoint', type=Path,
                        default=Path('roadscene_runs/vtmot_a1_coarse_adapt/best_adapted_a1.pth'))
    parser.add_argument('--fine-checkpoint', type=Path,
                        default=Path('roadscene_runs/vtmot_a1_fine_adapt/best_fine_adapted_a1.pth'))
    parser.add_argument('--audit-report', type=Path,
                        default=Path('experiments/vtmot_fusion_ab/frozen_registration_audit.json'))
    parser.add_argument('--cache-root', type=Path,
                        default=Path('roadscene_runs/vtmot_fusion_ab_cache'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be a new, empty directory')
    verified_audit(args)
    seed_all()
    install_mutual_guard()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.stage == 'prepare':
        prepare(args, device)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'check':
        print(json.dumps(code_check(args, device), indent=2), flush=True)
    else:
        run_train(args, device)


if __name__ == '__main__':
    main()
