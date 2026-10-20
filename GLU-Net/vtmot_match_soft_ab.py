"""Matched B0 continuation: original loss (M0) versus pre-MM soft matching CE (M1).

Only train/eval VTMOT manifests are instantiated. ``check`` opens at most two
training frames; full four-sequence evaluation runs only for ``report`` or
the explicit training stages on the server. No model definition is changed:
a temporary forward hook reads the global correlation immediately before
MutualMatching while the original inference path remains intact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import vtmot_dns_position_ablation as prior
import vtmot_b0_coarse_diagnose as geometry_audit
from datasets.vtmot_video import VTMOTVideos
from roadscene_coarse import sha256_file
from roadscene_fusion_hierarchical import augment, objective
from roadscene_refinement_audit import edge_quartile, pooled_truth
from vtmot_a1_adapt import (BATCH_SIZE, Frames, SEED, TRAIN_FRAME_STRIDE,
                            frozen_digest, loader, save_json, seed_all)
from vtmot_dns_adaptive_scale import frozen_local_digest
from vtmot_geometry import backward_warp
from vtmot_keyframe import edge_metrics, install_mutual_guard, lncc, nmi


RECIPE = 'vtmot_b0_soft_correspondence_v1'
ARMS = ('M0', 'M1')
STAGES = ('coarse', 'fine')
SOURCE_SHA256 = 'ca10574cd2bf40297fab65ee5b650fe8408fd28fef3b7a3355169c1267b2575a'
NEIGHBOR_TOP_K = 5


def source_checkpoint(args):
    comparison = json.loads(args.comparison.read_text(encoding='utf-8'))
    report = comparison['arms']['B0']['fine']
    path = args.comparison.parent / 'B0' / 'best_fine.pth'
    actual = sha256_file(path)
    if actual != SOURCE_SHA256 or actual != report['best_checkpoint_sha256']:
        raise ValueError(f'B0 source SHA-256 differs: actual={actual}; expected={SOURCE_SHA256}')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    expected = {'recipe': 'vtmot_dns_adaptive_scale_v1', 'arm': 'B0',
                'stage': 'fine', 'split_sha256': sha256_file(args.split_file),
                'pretrained_sha256': sha256_file(args.pretrained),
                'attention_checkpoint_sha256': sha256_file(args.attention_checkpoint),
                'test_accessed': False}
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f'B0 best fine checkpoint {key} mismatch')
    if report['parent_sha256'] != payload['parent_sha256']:
        raise ValueError('B0 best fine checkpoint coarse parent differs from report')
    return path, payload


def initialize(args, device):
    """Same selected B0 fine state, including identical original local weights."""
    seed_all()
    path, payload = source_checkpoint(args)
    model = prior.initialize('B', args, device)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    base = model.base
    if (base.coarse_dns_order != 'between_sa_ca' or
            base.coarse_dns.radii != (1, 2) or
            base.local_dcn32 is not None or base.local_dcn64 is not None or
            not base.cyclic_consistency or base.corr.normalization):
        raise AssertionError('source is not protected, fixed-scale SA -> DNS -> CA B0')
    if model.fusion is not None:
        raise AssertionError('local architecture changed')
    return model, sha256_file(path)


@contextmanager
def pre_mutual_scores(model):
    """Read the source-key by target-query cosine scores; never alter forward."""
    captured = []
    handle = model.base.corr.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output))
    try:
        yield captured
    finally:
        handle.remove()


def query_key_scores(correlation):
    # FeatureCorrelation returns [B,1,H_ir,W_ir,H_vi,W_vi].
    if correlation.ndim != 6 or correlation.shape[1] != 1:
        raise ValueError(f'unexpected pre-MM correlation {tuple(correlation.shape)}')
    b, _, hs, ws, ht, wt = correlation.shape
    if (hs, ws, ht, wt) != (16, 16, 16, 16):
        raise ValueError('the locked B0 global feature grid is not 16x16')
    return correlation[:, 0].reshape(b, hs * ws, ht * wt).transpose(1, 2)


def correspondence_labels(truth512, valid512, feature_hw=(16, 16),
                          minimum_valid_fraction=.8):
    """Sample VI-query-center GT and map its IR match into feature centers.

    The 512 raster is produced with a crop/resize affine including half-pixel
    centers. A 16-grid cell center is (x+.5)*512/16-.5 in that raster.
    The full-resolution flow is bilinearly sampled at that exact image pixel
    center (grid_sample align_corners=True maps pixel coordinates exactly).
    ``q = p + GT(p)`` maps back by ``(q+.5)/(512/16)-.5``. The area-valid
    fraction and four full-resolution neighbors must also be valid. This is
    a center-coordinate geometry proxy for a VGG feature with wide support;
    no invalid/outside q is made valid by clamping.
    """
    if truth512.ndim != 4 or truth512.shape[1:] != (2, 512, 512):
        raise ValueError('GT flow must be [B,2,512,512] in 512-pixel units')
    if valid512.shape != (truth512.shape[0], 512, 512):
        raise ValueError('valid mask shape mismatch')
    ht, wt = feature_hw
    if (ht, wt) != (16, 16):
        raise ValueError('only the existing 16x16 global grid is supervised')
    geometric = valid512.bool() & torch.isfinite(truth512).all(1)
    fraction = F.interpolate(geometric[:, None].float(), (ht, wt), mode='area')[:, 0]
    safe_flow = torch.where(geometric[:, None], truth512, torch.zeros_like(truth512))
    yy, xx = torch.meshgrid(torch.arange(ht, device=truth512.device),
                            torch.arange(wt, device=truth512.device), indexing='ij')
    px = (xx.to(truth512) + .5) * (512. / wt) - .5
    py = (yy.to(truth512) + .5) * (512. / ht) - .5
    sample_grid = torch.stack((2*px/511-1, 2*py/511-1), -1)[None].expand(
        truth512.shape[0], -1, -1, -1)
    center_flow = F.grid_sample(safe_flow, sample_grid, mode='bilinear',
                                padding_mode='zeros', align_corners=True)
    center_valid = F.grid_sample(geometric[:, None].float(), sample_grid,
                                 mode='bilinear', padding_mode='zeros',
                                 align_corners=True)[:, 0] > .9999
    qx = px[None] + center_flow[:, 0]
    qy = py[None] + center_flow[:, 1]
    cx = (qx + .5) / (512. / wt) - .5
    cy = (qy + .5) / (512. / ht) - .5
    selected = ((fraction > minimum_valid_fraction) & center_valid & torch.isfinite(cx) &
                torch.isfinite(cy) & (qx >= 0) & (qx <= 511) &
                (qy >= 0) & (qy <= 511) & (cx >= 0) & (cx <= wt - 1) &
                (cy >= 0) & (cy <= ht - 1))
    # Safe indexing only after validity is fixed; invalid rows receive zero
    # weight and are never included in the loss or metric denominators.
    safe_x = torch.where(selected, cx, torch.zeros_like(cx)).clamp(0, wt - 1)
    safe_y = torch.where(selected, cy, torch.zeros_like(cy)).clamp(0, ht - 1)
    x0 = safe_x.floor().long()
    y0 = safe_y.floor().long()
    x1 = (x0 + 1).clamp(max=wt - 1)
    y1 = (y0 + 1).clamp(max=ht - 1)
    dx, dy = safe_x - x0, safe_y - y0
    indexes = torch.stack((y0 * wt + x0, y0 * wt + x1,
                           y1 * wt + x0, y1 * wt + x1), dim=-1).reshape(-1, ht * wt, 4)
    weights = torch.stack(((1-dx)*(1-dy), dx*(1-dy),
                           (1-dx)*dy, dx*dy), dim=-1).reshape(-1, ht * wt, 4)
    weights = weights * selected.reshape(-1, ht * wt, 1)
    # Repeated corner indexes are merged by scatter_add in this dense audit.
    dense = weights.new_zeros(weights.shape[0], ht * wt, ht * wt)
    dense.scatter_add_(2, indexes, weights)
    valid_queries = selected.reshape(-1, ht * wt)
    if valid_queries.any():
        if (dense < -1e-7).any() or not torch.allclose(
                dense.sum(-1)[valid_queries],
                torch.ones_like(dense.sum(-1)[valid_queries]), atol=1e-5):
            raise AssertionError('four-corner soft labels are not a probability distribution')
    if torch.count_nonzero(dense[~valid_queries]):
        raise AssertionError('invalid correspondences acquired label mass')
    return {'indexes': indexes, 'weights': weights, 'dense': dense,
            'valid': valid_queries,
            'continuous_xy': torch.stack((cx, cy), -1).reshape(-1, ht*wt, 2),
            'valid_fraction': fraction.reshape(-1, ht*wt)}


def match_loss(scores, labels, temperature):
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    if scores.shape != labels['dense'].shape:
        raise ValueError('query/key layout and label shape differ')
    if not torch.isfinite(scores).all():
        raise ValueError('nonfinite pre-MM matching score')
    selected = labels['valid']
    if not selected.any():
        return scores.sum() * 0
    logits = scores.float() / temperature
    logp = F.log_softmax(logits, dim=-1)
    return -(labels['dense'][selected] * logp[selected]).sum(-1).mean()


@torch.no_grad()
def match_statistics(scores, labels):
    """GT rank is the best rank of any positive-weight bilinear corner."""
    selected = labels['valid']
    count = int(selected.sum())
    if not count:
        return {'queries': 0, 'rank_sum': 0., 'top1_hits': 0,
                'nearest_top1_hits': 0, 'top5_support_hits': 0,
                'neighbor1_hits': 0}
    chosen = scores[selected]
    support = labels['dense'][selected] > 0
    order = torch.argsort(chosen, dim=-1, descending=True)
    rank = torch.empty_like(order)
    sequence = torch.arange(chosen.shape[1], device=chosen.device)[None].expand_as(order)
    rank.scatter_(1, order, sequence)
    best_rank = rank.masked_fill(~support, chosen.shape[1]).min(-1).values + 1
    best = order[:, 0]
    xy = labels['continuous_xy'][selected]
    neighbor = ((best % 16 - xy[:, 0]).abs() <= 1) & (
        (best // 16 - xy[:, 1]).abs() <= 1)
    nearest_x = xy[:, 0].round().long().clamp(0, 15)
    nearest_y = xy[:, 1].round().long().clamp(0, 15)
    return {'queries': count, 'rank_sum': float(best_rank.float().sum()),
            'top1_hits': int(support.gather(1, best[:, None]).sum()),
            'nearest_top1_hits': int((best == nearest_y*16+nearest_x).sum()),
            'top5_support_hits': int(support.gather(1, order[:, :NEIGHBOR_TOP_K]).any(1).sum()),
            'neighbor1_hits': int(neighbor.sum())}


def load_stage_start(args, arm, stage, device):
    model, source_sha = initialize(args, device)
    if stage == 'coarse':
        return model, source_sha
    coarse_path = args.output / arm / 'best_coarse.pth'
    payload = torch.load(coarse_path, map_location='cpu', weights_only=False)
    expected = {'recipe': RECIPE, 'arm': arm, 'stage': 'coarse',
                'source_sha256': source_sha,
                'split_sha256': sha256_file(args.split_file), 'test_accessed': False}
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f'{coarse_path}: {key} mismatch')
    if (payload.get('lambda_match') != (args.lambda_match if arm == 'M1' else 0.) or
            payload.get('temperature') != args.temperature):
        raise ValueError('fine stage hyperparameters differ from selected coarse run')
    model.load_state_dict(payload['model_state_dict'], strict=True)
    if frozen_local_digest(model) != payload['local_start_digest']:
        raise AssertionError('coarse stage changed local/upscaler initialization')
    return model, sha256_file(coarse_path)


def checkpoint(path, args, model, optimizer, scheduler, arm, stage,
               epoch, steps, best, history, parent_sha, local_digest):
    payload = {'recipe': RECIPE, 'arm': arm, 'stage': stage,
               'source_sha256': SOURCE_SHA256, 'parent_sha256': parent_sha,
               'pretrained_sha256': sha256_file(args.pretrained),
               'attention_checkpoint_sha256': sha256_file(args.attention_checkpoint),
               'split_sha256': sha256_file(args.split_file),
               'lambda_match': args.lambda_match if arm == 'M1' else 0.,
               'temperature': args.temperature,
               'local_start_digest': local_digest,
               'model_state_dict': model.state_dict(),
               'optimizer_state_dict': optimizer.state_dict(),
               'scheduler_state_dict': scheduler.state_dict(),
               'epoch': epoch, 'optimizer_steps': steps,
               'best_val_final_epe_512px': best, 'history': history,
               'torch_rng_state': torch.get_rng_state(),
               'numpy_rng_state': np.random.get_state(),
               'python_rng_state': random.getstate(),
               'cuda_rng_state': (torch.cuda.get_rng_state()
                                  if torch.cuda.is_available() else None),
               'test_accessed': False}
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_epoch(model, optimizer, videos, device, epoch, arm, args,
                expected_steps=None):
    model.train()
    frames = Frames(videos, phase=(epoch-1) % TRAIN_FRAME_STRIDE)
    ordered = loader(frames, epoch, num_workers=args.data_workers)
    totals = {'existing': 0., 'match': 0., 'total': 0., 'queries': 0,
              'valid_pixels': 0, 'steps': 0, 'preclip_grad_max': 0.,
              'match_shared_grad_first_batch': 0.}
    order = hashlib.sha256()
    shared = list(model.base.pyramid._modules['level_4'].parameters()) + list(
        model.base.coarse_attention.parameters()) + list(model.base.coarse_dns.parameters())
    for raw in ordered:
        batch = augment(raw, epoch, enabled=True)
        for name in batch['name']:
            order.update((name + '\n').encode())
        optimizer.zero_grad(set_to_none=True)
        use_match = arm == 'M1' and model.base.train_coarse_encoder
        if use_match:
            with pre_mutual_scores(model) as captured:
                prediction = model(batch, device)
            if len(captured) != 1:
                raise AssertionError('global correlation must run exactly once')
            scores = query_key_scores(captured[0])
            labels = correspondence_labels(batch['flow_map'].to(device).float(),
                                           batch['correspondence_mask'].to(device))
            auxiliary = match_loss(scores, labels, args.temperature)
            query_count = int(labels['valid'].sum())
            if totals['steps'] == 0:
                match_grad = torch.autograd.grad(auxiliary, shared, retain_graph=True,
                                                 allow_unused=True)
                shared_norm = _grad_norm(match_grad)
                totals['match_shared_grad_first_batch'] = shared_norm
        else:
            prediction = model(batch, device)
            auxiliary = prediction['final512'].sum() * 0
            query_count = 0
        existing, _parts, valid_pixels = objective(prediction, batch, device,
                                                   hierarchical=True)
        loss = existing + (args.lambda_match * auxiliary if use_match else 0.)
        if not torch.isfinite(loss):
            raise RuntimeError(f'nonfinite loss at {batch["name"]}')
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group['params']], 1.)
        if not torch.isfinite(gradient):
            raise RuntimeError(f'nonfinite gradient at {batch["name"]}')
        optimizer.step()
        totals['existing'] += float(existing.detach()) * valid_pixels
        totals['match'] += float(auxiliary.detach()) * query_count
        totals['total'] += float(loss.detach()) * valid_pixels
        totals['queries'] += query_count
        totals['valid_pixels'] += valid_pixels
        totals['steps'] += 1
        totals['preclip_grad_max'] = max(totals['preclip_grad_max'], float(gradient))
    if expected_steps is None:
        expected_steps = prior.EXPECTED_TRAIN_FRAMES // TRAIN_FRAME_STRIDE // BATCH_SIZE
    if totals['steps'] != expected_steps:
        raise AssertionError('sample count or optimization budget changed')
    totals['image_order_sha256'] = order.hexdigest()
    totals['existing'] /= totals['valid_pixels']
    totals['total'] /= totals['valid_pixels']
    totals['match'] = totals['match'] / totals['queries'] if totals['queries'] else None
    return totals


def train_stage(args, arm, stage, train_videos, val_videos, device):
    directory = args.output / arm
    best_path = directory / f'best_{stage}.pth'
    if best_path.exists():
        raise FileExistsError(f'will not overwrite {best_path}')
    directory.mkdir(parents=True, exist_ok=True)
    model, parent_sha = load_stage_start(args, arm, stage, device)
    local_digest = frozen_local_digest(model)
    active = prior.active_parameters(model, stage)
    locked = frozen_digest(model)
    optimizer = torch.optim.AdamW(active, lr=prior.LR[stage])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=prior.EPOCHS[stage], eta_min=prior.ETA_MIN[stage])
    _, zero_summary, best = prior.validation(model, val_videos, device,
                                              args.data_workers)
    best_epoch = steps = 0
    history = []
    checkpoint(best_path, args, model, optimizer, scheduler, arm, stage,
               0, 0, best, history, parent_sha, local_digest)
    for epoch in range(1, prior.EPOCHS[stage]+1):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        update = train_epoch(model, optimizer, train_videos, device,
                             epoch, arm, args)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - started
        train_peak_mib = (torch.cuda.max_memory_allocated(device)/1024**2
                          if device.type == 'cuda' else None)
        steps += update['steps']
        started = time.perf_counter()
        _, validation, score = prior.validation(model, val_videos, device,
                                                 args.data_workers)
        val_seconds = time.perf_counter() - started
        lr_used = optimizer.param_groups[0]['lr']
        scheduler.step()
        if frozen_digest(model) != locked or (stage == 'coarse' and
                                               frozen_local_digest(model) != local_digest):
            raise AssertionError('frozen parameters or BN buffers changed')
        record = {'epoch': epoch, 'cumulative_steps': steps, 'train': update,
                  'validation': validation, 'lr_used': lr_used,
                  'train_wall_seconds': train_seconds,
                  'train_peak_allocated_mib': train_peak_mib,
                  'validation_wall_seconds': val_seconds,
                  'gates': prior.gates(model)}
        history.append(record)
        if score < best:
            best, best_epoch = score, epoch
            checkpoint(best_path, args, model, optimizer, scheduler,
                       arm, stage, epoch, steps, best, history,
                       parent_sha, local_digest)
        save_json(directory/f'history_{stage}.json', history)
        print(f'{arm} {stage} {epoch}/{prior.EPOCHS[stage]}: '
              f'val final={score:.4f}, best={best:.4f}@{best_epoch}, '
              f'coarse={validation["all"]["stage_epe_512px"]["coarse16"]:.4f}, '
              f'L_existing={update["existing"]:.4f}, L_match={update["match"]}, '
              f'queries={update["queries"]}, grad={update["preclip_grad_max"]:.3f}, '
              f'steps={steps}', flush=True)
    selected = torch.load(best_path, map_location='cpu', weights_only=False)
    report = {'recipe': RECIPE, 'arm': arm, 'stage': stage,
              'source_sha256': SOURCE_SHA256, 'parent_sha256': parent_sha,
              'best_checkpoint_sha256': sha256_file(best_path),
              'local_start_digest': local_digest,
              'selected_epoch': best_epoch,
              'selected_steps': selected['optimizer_steps'],
              'completed_epochs': prior.EPOCHS[stage], 'completed_steps': steps,
              'epoch_zero': zero_summary, 'selected_final_epe_512px': best,
              'selection': 'full 4-sequence valid-pixel-weighted final 512px EPE',
              'optimizer': 'AdamW', 'lr': prior.LR[stage],
              'scheduler': 'CosineAnnealingLR',
              'seed': SEED, 'batch_size': BATCH_SIZE,
              'train_frame_stride': TRAIN_FRAME_STRIDE,
              'train_sequences': len(train_videos.sequences),
              'validation_sequences': len(val_videos.sequences),
              'trainable_parameters': sum(p.numel() for p in active),
              'total_parameters': sum(p.numel() for p in model.parameters()),
              'loss': 'masked Charbonnier final + 0.1 local32 + 0.2 local64'
                      + (' + lambda_match pre-MM soft CE' if arm == 'M1' and stage == 'coarse' else ''),
              'lambda_match': args.lambda_match if arm == 'M1' else 0.,
              'temperature': args.temperature,
              'train_seconds': sum(r['train_wall_seconds'] for r in history),
              'peak_train_allocated_mib': max((r['train_peak_allocated_mib']
                                               for r in history), default=None),
              'validation_seconds': sum(r['validation_wall_seconds'] for r in history),
              'test_accessed': False}
    save_json(directory/f'report_{stage}.json', report)
    return report


def actual_window(center512, truth512, queries, size):
    """All 9x9 warped-source candidate positions at the actual local input."""
    yy, xx = torch.meshgrid(torch.arange(size, device=center512.device),
                            torch.arange(size, device=center512.device), indexing='ij')
    xy = torch.stack((xx, yy))[None].to(center512)
    source_xy = xy + center512 * (size / 512.)
    expected_xy = xy + truth512 * (size / 512.)
    candidate = F.unfold(source_xy, 9, padding=4).reshape(
        center512.shape[0], 2, 81, size, size)
    present = F.unfold(torch.ones_like(center512[:, :1]), 9, padding=4).reshape(
        center512.shape[0], 1, 81, size, size)[:, 0] > .5
    inbounds = (present & (candidate[:, 0] >= 0) & (candidate[:, 0] <= size-1) &
                (candidate[:, 1] >= 0) & (candidate[:, 1] <= size-1))
    distances = (candidate-expected_xy[:, :, None]).abs().amax(1)
    nearest = distances.masked_fill(~inbounds, float('inf')).amin(1)
    return {'half': queries & (nearest <= .5),
            'one': queries & (nearest <= 1.),
            'no_candidate': queries & ~inbounds.any(1),
            'queries': queries}


def _frame_metrics(model, videos, frame, device, temperature):
    """Predict before opening evaluation GT; score every method on its GT mask."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with pre_mutual_scores(model) as captured:
        stages = geometry_audit.infer(model, frame, device)
    if len(captured) != 1:
        raise AssertionError('expected one pre-MM score volume per frame')
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    forward_ms = (time.perf_counter()-started)*1000
    flows = geometry_audit.full_flows(stages)
    started = time.perf_counter()
    final_flow = flows['final512'][0].permute(1, 2, 0).cpu().numpy()
    warped_ir, warp_valid = backward_warp(frame['ir'], final_flow)
    warp_ms = (time.perf_counter()-started)*1000
    truth_np, valid_np = videos.flow_truth(
        frame['sequence'], frame['stem'], frame['original_hw'])
    truth = torch.from_numpy(truth_np).permute(2, 0, 1)[None].to(device)
    valid = torch.from_numpy(valid_np)[None].to(device)
    pixels = int(valid.sum())
    if not pixels:
        raise ValueError('validation frame has no valid GT pixels')
    labels = correspondence_labels(truth, valid)
    scores = query_key_scores(captured[0])
    corr = match_statistics(scores, labels)
    corr_ce = float(match_loss(scores, labels, temperature))
    row = {'sequence': frame['sequence'], 'stem': frame['stem'],
           'valid_pixels': pixels, 'forward_ms': forward_ms,
           'output_warp_ms': warp_ms,
           'corr_soft_ce': corr_ce,
           'predicted_source_inbounds_on_gt_fraction': float(
               (warp_valid.astype(bool) & valid_np).sum()/pixels),
           **{f'corr_{key}': value for key, value in corr.items()}}
    displacement = torch.linalg.vector_norm(truth, dim=1)
    large = valid & (displacement >= 64)
    edge = edge_quartile(torch.from_numpy(frame['vi']).permute(2, 0, 1)[None].to(device),
                         valid[0])[None]
    for name in ('raw16', 'deconv32', 'decoded32', 'decoded64',
                 'decoded128', 'final512'):
        error = torch.linalg.vector_norm(flows[name]-truth, dim=1)
        row[f'{name}_epe_sum'] = float(error[valid].sum())
        row[f'{name}_edge_epe_sum'] = float(error[edge].sum())
        row[f'{name}_large_epe_sum'] = float(error[large].sum())
        for threshold in (1, 3, 5):
            row[f'{name}_pck{threshold}_hits'] = int(((error < threshold) & valid).sum())
    row['edge_pixels'] = int(edge.sum())
    row['large_pixels'] = int(large.sum())
    row['final_frame_mean_epe_512px'] = row['final512_epe_sum']/pixels
    for size, before, after in ((32, 'warp32_center', 'decoded32'),
                                (64, 'warp64_center', 'decoded64'),
                                (128, 'warp128_center', 'decoded128')):
        native_gt, queries = pooled_truth(truth, valid, size)
        center = stages[before] * (2 if size == 32 else 1)
        updated = stages[after] * (2 if size == 32 else 1)
        coverage = actual_window(center, native_gt, queries, size)
        for key, mask in coverage.items():
            row[f'window{size}_{key}'] = int(mask.sum())
        local_error = torch.linalg.vector_norm(updated-native_gt, dim=1)
        for group in ('half', 'one'):
            mask = coverage[group]
            row[f'window{size}_{group}_after_epe_sum'] = float(local_error[mask].sum())
        if size == 32:
            eligible = F.interpolate(coverage['queries'][:, None].float(),
                                     (512, 512), mode='nearest')[:, 0].bool() & valid
            assigned = F.interpolate(coverage['half'][:, None].float(),
                                     (512, 512), mode='nearest')[:, 0].bool() & eligible
            other = eligible & ~assigned
            unassigned = valid & ~eligible
            final_error = torch.linalg.vector_norm(flows['final512']-truth, dim=1)
            row['first_window_inside_pixels'] = int(assigned.sum())
            row['first_window_outside_pixels'] = int(other.sum())
            row['first_window_unassigned_pixels'] = int(unassigned.sum())
            row['first_window_inside_final_epe_sum'] = float(final_error[assigned].sum())
            row['first_window_outside_final_epe_sum'] = float(final_error[other].sum())
            row['first_window_unassigned_final_epe_sum'] = float(final_error[unassigned].sum())
    retention, f1 = edge_metrics(frame['vi'], warped_ir, valid_np)
    row.update({'nmi_64bin_2mi_entropy_sum': nmi(frame['vi'], warped_ir, valid_np),
                'lncc17_squared_proxy': lncc(frame['vi'], warped_ir, valid_np),
                'edge_information_retention_proxy': retention,
                'edge_overlap_f1_3x3_proxy': f1})
    return row


def summarize_frames(rows):
    def ratio(numerator, denominator):
        return numerator/denominator if denominator else None
    pixels = sum(r['valid_pixels'] for r in rows)
    out = {'frames': len(rows), 'valid_pixels': pixels,
           'valid_gt_fraction': pixels/(len(rows)*512*512),
           'predicted_source_inbounds_on_gt_fraction': sum(
               r['predicted_source_inbounds_on_gt_fraction']*r['valid_pixels']
               for r in rows)/pixels,
           'flow_epe_512px': {}, 'pck_pixel_fraction': {},
           'cmr_frame_fraction': {}, 'corr_pre_mutual': {},
           'actual_local_windows': {},
           'forward_ms_mean': float(np.mean([r['forward_ms'] for r in rows])),
           'forward_ms_p95': float(np.percentile([r['forward_ms'] for r in rows], 95)),
           'forward_plus_warp_ms_mean': float(np.mean([
               r['forward_ms']+r['output_warp_ms'] for r in rows])),
           'worst_frames': [{'sequence': r['sequence'], 'stem': r['stem'],
                             'final_epe_512px': r['final_frame_mean_epe_512px']}
                            for r in sorted(rows, key=lambda r: -r['final_frame_mean_epe_512px'])[:5]]}
    for name in ('raw16', 'deconv32', 'decoded32', 'decoded64',
                 'decoded128', 'final512'):
        out['flow_epe_512px'][name] = sum(r[f'{name}_epe_sum'] for r in rows)/pixels
        out['pck_pixel_fraction'][name] = {str(t): sum(
            r[f'{name}_pck{t}_hits'] for r in rows)/pixels for t in (1, 3, 5)}
    out['cmr_frame_fraction'] = {str(t): sum(
        r['final_frame_mean_epe_512px'] < t for r in rows)/len(rows)
        for t in (1, 3, 5)}
    count = sum(r['corr_queries'] for r in rows)
    out['corr_pre_mutual'] = {
        'valid_queries': count,
        'soft_ce_temperature_fixed': ratio(sum(r['corr_soft_ce']*r['corr_queries']
                                                for r in rows), count),
        'gt_rank_mean_best_bilinear_corner': ratio(sum(r['corr_rank_sum'] for r in rows), count),
        'top1_bilinear_support': ratio(sum(r['corr_top1_hits'] for r in rows), count),
        'top1_nearest_single_key': ratio(sum(r['corr_nearest_top1_hits'] for r in rows), count),
        'top5_bilinear_support_recall': ratio(sum(r['corr_top5_support_hits'] for r in rows), count),
        'top1_within_one_grid_cell': ratio(sum(r['corr_neighbor1_hits'] for r in rows), count)}
    for size in (32, 64, 128):
        total = sum(r[f'window{size}_queries'] for r in rows)
        out['actual_local_windows'][str(size)] = {
            'radius_grid_cells': 4, 'valid_queries': total,
            'within_half_cell_fraction': ratio(sum(r[f'window{size}_half'] for r in rows), total),
            'within_one_cell_fraction': ratio(sum(r[f'window{size}_one'] for r in rows), total),
            'no_valid_candidate_fraction': ratio(sum(r[f'window{size}_no_candidate'] for r in rows), total),
            'within_half_after_epe_512px': ratio(sum(
                r[f'window{size}_half_after_epe_sum'] for r in rows),
                sum(r[f'window{size}_half'] for r in rows))}
    out['first32_actual_window_half_cell_groups'] = {
        group: {'pixels': sum(r[f'first_window_{group}_pixels'] for r in rows),
                'final_epe_512px': ratio(sum(r[f'first_window_{group}_final_epe_sum']
                                              for r in rows),
                                         sum(r[f'first_window_{group}_pixels'] for r in rows))}
        for group in ('inside', 'outside', 'unassigned')}
    for group, count_key in (('edge', 'edge_pixels'), ('large_motion_64plus', 'large_pixels')):
        count = sum(r[count_key] for r in rows)
        stem = 'edge' if group == 'edge' else 'large'
        out[group] = {'pixels': count,
                      'final_epe_512px': ratio(sum(r[f'final512_{stem}_epe_sum']
                                                    for r in rows), count)}
    for metric in ('nmi_64bin_2mi_entropy_sum', 'lncc17_squared_proxy',
                   'edge_information_retention_proxy', 'edge_overlap_f1_3x3_proxy'):
        out[metric] = float(np.mean([r[metric] for r in rows]))
    return out


def evaluate_selected(args, arm, stage, videos, device):
    path = args.output / arm / f'best_{stage}.pth'
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if (payload.get('recipe'), payload.get('arm'), payload.get('stage'),
        payload.get('source_sha256'), payload.get('test_accessed')) != (
            RECIPE, arm, stage, SOURCE_SHA256, False):
        raise ValueError(f'wrong selected checkpoint: {path}')
    model, _ = initialize(args, device)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    model.eval()
    model.base.train_coarse_encoder = False
    for p in model.parameters():
        p.requires_grad_(False)
    rows = []
    first_sequence = next(iter(videos.sequences))
    first_stem = videos.sequences[first_sequence][0]
    with torch.inference_mode():
        # Identical warmup count for all arms and selected stages.
        warmup = videos.input_frame(first_sequence, first_stem)
        for _ in range(2):
            geometry_audit.infer(model, warmup, device)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        for sequence, stems in videos.sequences.items():
            for stem in stems:
                frame = videos.input_frame(sequence, stem)
                rows.append(_frame_metrics(model, videos, frame, device,
                                           args.temperature))
            print(f'{arm} {stage}: evaluated {sequence} ({len(stems)} frames)', flush=True)
    if len(rows) != prior.EXPECTED_VALID_FRAMES:
        raise AssertionError('selected-checkpoint report must cover all 800 validation frames')
    directory = args.output / arm
    prior.save_csv(directory / f'per_frame_best_{stage}.csv', rows)
    summary = {'all': summarize_frames(rows),
               'by_sequence': {sequence: summarize_frames([
                   r for r in rows if r['sequence'] == sequence])
                   for sequence in videos.sequences}}
    result = {'recipe': RECIPE, 'arm': arm, 'stage': stage,
              'best_checkpoint_sha256': sha256_file(path),
              'selected_epoch': payload['epoch'],
              'validation': summary,
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'peak_inference_allocated_mib': (torch.cuda.max_memory_allocated(device)/1024**2
                                                if device.type == 'cuda' else None),
              'test_accessed': False}
    save_json(directory/f'selected_eval_{stage}.json', result)
    return rows, result


def comparison_report(args, videos, device):
    if tuple(videos.sequences) != prior.SPLIT_NAMES or sum(map(len, videos.sequences.values())) != 800:
        raise ValueError('comparison requires the fixed 4 validation sequences / 800 frames')
    reports, frames = {}, {}
    for arm in ARMS:
        reports[arm], frames[arm] = {}, {}
        for stage in STAGES:
            frames[arm][stage], reports[arm][stage] = evaluate_selected(
                args, arm, stage, videos, device)
        coarse_path = args.output/arm/'best_coarse.pth'
        fine_payload = torch.load(args.output/arm/'best_fine.pth',
                                  map_location='cpu', weights_only=False)
        if fine_payload['parent_sha256'] != sha256_file(coarse_path):
            raise AssertionError(f'{arm} fine checkpoint uses the wrong coarse parent')
    paired = []
    for row0, row1 in zip(frames['M0']['fine'], frames['M1']['fine']):
        if any(row0[key] != row1[key] for key in ('sequence', 'stem', 'valid_pixels')):
            raise AssertionError('M0/M1 validation frames or GT masks differ')
        paired.append({'sequence': row0['sequence'], 'stem': row0['stem'],
                       'valid_pixels': row0['valid_pixels'],
                       'M0_final_epe_512px': row0['final_frame_mean_epe_512px'],
                       'M1_final_epe_512px': row1['final_frame_mean_epe_512px'],
                       'M1_minus_M0_final_epe_512px': row1['final_frame_mean_epe_512px']
                                                     -row0['final_frame_mean_epe_512px'],
                       'M0_coarse_epe_512px': row0['raw16_epe_sum']/row0['valid_pixels'],
                       'M1_coarse_epe_512px': row1['raw16_epe_sum']/row1['valid_pixels']})
    prior.save_csv(args.output/'paired_per_frame.csv', paired)
    history = {}
    for arm in ARMS:
        history[arm] = {stage: json.loads((args.output/arm/f'history_{stage}.json').read_text(
            encoding='utf-8')) for stage in STAGES}
    for stage in STAGES:
        left, right = history['M0'][stage], history['M1'][stage]
        if len(left) != prior.EPOCHS[stage] or len(right) != len(left):
            raise AssertionError('arms have unequal training budgets')
        for a, b in zip(left, right):
            if (a['cumulative_steps'], a['lr_used'], a['train']['image_order_sha256']) != (
                    b['cumulative_steps'], b['lr_used'], b['train']['image_order_sha256']):
                raise AssertionError('M0/M1 update count, LR or sample order differs')
    result = {'recipe': RECIPE, 'source_sha256': SOURCE_SHA256,
              'temperature': args.temperature,
              'lambda_match_M1_coarse_only': args.lambda_match,
              'arms': reports,
              'M1_improved_frames': sum(r['M1_minus_M0_final_epe_512px'] < 0 for r in paired),
              'M1_worsened_frames': sum(r['M1_minus_M0_final_epe_512px'] > 0 for r in paired),
              'metric_definitions': {
                  'EPE': 'valid-pixel-weighted XY flow norm at 512x512 pixels',
                  'PCK': 'fraction of GT-valid pixels with EPE <1/3/5px',
                  'CMR': 'fraction of frames whose GT-valid mean EPE <1/3/5px; frames are correlated, not independent repetitions',
                  'GT_supervision': 'synthetic VI warp homography; mask is geometric validity, not a complete occlusion mask',
                  'corr_GT_rank': 'best rank among positive-weight bilinear corners; tie order from torch.argsort',
                  'corr_top1': 'top-1 key in the positive-weight bilinear corner support',
                  'corr_nearest_top1': 'top-1 key equals rounded continuous GT position, a distinct stricter metric',
                  'corr_top5_recall': 'at least one of top-5 keys in positive-weight bilinear support',
                  'corr_neighbor1': 'top-1 key within one source coarse cell per axis of continuous GT',
                  'score': 'cosine dot product of featureL2Norm source/target descriptors before nonnegative clamp and MutualMatching',
                  'NMI': '64-bin grayscale 2MI/(H(VI)+H(warped IR))',
                  'LNCC': '17x17 squared local NCC proxy',
                  'edge': 'Sobel magnitude retention and top-quartile edge F1 at 1px tolerance; proxies, not paper R_edge',
                  'actual_window': '9x9 candidates at p+d+flow(p+d), half/one native-cell L-infinity GT distance, GT-valid query denominator',
                  'Q_F_AB': 'not computed: no fusion output',
                  'timing': 'same two CUDA warmups; forward includes model preprocessing, excludes disk/GT/metrics; forward+output warp includes final D2H transfer and CPU warp, excludes image disk read'},
              'test_accessed': False}
    save_json(args.output/'comparison.json', result)
    return result


def synthetic_check():
    """No dataset or network involved: directions, soft mass and empty masks."""
    yy, xx = torch.meshgrid(torch.arange(512), torch.arange(512), indexing='ij')
    xy = torch.stack((xx, yy)).float()
    cases = {}
    for name, matrix in {
            'identity': ((1., 0., 0.), (0., 1., 0.)),
            'integer_cell_translation': ((1., 0., 32.), (0., 1., -32.)),
            'noninteger_cell_translation': ((1., 0., 8.), (0., 1., -12.)),
            'rotation_scale': ((1.03, -.06, 7.), (.06, 1.03, -9.)),
            'source_edge': ((1., 0., 480.), (0., 1., 0.)),
            }.items():
        mapped = torch.stack((matrix[0][0]*xy[0] + matrix[0][1]*xy[1] + matrix[0][2],
                              matrix[1][0]*xy[0] + matrix[1][1]*xy[1] + matrix[1][2]))
        flow = (mapped-xy)[None]
        valid = ((mapped[0] >= 0) & (mapped[0] <= 511) &
                 (mapped[1] >= 0) & (mapped[1] <= 511))[None]
        labels = correspondence_labels(flow, valid)
        queries = int(labels['valid'].sum())
        if name != 'source_edge' and queries == 0:
            raise AssertionError(f'{name}: no usable coarse GT query')
        if name == 'identity':
            expected = torch.arange(256)[None]
            assert torch.equal(labels['indexes'][0, :, 0], expected[0])
            assert torch.equal(labels['dense'][0, expected[0], expected[0]],
                               torch.ones(256))
        if name == 'integer_cell_translation':
            query = 8*16+8
            assert labels['valid'][0, query]
            assert labels['dense'][0, query, 7*16+9] == 1
        if name == 'noninteger_cell_translation':
            query = 8*16+8
            assert labels['valid'][0, query]
            expected_xy = torch.tensor([8.25, 7.625])
            assert torch.allclose(labels['continuous_xy'][0, query], expected_xy,
                                  atol=1e-5)
            assert (labels['dense'][0, query] > 0).sum() == 4
        if name == 'rotation_scale':
            query = 8*16+8
            p = torch.tensor([8.5*32-.5, 8.5*32-.5])
            q = torch.tensor([matrix[0][0]*p[0]+matrix[0][1]*p[1]+matrix[0][2],
                              matrix[1][0]*p[0]+matrix[1][1]*p[1]+matrix[1][2]])
            predicted = (q+.5)/32-.5
            assert torch.allclose(labels['continuous_xy'][0, query], predicted,
                                  atol=1e-4)
        if name == 'source_edge':
            assert queries < 256 and not labels['valid'][0, 8*16+8]
        cases[name] = {'valid_queries': queries,
                       'valid_row_sum_min': float(labels['dense'].sum(-1)[labels['valid']].min())
                                            if queries else None}
    empty = correspondence_labels(torch.zeros(1, 2, 512, 512),
                                  torch.zeros(1, 512, 512, dtype=torch.bool))
    scores = torch.zeros(1, 256, 256, requires_grad=True)
    empty_loss = match_loss(scores, empty, .1)
    empty_loss.backward()
    if float(empty_loss) != 0 or not torch.isfinite(scores.grad).all() or torch.count_nonzero(scores.grad):
        raise AssertionError('empty supervision has nonzero or nonfinite gradient')
    labels = correspondence_labels(torch.zeros(1, 2, 512, 512),
                                   torch.ones(1, 512, 512, dtype=torch.bool))
    flat = torch.arange(256)
    perfect = torch.zeros(1, 256, 256)
    perfect[0, flat, flat] = 1.
    before = float(match_loss(torch.zeros_like(perfect), labels, .1))
    after = float(match_loss(perfect, labels, .1))
    if not after < before:
        raise AssertionError('correct key score did not decrease soft correspondence CE')
    if query_key_scores(torch.zeros(1, 1, 16, 16, 16, 16)).shape != (1, 256, 256):
        raise AssertionError('query/key flatten layout changed')
    return {'synthetic_cases': cases, 'empty_loss': float(empty_loss),
            'identity_ce_before': before, 'identity_ce_after': after,
            'direction': 'VI target queries -> IR source keys',
            'grid_center_512px': '(index+0.5)*32-0.5',
            'soft_label_mass': 'four bilinear corners, duplicate boundary keys merged',
            'test_accessed': False}


def _grad_norm(grads):
    values = [g.detach().float().square().sum() for g in grads if g is not None]
    return float(torch.sqrt(torch.stack(values).sum())) if values else 0.


def small_training_check(args, device):
    if args.data_root is None:
        return {'training_frame_smoke': 'skipped; --data-root not supplied'}
    if device.type != 'cuda':
        raise RuntimeError('one-step local-correlation smoke needs CUDA/CuPy')
    videos = VTMOTVideos(args.data_root, args.split_file, 'train',
                         max_sequences=1, max_frames=2)
    dataset = Frames(videos)
    sample = dataset[0]
    batch = {key: value[None] if isinstance(value, torch.Tensor) else [value]
             for key, value in sample.items()}
    model, _ = initialize(args, device)
    active = prior.active_parameters(model, 'coarse')
    locked = frozen_digest(model)
    with torch.no_grad():
        out0 = model(batch, device)
        initial_flows = {name: out0[name].detach().cpu().clone()
                         for name in ('coarse16', 'local32', 'local64', 'final512')}
        with pre_mutual_scores(model) as captured:
            out1 = model(batch, device)
        differences = {name: float((out0[name]-out1[name]).abs().max())
                       for name in ('coarse16', 'local32', 'local64', 'final512')}
        if any(differences.values()) or len(captured) != 1:
            raise AssertionError('temporary score hook changed original inference')
        score = query_key_scores(captured[0]).detach()
        truth = batch['flow_map'].to(device).float()
        valid = batch['correspondence_mask'].to(device)
        labels = correspondence_labels(truth, valid)
        if not labels['valid'].any():
            raise AssertionError('selected small training sample has no match supervision')
        score_range = {'minimum': float(score.min()), 'maximum': float(score.max()),
                       'mean': float(score.mean()), 'std': float(score.std())}
        if not torch.isfinite(score).all():
            raise AssertionError('pre-MM score is nonfinite')
    del out0, out1, captured
    model.zero_grad(set_to_none=True)
    with pre_mutual_scores(model) as captured:
        output = model(batch, device)
    existing = objective(output, batch, device, hierarchical=True)[0]
    auxiliary = match_loss(query_key_scores(captured[0]), labels, args.temperature)
    shared = [p for module in (model.base.pyramid._modules['level_4'],
                              model.base.coarse_attention, model.base.coarse_dns)
              for p in module.parameters()]
    existing_grad = torch.autograd.grad(existing, shared, retain_graph=True,
                                        allow_unused=True)
    matching_grad = torch.autograd.grad(auxiliary, shared, retain_graph=True,
                                        allow_unused=True)
    existing_norm, matching_norm = _grad_norm(existing_grad), _grad_norm(matching_grad)
    if matching_norm <= 0 or not np.isfinite(matching_norm):
        raise AssertionError('pre-MM supervision has no finite shared-feature gradient')
    if float(existing + 0*auxiliary) != float(existing):
        raise AssertionError('lambda=0 changes the original loss')
    existing_value = float(existing.detach())
    auxiliary_value = float(auxiliary.detach())
    calibration = [{'name': sample['name'], 'score': score_range,
                    'existing_loss': existing_value, 'match_loss': auxiliary_value,
                    'existing_shared_gradient_norm': existing_norm,
                    'match_shared_gradient_norm': matching_norm}]
    model.zero_grad(set_to_none=True)
    del output, captured, existing_grad, matching_grad, existing, auxiliary
    torch.cuda.empty_cache()
    second = dataset[1]
    second_batch = {key: value[None] if isinstance(value, torch.Tensor) else [value]
                    for key, value in second.items()}
    with pre_mutual_scores(model) as captured:
        second_output = model(second_batch, device)
    second_scores = query_key_scores(captured[0])
    second_labels = correspondence_labels(second_batch['flow_map'].to(device).float(),
                                          second_batch['correspondence_mask'].to(device))
    second_existing = objective(second_output, second_batch, device,
                                hierarchical=True)[0]
    second_match = match_loss(second_scores, second_labels, args.temperature)
    second_base_grad = torch.autograd.grad(second_existing, shared,
                                           retain_graph=True, allow_unused=True)
    second_match_grad = torch.autograd.grad(second_match, shared,
                                            allow_unused=True)
    calibration.append({'name': second['name'],
                        'score': {'minimum': float(second_scores.detach().min()),
                                  'maximum': float(second_scores.detach().max()),
                                  'mean': float(second_scores.detach().mean()),
                                  'std': float(second_scores.detach().std())},
                        'existing_loss': float(second_existing.detach()),
                        'match_loss': float(second_match.detach()),
                        'existing_shared_gradient_norm': _grad_norm(second_base_grad),
                        'match_shared_gradient_norm': _grad_norm(second_match_grad)})
    model.zero_grad(set_to_none=True)
    del second_output, second_scores, second_existing, second_match
    del second_base_grad, second_match_grad, captured
    torch.cuda.empty_cache()
    # One optimizer step in each arm; reset weights and optimizer completely.
    steps = {}
    arm_initial_difference = {'M0': {name: 0. for name in initial_flows}}
    for arm in ARMS:
        if arm == 'M1':
            del model, optimizer
            torch.cuda.empty_cache()
            model, _ = initialize(args, device)
            active = prior.active_parameters(model, 'coarse')
            locked = frozen_digest(model)
            with torch.no_grad():
                start = model(batch, device)
            arm_initial_difference['M1'] = {name: float(
                (start[name].detach().cpu()-initial_flows[name]).abs().max())
                for name in initial_flows}
            if any(arm_initial_difference['M1'].values()):
                raise AssertionError('M0/M1 independently initialized flows differ')
            del start
        optimizer = torch.optim.AdamW(active, lr=prior.LR['coarse'])
        optimizer.zero_grad(set_to_none=True)
        with pre_mutual_scores(model) as captured:
            output = model(batch, device)
        existing_loss = objective(output, batch, device, hierarchical=True)[0]
        match = match_loss(query_key_scores(captured[0]), labels, args.temperature)
        combined = existing_loss + (args.lambda_match*match if arm == 'M1' else 0.)
        if not torch.isfinite(combined):
            raise AssertionError(f'{arm} small-sample loss is nonfinite')
        combined.backward()
        norm = torch.nn.utils.clip_grad_norm_(active, 1.)
        if not torch.isfinite(norm):
            raise AssertionError(f'{arm} small-sample gradient is nonfinite')
        optimizer.step()
        if frozen_digest(model) != locked or any(
                m.training for m in model.base.modules()
                if isinstance(m, nn.modules.batchnorm._BatchNorm)):
            raise AssertionError(f'{arm} updated frozen parameters or BN state')
        steps[arm] = {'existing': float(existing_loss.detach()),
                      'match': float(match.detach()), 'total': float(combined.detach()),
                      'preclip_grad_norm': float(norm),
                      'optimizer_step': 'finite', 'frozen_digest_unchanged': True}
    result = {'training_sample': sample['name'], 'score_before_mutual': score_range,
              'pre_mutual_shape_query_key': list(score.shape),
              'valid_match_queries': int(labels['valid'].sum()),
              'initial_flow_hook_max_abs_difference': differences,
              'arm_initial_flow_max_abs_difference': arm_initial_difference,
              'lambda_zero_existing_loss_exact': True,
              'existing_shared_gradient_norm': existing_norm,
              'match_shared_gradient_norm': matching_norm,
              'weighted_match_to_existing_gradient_norm_ratio':
                  args.lambda_match*matching_norm/max(existing_norm, 1e-12),
              'two_train_frame_calibration': calibration,
              'existing_loss': existing_value,
              'match_loss': auxiliary_value,
              'temperature': args.temperature, 'lambda_match': args.lambda_match,
              'one_step_each_arm': steps,
              'peak_allocated_mib': torch.cuda.max_memory_allocated(device)/1024**2,
              'test_accessed': False}
    del model, optimizer, active
    torch.cuda.empty_cache()
    check_args = argparse.Namespace(**vars(args))
    check_args.data_workers = 0
    loop_results = {}
    for arm in ARMS:
        loop_model, _ = initialize(args, device)
        loop_active = prior.active_parameters(loop_model, 'coarse')
        loop_optimizer = torch.optim.AdamW(loop_active, lr=prior.LR['coarse'])
        loop_results[arm] = train_epoch(loop_model, loop_optimizer, videos,
                                        device, 1, arm, check_args,
                                        expected_steps=1)
        del loop_model, loop_active, loop_optimizer
        torch.cuda.empty_cache()
    if loop_results['M0']['image_order_sha256'] != loop_results['M1']['image_order_sha256']:
        raise AssertionError('real train loop gave M0/M1 different sample order')
    result['real_train_loop_one_step_each_arm'] = loop_results
    return result


def small_report_check(args, device):
    if args.data_root is None:
        return {'evaluation_smoke': 'skipped; --data-root not supplied'}
    videos = VTMOTVideos(args.data_root, args.split_file, 'train',
                         max_sequences=1, max_frames=1)
    sequence = next(iter(videos.sequences))
    frame = videos.input_frame(sequence, videos.sequences[sequence][0])
    model, _ = initialize(args, device)
    model.eval()
    model.base.train_coarse_encoder = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.inference_mode():
        row = _frame_metrics(model, videos, frame, device, args.temperature)
        summary = summarize_frames([row])
        truth_np, valid_np = videos.flow_truth(
            frame['sequence'], frame['stem'], frame['original_hw'])
        truth = torch.from_numpy(truth_np).permute(2, 0, 1)[None].to(device)
        valid = torch.from_numpy(valid_np)[None].to(device)
        captured = geometry_audit.infer(model, frame, device)
        native, queries = pooled_truth(truth, valid, 32)
        exact = geometry_audit.actual_candidates(captured['warp32_center']*2,
                                                 native, queries, 32)
    if (row['window32_half'] != exact['within_half_grid'] or
            row['window32_one'] != exact['within_one_grid'] or
            row['window32_no_candidate'] != exact['no_valid_candidate']):
        raise AssertionError('vectorized window candidates differ from audited implementation')
    if not all(np.isfinite(value) for value in (
            summary['flow_epe_512px']['final512'],
            summary['nmi_64bin_2mi_entropy_sum'],
            summary['lncc17_squared_proxy'])):
        raise AssertionError('small report emitted nonfinite metrics')
    return {'training_frame': f'{frame["sequence"]}/{frame["stem"]}',
            'report_fields': sorted(row),
            'vectorized_actual_window_matches_audit': True,
            'valid_pixels': row['valid_pixels'],
            'inference_report_finite': True, 'test_accessed': False}


def verify_manifest(args, stage):
    if stage == 'check':
        return None, None
    train = VTMOTVideos(args.data_root, args.split_file, 'train') if stage != 'report' else None
    valid = VTMOTVideos(args.data_root, args.split_file, 'eval')
    if tuple(valid.sequences) != prior.SPLIT_NAMES or sum(map(len, valid.sequences.values())) != 800:
        raise ValueError('validation manifest must be the fixed 4 sequences / 800 frames')
    if train is not None and (len(train.sequences) != 38 or
                              sum(map(len, train.sequences.values())) != 7600):
        raise ValueError('training manifest must contain 38 sequences / 7600 frames')
    return train, valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('check', 'coarse', 'fine', 'report'))
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--comparison', type=Path, default=Path(
        'roadscene_runs/vtmot_dns_adaptive_scale_v1/comparison.json'))
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--split-file', type=Path, default=Path(
        'experiments/vtmot_a1_adapt/split.json'))
    parser.add_argument('--pretrained', type=Path, default=Path(
        'pre_trained_models/GLUNet_DPED_CityScape_ADE.pth'))
    parser.add_argument('--attention-checkpoint', type=Path, default=Path(
        'roadscene_runs/sa_ca/best_attention.pth'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--temperature', type=float, default=.1)
    parser.add_argument('--lambda-match', type=float, default=.1)
    parser.add_argument('--data-workers', type=int, default=0)
    parser.add_argument('--allow-800-frame-eval', action='store_true')
    args = parser.parse_args()
    if args.temperature <= 0 or args.lambda_match < 0 or args.data_workers < 0:
        parser.error('temperature>0, lambda-match>=0, data-workers>=0 required')
    if args.stage in STAGES and args.arm is None:
        parser.error('coarse/fine stages require --arm M0 or M1')
    if args.stage != 'check' and args.data_root is None:
        parser.error('training/report require --data-root')
    if args.stage == 'report' and not args.allow_800_frame_eval:
        parser.error('report requires explicit --allow-800-frame-eval')
    install_mutual_guard()
    source_checkpoint(args)  # Fail before new output or model setup on SHA mismatch.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.stage == 'check':
        result = {'source_sha256': SOURCE_SHA256,
                  'synthetic': synthetic_check(),
                  'training_smoke': small_training_check(args, device),
                  'evaluation_smoke': small_report_check(args, device),
                  'mutual_guard_all_zero_safe': True,
                  'test_accessed': False}
        args.output.mkdir(parents=True, exist_ok=True)
        save_json(args.output/'check.json', result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    train, valid = verify_manifest(args, args.stage)
    if args.stage == 'report':
        result = comparison_report(args, valid, device)
        print(json.dumps({'M0_final_epe_512px': result['arms']['M0']['fine']
                          ['validation']['all']['flow_epe_512px']['final512'],
                          'M1_final_epe_512px': result['arms']['M1']['fine']
                          ['validation']['all']['flow_epe_512px']['final512'],
                          'test_accessed': False}, indent=2))
        return
    if device.type != 'cuda':
        raise RuntimeError('full local correlation training requires CUDA/CuPy')
    seed_all()
    report = train_stage(args, args.arm, args.stage, train, valid, device)
    print(json.dumps({'arm': args.arm, 'stage': args.stage,
                      'selected_epoch': report['selected_epoch'],
                      'selected_final_epe_512px': report['selected_final_epe_512px'],
                      'test_accessed': False}, indent=2))


if __name__ == '__main__':
    main()
