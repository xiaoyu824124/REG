"""Single-layer 64-grid DNS ablation from the selected VTMOT B0 fine model.

Only the new shared, zero-gated DNS trains. The existing SA->DNS->CA coarse
path, VGG, 32/64/128 local decoders, BN and flow upsamplers stay frozen.
Training uses the existing 38-sequence VTMOT recipe; model selection uses the
four fixed validation sequences and final 512px valid-pixel EPE. No test split.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import vtmot_dns_position_ablation as prior
from datasets.vtmot_video import VTMOTVideos
from models.our_models.coarse_dns import CoarseDNS
from roadscene_coarse import sha256_file
from roadscene_fusion_hierarchical import objective
from roadscene_local256 import edge_quartile
from roadscene_refinement_audit import pooled_truth
from vtmot_a1_adapt import (BATCH_SIZE, Frames, SEED, TRAIN_FRAME_STRIDE,
                            frozen_digest, save_json, seed_all, train_epoch)
from vtmot_geometry import backward_warp
from vtmot_keyframe import (edge_metrics, install_mutual_guard, lncc,
                            mse_ncc, nmi, sobel)


RECIPE = 'vtmot_local_dns64_v1'
STAGES = ('coarse16', 'local32', 'local64', 'final512')
THRESHOLDS = (1, 3, 5)
GROUPS = ('all', 'edge_top_quartile', 'large_motion_64plus',
          'window64_inside', 'window64_outside')
EPOCHS = 30
LR = 5e-6
ETA_MIN = 5e-7
GHOST_STEMS = ('000000', '000100')  # fixed before inspecting this experiment


def load_model(args, device, with_local_dns):
    """Strictly load the complete selected B0 fine weights, then add DNS."""
    seed_all()
    payload = torch.load(args.b0_checkpoint, map_location='cpu',
                         weights_only=False)
    expected = {
        'recipe': 'vtmot_dns_adaptive_scale_v1', 'arm': 'B0', 'stage': 'fine',
        'pretrained_sha256': sha256_file(args.pretrained),
        'attention_checkpoint_sha256': sha256_file(args.attention_checkpoint),
        'split_sha256': sha256_file(args.split_file), 'test_accessed': False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f'B0 fine checkpoint {key} mismatch')
    model = prior.initialize('B', args, device)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    if model.base.coarse_dns_order != 'between_sa_ca':
        raise AssertionError('B0 is not SA->DNS->CA')
    if with_local_dns:
        # VGG 512-input level_3 features are [B,256,64,64]. CoarseDNS is a
        # shared 2D residual descriptor, used here without adaptive weighting.
        model.base.local_dns64 = CoarseDNS(channels=256, hidden=128,
                                           radii=(1, 2)).to(device)
        if float(model.base.local_dns64.gate) != 0:
            raise AssertionError('local DNS gate must start at zero')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.base.train_coarse_encoder = False
    model.eval()
    return model


def freeze_for_training(model):
    if model.base.local_dns64 is None:
        raise ValueError('training requires local DNS')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    active = list(model.base.local_dns64.parameters())
    for parameter in active:
        parameter.requires_grad_(True)
    model.base.train_coarse_encoder = False
    model.train()  # FineArm.train keeps the base and BN in eval mode.
    if model.base.training or any(
            module.training for module in model.base.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)):
        raise AssertionError('frozen base or BN entered train mode')
    if any(p.requires_grad for name, p in model.named_parameters()
           if not name.startswith('base.local_dns64.')):
        raise AssertionError('old model parameter became trainable')
    return active


def input_only(batch):
    # Explicitly keep GT, masks and visible_gt out of the model interface.
    return {name: batch[name] for name in ('source_image', 'target_image')}


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def window64(pre_flow512, truth512, valid512):
    """Actual pre-64 warp centre and 9x9 correlation window, 512px units."""
    if pre_flow512.shape[-2:] != (64, 64):
        raise ValueError('expected actual 64-grid pre-warp flow')
    gt64, selected = pooled_truth(truth512, valid512, 64)
    residual = gt64 - pre_flow512
    inside = selected & (residual.abs().amax(dim=1) <= 32.)
    outside = selected & ~inside
    yy, xx = torch.meshgrid(
        torch.arange(64, device=pre_flow512.device),
        torch.arange(64, device=pre_flow512.device), indexing='ij')
    center_x = xx + pre_flow512[:, 0] / 8.
    center_y = yy + pre_flow512[:, 1] / 8.
    warp_center_inside = ((center_x >= 0) & (center_x <= 63) &
                          (center_y >= 0) & (center_y <= 63) & selected)
    pixel_inside = F.interpolate(inside[:, None].float(), (512, 512),
                                 mode='nearest')[:, 0].bool() & valid512
    pixel_outside = F.interpolate(outside[:, None].float(), (512, 512),
                                  mode='nearest')[:, 0].bool() & valid512
    return {'selected': selected, 'inside': inside, 'outside': outside,
            'residual': residual,
            'warp_center_inside': warp_center_inside,
            'pixel_inside': pixel_inside, 'pixel_outside': pixel_outside}


def initialization_check(args, device):
    install_mutual_guard()
    baseline = load_model(args, device, False)
    candidate = load_model(args, device, True)
    torch.manual_seed(SEED + 31)
    batch = {'source_image': torch.randint(0, 256, (1, 3, 512, 512),
                                           dtype=torch.uint8),
             'target_image': torch.randint(0, 256, (1, 3, 512, 512),
                                           dtype=torch.uint8)}
    with torch.inference_mode():
        baseline.base.capture_local64_pre_flow = True
        candidate.base.capture_local64_pre_flow = True
        output0 = baseline(batch, device)
        output1 = candidate(batch, device)
        differences = {stage: float((output0[stage]-output1[stage]).abs().max())
                       for stage in STAGES}
        pre_difference = float((baseline.base.local64_pre_flow -
                                candidate.base.local64_pre_flow).abs().max())
        if any(value != 0 for value in differences.values()) or pre_difference != 0:
            raise AssertionError('zero-gated local DNS changed B0 output')
        if baseline.base.local64_pre_flow.shape != (1, 2, 64, 64):
            raise AssertionError('unexpected pre-64 flow shape')
        # Known 512px displacements: 32px is within the 64-grid radius 4;
        # 40px is outside. The source warp displacement is flow/8.
        pre = torch.zeros(1, 2, 64, 64, device=device)
        truth = torch.zeros(1, 2, 512, 512, device=device)
        valid = torch.ones(1, 512, 512, dtype=torch.bool, device=device)
        truth[:, 0] = 32
        if not window64(pre, truth, valid)['inside'].all():
            raise AssertionError('32px synthetic correspondence left 64 window')
        truth[:, 0] = 40
        if not window64(pre, truth, valid)['outside'].all():
            raise AssertionError('40px synthetic correspondence entered 64 window')
    candidate.base.capture_local64_pre_flow = False
    active = freeze_for_training(candidate)
    original = frozen_digest(candidate)
    synthetic = dict(batch)
    synthetic['flow_map'] = torch.zeros(1, 2, 512, 512)
    synthetic['correspondence_mask'] = torch.ones(1, 512, 512, dtype=torch.bool)
    candidate.zero_grad(set_to_none=True)
    output = candidate(input_only(synthetic), device)
    loss = objective(output, synthetic, device, hierarchical=True)[0]
    loss.backward()
    gate_grad = candidate.base.local_dns64.gate.grad
    if gate_grad is None or not torch.isfinite(gate_grad) or float(gate_grad.abs()) == 0:
        raise AssertionError('local DNS gate lacks a finite input gradient')
    if any(p.grad is not None for name, p in candidate.named_parameters()
           if not name.startswith('base.local_dns64.')):
        raise AssertionError('frozen decoder/encoder received parameter gradient')
    if frozen_digest(candidate) != original:
        raise AssertionError('initial backward modified frozen tensors/BN')
    # A nonzero gate must expose gradients in the normally initialized
    # projection, verifying the frozen decoder was not wrapped in no_grad.
    candidate.zero_grad(set_to_none=True)
    with torch.no_grad():
        candidate.base.local_dns64.gate.fill_(.01)
    output = candidate(input_only(synthetic), device)
    loss = objective(output, synthetic, device, hierarchical=True)[0]
    loss.backward()
    projection_grad = candidate.base.local_dns64.projection[-1].weight.grad
    if (projection_grad is None or not torch.isfinite(projection_grad).all() or
            float(projection_grad.abs().sum()) == 0):
        raise AssertionError('local DNS projection has no gradient through frozen decoder')
    return {
        'source_b0_sha256': sha256_file(args.b0_checkpoint),
        'order': 'SA->DNS->CA at 16; shared DNS on each 64-grid modality before IR warp',
        'direction': 'VI target -> IR source',
        'grid64_features': [1, 256, 64, 64],
        'flow64_warp_conversion': '512-image-pixel flow / 8 = 64-grid pixels',
        'window': '9x9, radius 4 grid pixels = 32 image pixels per axis',
        'zero_gate_flow_max_abs_difference': differences,
        'pre64_flow_max_abs_difference': pre_difference,
        'zero_gate_gradient': float(gate_grad.detach()),
        'nonzero_gate_projection_gradient_l1': float(projection_grad.abs().sum()),
        'synthetic_32px_inside_40px_outside': True,
        'frozen_parameter_digest_unchanged': True,
        'base_parameters': sum(p.numel() for p in baseline.parameters()),
        'candidate_parameters': sum(p.numel() for p in candidate.parameters()),
        'new_dns_parameters': sum(p.numel() for p in active),
        'test_accessed': False,
    }


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def baseline_window_diagnostic(model, videos, device, output, workers):
    """Read-only B0 pre-64 residual and coverage before any DNS training."""
    model.eval()
    model.base.capture_local64_pre_flow = True
    options = {'num_workers': workers, 'pin_memory': device.type == 'cuda'}
    if workers:
        options['prefetch_factor'] = 2
    rows = []
    components = {'dx': [], 'dy': [], 'norm': []}
    for batch in DataLoader(Frames(videos), batch_size=1, shuffle=False,
                            **options):
        model(input_only(batch), device)
        truth = batch['flow_map'].to(device).float()
        valid = batch['correspondence_mask'].to(device).bool()
        window = window64(model.base.local64_pre_flow, truth, valid)
        selected = window['selected']
        residual = window['residual']
        dx = residual[:, 0][selected].float().cpu().numpy()
        dy = residual[:, 1][selected].float().cpu().numpy()
        norm = np.hypot(dx, dy)
        for key, data in (('dx', dx), ('dy', dy), ('norm', norm)):
            components[key].append(data)
        rows.append({'name': batch['name'][0],
                     'valid_queries': int(selected.sum()),
                     'inside_queries': int(window['inside'].sum()),
                     'outside_queries': int(window['outside'].sum()),
                     'warp_center_inside_queries':
                         int(window['warp_center_inside'].sum()),
                     'residual_norm_mean_512px': float(norm.mean()),
                     'residual_norm_median_512px': float(np.median(norm)),
                     'residual_norm_p95_512px': float(np.percentile(norm, 95))})
    model.base.capture_local64_pre_flow = False
    if len(rows) != prior.EXPECTED_VALID_FRAMES:
        raise AssertionError('baseline diagnostic did not cover 800 frames')
    totals = {key: sum(row[key] for row in rows)
              for key in ('valid_queries', 'inside_queries', 'outside_queries',
                          'warp_center_inside_queries')}
    residual_distribution = {}
    for key, parts in components.items():
        values = np.concatenate(parts)
        residual_distribution[key] = {
            'mean': float(values.mean()), 'median': float(np.median(values)),
            'p95': float(np.percentile(values, 95)),
            'abs_p95': float(np.percentile(np.abs(values), 95))}
    report = {
        'definition': 'GT area pooling to 64, selected if >0.8 GT-valid; '
                      'inside iff max absolute XY residual <=32 image px; '
                      'query denominator is selected 64-grid cells',
        'frames': len(rows), 'query_counts': totals,
        'inside_fraction': totals['inside_queries']/totals['valid_queries'],
        'warp_center_inside_fraction':
            totals['warp_center_inside_queries']/totals['valid_queries'],
        'residual_distribution_512px': residual_distribution,
        'test_accessed': False}
    save_csv(output/'baseline_window_per_frame.csv', rows)
    save_json(output/'baseline_window_diagnostic.json', report)
    return report


def checkpoint(path, args, model, optimizer, scheduler, epoch, steps, score,
               history):
    payload = {
        'recipe': RECIPE, 'source_b0_sha256': sha256_file(args.b0_checkpoint),
        'split_sha256': sha256_file(args.split_file),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch, 'optimizer_steps': steps,
        'best_val_final_epe_512px': score, 'history': history,
        'torch_rng_state': torch.get_rng_state(),
        'numpy_rng_state': np.random.get_state(),
        'python_rng_state': random.getstate(),
        'cuda_rng_state': (torch.cuda.get_rng_state()
                           if torch.cuda.is_available() else None),
        'test_accessed': False}
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train(args, train_videos, val_videos, device):
    best_path = args.output/'best_local_dns64.pth'
    if best_path.exists():
        raise FileExistsError(f'will not overwrite {best_path}')
    model = load_model(args, device, True)
    active = freeze_for_training(model)
    locked = frozen_digest(model)
    optimizer = torch.optim.AdamW(active, lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=ETA_MIN)
    _, epoch_zero, best = prior.validation(model, val_videos, device,
                                           args.data_workers)
    selected_epoch = steps = 0
    history = []
    peak_train_mib = 0.
    checkpoint(best_path, args, model, optimizer, scheduler, 0, 0, best, history)
    for epoch in range(1, EPOCHS+1):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        update = train_epoch(model, optimizer, train_videos, device, epoch,
                             num_workers=args.data_workers)
        if device.type == 'cuda':
            epoch_peak_mib = torch.cuda.max_memory_allocated(device)/1024**2
            peak_train_mib = max(peak_train_mib, epoch_peak_mib)
            update['peak_train_allocated_mib'] = epoch_peak_mib
        if update['steps'] != 760:
            raise AssertionError('unexpected optimizer step count')
        steps += update['steps']
        _, validation, score = prior.validation(model, val_videos, device,
                                                args.data_workers)
        lr_used = optimizer.param_groups[0]['lr']
        scheduler.step()
        history.append({'epoch': epoch, 'cumulative_steps': steps,
                        'train': update, 'validation': validation,
                        'lr_used': lr_used,
                        'local_dns_gate': float(model.base.local_dns64.gate.detach())})
        if frozen_digest(model) != locked:
            raise AssertionError('encoder/SA-DNS-CA/decoder/BN/upscaler changed')
        if score < best:
            best, selected_epoch = score, epoch
            checkpoint(best_path, args, model, optimizer, scheduler,
                       epoch, steps, best, history)
        save_json(args.output/'history.json', history)
        print(f'local DNS64 {epoch}/{EPOCHS}: val final={score:.4f}, '
              f'best={best:.4f}@{selected_epoch}, '
              f'gate={float(model.base.local_dns64.gate.detach()):.6f}, '
              f'steps={steps}', flush=True)
    return {'selected_epoch': selected_epoch, 'selected_steps':
            torch.load(best_path, map_location='cpu', weights_only=False)
                 ['optimizer_steps'],
            'completed_steps': steps, 'best_val_final_epe_512px': best,
            'epoch_zero': epoch_zero,
            'peak_train_allocated_mib': (peak_train_mib if device.type == 'cuda'
                                         else None),
            'best_checkpoint_sha256': sha256_file(best_path)}


def ghost_overlay(target_rgb, warped_ir_rgb, valid):
    """Red target edges and cyan registered-IR edges on fixed pixel positions."""
    target_edge = sobel(target_rgb)
    warped_edge = sobel(warped_ir_rgb)
    common_scale = max(float(np.percentile(target_edge[valid], 95)), 1e-6)
    red = np.clip(target_edge/common_scale, 0, 1)
    cyan = np.clip(warped_edge/common_scale, 0, 1)
    panel = np.stack((red, cyan, cyan), axis=-1)
    panel[~valid] = 0
    return (panel*255).astype(np.uint8)


def save_ghosts(output, baseline, candidate):
    target = output/'ghosts'
    target.mkdir(parents=True, exist_ok=True)
    if set(baseline) != set(candidate):
        raise AssertionError('ghost comparison uses different frame positions')
    for name in sorted(baseline):
        canvas = Image.new('RGB', (1024, 540), 'black')
        canvas.paste(Image.fromarray(baseline[name]), (0, 28))
        canvas.paste(Image.fromarray(candidate[name]), (512, 28))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), 'B0: VI red / warped IR cyan', fill='white')
        draw.text((520, 8), 'DNS64: VI red / warped IR cyan', fill='white')
        canvas.save(target/f'{name.replace("/", "_")}.png')


def summarize_rows(rows, error_file, sequence_offsets, sequence_names=None):
    """Exact global pixel quantiles from a temporary file-backed float array."""
    count = sum(int(row['valid_pixels']) for row in rows)
    if error_file.stat().st_size != count*4:
        raise AssertionError('pixel-error file size does not match GT-valid count')
    errors = np.memmap(error_file, mode='r', dtype=np.float32, shape=(count,))

    def collect(chosen, section):
        pixels = sum(int(row['valid_pixels']) for row in chosen)
        start, stop = section
        if stop-start != pixels:
            raise AssertionError('sequence pixel offsets differ from frame masks')
        median, p95 = np.quantile(errors[start:stop], [.5, .95])
        result = {'frames': len(chosen), 'valid_pixels': pixels,
                  'final_epe_512px': sum(row['final512_epe_512px'] *
                                         row['valid_pixels'] for row in chosen)/pixels,
                  'pixel_error_median_512px': float(median),
                  'pixel_error_p95_512px': float(p95),
                  'pck_512px': {str(t): sum(row[f'pck_{t}_hits'] for row in chosen)/pixels
                                for t in THRESHOLDS},
                  'cmr_frame_mean_epe_512px': {
                      str(t): sum(row['final512_epe_512px'] < t for row in chosen)/len(chosen)
                      for t in THRESHOLDS},
                  'cmr_frame_hits': {
                      str(t): sum(row['final512_epe_512px'] < t for row in chosen)
                      for t in THRESHOLDS},
                  'worst_frame': max(chosen, key=lambda row: row['final512_epe_512px'])['name'],
                  'worst_frame_epe_512px': max(row['final512_epe_512px'] for row in chosen),
                  'predicted_source_inside_fraction': sum(
                      row['predicted_source_inside_pixels'] for row in chosen)/pixels,
                  'inference_forward_ms_mean': float(np.mean(
                      [row['inference_forward_ms'] for row in chosen])),
                  'inference_forward_ms_p95': float(np.percentile(
                      [row['inference_forward_ms'] for row in chosen], 95)),
                  'registration_output_ms_mean': float(np.mean(
                      [row['registration_output_ms'] for row in chosen])),
                  'registration_output_ms_p95': float(np.percentile(
                      [row['registration_output_ms'] for row in chosen], 95)),
                  'frame_mean_metrics': {
                      key: float(np.mean([row[key] for row in chosen]))
                      for key in ('nmi_64bin_2mi_entropy_sum',
                                  'lncc17_squared_proxy',
                                  'edge_overlap_f1_3x3_proxy',
                                  'edge_information_retention_proxy',
                                  'visible_gt_rgb_mse_01',
                                  'visible_gt_rgb_ncc_01')},
                  'stage_epe_512px': {}, 'groups': {},
                  'window64_queries': {}}
        for stage in STAGES:
            result['stage_epe_512px'][stage] = sum(
                row[f'{stage}_epe_512px']*row['valid_pixels']
                for row in chosen)/pixels
        for group in GROUPS:
            region_pixels = sum(row[f'{group}_pixels'] for row in chosen)
            result['groups'][group] = {'pixels': region_pixels,
                                      'stage_epe_512px': {}}
            for stage in STAGES:
                result['groups'][group]['stage_epe_512px'][stage] = (
                    sum(row[f'{stage}_{group}_epe_sum_512px'] for row in chosen)
                    / region_pixels if region_pixels else None)
        for field in ('valid', 'inside', 'outside', 'warp_center_inside'):
            result['window64_queries'][field] = sum(
                row[f'window64_{field}_queries'] for row in chosen)
        result['window64_queries']['inside_fraction'] = (
            result['window64_queries']['inside'] /
            max(1, result['window64_queries']['valid']))
        return result

    all_result = collect(rows, (0, count))
    by_sequence = {}
    for name in (sequence_names if sequence_names is not None
                 else prior.SPLIT_NAMES):
        chosen = [row for row in rows if row['sequence'] == name]
        by_sequence[name] = collect(chosen, sequence_offsets[name])
    del errors
    return {'all': all_result, 'by_sequence': by_sequence}


@torch.inference_mode()
def evaluate_one(model, videos, device, output, arm, workers,
                 sequence_names=None, expected_frames=None):
    """One frozen model, identical frame set/mask; visible_gt only post-flow."""
    sequence_names = (prior.SPLIT_NAMES if sequence_names is None
                      else tuple(sequence_names))
    expected_frames = (prior.EXPECTED_VALID_FRAMES if expected_frames is None
                       else expected_frames)
    model.eval()
    model.base.capture_local64_pre_flow = True
    dataset = Frames(videos)
    first_sequence, first_stem = dataset.items[0]
    first_frame = videos.input_frame(first_sequence, first_stem)
    first_input = {
        'source_image': torch.from_numpy(first_frame['ir'].transpose(2, 0, 1).copy())[None],
        'target_image': torch.from_numpy(first_frame['vi'].transpose(2, 0, 1).copy())[None]}
    model(first_input, device)  # same one-forward CUDA warm-up per arm
    del first_input, first_frame
    synchronize(device)
    rows, ghosts = [], {}
    sequence_offsets = {}
    written = 0
    peak_inference_mib = 0.
    with tempfile.NamedTemporaryFile(prefix=f'{arm}_errors_', suffix='.f32',
                                     dir=output, delete=False) as stream:
        error_path = Path(stream.name)
        for sequence, stem in dataset.items:
            name = f'{sequence}/{stem}'
            if sequence not in sequence_offsets:
                sequence_offsets[sequence] = [written, None]
            synchronize(device)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            output_start = time.perf_counter()
            frame = videos.input_frame(sequence, stem)
            ir, vi = frame['ir'], frame['vi']
            input_batch = {
                'source_image': torch.from_numpy(ir.transpose(2, 0, 1).copy())[None],
                'target_image': torch.from_numpy(vi.transpose(2, 0, 1).copy())[None]}
            forward_start = time.perf_counter()
            prediction = model(input_batch, device)
            synchronize(device)
            forward_ms = (time.perf_counter()-forward_start)*1000
            if device.type == 'cuda':
                peak_inference_mib = max(
                    peak_inference_mib,
                    torch.cuda.max_memory_allocated(device)/1024**2)
            flow = prediction['final512'][0].permute(1, 2, 0).float().cpu().numpy()
            warped_ir, source_inside = backward_warp(ir, flow)
            output_ms = (time.perf_counter()-output_start)*1000
            # GT and visible_gt are first accessed after the model emitted flow.
            truth_eval, valid_eval, visible_gt = videos.evaluation_truth(
                sequence, stem, frame['original_hw'])
            truth = torch.from_numpy(truth_eval.transpose(2, 0, 1).copy())[
                None].to(device).float()
            valid = torch.from_numpy(valid_eval.copy())[None].to(device).bool()
            valid_np = valid_eval.astype(bool)
            warped_gt, _ = backward_warp(visible_gt, flow)
            same_mse, same_ncc = mse_ncc(vi, warped_gt, valid_np)
            retention, edge_f1 = edge_metrics(vi, warped_ir, valid_np)
            window = window64(model.base.local64_pre_flow, truth, valid)
            edge = edge_quartile(input_batch['target_image'].to(device), valid[0])
            large = valid[0] & (torch.linalg.vector_norm(truth[0], dim=0) >= 64.)
            masks = {'all': valid[0], 'edge_top_quartile': edge,
                     'large_motion_64plus': large,
                     'window64_inside': window['pixel_inside'][0],
                     'window64_outside': window['pixel_outside'][0]}
            row = {'name': name, 'sequence': sequence, 'stem': stem,
                   'valid_pixels': int(valid.sum()),
                   'gt_mask_sha256': hashlib.sha256(
                       np.ascontiguousarray(valid_np).tobytes()).hexdigest(),
                   'inference_forward_ms': forward_ms,
                   'registration_output_ms': output_ms,
                   'predicted_source_inside_pixels': int(
                       (source_inside & valid_np).sum()),
                   'nmi_64bin_2mi_entropy_sum': nmi(vi, warped_ir, valid_np),
                   'lncc17_squared_proxy': lncc(vi, warped_ir, valid_np),
                   'edge_overlap_f1_3x3_proxy': edge_f1,
                   'edge_information_retention_proxy': retention,
                   'visible_gt_rgb_mse_01': same_mse,
                   'visible_gt_rgb_ncc_01': same_ncc,
                   'window64_valid_queries': int(window['selected'].sum()),
                   'window64_inside_queries': int(window['inside'].sum()),
                   'window64_outside_queries': int(window['outside'].sum()),
                   'window64_warp_center_inside_queries':
                       int(window['warp_center_inside'].sum()),
                   'window64_pre_flow_sha256': hashlib.sha256(
                       model.base.local64_pre_flow.detach().cpu().contiguous()
                       .numpy().tobytes()).hexdigest()}
            for group, mask in masks.items():
                row[f'{group}_pixels'] = int(mask.sum())
            for stage in STAGES:
                error = torch.linalg.vector_norm(prediction[stage][0]-truth[0],
                                                  dim=0)
                row[f'{stage}_epe_512px'] = float(error[valid[0]].mean())
                for group, mask in masks.items():
                    row[f'{stage}_{group}_epe_sum_512px'] = float(
                        error[mask].sum())
                if stage == 'final512':
                    error_np = error.float().cpu().numpy()
            selected_errors = error_np[valid_np].astype(np.float32)
            if not len(selected_errors) or not np.isfinite(selected_errors).all():
                raise AssertionError('empty or nonfinite valid EPE')
            stream.write(selected_errors.tobytes())
            written += len(selected_errors)
            sequence_offsets[sequence][1] = written
            row['pixel_error_median_512px'] = float(np.median(selected_errors))
            row['pixel_error_p95_512px'] = float(np.percentile(selected_errors, 95))
            for threshold in THRESHOLDS:
                hits = int(np.count_nonzero(selected_errors < threshold))
                row[f'pck_{threshold}_hits'] = hits
                row[f'pck_{threshold}_fraction'] = hits/len(selected_errors)
                row[f'cmr_{threshold}_frame_hit'] = int(
                    row['final512_epe_512px'] < threshold)
            for group in GROUPS:
                count = row[f'{group}_pixels']
                for stage in STAGES:
                    row[f'{stage}_{group}_epe_512px'] = (
                        row[f'{stage}_{group}_epe_sum_512px']/count
                        if count else None)
            finite_keys = ('inference_forward_ms', 'registration_output_ms',
                           'nmi_64bin_2mi_entropy_sum', 'lncc17_squared_proxy',
                           'edge_overlap_f1_3x3_proxy',
                           'edge_information_retention_proxy',
                           'visible_gt_rgb_mse_01', 'visible_gt_rgb_ncc_01')
            if any(not np.isfinite(row[key]) for key in finite_keys):
                raise AssertionError(f'nonfinite metric at {name}')
            rows.append(row)
            if stem in GHOST_STEMS:
                ghosts[name] = ghost_overlay(vi, warped_ir, valid_np)
            del prediction, truth, valid, input_batch
    model.base.capture_local64_pre_flow = False
    try:
        if (len(rows) != expected_frames or
                tuple(sequence_offsets) != sequence_names or
                len(ghosts) != len(sequence_names)*len(GHOST_STEMS)):
            raise AssertionError('evaluation frame manifest/ghost positions differ')
        summary = summarize_rows(rows, error_path, sequence_offsets,
                                 sequence_names)
    finally:
        error_path.unlink(missing_ok=True)
    summary['peak_inference_allocated_mib'] = (
        peak_inference_mib if device.type == 'cuda' else None)
    summary['parameters_total'] = sum(p.numel() for p in model.parameters())
    summary['parameters_new_dns'] = (
        sum(p.numel() for p in model.base.local_dns64.parameters())
        if model.base.local_dns64 is not None else 0)
    save_csv(output/f'per_frame_{arm}.csv', rows)
    return rows, summary, ghosts


def evaluate_pair(args, videos, device, train_report):
    baseline = load_model(args, device, False)
    baseline_rows, baseline_summary, baseline_ghosts = evaluate_one(
        baseline, videos, device, args.output, 'B0', args.data_workers)
    del baseline
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    candidate = load_model(args, device, True)
    best = torch.load(args.output/'best_local_dns64.pth', map_location='cpu',
                      weights_only=False)
    if (best.get('recipe') != RECIPE or
            best.get('source_b0_sha256') != sha256_file(args.b0_checkpoint) or
            best.get('split_sha256') != sha256_file(args.split_file) or
            best.get('test_accessed') is not False):
        raise ValueError('selected local DNS checkpoint provenance mismatch')
    candidate.load_state_dict(best['model_state_dict'], strict=True)
    candidate_rows, candidate_summary, candidate_ghosts = evaluate_one(
        candidate, videos, device, args.output, 'local_dns64', args.data_workers)
    save_ghosts(args.output, baseline_ghosts, candidate_ghosts)
    paired = []
    for base, new in zip(baseline_rows, candidate_rows):
        if (base['name'] != new['name'] or
                base['valid_pixels'] != new['valid_pixels'] or
                base['gt_mask_sha256'] != new['gt_mask_sha256'] or
                base['window64_pre_flow_sha256'] !=
                new['window64_pre_flow_sha256']):
            raise AssertionError('arms differ in frame, valid pixels or frozen pre64 flow')
        paired.append({'name': base['name'], 'sequence': base['sequence'],
                       'stem': base['stem'], 'valid_pixels': base['valid_pixels'],
                       'B0_final_epe_512px': base['final512_epe_512px'],
                       'local_dns64_final_epe_512px': new['final512_epe_512px'],
                       'local_dns64_minus_B0_epe_512px':
                           new['final512_epe_512px']-base['final512_epe_512px'],
                       'B0_pixel_p95_512px': base['pixel_error_p95_512px'],
                       'local_dns64_pixel_p95_512px': new['pixel_error_p95_512px']})
    save_csv(args.output/'paired_per_frame.csv', paired)
    per_sequence = []
    for name in prior.SPLIT_NAMES:
        left = baseline_summary['by_sequence'][name]
        right = candidate_summary['by_sequence'][name]
        record = {'sequence': name, 'frames': left['frames'],
                  'valid_pixels': left['valid_pixels'],
                  'B0_final_epe_512px': left['final_epe_512px'],
                  'local_dns64_final_epe_512px': right['final_epe_512px'],
                  'local_dns64_minus_B0_epe_512px':
                      right['final_epe_512px']-left['final_epe_512px'],
                  'improved_frames': sum(
                      row['local_dns64_minus_B0_epe_512px'] < 0
                      for row in paired if row['sequence'] == name),
                  'B0_worst_frame': left['worst_frame'],
                  'local_dns64_worst_frame': right['worst_frame']}
        for label, result in (('B0', left), ('local_dns64', right)):
            for threshold in THRESHOLDS:
                record[f'{label}_pck_{threshold}'] = result['pck_512px'][str(threshold)]
                record[f'{label}_cmr_{threshold}'] = result[
                    'cmr_frame_mean_epe_512px'][str(threshold)]
            for metric in ('pixel_error_median_512px', 'pixel_error_p95_512px',
                           'worst_frame_epe_512px', 'predicted_source_inside_fraction',
                           'inference_forward_ms_mean', 'inference_forward_ms_p95',
                           'registration_output_ms_mean',
                           'registration_output_ms_p95'):
                record[f'{label}_{metric}'] = result[metric]
            for stage in STAGES:
                record[f'{label}_{stage}_epe_512px'] = result[
                    'stage_epe_512px'][stage]
            for group in GROUPS:
                record[f'{label}_{group}_final_epe_512px'] = result[
                    'groups'][group]['stage_epe_512px']['final512']
                record[f'{label}_{group}_pixels'] = result['groups'][group]['pixels']
            for metric, value in result['frame_mean_metrics'].items():
                record[f'{label}_{metric}'] = value
            record[f'{label}_window64_inside_fraction'] = result[
                'window64_queries']['inside_fraction']
        per_sequence.append(record)
    save_csv(args.output/'per_sequence.csv', per_sequence)
    report = {'recipe': RECIPE, 'source_b0_sha256': sha256_file(args.b0_checkpoint),
              'selected_checkpoint_sha256':
                  sha256_file(args.output/'best_local_dns64.pth'),
              'split_sha256': sha256_file(args.split_file),
              'selection': 'lowest full 4-sequence validation valid-pixel-weighted '
                           'final 512px EPE; fixed before training',
              'training': train_report,
              'B0': baseline_summary, 'local_dns64': candidate_summary,
              'selected_local_dns_gate': float(candidate.base.local_dns64.gate.detach()),
              'improved_frames': sum(row['local_dns64_minus_B0_epe_512px'] < 0
                                     for row in paired),
              'worsened_frames': sum(row['local_dns64_minus_B0_epe_512px'] > 0
                                     for row in paired),
              'per_sequence': per_sequence,
              'metric_definitions': {
                  'direction': 'VI target to IR source, backward XY flow',
                  'flow_unit': 'all stage outputs and EPE in 512x512 image pixels',
                  'mask': 'same GT-valid VI->IR mask for both arms and all '
                          'metrics; predicted out-of-bounds remains in denominator',
                  'epe': 'L2 XY error, sum over all valid pixels / valid count',
                  'pck': 'valid pixels with EPE strictly <1/3/5px / all valid pixels',
                  'cmr': 'frames whose own valid-pixel mean EPE is strictly '
                         '<1/3/5px / frame count; not pixel PCK',
                  'pixel_quantiles': 'exact pooled valid-pixel median/P95 from '
                                     'temporary float32 file, not mean frame quantiles',
                  'nmi_64bin_2mi_entropy_sum': 'grayscale uint8//4, 64 bins; '
                      '2*MI/(H(VI)+H(warped IR)), frame arithmetic mean',
                  'lncc17_squared_proxy': '17x17 local NCC squared, constant '
                      'box boundary, valid-pixel mean then frame arithmetic mean',
                  'edge_top_quartile': 'VI Sobel magnitude on RGB/255 with '
                      '3x3 /8 kernels; per-frame 75th percentile of GT-valid '
                      'pixels; region EPE pooled by region pixel count',
                  'edge_overlap_f1_3x3_proxy': 'per-image GT-valid top-quartile '
                      'Sobel edges, 3x3 tolerance, frame arithmetic mean',
                  'edge_information_retention_proxy': 'sum min(VI,warped IR) '
                      'Sobel magnitudes / sum VI magnitudes, frame mean',
                  'visible_gt_rgb_mse_01': 'predicted VI->IR warp samples '
                      'visible_gt; compare visible_mis in RGB [0,1] over '
                      'unchanged GT-valid mask; frame arithmetic mean',
                  'visible_gt_rgb_ncc_01': 'same RGB [0,1] flattened NCC '
                      'with existing project epsilon; frame arithmetic mean',
                  'window64': 'actual pre64 warp centre in 512px units; '
                      'GT area pooled to64 with valid fraction>0.8; 9x9 '
                      'radius4 window means max XY residual<=32px; query '
                      'fraction separate from nearest-assigned pixel regions',
                  'large_motion': 'GT flow L2 magnitude >=64px',
                  'timing': 'CUDA-synchronized forward includes model '
                      'preprocessing+forward; registration output additionally '
                      'includes RGB input read/crop/resize, tensor creation, '
                      'flow transfer and IR backward warp. Both exclude '
                      'GT/visible_gt reading and metric computation; '
                      'one forward warm-up per arm',
                  'aggregation': 'EPE/PCK/region by valid pixels; CMR and '
                      'image proxies by frames; latency by frames',
                  'Q_F_AB': 'missing: no fusion output',
                  'paper_R_edge': 'missing: project edge F1 is a proxy',
                  'paper_LNCC': 'not claimed comparable to paper; squared proxy'},
              'test_accessed': False}
    save_json(args.output/'comparison.json', report)
    return report


def metric_synthetic_check(output):
    """A frame-level hit is distinct from an individual pixel-level hit."""
    first = np.array([0., 4.], np.float32)
    second = np.array([4., 4.], np.float32)
    pck3 = sum(np.count_nonzero(item < 3) for item in (first, second))/4
    cmr3 = sum(float(item.mean() < 3) for item in (first, second))/2
    if (pck3, cmr3) != (.25, .5):
        raise AssertionError('synthetic PCK/CMR aggregation mismatch')
    image = np.array([[[0, 128, 255], [255, 128, 0]]], np.uint8)
    mse, ncc = mse_ncc(image, image, np.ones((1, 2), dtype=bool))
    if mse != 0 or not np.isfinite(ncc):
        raise AssertionError('RGB [0,1] same-modality metric check failed')
    rows = []
    metric_names = ('nmi_64bin_2mi_entropy_sum', 'lncc17_squared_proxy',
                    'edge_overlap_f1_3x3_proxy',
                    'edge_information_retention_proxy',
                    'visible_gt_rgb_mse_01', 'visible_gt_rgb_ncc_01')
    for index, errors in enumerate((first, second)):
        row = {'name': f'synthetic/{index:06d}', 'sequence': 'synthetic',
               'valid_pixels': 2, 'final512_epe_512px': float(errors.mean()),
               'predicted_source_inside_pixels': 2,
               'inference_forward_ms': 1., 'registration_output_ms': 2.}
        row.update({name: 0. for name in metric_names})
        for threshold in THRESHOLDS:
            row[f'pck_{threshold}_hits'] = int((errors < threshold).sum())
        for stage in STAGES:
            row[f'{stage}_epe_512px'] = float(errors.mean())
        for group in GROUPS:
            count = 2 if group == 'all' else 0
            row[f'{group}_pixels'] = count
            for stage in STAGES:
                row[f'{stage}_{group}_epe_sum_512px'] = (
                    float(errors.sum()) if count else 0.)
        row.update({'window64_valid_queries': 1,
                    'window64_inside_queries': 1,
                    'window64_outside_queries': 0,
                    'window64_warp_center_inside_queries': 1})
        rows.append(row)
    with tempfile.NamedTemporaryFile(dir=output, suffix='.f32',
                                     delete=False) as stream:
        error_path = Path(stream.name)
        stream.write(np.concatenate((first, second)).astype(np.float32).tobytes())
    try:
        result = summarize_rows(rows, error_path,
                                {'synthetic': (0, 4)}, ('synthetic',))
    finally:
        error_path.unlink(missing_ok=True)
    if (result['all']['pck_512px']['3'] != pck3 or
            result['all']['cmr_frame_mean_epe_512px']['3'] != cmr3 or
            result['all']['pixel_error_median_512px'] != 4. or
            result['by_sequence']['synthetic']['valid_pixels'] != 4):
        raise AssertionError('full metric aggregation smoke check failed')
    return {'pck_at_3px': pck3, 'cmr_at_3px': cmr3,
            'pooled_pixel_median_512px': result['all']['pixel_error_median_512px'],
            'visible_gt_identity_mse_rgb_01': mse,
            'visible_gt_identity_ncc_rgb_01': ncc}


def evaluation_synthetic_check(args, device):
    """Exercise the entire evaluation path on two generated frames only."""
    class TinyVideos:
        def __init__(self):
            self.sequences = {'synthetic': ['000000', '000100']}
            self.visible = np.random.default_rng(SEED).integers(
                0, 256, (512, 512, 3), dtype=np.uint8)
            self.infrared = np.roll(self.visible, 2, axis=1).copy()

        def input_frame(self, sequence, stem):
            if sequence != 'synthetic' or stem not in self.sequences[sequence]:
                raise AssertionError('synthetic check accessed another sequence')
            return {'ir': self.infrared, 'vi': self.visible,
                    'original_hw': (512, 512)}

        def evaluation_truth(self, sequence, stem, original_hw):
            if original_hw != (512, 512):
                raise AssertionError('synthetic source geometry changed')
            return (np.zeros((512, 512, 2), np.float32),
                    np.ones((512, 512), bool), self.visible)

    model = load_model(args, device, False)
    rows, summary, ghosts = evaluate_one(
        model, TinyVideos(), device, args.output, 'synthetic_B0', 0,
        sequence_names=('synthetic',), expected_frames=2)
    if (len(rows) != 2 or summary['all']['valid_pixels'] != 2*512*512 or
            not np.isfinite(summary['all']['final_epe_512px']) or
            len(ghosts) != 2):
        raise AssertionError('synthetic full evaluation path failed')
    del model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return {'frames': len(rows), 'valid_pixels': summary['all']['valid_pixels'],
            'all_metric_paths_finite': True, 'test_accessed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('check', 'all', 'report'), required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--split-file', type=Path, required=True)
    parser.add_argument('--pretrained', type=Path, required=True)
    parser.add_argument('--attention-checkpoint', type=Path, required=True)
    parser.add_argument('--b0-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-workers', type=int, default=0)
    args = parser.parse_args()
    if args.data_workers < 0:
        parser.error('--data-workers must be nonnegative')
    if args.stage != 'check' and args.data_root is None:
        parser.error('--data-root is required for all/report')
    if args.stage == 'all' and args.output.exists() and any(args.output.iterdir()):
        parser.error('--output must be a new empty directory')
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_all()
    install_mutual_guard()
    metric_check = metric_synthetic_check(args.output)
    geometry_check = initialization_check(args, device)
    checks = {'geometry_and_gradient': geometry_check,
              'metric_aggregation': metric_check,
              'synthetic_evaluation_pipeline':
                  evaluation_synthetic_check(args, device),
              'device': str(device), 'test_accessed': False}
    save_json(args.output/'initialization_check.json', checks)
    if args.stage == 'check':
        print(json.dumps(checks, ensure_ascii=False, indent=2))
        return
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    train_videos = VTMOTVideos(args.data_root, args.split_file, 'train')
    val_videos = VTMOTVideos(args.data_root, args.split_file, 'eval')
    if (len(train_videos.sequences) != 38 or
            sum(map(len, train_videos.sequences.values())) !=
            prior.EXPECTED_TRAIN_FRAMES or
            tuple(val_videos.sequences) != prior.SPLIT_NAMES or
            sum(map(len, val_videos.sequences.values())) !=
            prior.EXPECTED_VALID_FRAMES):
        raise ValueError('expected exactly 38 training and four fixed validation '
                         'sequences with 200 canonical frames each')
    if args.stage == 'all':
        baseline = load_model(args, device, False)
        baseline_window_diagnostic(baseline, val_videos, device,
                                   args.output, args.data_workers)
        del baseline
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        train_report = train(args, train_videos, val_videos, device)
        save_json(args.output/'train_report.json', train_report)
    else:
        best = args.output/'best_local_dns64.pth'
        if not best.is_file():
            parser.error(f'no selected DNS checkpoint at {best}')
        train_report_path = args.output/'train_report.json'
        train_report = (json.loads(train_report_path.read_text(encoding='utf-8'))
                        if train_report_path.is_file() else
                        {'selected_epoch': torch.load(best, map_location='cpu',
                                                      weights_only=False)['epoch']})
    result = evaluate_pair(args, val_videos, device, train_report)
    short = {'B0_final_epe_512px': result['B0']['all']['final_epe_512px'],
             'local_dns64_final_epe_512px': result['local_dns64']['all']['final_epe_512px'],
             'improved_frames': result['improved_frames'],
             'comparison_json': str(args.output/'comparison.json')}
    print(json.dumps(short, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
