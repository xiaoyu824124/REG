"""Frozen-A1 VTMOT control: geometric propagation with/without causal match state.

Stages: check (synthetic or five real frames), train (also prepares the train
schedule), eval. Test sequences are deliberately inaccessible through VTMOTVideos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from datasets.vtmot_video import VTMOTVideos
from vtmot_geometry import backward_warp, grid, sample, self_test
from vtmot_keyframe import (frame_metrics, install_mutual_guard, load_a1,
                            gray, predict_a1, save_ghost, save_json, sha256,
                            summarise, temporal_fast_self_test,
                            temporal_pair_fast, temporal_quality, write_rows)
from vtmot_local_correction_ab import (A1_SHA, SPLIT_NAMES, TRIGGER,
                                       region_summary)


SEED = 2026
EPOCHS = 30
LR = 1e-4
RADIUS = 2
GRID = 128
SCALE = 512 // GRID
ANCHORS = (20, 70, 120, 170)
MATCH_CHANNELS = (2 * RADIUS + 1) ** 2
RECIPE = 'vtmot_causal_match_state_ab_v1'


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def tensor_digest(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def xy_grid(height, width, device):
    yy, xx = torch.meshgrid(torch.arange(height, device=device),
                            torch.arange(width, device=device), indexing='ij')
    return torch.stack((xx, yy), dim=-1).float()[None]


def sample_torch(field, coordinates, mode='bilinear'):
    height, width = field.shape[-2:]
    inside = ((coordinates[..., 0] >= 0) & (coordinates[..., 0] <= width - 1) &
              (coordinates[..., 1] >= 0) & (coordinates[..., 1] <= height - 1) &
              torch.isfinite(coordinates).all(dim=-1))
    safe = torch.nan_to_num(coordinates, nan=-2., posinf=-2., neginf=-2.)
    norm = torch.stack((safe[..., 0] * (2. / max(1, width - 1)) - 1,
                        safe[..., 1] * (2. / max(1, height - 1)) - 1), dim=-1)
    return F.grid_sample(field, norm, mode=mode, padding_mode='zeros',
                         align_corners=True), inside[:, None]


def local_correlation(vi_feature, ir_feature, base_grid_flow):
    """All tensors are on the current VI feature grid; source is current IR."""
    if vi_feature.shape != ir_feature.shape or vi_feature.shape[0] != 1:
        raise ValueError('expected one paired 128-grid feature map')
    _, _, height, width = vi_feature.shape
    if base_grid_flow.shape != (1, 2, height, width):
        raise ValueError('base flow must use feature-grid pixel units')
    coordinates = xy_grid(height, width, vi_feature.device) + base_grid_flow.permute(0, 2, 3, 1)
    vi_norm = F.normalize(vi_feature, dim=1, eps=1e-6)
    ir_norm = F.normalize(ir_feature, dim=1, eps=1e-6)
    evidence = []
    for dy in range(-RADIUS, RADIUS + 1):
        for dx in range(-RADIUS, RADIUS + 1):
            offset = coordinates + coordinates.new_tensor([dx, dy])
            sampled, inside = sample_torch(ir_norm, offset)
            evidence.append((vi_norm * sampled).sum(dim=1, keepdim=True) * inside)
    _, center_inside = sample_torch(ir_norm[:, :1], coordinates)
    return torch.cat(evidence, dim=1), center_inside


def warp_history(previous_state, previous_mask, vi_temporal_grid_flow,
                 vi_fb_valid, current_mask):
    if previous_state is None:
        return torch.zeros_like(current_mask).expand(-1, MATCH_CHANNELS, -1, -1), \
            torch.zeros_like(current_mask, dtype=torch.bool)
    height, width = previous_state.shape[-2:]
    coordinates = xy_grid(height, width, previous_state.device) + \
        vi_temporal_grid_flow.permute(0, 2, 3, 1)
    aligned, inside = sample_torch(previous_state, coordinates)
    old_valid, _ = sample_torch(previous_mask.float(), coordinates)
    valid = inside & (old_valid > .999) & vi_fb_valid & current_mask
    return aligned * valid, valid


class CausalMatchHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(2 * MATCH_CHANNELS + 3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.residual = nn.Conv2d(32, 2, 3, padding=1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, current, state, base_grid_flow, valid):
        inputs = torch.cat((current, state, base_grid_flow / 16., valid.float()), dim=1)
        return self.residual(self.body(inputs)) * valid


def feature_pair(model, frame, device):
    """Only the frozen VGG quarter-resolution stage; no A1 decoder call."""
    source = torch.from_numpy(frame['ir'].transpose(2, 0, 1).copy())[None].to(device)
    target = torch.from_numpy(frame['vi'].transpose(2, 0, 1).copy())[None].to(device)
    with torch.no_grad():
        source, target, *_ = model.base.pre_process_data(source, target, device=device)
        vi_feature = model.base.pyramid(target, quarter_resolution_only=True)[0]
        ir_feature = model.base.pyramid(source, quarter_resolution_only=True)[0]
    if vi_feature.shape[-2:] != (GRID, GRID) or ir_feature.shape != vi_feature.shape:
        raise RuntimeError(f'expected matching 128-grid VGG features: {vi_feature.shape}')
    return vi_feature.detach(), ir_feature.detach()


def branch_step(head, model, frame, base_flow, base_valid, temporal,
                previous_state, previous_mask, device, keyframe):
    """Return 512px VI->IR flow and a detached VI-grid state.

    The B residual never enters the A geometric propagation trajectory.
    """
    vi_feature, ir_feature = feature_pair(model, frame, device)
    flow = torch.from_numpy(base_flow.transpose(2, 0, 1).copy())[None].to(device)
    base_grid_flow = F.interpolate(flow, (GRID, GRID), mode='bilinear',
                                   align_corners=False) / SCALE
    evidence, inside = local_correlation(vi_feature, ir_feature, base_grid_flow)
    base_mask = torch.from_numpy(base_valid.copy())[None, None].to(device)
    current_mask = inside & F.interpolate(base_mask.float(), (GRID, GRID),
                                          mode='nearest').bool()
    evidence = evidence * current_mask
    if keyframe:
        state, state_mask = evidence.detach(), current_mask.detach()
        return flow, state, state_mask
    if temporal is None or previous_state is None:
        raise ValueError('non-keyframe requires causal history and temporal flow')
    vi_temporal = torch.from_numpy(temporal['vi'].transpose(2, 0, 1).copy())[None].to(device)
    vi_temporal = F.interpolate(vi_temporal, (GRID, GRID), mode='bilinear',
                                align_corners=False) / SCALE
    vi_ok = temporal['vi_bounds'] & (temporal['vi_error'] <= TRIGGER['fb_pixel_threshold'])
    vi_ok = torch.from_numpy(vi_ok.copy())[None, None].to(device)
    vi_ok = F.interpolate(vi_ok.float(), (GRID, GRID), mode='nearest').bool()
    old, old_valid = warp_history(previous_state, previous_mask, vi_temporal,
                                  vi_ok, current_mask)
    state = (evidence + old) / (1. + old_valid.float())
    residual_grid = head(evidence, state, base_grid_flow, current_mask)
    residual_full = F.interpolate(residual_grid, (512, 512), mode='bilinear',
                                  align_corners=False) * SCALE
    candidate = flow + residual_full * base_mask
    source_xy = xy_grid(512, 512, device) + candidate.permute(0, 2, 3, 1)
    inside = ((source_xy[..., 0] >= 0) & (source_xy[..., 0] <= 511) &
              (source_xy[..., 1] >= 0) & (source_xy[..., 1] <= 511))[:, None]
    result = flow + residual_full * base_mask * inside
    return result, state.detach(), current_mask.detach()


def frozen_schedule(path, split_file, expected_names, expected_frames):
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload['a1_sha256'] != A1_SHA or payload['frames'] != expected_frames or \
       set(payload['keyframes']) != set(expected_names):
        raise ValueError('schedule/A1/frame provenance mismatch')
    if 'split_sha256' in payload and payload['split_sha256'] != sha256(split_file):
        raise ValueError('training schedule split hash mismatch')
    if any('000000' not in stems for stems in payload['keyframes'].values()):
        raise ValueError('each sequence must start with an A1 keyframe')
    return {name: set(stems) for name, stems in payload['keyframes'].items()}


def prepare_train(args, dataset, model, device):
    schedule = {}
    for sequence, stems in dataset.sequences.items():
        chosen, previous, previous_flow, previous_valid, age = [], None, None, None, 0
        for index, stem in enumerate(stems):
            frame = dataset.input_frame(sequence, stem)
            if index == 0:
                reason, quality = 'sequence_start', None
            else:
                temporal = temporal_pair_fast(previous, frame)
                quality = temporal_quality(temporal, previous_flow, previous_valid,
                                           TRIGGER['fb_pixel_threshold'])
                if quality['valid_fraction'] < TRIGGER['minimum_valid_fraction']:
                    reason = 'low_temporal_coverage'
                elif quality['fb_median_px'] > TRIGGER['maximum_fb_median_px']:
                    reason = 'forward_backward_error'
                elif age >= TRIGGER['fixed_interval'] - 1:
                    reason = 'fixed_interval'
                else:
                    reason = None
            if reason:
                flow, valid, _ = predict_a1(model, frame, device)
                chosen.append(stem)
                age = 0
            else:
                flow, valid = quality['flow'], quality['valid']
                age += 1
            previous, previous_flow, previous_valid = frame, flow, valid
        schedule[sequence] = chosen
        print(f'prepared {sequence}: {len(chosen)} fixed keyframes', flush=True)
    save_json(args.output / 'train_schedule.json', {
        'recipe': RECIPE, 'split_sha256': sha256(args.split_file),
        'a1_sha256': A1_SHA, 'frames': sum(map(len, dataset.sequences.values())),
        'thresholds': TRIGGER, 'keyframes': schedule, 'test_accessed': False})
    return schedule


def synthetic_check():
    seed_all()
    self_test()
    temporal_fast_self_test()
    head = CausalMatchHead()
    vi = torch.randn(1, 8, 16, 16)
    ir = torch.roll(vi, shifts=(-1, 2), dims=(-2, -1))
    cost, valid = local_correlation(vi, ir, torch.zeros(1, 2, 16, 16))
    peak = int(cost[0, :, 4:12, 4:12].mean(dim=(1, 2)).argmax())
    if peak != (RADIUS - 1) * (2 * RADIUS + 1) + (RADIUS + 2):
        raise AssertionError(f'VI->IR local correlation sign failed: {peak}')
    old = torch.randn(1, MATCH_CHANNELS, 16, 16)
    old_mask = torch.ones(1, 1, 16, 16, dtype=torch.bool)
    current_mask = torch.ones_like(old_mask)
    offset = torch.zeros(1, 2, 16, 16)
    offset[:, 0] = 1
    aligned, mask = warp_history(old, old_mask, offset, current_mask, current_mask)
    if (not torch.allclose(aligned[..., 2:14, 2:14], old[..., 2:14, 3:15], atol=1e-5) or
            not mask[..., 2:14, 2:14].all()):
        raise AssertionError('history warp is not current VI -> previous VI')
    zero = head(cost, cost, torch.zeros(1, 2, 16, 16), valid)
    if torch.count_nonzero(zero):
        raise AssertionError('zero-initialized residual must be exactly zero')
    zero.sum().backward()
    if head.residual.weight.grad is None or not torch.isfinite(head.residual.weight.grad).all():
        raise AssertionError('residual head gradient is invalid')
    # Exercise the actual branch interface on two synthetic frames.  The
    # second frame uses only an identity current-VI -> previous-VI motion.
    class ToyPyramid:
        @staticmethod
        def pre_process_data(source, target, device):
            return source.float()/255., target.float()/255., None, None

        @staticmethod
        def pyramid(image, quarter_resolution_only=False):
            quarter = F.interpolate(image, (GRID, GRID), mode='bilinear',
                                    align_corners=False)
            return [torch.cat((quarter, quarter, quarter[:, :2]), dim=1)]

    toy_model = SimpleNamespace(base=ToyPyramid())
    pattern = np.zeros((512, 512, 3), np.uint8)
    pattern[96:192, 128:224] = 255
    toy_frame = {'vi': pattern, 'ir': pattern.copy()}
    baseline = np.zeros((512, 512, 2), np.float32)
    support = np.ones((512, 512), bool)
    identity_temporal = {'vi': baseline.copy(), 'vi_bounds': support.copy(),
                         'vi_error': np.zeros((512, 512), np.float32)}
    first, state, state_mask = branch_step(
        head, toy_model, toy_frame, baseline, support, None, None, None,
        torch.device('cpu'), True)
    second, _, _ = branch_step(
        head, toy_model, toy_frame, baseline, support, identity_temporal,
        state, state_mask, torch.device('cpu'), False)
    if not (torch.count_nonzero(first) == 0 and torch.count_nonzero(second) == 0):
        raise AssertionError('full branch must exactly reproduce A at initialization')
    if not (state.shape == (1, MATCH_CHANNELS, GRID, GRID) and state_mask.all()):
        raise AssertionError('historical match state shape/support is wrong')
    return {'geometry': 'passed', 'temporal': 'passed', 'matching_peak_xy': [2, -1],
            'history_warp': 'passed', 'zero_residual': 'exact',
            'head_gradient': 'finite', 'two_frame_branch': 'exact A',
            'test_accessed': False}


def rollout(dataset, sequence, stems, schedule, model, head, device, timed=False):
    """A is autonomous; B state sees only present/past inputs and A proposals."""
    previous = previous_a = previous_valid = state = state_mask = None
    age = 0
    for index, stem in enumerate(stems):
        frame = dataset.input_frame(sequence, stem)
        temporal = None
        temporal_ms = propagation_ms = a1_ms = branch_ms = 0.
        if index:
            temporal = temporal_pair_fast(previous, frame)
            temporal_ms = temporal['ms']
            start = time.perf_counter()
            quality = temporal_quality(temporal, previous_a, previous_valid,
                                       TRIGGER['fb_pixel_threshold'])
            propagation_ms = 1000 * (time.perf_counter() - start)
        keyframe = stem in schedule[sequence]
        if index == 0 and not keyframe:
            raise ValueError(f'rollout must begin at a frozen keyframe: {sequence}/{stem}')
        if keyframe:
            start = time.perf_counter()
            a_flow, a_valid, _ = predict_a1(model, frame, device)
            a1_ms = 1000 * (time.perf_counter() - start)
            age = 0
        else:
            a_flow, a_valid = quality['flow'], quality['valid']
            age += 1
        synchronize(device)
        start = time.perf_counter()
        b_tensor, state, state_mask = branch_step(
            head, model, frame, a_flow, a_valid, temporal, state, state_mask,
            device, keyframe)
        synchronize(device)
        branch_ms = 1000 * (time.perf_counter() - start)
        yield {'sequence': sequence, 'stem': stem, 'index': index, 'frame': frame,
               'temporal': temporal, 'a_flow': a_flow, 'a_valid': a_valid,
               'b_tensor': b_tensor, 'keyframe': keyframe, 'age': age,
               'timing': {'temporal_ms': temporal_ms,
                          'propagation_ms': propagation_ms,
                          'a1_ms': a1_ms, 'branch_ms': branch_ms}}
        previous, previous_a, previous_valid = frame, a_flow, a_valid


def validation_epe(dataset, schedule, model, head, device, check_zero=False):
    head.eval()
    totals = {'a': 0., 'b': 0., 'pixels': 0, 'max_initial_difference': 0.}
    with torch.no_grad():
        for sequence, stems in dataset.sequences.items():
            for item in rollout(dataset, sequence, stems, schedule, model,
                                head, device):
                b_flow = item['b_tensor'][0].permute(1, 2, 0).cpu().numpy()
                if check_zero:
                    difference = float(np.max(np.abs(item['a_flow'] - b_flow)))
                    totals['max_initial_difference'] = max(totals['max_initial_difference'],
                                                            difference)
                    if not np.array_equal(item['a_flow'], b_flow):
                        raise AssertionError(f'zero head changed A at {sequence}/{item["stem"]}: '
                                             f'{difference}')
                truth, valid = dataset.flow_truth(sequence, item['stem'],
                                                  item['frame']['original_hw'])
                pixels = int(valid.sum())
                totals['a'] += float(np.linalg.norm(item['a_flow']-truth, axis=-1)[valid].sum())
                totals['b'] += float(np.linalg.norm(b_flow-truth, axis=-1)[valid].sum())
                totals['pixels'] += pixels
    return {'a_epe_512px': totals['a']/totals['pixels'],
            'b_epe_512px': totals['b']/totals['pixels'],
            'valid_pixels': totals['pixels'],
            'max_initial_difference': totals['max_initial_difference']}


def clip_stems(stems, keyframes, anchor):
    start = max(i for i in range(anchor + 1) if stems[i] in keyframes)
    return stems[start:min(anchor + 5, len(stems))]


def train_one_epoch(dataset, schedule, model, head, optimizer, device, epoch):
    head.train()
    order = [(sequence, anchor) for sequence in dataset.sequences for anchor in ANCHORS]
    random.Random(SEED + epoch).shuffle(order)
    steps, observed, loss_total = 0, 0, 0.
    order_digest = hashlib.sha256()
    for sequence, anchor in order:
        stems = dataset.sequences[sequence]
        clip = clip_stems(stems, schedule[sequence], anchor)
        order_digest.update(f'{sequence}:{anchor}\n'.encode())
        losses = []
        for item in rollout(dataset, sequence, clip, schedule, model, head, device):
            frame_index = int(item['stem'])
            if frame_index <= anchor or frame_index > anchor + 4 or item['keyframe']:
                continue
            truth, valid = dataset.flow_truth(sequence, item['stem'],
                                              item['frame']['original_hw'])
            if not valid.any():
                continue
            truth_t = torch.from_numpy(truth.transpose(2, 0, 1).copy())[None].to(device)
            valid_t = torch.from_numpy(valid.copy())[None].to(device)
            # Existing final-flow Charbonnier only. Frozen 32/64 stages cannot
            # receive gradients from this branch, so no dummy auxiliary loss.
            distance = torch.sqrt((item['b_tensor'] - truth_t).square().sum(dim=1) + .01)
            losses.append(distance[valid_t].mean())
            observed += 1
        if not losses:
            continue
        optimizer.zero_grad(set_to_none=True)
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f'nonfinite loss in {sequence} anchor={anchor}')
        loss.backward()
        preclip = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
        if not torch.isfinite(preclip):
            raise RuntimeError(f'nonfinite head gradient in {sequence} anchor={anchor}')
        optimizer.step()
        steps += 1
        loss_total += float(loss.detach())
    return {'epoch': epoch, 'steps': steps, 'supervised_frames': observed,
            'train_loss': loss_total/max(steps, 1),
            'image_order_sha256': order_digest.hexdigest(),
            'learning_rate': optimizer.param_groups[0]['lr']}


def run_training(args, trainset, valset, schedule_train, schedule_val,
                 model, head, device):
    frozen_before = tensor_digest(model)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6)
    initial = validation_epe(valset, schedule_val, model, head, device,
                             check_zero=True)
    frozen = json.loads(args.frozen_schedule.read_text(encoding='utf-8'))
    if (initial['valid_pixels'] != frozen['historical_gt_valid_pixels'] or
            abs(initial['a_epe_512px'] -
                frozen['historical_fixed_fast_epe_512px']) > .05):
        raise RuntimeError('A does not reproduce historical fixed_fast mask/EPE')
    if initial['a_epe_512px'] != initial['b_epe_512px']:
        raise RuntimeError('zero head is not the exact A baseline')
    history, best, best_epoch, steps = [], initial['b_epe_512px'], 0, 0
    schedule_sha = sha256(args.frozen_schedule)

    def save_best(epoch, score):
        torch.save({'recipe': RECIPE, 'stage': 'train', 'epoch': epoch,
                    'selected_val_epe_512px': score, 'cumulative_steps': steps,
                    'head_state_dict': head.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'torch_rng': torch.get_rng_state(),
                    'numpy_rng': np.random.get_state(),
                    'python_rng': random.getstate(),
                    'a1_sha256': A1_SHA, 'split_sha256': sha256(args.split_file),
                    'frozen_schedule_sha256': schedule_sha,
                    'train_schedule_sha256': sha256(args.output/'train_schedule.json'),
                    'frozen_model_digest': frozen_before,
                    'test_accessed': False}, args.output/'best_head.pth')

    save_best(0, best)
    print(f'epoch 0: A={initial["a_epe_512px"]:.4f}, '
          f'B={initial["b_epe_512px"]:.4f}, exact difference=0', flush=True)
    for epoch in range(1, EPOCHS + 1):
        record = train_one_epoch(trainset, schedule_train, model, head,
                                 optimizer, device, epoch)
        steps += record['steps']
        scheduler.step()
        val = validation_epe(valset, schedule_val, model, head, device)
        record['validation'] = val
        record['cumulative_steps'] = steps
        history.append(record)
        if val['b_epe_512px'] < best:
            best, best_epoch = val['b_epe_512px'], epoch
            save_best(epoch, best)
        save_json(args.output/'history.json', history)
        print(f'epoch {epoch}/{EPOCHS}: val A={val["a_epe_512px"]:.4f}, '
              f'B={val["b_epe_512px"]:.4f}, best={best:.4f} at {best_epoch}, '
              f'steps={steps}', flush=True)
    frozen_after = tensor_digest(model)
    if frozen_after != frozen_before:
        raise RuntimeError('frozen A1 parameters or BN buffers changed')
    save_json(args.output/'training_report.json', {
        'recipe': RECIPE, 'selected_epoch': best_epoch,
        'selected_val_epe_512px': best, 'initial': initial,
        'epochs': EPOCHS, 'cumulative_steps': steps,
        'frozen_a1_digest_before': frozen_before,
        'frozen_a1_digest_after': frozen_after,
        'training_samples': '38 train sequences; anchors 20/70/120/170; '
                            'four following frames each; frozen keyframe schedule',
        'loss': 'existing final-flow masked Charbonnier',
        'test_accessed': False})


def weighted_region(rows, error_key, count_key):
    count = sum(int(row[count_key]) for row in rows)
    value = sum(float(row[error_key]) * int(row[count_key]) for row in rows
                if row[error_key] is not None and row[count_key])
    return {'pixels': count, 'value': value/count if count else None}


def enrich_summary(rows, base):
    names = list(SPLIT_NAMES) + ['all']
    for name in names:
        subset = rows if name == 'all' else [r for r in rows if r['sequence'] == name]
        target = base if name == 'all' else base['per_sequence'][name]
        valid = sum(r['gt_valid_pixels'] for r in subset)
        target['pck_512px'] = {str(radius):
            sum(r[f'pck_{radius}_pixels'] for r in subset)/valid for radius in (1, 3, 5)}
        target['motion_compensated_temporal_mae_0to1'] = weighted_region(
            subset, 'motion_compensated_temporal_mae_0to1',
            'motion_compensated_temporal_pixels')
        target['drift_by_keyframe_age'] = {
            str(age): weighted_region([r for r in subset if r['age_since_keyframe'] == age],
                                      'epe_512px', 'gt_valid_pixels')
            for age in range(5)}
        target['fps_compute_only'] = (len(subset)/target['total_compute_seconds']
                                      if target['total_compute_seconds'] else None)
    return base


def evaluate_full(args, dataset, schedule, model, head, device, selected):
    head.eval()
    arms = {'A_geometric_propagation': [], 'B_causal_match_state': []}
    ghosts = {'photo-0319-18': {'000013', '000022', '000186'}}
    wall_start = time.perf_counter()
    with torch.no_grad():
        for sequence, stems in dataset.sequences.items():
            previous_registered = {name: None for name in arms}
            previous_gt_valid = None
            previous_a_valid = None
            for item in rollout(dataset, sequence, stems, schedule, model,
                                head, device, timed=True):
                frame, temporal = item['frame'], item['temporal']
                a_flow = item['a_flow']
                b_flow = item['b_tensor'][0].permute(1, 2, 0).cpu().numpy()
                if not np.isfinite(b_flow).all():
                    raise RuntimeError(f'nonfinite residual flow: {sequence}/{item["stem"]}')
                truth, gt_valid = dataset.flow_truth(sequence, item['stem'],
                                                     frame['original_hw'])
                motion = (np.linalg.norm(temporal['vi'], axis=-1) >= 3.
                          if temporal is not None else np.zeros((512, 512), bool))
                occlusion_proxy = ~item['a_valid'] if temporal is not None else \
                    np.zeros((512, 512), bool)
                for name, flow, compute_ms in (
                        ('A_geometric_propagation', a_flow,
                         sum(item['timing'][key] for key in
                             ('temporal_ms', 'propagation_ms', 'a1_ms'))),
                        ('B_causal_match_state', b_flow,
                         sum(item['timing'].values()))):
                    # The same A support is reported for both arms; GT masking
                    # never depends on predicted-valid pixels.
                    metrics, registered = frame_metrics(
                        dataset, frame, flow, item['a_valid'],
                        previous_registered[name])
                    if metrics['gt_valid_pixels'] != int(gt_valid.sum()):
                        raise AssertionError('A/B evaluator masks differ')
                    error = np.linalg.norm(flow-truth, axis=-1)
                    temporal_pixels, temporal_mae = 0, None
                    if temporal is not None and previous_registered[name] is not None:
                        prev_warped, inside = backward_warp(previous_registered[name],
                                                           temporal['vi'])
                        old_gt, _ = sample(previous_gt_valid.astype(np.float32),
                                           grid((512, 512)) + temporal['vi'])
                        old_pred, _ = sample(previous_a_valid.astype(np.float32),
                                             grid((512, 512)) + temporal['vi'])
                        common = (gt_valid & (old_gt > .999) & (old_pred > .999) &
                                  item['a_valid'] & inside & temporal['vi_bounds'] &
                                  (temporal['vi_error'] <= TRIGGER['fb_pixel_threshold']))
                        temporal_pixels = int(common.sum())
                        if temporal_pixels:
                            temporal_mae = float(np.abs(
                                gray(registered).astype(np.float32) -
                                gray(prev_warped).astype(np.float32))[common].mean()/255.)
                    row = {'sequence': sequence, 'stem': item['stem'],
                           'frame_index': item['index'],
                           'keyframe': int(item['keyframe']),
                           'reason': ('frozen_keyframe' if item['keyframe'] else 'propagated'),
                           'age_since_keyframe': item['age'],
                           'frame_ms': compute_ms,
                           'temporal_ms': item['timing']['temporal_ms'],
                           'propagation_ms': item['timing']['propagation_ms'],
                           'a1_ms': item['timing']['a1_ms'],
                           'branch_ms': (item['timing']['branch_ms']
                                         if name.startswith('B_') else 0.),
                           'pck_1_pixels': int(((error < 1) & gt_valid).sum()),
                           'pck_3_pixels': int(((error < 3) & gt_valid).sum()),
                           'pck_5_pixels': int(((error < 5) & gt_valid).sum()),
                           'moving_region_pixels': int((gt_valid & motion).sum()),
                           'moving_region_epe_512px': (float(error[gt_valid & motion].mean())
                                                       if (gt_valid & motion).any() else None),
                           'occlusion_proxy_pixels': int((gt_valid & occlusion_proxy).sum()),
                           'occlusion_proxy_epe_512px': (
                               float(error[gt_valid & occlusion_proxy].mean())
                               if (gt_valid & occlusion_proxy).any() else None),
                           'motion_compensated_temporal_pixels': temporal_pixels,
                           'motion_compensated_temporal_mae_0to1': temporal_mae,
                           **metrics}
                    arms[name].append(row)
                    previous_registered[name] = registered
                    if item['stem'] in ghosts.get(sequence, ()):
                        save_ghost(frame['vi'], registered,
                                   args.output/name/'ghosts'/f'{sequence}_{item["stem"]}.png')
                previous_gt_valid, previous_a_valid = gt_valid, item['a_valid']
            print(f'eval {sequence}: A={np.mean([r["epe_512px"] for r in arms["A_geometric_propagation"] if r["sequence"] == sequence]):.4f}, '
                  f'B={np.mean([r["epe_512px"] for r in arms["B_causal_match_state"] if r["sequence"] == sequence]):.4f}',
                  flush=True)
    a, b = arms.values()
    if [(r['sequence'], r['stem'], r['gt_valid_pixels'], r['keyframe']) for r in a] != \
       [(r['sequence'], r['stem'], r['gt_valid_pixels'], r['keyframe']) for r in b]:
        raise AssertionError('A/B frame, mask or keyframe mismatch')
    summaries = {}
    for name, rows in arms.items():
        summary = enrich_summary(rows, summarise(rows)['all'])
        summaries[name] = summary
        write_rows(args.output/name/'per_frame.csv', rows)
        save_json(args.output/name/'report.json', {
            'recipe': RECIPE, 'selected_epoch': selected['epoch'],
            'a1_sha256': A1_SHA, 'split_sha256': sha256(args.split_file),
            'summary': summary, 'regions': region_summary(rows), 'test_accessed': False})
    a_sum, b_sum = summaries.values()
    frozen = json.loads(args.frozen_schedule.read_text(encoding='utf-8'))
    if (a_sum['gt_valid_pixels'] != frozen['historical_gt_valid_pixels'] or
            abs(a_sum['pixel_weighted_epe_512px'] -
                frozen['historical_fixed_fast_epe_512px']) > .05):
        raise RuntimeError('A arm did not reproduce historical fixed_fast result')
    every = json.loads(args.every_frame_report.read_text(encoding='utf-8'))
    if (every['a1_sha256'] != A1_SHA or every['summary']['frames'] != 800 or
            every['summary']['gt_valid_pixels'] != a_sum['gt_valid_pixels']):
        raise ValueError('every-frame A1 reference has a different weight/mask')
    paired = [{'sequence': ra['sequence'], 'stem': ra['stem'],
               'keyframe': ra['keyframe'], 'age_since_keyframe': ra['age_since_keyframe'],
               'a_epe_512px': ra['epe_512px'], 'b_epe_512px': rb['epe_512px'],
               'a_minus_b_512px': ra['epe_512px']-rb['epe_512px']}
              for ra, rb in zip(a, b)]
    write_rows(args.output/'paired_per_frame.csv', paired)
    improved_sequences = sum(b_sum['per_sequence'][name]['epe_512px'] <
                             a_sum['per_sequence'][name]['epe_512px']
                             for name in SPLIT_NAMES)
    gain = a_sum['pixel_weighted_epe_512px'] - b_sum['pixel_weighted_epe_512px']
    extra_time = b_sum['total_compute_seconds'] - a_sum['total_compute_seconds']
    save_json(args.output/'comparison.json', {
        'recipe': RECIPE, 'a1_sha256': A1_SHA,
        'head_checkpoint_sha256': sha256(args.head_checkpoint),
        'selected_epoch': selected['epoch'], 'arms': summaries,
        'every_frame_a1_reference': {
            'epe_512px': every['summary']['pixel_weighted_epe_512px'],
            'total_compute_seconds': every['summary']['total_compute_seconds']},
        'paired_improved_frames': sum(row['a_minus_b_512px'] > 0 for row in paired),
        'improved_sequences': improved_sequences,
        'overall_gain_512px': gain, 'extra_compute_seconds': extra_time,
        'motion_and_age': {name: region_summary(rows) for name, rows in arms.items()},
        'worst_b_minus_a_frames': sorted(
            paired, key=lambda row: row['a_minus_b_512px'])[:10],
        'predeclared_rule': {'minimum_epe_gain_512px': .1,
                             'minimum_improved_sequences': 3,
                             'stop_if_no_gain_and_more_time': True},
        'stop_state_branch': gain < .1 and extra_time > 0,
        'wall_seconds_including_io_metrics': time.perf_counter() - wall_start,
        'timing_scope': 'compute includes 4 temporal Farneback, A1, geometry, '
                        'frozen VGG features, local matching/state/head; '
                        'excludes disk IO and GT/metrics',
        'test_accessed': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('check', 'train', 'eval'))
    for name in ('data-root', 'split-file', 'pretrained', 'coarse-checkpoint',
                 'source-a1-checkpoint', 'source-coarse-adapt-checkpoint',
                 'a1-checkpoint', 'frozen-schedule', 'every-frame-report',
                 'head-checkpoint', 'output'):
        parser.add_argument(f'--{name}', type=Path)
    args = parser.parse_args()
    print(json.dumps(synthetic_check(), indent=2), flush=True)
    if args.stage == 'check' and args.data_root is None:
        return
    for name in ('data_root', 'split_file', 'pretrained', 'coarse_checkpoint',
                 'source_a1_checkpoint', 'source_coarse_adapt_checkpoint',
                 'a1_checkpoint', 'frozen_schedule', 'output'):
        if getattr(args, name) is None:
            parser.error(f'--{name.replace("_", "-")} is required')
    if sha256(args.a1_checkpoint) != A1_SHA:
        raise ValueError('frozen A1 SHA-256 mismatch')
    if args.stage == 'train' and args.output.exists() and any(args.output.iterdir()):
        parser.error('training output must be a new empty directory')
    if args.stage == 'eval' and args.every_frame_report is None:
        parser.error('--every-frame-report is required for formal comparison')
    if args.stage == 'eval' and (args.output/'comparison.json').exists():
        parser.error('evaluation report exists; use a new run rather than overwrite')
    seed_all()
    install_mutual_guard()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_a1(args, device)
    head = CausalMatchHead().to(device)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError('frozen A1 parameters require gradients')
    valset = VTMOTVideos(args.data_root, args.split_file, split='eval',
                         max_sequences=1 if args.stage == 'check' else 0,
                         max_frames=5 if args.stage == 'check' else 0)
    if args.stage != 'check' and (tuple(valset.sequences) != SPLIT_NAMES or
                                  sum(map(len, valset.sequences.values())) != 800):
        raise ValueError('expected four fixed validation sequences / 800 frames')
    schedule_val = frozen_schedule(args.frozen_schedule, args.split_file,
                                   SPLIT_NAMES, 800)
    first = next(iter(valset.sequences))
    predict_a1(model, valset.input_frame(first, valset.sequences[first][0]), device)
    if args.stage == 'check':
        head.eval()
        sequence, stems = next(iter(valset.sequences.items()))
        with torch.no_grad():
            for item in rollout(valset, sequence, stems, schedule_val,
                                model, head, device):
                b = item['b_tensor'][0].permute(1, 2, 0).cpu().numpy()
                if not np.array_equal(b, item['a_flow']):
                    raise AssertionError(f'actual A/B zero-head mismatch: {item["stem"]}')
        print('real five-frame zero-head A/B flow identity passed; no GT read', flush=True)
        return
    if args.stage == 'train':
        args.output.mkdir(parents=True, exist_ok=True)
        trainset = VTMOTVideos(args.data_root, args.split_file, split='train')
        if len(trainset.sequences) != 38:
            raise ValueError('expected all 38 training sequences')
        schedule_train = prepare_train(args, trainset, model, device)
        schedule_train = {k: set(v) for k, v in schedule_train.items()}
        run_training(args, trainset, valset, schedule_train, schedule_val,
                     model, head, device)
        return
    args.head_checkpoint = args.head_checkpoint or args.output/'best_head.pth'
    selected = torch.load(args.head_checkpoint, map_location='cpu', weights_only=False)
    if (selected.get('recipe') != RECIPE or selected.get('a1_sha256') != A1_SHA or
            selected.get('split_sha256') != sha256(args.split_file) or
            selected.get('frozen_schedule_sha256') != sha256(args.frozen_schedule) or
            selected.get('test_accessed') is not False):
        raise ValueError('head checkpoint provenance mismatch')
    head.load_state_dict(selected['head_state_dict'], strict=True)
    evaluate_full(args, valset, schedule_val, model, head, device, selected)


if __name__ == '__main__':
    main()
