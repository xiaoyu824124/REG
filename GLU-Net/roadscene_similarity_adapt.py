"""Matched fine adaptation of raw versus train-fixed similarity coarse flow.

Only train/val are opened. A and B start from one best_coarse.pth, keep the
coarse model and original flow upsamplers frozen, and update the same original
GLU-Net local layers with an identical step-based learning-rate schedule.
"""

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from datasets.roadscene import RoadScenePairs
from experiments.global_geometry_readonly.diagnose import (
    MAX_ANGLE_DEG, candidate, flow_from_similarity)
from roadscene_coarse import sha256_file
from roadscene_local_ablation import checked_payload, new_model
from roadscene_local256 import edge_map, inputs, warp_image
from roadscene_refinement_audit import edge_quartile, pooled_truth
from roadscene_staged_mutual_fix import FIX
import roadscene_staged as staged


SEED = 2026
ARMS = ('A_raw', 'B_similarity')
EPOCHS = 80
BATCH_SIZE = 2
FINE_WARMUP_EPOCHS = 5
STAGES = ('coarse16', 'local32', 'local64', 'final128')
WINDOWS = {'local32': (32, 4), 'local64': (64, 4),
           'final128': (128, 4)}
RULE = {'points': 'decoder coarse-flow endpoints',
        'confidence': 'post-MutualMatching correlation top1 minus top2',
        'margin_percentile': 50, 'ransac_threshold_512px': 8,
        'max_abs_rotation_deg': 50, 'allowed_scale': [.8, 1.2],
        'minimum_inliers': 8, 'ransac_seed': '2026 + decimal pair ID',
        'fallback': 'original coarse flow',
        'source': 'train176 selection in global_geometry_readonly_v2',
        'implementation_sha256': sha256_file(
            Path(__file__).resolve().parent / 'experiments/global_geometry_readonly/diagnose.py')}
ADOPTION = {'minimum_full23_final_gain_512px': .1,
            'minimum_improved_pairs': 15,
            'minimum_gain_each_difficult_pair_512px': .1,
            'maximum_large_motion_regression_512px': .1,
            'maximum_first_window_outside_regression_512px': .1}


def tensor_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode('utf-8'))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def freeze_digest(model):
    selected = staged.groups(model)
    active_ids = {id(p) for p in selected['local'] + selected['dcn']}
    state = {name: p.detach() for name, p in model.named_parameters()
             if id(p) not in active_ids}
    state.update({f'buffer:{name}': b.detach()
                  for name, b in model.named_buffers()})
    return tensor_digest(state)


def upsampler_digest(model):
    state = {}
    for name in staged.FROZEN_FLOW_UPSAMPLERS:
        state.update({f'{name}.{key}': value
                      for key, value in getattr(model, name).state_dict().items()})
    return tensor_digest(state)


def local_digest(model):
    selected = staged.groups(model)
    local_ids = {id(p) for p in selected['local'] + selected['dcn']}
    return tensor_digest({name: p for name, p in model.named_parameters()
                          if id(p) in local_ids})


def optimizer_signature(optimizer):
    return [{key: group[key] for key in ('group_name', 'lr', 'weight_decay',
                                         'betas', 'eps')}
            | {'parameter_shapes': [tuple(p.shape) for p in group['params']]}
            for group in optimizer.param_groups]


def geometry_check():
    matrix = np.array([[1, 0, 32], [0, 1, -16]], dtype=np.float32)
    flow512 = flow_from_similarity(matrix, 16)
    if (not np.allclose(flow512[0], 32) or
            not np.allclose(flow512[1], -16) or
            not np.allclose(flow512 / 2, np.array([16, -8])[:, None, None])):
        raise RuntimeError('Similarity direction or 512-to-256 flow conversion failed')
    return {'target_to_source_translation_512px': [32, -16],
            'inserted_coarse_flow_256px': [16, -8],
            'insertion': 'after 16-grid decoder4, before pretrained deconv4'}


def set_names(model, batch):
    model._active_pair_names = tuple(batch['name'])


def install_fit(model):
    original = model.coarsest_resolution_flow
    model._fit_last = []

    def fitted(c14, c24, h_256, w_256, return_corr=False):
        if (h_256, w_256) != (256, 256):
            raise ValueError('Similarity fit requires the original 256 input')
        raw, corr = original(c14, c24, h_256, w_256, return_corr=True)
        model._raw_coarse_for_shared_region = raw.detach()
        names = getattr(model, '_active_pair_names', None)
        if names is None or len(names) != raw.shape[0]:
            raise RuntimeError('Pair IDs are required for fixed RANSAC seeds')
        scores, _ = torch.topk(corr.flatten(2), 2, dim=1)
        margin = (scores[:, 0] - scores[:, 1]).detach().cpu().numpy()
        yy, xx = np.indices((16, 16), dtype=np.float32)
        points = np.stack((xx, yy), axis=-1).reshape(-1, 2) * 32
        output, details = [], []
        for index, name in enumerate(names):
            if not name.isdecimal():
                raise ValueError(f'RoadScene pair ID must be decimal: {name}')
            raw512 = (raw[index].detach().float().cpu().numpy() * 2)
            pred512, success, candidates, inliers = candidate(
                {'target': points, 'coarse512': raw512,
                 'margin': margin[index].reshape(-1)},
                'decoder', RULE['margin_percentile'],
                RULE['ransac_threshold_512px'], int(name) + SEED)
            output.append(torch.from_numpy(np.ascontiguousarray(pred512 / 2))
                          .to(device=raw.device, dtype=raw.dtype))
            details.append({'name': name, 'fit_success': bool(success),
                            'candidate_points': int(candidates),
                            'ransac_inliers': int(inliers)})
        model._fit_last = details
        flow = torch.stack(output)
        return (flow, corr) if return_corr else flow

    model.coarsest_resolution_flow = fitted


def make_arm(arm, parent, device):
    model = new_model(parent, device)
    if arm == 'B_similarity':
        install_fit(model)
    model.eval()
    model.train_coarse_encoder = False
    return model


def configure(model, epoch, warmup=FINE_WARMUP_EPOCHS):
    selected, active = staged.configure_trainability(
        model, 'fine', epoch, fine_warmup=warmup)
    model.eval()  # Freeze all original BatchNorm running statistics.
    if any(p.requires_grad for p in selected['coarse']):
        raise RuntimeError('Coarse parameters were unfrozen')
    if any(p.requires_grad for name in staged.FROZEN_FLOW_UPSAMPLERS
           for p in getattr(model, name).parameters()):
        raise RuntimeError('Pretrained flow upsampler was unfrozen')
    return selected, active


def optimizer_and_scheduler(model, total_epochs):
    # The original fine-stage AdamW groups and rates are retained. Unlike its
    # validation-driven plateau scheduler, this fixed cosine factor gives A/B
    # identical realized learning rates at every optimizer step.
    optimizer, _ = staged.make_optimizer(model, 'fine')
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: .1 + .9 *
        (1 + math.cos(math.pi * min(step, total_epochs) / total_epochs)) / 2)
    return optimizer, scheduler


def masked_epe(prediction, truth, valid):
    error = torch.linalg.vector_norm(prediction - truth, dim=1)
    return float(error[valid].sum()), int(valid.sum())


def window_coverage(prediction, truth, valid, size, radius):
    pooled, selected = pooled_truth(truth, valid, size)
    delta = pooled - prediction
    inside = (delta.abs().amax(dim=1) <= radius * 512 / size) & selected
    return {'valid_queries': int(selected.sum()),
            'inside_queries': int(inside.sum()),
            'outside_queries': int((selected & ~inside).sum()),
            'outside_epe_sum': float(torch.linalg.vector_norm(delta, dim=1)
                                     [selected & ~inside].sum())}


@torch.no_grad()
def evaluate(model, dataset, device, code_check=False, timing=False):
    model.eval()
    rows = []
    repeats = 1 if code_check else 5
    warmups = 0 if code_check else 3
    for batch in DataLoader(dataset, batch_size=1, shuffle=False):
        name = batch['name'][0]
        set_names(model, batch)
        truth = batch['flow_map'].to(device).float()
        valid = batch['correspondence_mask'].to(device).bool()
        valid_image = valid[0]

        def forward():
            source, target, source256, target256 = inputs(model, batch, device)
            return model(target, source, target256, source256)

        durations = []
        if timing:
            for _ in range(warmups):
                forward()
            for _ in range(repeats):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()
                forward()
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                durations.append((time.perf_counter() - start) * 1000)
        flows256, flows512 = forward()
        fit = getattr(model, '_fit_last', [])
        coarse16, local32 = flows256
        local64, final128 = flows512
        stage = {'coarse16': F.interpolate(coarse16, (512, 512),
                                          mode='bilinear', align_corners=False) * 2,
                 'local32': F.interpolate(local32, (512, 512),
                                         mode='bilinear', align_corners=False) * 2,
                 'local64': F.interpolate(local64, (512, 512),
                                         mode='bilinear', align_corners=False),
                 'final128': F.interpolate(final128, (512, 512),
                                          mode='bilinear', align_corners=False)}
        row = {'name': name, 'valid_pixels': int(valid.sum()),
               'timing_ms': float(np.median(durations)) if durations else None,
               'fit_success': fit[0]['fit_success'] if fit else None,
               'fit_candidates': fit[0]['candidate_points'] if fit else None,
               'fit_inliers': fit[0]['ransac_inliers'] if fit else None}
        for label in STAGES:
            error = torch.linalg.vector_norm(stage[label] - truth, dim=1)
            row[f'{label}_error_sum_512px'] = float(error[valid].sum())
            row[f'{label}_epe_512px'] = float(error[valid].mean())
        pooled16, selected16 = pooled_truth(truth, valid, 16)
        row['coarse16_grid_valid'] = int(selected16.sum())
        row['coarse16_grid_epe_512px'] = float(
            torch.linalg.vector_norm(coarse16 * 2 - pooled16, dim=1)
            [selected16].mean())
        pre = {'local32': model.deconv4(coarse16) * 2,
               'local64': F.interpolate(local32, (64, 64),
                                        mode='bilinear', align_corners=False) * 2,
               'final128': model.deconv2(local64)}
        for label, (size, radius) in WINDOWS.items():
            cover = window_coverage(pre[label], truth, valid, size, radius)
            for key, value in cover.items():
                row[f'{label}_window_{key}'] = value
        final_error = torch.linalg.vector_norm(stage['final128'] - truth, dim=1)[0]
        truth32, selected32 = pooled_truth(truth, valid, 32)
        shared_pre32 = (model.deconv4(model._raw_coarse_for_shared_region) * 2
                        if hasattr(model, '_raw_coarse_for_shared_region')
                        else pre['local32'])
        outside32 = selected32 & (
            (truth32 - shared_pre32).abs().amax(dim=1) > 64)
        outside_full = valid_image & F.interpolate(
            outside32[:, None].float(), (512, 512), mode='nearest')[0, 0].bool()
        row['first32_outside_pixels'] = int(outside_full.sum())
        row['first32_outside_error_sum_512px'] = float(final_error[outside_full].sum())
        row['first32_outside_epe_512px'] = (
            float(final_error[outside_full].mean()) if outside_full.any() else None)
        edge = edge_quartile(batch['target_image'].to(device), valid_image)
        large = valid_image & (torch.linalg.vector_norm(truth[0], dim=0) >= 64)
        for group, mask in (('edge', edge), ('large64', large)):
            row[f'{group}_pixels'] = int(mask.sum())
            row[f'{group}_error_sum_512px'] = float(final_error[mask].sum())
            row[f'{group}_epe_512px'] = (float(final_error[mask].mean())
                                         if mask.any() else None)
        for threshold in (1, 3, 5):
            row[f'pixels_below_{threshold}px'] = int(
                ((final_error < threshold) & valid_image).sum())
            row[f'fraction_below_{threshold}px'] = (
                row[f'pixels_below_{threshold}px'] / row['valid_pixels'])
        rows.append(row)
    return rows


def summarize(rows):
    if not rows:
        return {'pairs': 0, 'valid_pixels': 0, 'reason': 'no images in this subset'}
    pixels = sum(r['valid_pixels'] for r in rows)
    result = {'pairs': len(rows), 'valid_pixels': pixels,
              'stage_epe_512px': {
                  label: sum(r[f'{label}_error_sum_512px'] for r in rows) / pixels
                  for label in STAGES},
              'coarse16_grid_epe_512px': sum(
                  r['coarse16_grid_epe_512px'] * r['coarse16_grid_valid']
                  for r in rows) / sum(r['coarse16_grid_valid'] for r in rows),
              'pixel_fraction_below': {str(t): sum(
                  r[f'pixels_below_{t}px'] for r in rows) / pixels
                  for t in (1, 3, 5)}}
    result['window_coverage'] = {}
    for label in WINDOWS:
        valid = sum(r[f'{label}_window_valid_queries'] for r in rows)
        inside = sum(r[f'{label}_window_inside_queries'] for r in rows)
        result['window_coverage'][label] = {
            'valid_queries': valid, 'inside_queries': inside,
            'inside_fraction': inside / valid}
    result['regions'] = {}
    for group in ('edge', 'large64', 'first32_outside'):
        count = sum(r[f'{group}_pixels'] for r in rows)
        result['regions'][group] = {
            'pixels': count,
            'epe_512px': (sum(r[f'{group}_error_sum_512px'] for r in rows) / count
                          if count else None)}
    times = [r['timing_ms'] for r in rows if r['timing_ms'] is not None]
    result['inference_ms_pair_mean_of_medians'] = float(np.mean(times)) if times else None
    result['fit_success_pairs'] = sum(r['fit_success'] is True for r in rows)
    return result


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def capture_rng(device):
    return {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(),
            'python': random.getstate(),
            'cuda': torch.cuda.get_rng_state() if device.type == 'cuda' else None}


def restore_rng(state, device):
    torch.set_rng_state(state['torch'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda'])


def save_checkpoint(path, model, optimizer, scheduler, arm, epoch, steps,
                    best, history, parent_sha, base_sha, code_check,
                    frozen_sha, upsampler_sha, device, finished):
    payload = {'arm': arm, 'epoch': epoch, 'steps': steps,
               'best_val_final_epe_512px': best, 'history': history,
               'model_state_dict': model.state_dict(),
               'optimizer_state_dict': optimizer.state_dict(),
               'scheduler_state_dict': scheduler.state_dict(),
               'rng': capture_rng(device), 'parent_sha256': parent_sha,
               'pretrained_sha256': base_sha, 'matching_fix': FIX,
               'rule': RULE, 'code_sha256': sha256_file(Path(__file__)),
               'frozen_sha256': frozen_sha, 'upsampler_sha256': upsampler_sha,
               'code_check_only': code_check, 'finished': finished}
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_arm(arm, parent, trainset, valset, args, device,
              initial_state_sha, initial_up_sha, initial_local_sha,
              reference_optimizer, reference_orders=None):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if device.type == 'cuda':
        torch.cuda.manual_seed(SEED)
    model = make_arm(arm, parent, device)
    if tensor_digest(model.state_dict()) != initial_state_sha:
        raise RuntimeError(f'{arm} did not start from the common coarse state')
    if upsampler_digest(model) != initial_up_sha:
        raise RuntimeError(f'{arm} changed the pretrained flow upsampler')
    if local_digest(model) != initial_local_sha:
        raise RuntimeError(f'{arm} changed the initial local decoder or DCN')
    epochs = 2 if args.code_check else EPOCHS
    warmup = 1 if args.code_check else FINE_WARMUP_EPOCHS
    optimizer, scheduler = optimizer_and_scheduler(model, epochs)
    if optimizer_signature(optimizer) != reference_optimizer or optimizer.state:
        raise RuntimeError(f'{arm} optimizer initialization differs')
    frozen_sha = freeze_digest(model)
    path = args.output / arm
    path.mkdir(parents=True, exist_ok=True)
    latest = path / 'latest.pth'
    best_file = path / 'best.pth'
    expected_steps = len(trainset) // (1 if args.code_check else BATCH_SIZE) * epochs
    if args.resume and latest.is_file():
        state = torch.load(latest, map_location='cpu', weights_only=False)
        if (state['arm'] != arm or state['parent_sha256'] != args.coarse_sha or
                state['pretrained_sha256'] != args.pretrained_sha or
                state['code_sha256'] != sha256_file(Path(__file__)) or
                state['rule'] != RULE or state['code_check_only'] != args.code_check):
            raise RuntimeError(f'{arm} resume provenance mismatch')
        model.load_state_dict(state['model_state_dict'], strict=True)
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        restore_rng(state['rng'], device)
        history, steps, best = state['history'], state['steps'], state['best_val_final_epe_512px']
        first_epoch = state['epoch'] + 1
        if state['finished']:
            return history, best_file, [r['image_order_sha256'] for r in history[1:]]
    else:
        if latest.exists() or best_file.exists():
            raise FileExistsError(f'{arm} has existing checkpoints; use --resume')
        configure(model, 0, warmup)
        initial_rows = evaluate(model, valset, device, args.code_check)
        best = summarize(initial_rows)['stage_epe_512px']['final128']
        history = [{'epoch': 0, 'steps': 0, 'val_final_epe_512px': best,
                    'image_order_sha256': None, 'train_loss': None,
                    'learning_rates': [g['lr'] for g in optimizer.param_groups]}]
        steps, first_epoch = 0, 1
        save_checkpoint(best_file, model, optimizer, scheduler, arm, 0, steps,
                        best, history, args.coarse_sha, args.pretrained_sha,
                        args.code_check, frozen_sha, initial_up_sha, device, False)
        save_checkpoint(latest, model, optimizer, scheduler, arm, 0, steps,
                        best, history, args.coarse_sha, args.pretrained_sha,
                        args.code_check, frozen_sha, initial_up_sha, device, False)
    for epoch in range(first_epoch, epochs + 1):
        selected, active = configure(model, epoch, warmup)
        loader = DataLoader(trainset, batch_size=1 if args.code_check else BATCH_SIZE,
                            shuffle=True,
                            generator=torch.Generator().manual_seed(SEED + epoch),
                            num_workers=0, pin_memory=device.type == 'cuda')
        digest = hashlib.sha256()
        losses = []
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for batch in loader:
            for name in batch['name']:
                digest.update(name.encode('utf-8') + b'\n')
            set_names(model, batch)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = staged.dense_training_loss(
                model, batch, device, coarse_auxiliary=False)
            if not torch.isfinite(loss):
                raise RuntimeError(f'{arm} nonfinite loss at epoch {epoch}')
            loss.backward()
            if any(p.grad is not None for p in selected['coarse']):
                raise RuntimeError('Frozen coarse received gradient')
            if not any(p.grad is not None and torch.isfinite(p.grad).all()
                       and p.grad.abs().sum() > 0
                       for group in active for p in selected[group]):
                raise RuntimeError(f'{arm} has no finite local gradient')
            torch.nn.utils.clip_grad_norm_(
                [p for group in active for p in selected[group]], 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            steps += 1
        if reference_orders is not None and digest.hexdigest() != reference_orders[epoch-1]:
            raise RuntimeError('A and B training pair order differs')
        rows = evaluate(model, valset, device, args.code_check)
        metric = summarize(rows)['stage_epe_512px']['final128']
        scheduler.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        history.append({'epoch': epoch, 'steps': steps,
                        'val_final_epe_512px': metric,
                        'val_stage_epe_512px': summarize(rows)['stage_epe_512px'],
                        'val_window_coverage': summarize(rows)['window_coverage'],
                        'train_loss': float(np.mean(losses)),
                        'image_order_sha256': digest.hexdigest(),
                        'learning_rates': [g['lr'] for g in optimizer.param_groups],
                        'peak_train_allocated_mib': (
                            torch.cuda.max_memory_allocated() / 2**20
                            if device.type == 'cuda' else None),
                        'epoch_seconds': time.perf_counter() - start})
        if metric < best:
            best = metric
            save_checkpoint(best_file, model, optimizer, scheduler, arm,
                            epoch, steps, best, history, args.coarse_sha,
                            args.pretrained_sha, args.code_check, frozen_sha,
                            initial_up_sha, device, False)
        save_checkpoint(latest, model, optimizer, scheduler, arm,
                        epoch, steps, best, history, args.coarse_sha,
                        args.pretrained_sha, args.code_check, frozen_sha,
                        initial_up_sha, device, epoch == epochs)
        (path / 'history.json').write_text(json.dumps(history, indent=2),
                                           encoding='utf-8')
        print(f'{arm} epoch {epoch}/{epochs}: val final={metric:.4f}, '
              f'best={best:.4f}, steps={steps}', flush=True)
    if steps != expected_steps:
        raise RuntimeError(f'{arm} did not finish the fixed update budget: {steps}')
    if freeze_digest(model) != frozen_sha or upsampler_digest(model) != initial_up_sha:
        raise RuntimeError(f'{arm} changed frozen coarse/BN/upsampler state')
    return history, best_file, [r['image_order_sha256'] for r in history[1:]]


def selected_arm(arm, parent, path, device, args):
    model = make_arm(arm, parent, device)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if (checkpoint['arm'] != arm or checkpoint['parent_sha256'] != args.coarse_sha
            or checkpoint['rule'] != RULE):
        raise RuntimeError('Selected checkpoint provenance mismatch')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    return model, checkpoint


def ghost_panel(path, source, target, flow_a, flow_b, valid):
    target_edge = edge_map(target)
    target_edge = target_edge / target_edge[valid].quantile(.95).clamp_min(1e-6)
    canvas = Image.new('RGB', (1024, 540))
    draw = ImageDraw.Draw(canvas)
    for index, (label, flow) in enumerate((('A raw coarse', flow_a),
                                            ('B predicted similarity', flow_b))):
        warped = warp_image(source, flow)
        ir_edge = edge_map(warped)
        ir_edge = ir_edge / ir_edge[valid].quantile(.95).clamp_min(1e-6)
        overlay = torch.zeros(3, 512, 512, device=source.device)
        overlay[0] = target_edge.clamp(0, 1)
        overlay[1] = ir_edge.clamp(0, 1)
        overlay[2] = ir_edge.clamp(0, 1)
        overlay[:, ~valid] = 0
        rgb = (overlay.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        canvas.paste(Image.fromarray(rgb), (512 * index, 28))
        draw.text((512 * index + 8, 8), label, fill='white')
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


@torch.no_grad()
def final_report(parent, valset, best_files, args, device):
    data = {}
    models = {}
    for arm in ARMS:
        model, checkpoint = selected_arm(arm, parent, best_files[arm], device, args)
        models[arm] = model
        rows = evaluate(model, valset, device, args.code_check, timing=True)
        write_csv(args.output / f'per_image_{arm}.csv', rows)
        data[arm] = {'selected_epoch': checkpoint['epoch'],
                     'selected_steps': checkpoint['steps'],
                     'metrics_23': summarize(rows), 'rows': rows,
                     'total_parameters': sum(p.numel() for p in model.parameters())}
    a = {r['name']: r for r in data['A_raw']['rows']}
    b = {r['name']: r for r in data['B_similarity']['rows']}
    if a.keys() != b.keys():
        raise RuntimeError('A/B validation image IDs do not pair')
    paired = []
    for name in sorted(a):
        left, right = a[name], b[name]
        if left['valid_pixels'] != right['valid_pixels']:
            raise RuntimeError('A/B valid masks differ')
        if left['first32_outside_pixels'] != right['first32_outside_pixels']:
            raise RuntimeError('A/B fixed first-window-outside masks differ')
        paired.append({'name': name, 'valid_pixels': left['valid_pixels'],
                       'A_final_epe_512px': left['final128_epe_512px'],
                       'B_final_epe_512px': right['final128_epe_512px'],
                       'B_minus_A_epe_512px': right['final128_epe_512px'] - left['final128_epe_512px'],
                       'A_coarse_epe_512px': left['coarse16_epe_512px'],
                       'B_coarse_epe_512px': right['coarse16_epe_512px'],
                       'A_edge_epe_512px': left['edge_epe_512px'],
                       'B_edge_epe_512px': right['edge_epe_512px'],
                       'A_large64_epe_512px': left['large64_epe_512px'],
                       'B_large64_epe_512px': right['large64_epe_512px'],
                       'first32_outside_pixels': left['first32_outside_pixels'],
                       'A_first32_outside_epe_512px': left['first32_outside_epe_512px'],
                       'B_first32_outside_epe_512px': right['first32_outside_epe_512px'],
                       **{f'{arm}_fraction_below_{t}px': row[f'fraction_below_{t}px']
                          for arm, row in (('A', left), ('B', right)) for t in (1, 3, 5)}})
    write_csv(args.output / 'per_image_paired.csv', paired)
    for name in ('000002', '000014'):
        if name not in a:
            continue
        batch = next(x for x in DataLoader(valset, batch_size=1, shuffle=False)
                     if x['name'][0] == name)
        flows = {}
        for arm, model in models.items():
            set_names(model, batch)
            source, target, source256, target256 = inputs(model, batch, device)
            _, local = model(target, source, target256, source256)
            flows[arm] = F.interpolate(local[-1], (512, 512),
                                       mode='bilinear', align_corners=False)[0]
        ghost_panel(args.output/'edge_ghosts'/f'{name}.png',
                    batch['source_image'][0].to(device),
                    batch['target_image'][0].to(device),
                    flows['A_raw'], flows['B_similarity'],
                    batch['correspondence_mask'][0].to(device).bool())
    metrics_a = data['A_raw']['metrics_23']
    metrics_b = data['B_similarity']['metrics_23']
    gain = (metrics_a['stage_epe_512px']['final128'] -
            metrics_b['stage_epe_512px']['final128'])
    difficult = {name: a[name]['final128_epe_512px'] - b[name]['final128_epe_512px']
                 for name in ('000002', '000014') if name in a}
    improved = sum(row['B_minus_A_epe_512px'] < 0 for row in paired)
    def regression(group):
        old = metrics_a['regions'][group]['epe_512px']
        new = metrics_b['regions'][group]['epe_512px']
        return new - old if old is not None and new is not None else None
    large_reg = regression('large64')
    outside_reg = regression('first32_outside')
    decision = {'final_gain_512px': gain, 'improved_pairs': improved,
                'difficult_gain_512px': difficult,
                'large_motion_regression_512px': large_reg,
                'first_window_outside_regression_512px': outside_reg,
                'adoption_rule': ADOPTION,
                'adopt_B': (not args.code_check and
                            gain >= ADOPTION['minimum_full23_final_gain_512px'] and
                            improved >= ADOPTION['minimum_improved_pairs'] and
                            len(difficult) == 2 and all(
                                value >= ADOPTION['minimum_gain_each_difficult_pair_512px']
                                for value in difficult.values()) and
                            large_reg is not None and
                            large_reg <= ADOPTION['maximum_large_motion_regression_512px'] and
                            outside_reg is not None and
                            outside_reg <= ADOPTION['maximum_first_window_outside_regression_512px'])}
    report = {'code_check_only': args.code_check, 'test_pairs_accessed': False,
              'coarse_checkpoint_sha256': args.coarse_sha,
              'historical_fine_checkpoint_sha256': args.historical_fine_sha,
              'pretrained_sha256': args.pretrained_sha,
              'rule': RULE, 'budget': {'epochs_each': 2 if args.code_check else EPOCHS,
                                      'updates_each': len(args.trainset) * (2 if args.code_check else EPOCHS) //
                                      (1 if args.code_check else BATCH_SIZE),
                                      'batch_size': 1 if args.code_check else BATCH_SIZE,
                                      'seed': SEED, 'fine_warmup_epochs': 1 if args.code_check else FINE_WARMUP_EPOCHS,
                                      'optimizer': 'AdamW local=1e-5, DCN=1e-4',
                                      'schedule': 'same fixed epoch cosine factor 1.0 to 0.1',
                                      'loss': 'masked final 512px Charbonnier',
                                      'selection': 'lowest full23 valid-pixel-weighted final EPE'},
              'training_scope': {
                  'coarse_encoder_dns_attention_global_decoder_frozen': True,
                  'batchnorm_running_statistics_frozen': True,
                  'flow_upsamplers_frozen': list(staged.FROZEN_FLOW_UPSAMPLERS),
                  'first_epochs_train': ['local_dcn32'],
                  'remaining_epochs_train': [*staged.LOCAL_MODULES, 'local_dcn32'],
                  'no_new_network_modules': True},
              'environment': {'torch': torch.__version__,
                              'cuda_runtime': torch.version.cuda,
                              'device': str(device)},
              'initial_state_sha256': args.initial_state_sha,
              'initial_local_sha256': args.initial_local_sha,
              'initial_upsampler_sha256': args.initial_up_sha,
              'initial_optimizer_signature': args.initial_optimizer_signature,
              'geometry_check': geometry_check(),
              'historical_fine_reference': {
                  'selected_epoch': args.historical_fine_epoch,
                  'validation_final_epe_512px': args.historical_fine_epe},
              'arms': {arm: {k: v for k, v in record.items() if k != 'rows'}
                       for arm, record in data.items()},
              'diagnostic21': {
                  arm: summarize([r for r in data[arm]['rows']
                                  if r['name'] not in ('000002', '000014')])
                  for arm in ARMS},
              'difficult2': {
                  arm: summarize([r for r in data[arm]['rows']
                                  if r['name'] in ('000002', '000014')])
                  for arm in ARMS},
              'decision': decision}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2),
                                             encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--pretrained', type=Path, required=True)
    parser.add_argument('--coarse-checkpoint', type=Path, required=True)
    parser.add_argument('--historical-fine-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--code-check', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if MAX_ANGLE_DEG != RULE['max_abs_rotation_deg']:
        raise RuntimeError('The frozen train-derived rotation guard changed')
    if args.code_check and args.resume:
        parser.error('Code check cannot resume')
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        parser.error('Output must be a new empty directory, or use --resume')
    args.output.mkdir(parents=True, exist_ok=True)
    args.coarse_sha = sha256_file(args.coarse_checkpoint)
    args.pretrained_sha = sha256_file(args.pretrained)
    args.historical_fine_sha = sha256_file(args.historical_fine_checkpoint)
    parent = checked_payload(args.coarse_checkpoint, args.pretrained, smoke=False)
    historical = torch.load(args.historical_fine_checkpoint, map_location='cpu',
                            weights_only=False)
    if historical.get('parent_sha256') != args.coarse_sha:
        raise ValueError('Historical best_fine is not from this best_coarse')
    args.historical_fine_epoch = historical['epoch']
    args.historical_fine_epe = historical['selection_metric']
    geometry_check()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import models.our_models.GLUNet as glunet
    original_matching = glunet.MutualMatching
    glunet.MutualMatching = lambda corr: original_matching(corr.clamp_min(0))
    try:
        train = RoadScenePairs(args.data_root, 'train')
        val = RoadScenePairs(args.data_root, 'val')
        if args.code_check:
            train, val = Subset(train, [0]), Subset(val, [0])
        args.trainset = train
        reference = make_arm('A_raw', parent, device)
        args.initial_state_sha = tensor_digest(reference.state_dict())
        args.initial_up_sha = upsampler_digest(reference)
        args.initial_local_sha = local_digest(reference)
        reference_optimizer, _ = optimizer_and_scheduler(
            reference, 2 if args.code_check else EPOCHS)
        args.initial_optimizer_signature = optimizer_signature(reference_optimizer)
        del reference
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        history_a, best_a, orders = train_arm(
            'A_raw', parent, train, val, args, device,
            args.initial_state_sha, args.initial_up_sha,
            args.initial_local_sha, args.initial_optimizer_signature)
        history_b, best_b, _ = train_arm(
            'B_similarity', parent, train, val, args, device,
            args.initial_state_sha, args.initial_up_sha,
            args.initial_local_sha, args.initial_optimizer_signature, orders)
        if (len(history_a) != len(history_b) or
                history_a[-1]['steps'] != history_b[-1]['steps'] or
                any(a['learning_rates'] != b['learning_rates']
                    for a, b in zip(history_a, history_b))):
            raise RuntimeError('A/B did not use the same update budget')
        report = final_report(parent, val, {'A_raw': best_a,
                                             'B_similarity': best_b}, args, device)
        print(json.dumps({'code_check_only': args.code_check,
                          'A_final_epe_512px': report['arms']['A_raw']['metrics_23']
                          ['stage_epe_512px']['final128'],
                          'B_final_epe_512px': report['arms']['B_similarity']['metrics_23']
                          ['stage_epe_512px']['final128'],
                          'decision': report['decision']}, indent=2), flush=True)
    finally:
        glunet.MutualMatching = original_matching


if __name__ == '__main__':
    if os.environ.get('REG_FIXED_TEMP'):
        scratch = Path(os.environ['REG_FIXED_TEMP'])
        if not scratch.is_dir():
            raise FileNotFoundError(scratch)
        @contextlib.contextmanager
        def fixed_temp(*_args, **_kwargs):
            yield str(scratch)
        tempfile.TemporaryDirectory = fixed_temp
    if hasattr(os, 'add_dll_directory'):
        _dll_handle = os.add_dll_directory(str(Path(torch.__file__).parent/'lib'))
    main()
