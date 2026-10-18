"""Matched B-position DNS scale ablation on VTMOT (train/validation only).

B0: trained SA -> DNS(r=1,2) -> CA, continued without architecture change.
B1: same checkpoint plus R*softmax per-position radius weights after DNS
channel squeezing. Both complete equal coarse and fine optimization budgets.
The 6 held-out test sequences are never instantiated by this program.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

import vtmot_dns_position_ablation as prior
from datasets.vtmot_video import VTMOTVideos
from models.our_models.coarse_dns import CoarseDNS
from roadscene_coarse import sha256_file
from roadscene_fusion_hierarchical import objective
from vtmot_a1_adapt import (BATCH_SIZE, Frames, SEED, TRAIN_FRAME_STRIDE,
                            frozen_digest, save_json, seed_all, train_epoch)
from vtmot_gpu_epoch_cache import build_gpu_epoch_cache, expected_order_sha256
from vtmot_keyframe import install_mutual_guard


ARMS = ('B0', 'B1')
STAGES = ('coarse', 'fine')
RECIPE = 'vtmot_dns_adaptive_scale_v1'


def source_model(args, arm, device):
    """Load the identical trained B fine model before either continuation."""
    seed_all()
    payload = torch.load(args.b_checkpoint, map_location='cpu', weights_only=False)
    expected = {
        'recipe': 'vtmot_dns_position_v1', 'arm': 'B', 'stage': 'fine',
        'pretrained_sha256': sha256_file(args.pretrained),
        'attention_checkpoint_sha256': sha256_file(args.attention_checkpoint),
        'split_sha256': sha256_file(args.split_file), 'test_accessed': False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f'B source checkpoint {key} mismatch')
    model = prior.initialize('B', args, device)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    dns = model.base.coarse_dns
    if dns.radii != (1, 2) or model.base.coarse_dns_order != 'between_sa_ca':
        raise AssertionError('source must have DNS(1,2) between SA and CA')
    if arm == 'B1':
        adaptive = CoarseDNS(
            channels=dns.channels, hidden=dns.projection[0].out_channels,
            radii=dns.radii, eps=dns.eps, adaptive_scale=True).to(device)
        missing, unexpected = adaptive.load_state_dict(dns.state_dict(), strict=False)
        if set(missing) != {'scale_predictor.weight', 'scale_predictor.bias'} or unexpected:
            raise AssertionError(f'adaptive DNS load mismatch: {missing}, {unexpected}')
        if any(torch.count_nonzero(value) for value in
               adaptive.scale_predictor.state_dict().values()):
            raise AssertionError('scale logits must start at zero')
        model.base.coarse_dns = adaptive
    elif arm != 'B0':
        raise ValueError(arm)
    # Constructing B1's zeroed head consumes initializer RNG draws. Restore
    # the common seed so subsequent optimization starts identically in B0/B1.
    seed_all()
    return model


def frozen_local_digest(model):
    """Audit identical local decoder and pretrained upsampler initialization."""
    names = set(prior.LOCAL_NAMES + prior.UP_NAMES)
    digest = hashlib.sha256()
    for name, value in model.base.state_dict().items():
        if name.split('.')[0] in names:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_stage_start(args, arm, stage, device):
    if stage == 'coarse':
        model = source_model(args, arm, device)
        return model, sha256_file(args.b_checkpoint)
    path = args.output / arm / 'best_coarse.pth'
    payload = torch.load(path, map_location='cpu', weights_only=False)
    expected = {'recipe': RECIPE, 'arm': arm, 'stage': 'coarse',
                'source_sha256': sha256_file(args.b_checkpoint),
                'split_sha256': sha256_file(args.split_file),
                'test_accessed': False}
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f'{arm} coarse parent {key} mismatch')
    model = source_model(args, arm, device)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    if frozen_local_digest(model) != payload['local_start_digest']:
        raise AssertionError('coarse training changed local/upscaler initialization')
    return model, sha256_file(path)


@torch.no_grad()
def check(args, device):
    install_mutual_guard()
    b0 = source_model(args, 'B0', device).eval()
    b1 = source_model(args, 'B1', device).eval()
    dns0, dns1 = b0.base.coarse_dns, b1.base.coarse_dns
    if float(dns0.gate) == 0 or float(b0.base.coarse_attention.gate) == 0:
        raise AssertionError('expected trained nonzero DNS and attention gates')
    torch.manual_seed(SEED + 17)
    target = torch.randn(1, dns0.channels, 16, 16, device=device)
    source = torch.randn_like(target)
    pair0, pair1 = dns0(target, source), dns1(target, source)
    feature_diff = max(float((x-y).abs().max()) for x, y in zip(pair0, pair1))
    squeezed = dns1.squeezed_descriptor(target)
    # Compare against the exact pre-ablation forward expression, rather than
    # only comparing the two newly constructed model instances.
    legacy = target + dns0.gate * dns0.projection(dns0.squeezed_descriptor(target))
    legacy_diff = float((legacy-pair0[0]).abs().max())
    weights = dns1.scale_weight_map(squeezed)
    weight_diff = float((weights-1).abs().max())
    flow0, corr0 = b0.base.coarsest_resolution_flow(
        target, source, 256, 256, return_corr=True)
    flow1, corr1 = b1.base.coarsest_resolution_flow(
        target, source, 256, 256, return_corr=True)
    coarse_diff = float((flow0-flow1).abs().max())
    corr_diff = float((corr0-corr1).abs().max())
    if any(x != 0 for x in (feature_diff, legacy_diff, weight_diff,
                            coarse_diff, corr_diff)):
        raise AssertionError('B0/B1 warm starts differ before training')
    source_local = frozen_local_digest(b0)
    if source_local != frozen_local_digest(b1):
        raise AssertionError('initial local/upscaler tensors differ')
    result = {
        'source_sha256': sha256_file(args.b_checkpoint),
        'trained_gates': prior.gates(b0),
        'order': 'SA -> DNS -> CA -> Global Correlation',
        'radii': dns0.radii,
        'source_local_upscaler_digest': source_local,
        'initial_max_abs_difference': {
            'dns_features': feature_diff, 'legacy_dns_expression': legacy_diff,
            'scale_weights_from_one': weight_diff,
            'correlation': corr_diff, 'coarse_flow': coarse_diff},
        'parameter_count': {arm: sum(p.numel() for p in model.parameters())
                            for arm, model in (('B0', b0), ('B1', b1))},
        'test_accessed': False,
    }
    del target, source, squeezed, weights, pair0, pair1, flow0, flow1, corr0, corr1
    if args.full_forward_check:
        if device.type != 'cuda':
            raise RuntimeError('full local correlation check needs CUDA/CuPy')
        source_image = torch.randint(0, 256, (1, 3, 512, 512),
                                     dtype=torch.uint8)
        target_image = torch.randint(0, 256, (1, 3, 512, 512),
                                     dtype=torch.uint8)
        batch = {'source_image': source_image, 'target_image': target_image,
                 'flow_map': torch.zeros(1, 2, 512, 512),
                 'correspondence_mask': torch.ones(1, 512, 512, dtype=torch.bool)}
        out0, out1 = b0(batch, device), b1(batch, device)
        full_diff = {key: float((out0[key]-out1[key]).abs().max())
                     for key in ('coarse16', 'local32', 'local64', 'final512')}
        if any(value != 0 for value in full_diff.values()):
            raise AssertionError('initial full flow differs')
        result['initial_max_abs_difference']['full_flows'] = full_diff
        # Zero logits must receive a finite update gradient through the
        # trained, nonzero DNS and attention gates.
        del out0, out1
        for p in b1.parameters():
            p.requires_grad_(False)
        for p in dns1.scale_predictor.parameters():
            p.requires_grad_(True)
        with torch.enable_grad():
            output = b1(batch, device)
            loss = objective(output, batch, device, hierarchical=True)[0]
            loss.backward()
        gradients = [p.grad for p in dns1.scale_predictor.parameters()]
        if not all(g is not None and torch.isfinite(g).all() for g in gradients):
            raise AssertionError('scale head gradient is missing/nonfinite')
        result['scale_head_gradient_norm'] = float(torch.linalg.vector_norm(
            torch.cat([g.flatten() for g in gradients])))
        if result['scale_head_gradient_norm'] == 0:
            raise AssertionError('scale head has zero gradient')
    return result


def save_checkpoint(path, args, model, optimizer, scheduler, arm, stage,
                    epoch, steps, score, history, parent_sha, local_digest):
    payload = {
        'recipe': RECIPE, 'arm': arm, 'stage': stage,
        'source_sha256': sha256_file(args.b_checkpoint),
        'pretrained_sha256': sha256_file(args.pretrained),
        'attention_checkpoint_sha256': sha256_file(args.attention_checkpoint),
        'split_sha256': sha256_file(args.split_file),
        'parent_sha256': parent_sha, 'local_start_digest': local_digest,
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
        'test_accessed': False,
    }
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    os.replace(temp, path)


class ScaleStatistics:
    """Validation DNS weight/delta distribution, both modalities, all frames."""

    def __init__(self, dns):
        self.dns = dns
        self.count = 0
        self.weight_sum = torch.zeros(2, len(dns.radii), dtype=torch.float64)
        self.weight_square_sum = torch.zeros_like(self.weight_sum)
        self.high = torch.zeros_like(self.weight_sum)
        self.low = torch.zeros_like(self.weight_sum)
        self.positions = [0, 0]
        self.delta_sum = [0., 0.]
        self.feature_sum = [0., 0.]

    @torch.no_grad()
    def hook(self, _module, inputs, outputs):
        for index, (features, enhanced) in enumerate(zip(inputs, outputs)):
            squeezed = self.dns.squeezed_descriptor(features)
            weights = self.dns.scale_weight_map(squeezed).detach()
            self.weight_sum[index] += weights.double().sum((0, 2, 3)).cpu()
            self.weight_square_sum[index] += weights.double().square().sum((0, 2, 3)).cpu()
            self.high[index] += (weights > 1.5).sum((0, 2, 3)).cpu()
            self.low[index] += (weights < .5).sum((0, 2, 3)).cpu()
            self.positions[index] += weights.shape[0] * weights.shape[2] * weights.shape[3]
            self.delta_sum[index] += float((enhanced-features).abs().sum())
            self.feature_sum[index] += float(features.abs().sum())
        self.count += 1

    def report(self):
        if self.count != prior.EXPECTED_VALID_FRAMES:
            raise AssertionError(f'DNS statistics covered {self.count} validation frames')
        denominator = torch.tensor(self.positions, dtype=torch.float64)[:, None]
        mean = self.weight_sum / denominator
        variance = (self.weight_square_sum / denominator - mean.square()).clamp_min(0)
        return {'frames': self.count, 'modalities': 2,
                'positions_per_scale_per_modality': self.positions,
                'radii': list(self.dns.radii),
                'modality_order': ['VI_target', 'IR_source'],
                'weight_mean': mean.tolist(),
                'weight_std': variance.sqrt().tolist(),
                'fraction_weight_gt_1_5': (self.high/denominator).tolist(),
                'fraction_weight_lt_0_5': (self.low/denominator).tolist(),
                'dns_gate': float(self.dns.gate.detach()),
                'relative_l1_feature_change': [
                    delta/max(feature, 1e-12) for delta, feature in
                    zip(self.delta_sum, self.feature_sum)]}


def train_stage(args, arm, stage, train_videos, val_videos, device):
    directory = args.output / arm
    best_path = directory / f'best_{stage}.pth'
    if best_path.exists():
        raise FileExistsError(f'will not overwrite {best_path}')
    directory.mkdir(parents=True, exist_ok=True)
    model, parent_sha = load_stage_start(args, arm, stage, device)
    local_digest = frozen_local_digest(model)
    if args.gpu_epoch_cache and device.type != 'cuda':
        raise RuntimeError('--gpu-epoch-cache requires CUDA')
    active = prior.active_parameters(model, stage)
    locked = frozen_digest(model)
    optimizer = torch.optim.AdamW(active, lr=prior.LR[stage])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=prior.EPOCHS[stage], eta_min=prior.ETA_MIN[stage])
    _, epoch_zero, best = prior.validation(model, val_videos, device,
                                           args.data_workers)
    best_epoch = steps = 0
    history = []
    save_checkpoint(best_path, args, model, optimizer, scheduler,
                    arm, stage, 0, steps, best, history, parent_sha, local_digest)
    for epoch in range(1, prior.EPOCHS[stage]+1):
        train_start = time.perf_counter()
        cache_stats = None
        if args.gpu_epoch_cache:
            phase_frames = Frames(train_videos,
                                  phase=(epoch-1) % TRAIN_FRAME_STRIDE)
            cached, cache_stats = build_gpu_epoch_cache(
                phase_frames, device, num_workers=args.data_workers,
                reserve_gib=args.gpu_cache_reserve_gib)
            update = train_epoch(model, optimizer, train_videos, device,
                                 epoch, cached_frames=cached)
            if update['image_order_sha256'] != expected_order_sha256(
                    phase_frames, epoch, SEED, BATCH_SIZE):
                raise AssertionError('CUDA image cache changed sample order')
            del cached
            torch.cuda.empty_cache()
        else:
            update = train_epoch(model, optimizer, train_videos, device,
                                 epoch, num_workers=args.data_workers)
        train_seconds = time.perf_counter()-train_start
        if update['steps'] != prior.EXPECTED_TRAIN_FRAMES // TRAIN_FRAME_STRIDE // BATCH_SIZE:
            raise AssertionError('unexpected optimizer step count')
        steps += update['steps']
        validation_start = time.perf_counter()
        _, val_summary, score = prior.validation(model, val_videos, device,
                                                  args.data_workers)
        val_seconds = time.perf_counter()-validation_start
        lr_used = optimizer.param_groups[0]['lr']
        scheduler.step()
        record = {'epoch': epoch, 'steps': steps, 'train': update,
                  'validation': val_summary, 'lr_used': lr_used,
                  'gates': prior.gates(model),
                  'train_wall_seconds': train_seconds,
                  'validation_wall_seconds': val_seconds,
                  'gpu_cache': cache_stats}
        history.append(record)
        if frozen_digest(model) != locked:
            raise AssertionError(f'{arm} {stage}: frozen parameters or BN changed')
        if stage == 'coarse' and frozen_local_digest(model) != local_digest:
            raise AssertionError('coarse training modified local/upscaler tensors')
        if score < best:
            best, best_epoch = score, epoch
            save_checkpoint(best_path, args, model, optimizer, scheduler,
                            arm, stage, epoch, steps, best, history,
                            parent_sha, local_digest)
        save_json(directory/f'history_{stage}.json', history)
        print(f'{arm} {stage} {epoch}/{prior.EPOCHS[stage]}: '
              f'val final={score:.4f}, best={best:.4f}@{best_epoch}, '
              f'coarse={val_summary["all"]["stage_epe_512px"]["coarse16"]:.4f}, '
              f'gates={prior.gates(model)}, steps={steps}', flush=True)
    selected = torch.load(best_path, map_location='cpu', weights_only=False)
    model.load_state_dict(selected['model_state_dict'], strict=True)
    # Keep the extra diagnostic hook outside the timed image evaluation.
    image_rows = prior.image_evaluation(model, val_videos, device,
                                        args.data_workers)
    image_summary = prior.metric_summary(image_rows)
    prior.save_csv(directory/f'per_frame_best_{stage}.csv', image_rows)
    stats = ScaleStatistics(model.base.coarse_dns)
    handle = model.base.coarse_dns.register_forward_hook(stats.hook)
    try:
        prior.validation(model, val_videos, device, args.data_workers)
    finally:
        handle.remove()
    report = {
        'recipe': RECIPE, 'arm': arm, 'stage': stage,
        'source_b_fine_sha256': sha256_file(args.b_checkpoint),
        'parent_sha256': parent_sha,
        'local_start_digest': local_digest,
        'best_checkpoint_sha256': sha256_file(best_path),
        'selected_epoch': best_epoch,
        'selected_steps': selected['optimizer_steps'],
        'completed_epochs': prior.EPOCHS[stage], 'completed_steps': steps,
        'epoch_zero': epoch_zero,
        'best_validation_final_epe_512px': best,
        'best_image_metrics': image_summary,
        'best_gates': prior.gates(model),
        'scale_statistics': stats.report(),
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'trainable_count': sum(p.numel() for p in active),
        'selection': '4 validation sequences, valid-pixel-weighted 512px final EPE',
        'loss': 'L_final + 0.1 L_32 + 0.2 L_64, masked Charbonnier',
        'data_order': 'seed 2026; 38 training sequences; deterministic epoch shuffle',
        'budget': f'{prior.EPOCHS[stage]} epochs x 760 steps',
        'optimizer': 'AdamW, cosine schedule, gradient clip 1.0',
        'learning_rate': prior.LR[stage],
        'inference_time_scope': 'CUDA-synchronized model preprocessing+forward, '
                                'excludes disk IO, GT, warp, and metrics',
        'test_accessed': False,
    }
    if stage == 'coarse':
        with prior.frozen_inference(model):
            corr_rows, corr_summary = prior.correlation_diagnostics(
                model, val_videos, device, args.data_workers)
        prior.save_csv(directory/'coarse_correlation_per_frame.csv', corr_rows)
        report['correlation'] = corr_summary
    save_json(directory/f'report_{stage}.json', report)
    return report


def final_report(args):
    reports, frames = {}, {}
    for arm in ARMS:
        directory = args.output/arm
        reports[arm] = {stage: json.loads(
            (directory/f'report_{stage}.json').read_text(encoding='utf-8'))
            for stage in STAGES}
        with (directory/'per_frame_best_fine.csv').open(
                newline='', encoding='utf-8-sig') as stream:
            frames[arm] = list(csv.DictReader(stream))
        if len(frames[arm]) != prior.EXPECTED_VALID_FRAMES:
            raise AssertionError('report must cover all 800 validation frames')
        if reports[arm]['fine']['parent_sha256'] != reports[arm]['coarse']['best_checkpoint_sha256']:
            raise AssertionError('fine did not start from its own best coarse checkpoint')
        if reports[arm]['coarse']['local_start_digest'] != reports[arm]['fine']['local_start_digest']:
            raise AssertionError('local initialization changed between stages')
    if reports['B0']['coarse']['local_start_digest'] != reports['B1']['coarse']['local_start_digest']:
        raise AssertionError('B0/B1 used different local initialization')
    for stage in STAGES:
        left = json.loads((args.output/'B0'/f'history_{stage}.json').read_text(
            encoding='utf-8'))
        right = json.loads((args.output/'B1'/f'history_{stage}.json').read_text(
            encoding='utf-8'))
        if len(left) != prior.EPOCHS[stage] or len(right) != len(left):
            raise AssertionError('B0/B1 training budgets differ')
        for record0, record1 in zip(left, right):
            if (record0['steps'] != record1['steps'] or
                    record0['lr_used'] != record1['lr_used'] or
                    record0['train']['image_order_sha256'] !=
                    record1['train']['image_order_sha256']):
                raise AssertionError('B0/B1 optimizer steps, LR, or sample order differ')
    for stage in STAGES:
        score0 = reports['B0'][stage]['epoch_zero']['all']['stage_epe_512px']['final512']
        score1 = reports['B1'][stage]['epoch_zero']['all']['stage_epe_512px']['final512']
        if stage == 'coarse' and score0 != score1:
            raise AssertionError('B0/B1 did not start coarse training identically')
    paired = []
    for row0, row1 in zip(frames['B0'], frames['B1']):
        if any(row0[key] != row1[key] for key in ('sequence', 'stem', 'valid_pixels')):
            raise AssertionError('different frames or valid mask counts')
        paired.append({'sequence': row0['sequence'], 'stem': row0['stem'],
                       'valid_pixels': row0['valid_pixels'],
                       'B0_final_epe_512px': row0['final512_epe_512px'],
                       'B1_final_epe_512px': row1['final512_epe_512px'],
                       'B1_minus_B0_final_epe_512px':
                           float(row1['final512_epe_512px'])-float(row0['final512_epe_512px']),
                       'B0_coarse_epe_512px': row0['coarse16_epe_512px'],
                       'B1_coarse_epe_512px': row1['coarse16_epe_512px']})
    prior.save_csv(args.output/'paired_per_frame.csv', paired)
    b0 = reports['B0']['fine']['best_image_metrics']['all']['final512']['epe_512px']
    b1 = reports['B1']['fine']['best_image_metrics']['all']['final512']['epe_512px']
    per_sequence_gain = {}
    for seq in prior.SPLIT_NAMES:
        s0 = reports['B0']['fine']['best_image_metrics']['by_sequence'][seq]['final512']['epe_512px']
        s1 = reports['B1']['fine']['best_image_metrics']['by_sequence'][seq]['final512']['epe_512px']
        per_sequence_gain[seq] = s0-s1
    result = {
        'recipe': RECIPE,
        'criterion': 'matched continuation; compare selected final 512px EPE, '
                     'paired frames, 4 sequences and inference cost',
        'B0_final_epe_512px': b0, 'B1_final_epe_512px': b1,
        'B1_gain_512px': b0-b1,
        'improved_frames_B1_vs_B0': sum(
            float(r['B1_final_epe_512px']) < float(r['B0_final_epe_512px'])
            for r in paired),
        'sequence_gain_512px': per_sequence_gain,
        'arms': reports,
        'metric_definitions': {
            'epe': 'valid-pixel-weighted XY flow error at 512x512',
            'pck': 'valid-pixel fraction with EPE strictly < 1/3/5 px',
            'nmi_64bin_2mi_entropy_sum': 'grayscale 64 bins; 2MI/(H(VI)+H(warped IR))',
            'lncc17_squared_proxy': '17x17 squared local NCC, valid-pixel mean',
            'edge_information_retention_proxy': 'Sobel-magnitude retention',
            'edge_overlap_f1_3x3_proxy': 'top-quartile Sobel edge F1, 1px tolerance',
            'Q_F_AB': 'not computed; no fusion output'},
        'decision': 'Treat any first-run gain as provisional; repeat matched '
                    'B0/B1 before adding a third DNS radius',
        'test_accessed': False,
    }
    save_json(args.output/'comparison.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True,
                        choices=('check', 'coarse', 'fine', 'all', 'report'))
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--split-file', type=Path, required=True)
    parser.add_argument('--pretrained', type=Path, required=True)
    parser.add_argument('--attention-checkpoint', type=Path, required=True)
    parser.add_argument('--b-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-workers', type=int, default=0)
    parser.add_argument('--gpu-epoch-cache', action='store_true')
    parser.add_argument('--gpu-cache-reserve-gib', type=float, default=4.)
    parser.add_argument('--full-forward-check', action='store_true')
    args = parser.parse_args()
    if args.data_workers < 0 or args.gpu_cache_reserve_gib <= 0:
        parser.error('data-workers must be >=0 and cache reserve must be >0')
    if args.stage == 'report':
        print(json.dumps(final_report(args), ensure_ascii=False, indent=2))
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.stage == 'check':
        print(json.dumps(check(args, device), ensure_ascii=False, indent=2))
        return
    if args.stage in STAGES and args.arm is None:
        parser.error('--arm is required for a single stage')
    if args.data_root is None:
        parser.error('--data-root is required for training')
    seed_all()
    install_mutual_guard()
    train_videos = VTMOTVideos(args.data_root, args.split_file, 'train')
    val_videos = VTMOTVideos(args.data_root, args.split_file, 'eval')
    if (len(train_videos.sequences) != 38 or
            sum(map(len, train_videos.sequences.values())) != prior.EXPECTED_TRAIN_FRAMES or
            tuple(val_videos.sequences) != prior.SPLIT_NAMES or
            sum(map(len, val_videos.sequences.values())) != prior.EXPECTED_VALID_FRAMES):
        raise ValueError('unexpected train/validation manifest')
    if args.stage == 'all':
        existing = [args.output/arm/f'best_{stage}.pth'
                    for arm in ARMS for stage in STAGES
                    if (args.output/arm/f'best_{stage}.pth').exists()]
        if existing:
            raise FileExistsError(f'new output required; found {existing[0]}')
        args.full_forward_check = True
        save_json(args.output/'initialization_check.json', check(args, device))
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        for stage in STAGES:
            for arm in ARMS:
                print(f'=== {arm} {stage} ===', flush=True)
                train_stage(args, arm, stage, train_videos, val_videos, device)
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        result = final_report(args)
        print(json.dumps({'B0_final_epe_512px': result['B0_final_epe_512px'],
                          'B1_final_epe_512px': result['B1_final_epe_512px'],
                          'test_accessed': False}, indent=2))
    else:
        result = train_stage(args, args.arm, args.stage, train_videos,
                             val_videos, device)
        print(json.dumps({'arm': args.arm, 'stage': args.stage,
                          'selected_epoch': result['selected_epoch'],
                          'best_val_final_epe_512px':
                              result['best_validation_final_epe_512px'],
                          'test_accessed': False}, indent=2))


if __name__ == '__main__':
    main()
