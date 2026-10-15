"""Matched VTMOT coarse DNS-position ablation followed by local-only training.

Arms: S=SA/CA only, A=DNS->SA->CA, B=SA->DNS->CA, C=SA->CA->DNS.
All four start from the same original GLU-Net pretrained weights plus the
RoadScene no-DNS SA/CA checkpoint (attention and decoder4). No previously
DNS-trained or fine-adapted weights are used to initialize an arm.

The VTMOT single-frame protocol is fixed: 38 train sequences, four validation
sequences, 512px VI->IR GT mask, seed 2026, deterministic geometric flips,
batch 2, hierarchical Charbonnier loss, AdamW, cosine schedule, and exactly
30 coarse plus 30 local epochs (760 optimizer steps per epoch). Validation
selects the best checkpoint but does not shorten any arm's step budget.
The test split is never instantiated.

Optional --gpu-epoch-cache stages the current training stride phase on CUDA
before each epoch. It changes only data transport. The --stage cache-check
verifies all cached inputs, shuffle order, augmented batch, model output and
loss against the original streaming path before enabling this option.
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
from torch import nn
from torch.utils.data import DataLoader

from datasets.vtmot_video import VTMOTVideos
from models.our_models.GLUNet import GLUNet_model
from models.our_models.coarse_attention import CoarseSelfCrossAttention
from models.our_models.coarse_dns import CoarseDNS
from roadscene_coarse import (coarse_statistics, load_base_weights, prepare,
                              sha256_file)
from roadscene_fusion_hierarchical import FineArm, augment, objective
from roadscene_no_dcn import LOCAL_NAMES, UP_NAMES, check_no_dcn
from vtmot_a1_adapt import (BATCH_SIZE, Frames, LEARNING_RATE as COARSE_LR,
                            MAX_EPOCHS as COARSE_EPOCHS, SEED, TRAIN_FRAME_STRIDE,
                            evaluate_vtmot, frozen_digest, frozen_inference,
                            loader, save_json, seed_all, train_epoch)
from vtmot_a1_fine_adapt import LEARNING_RATE as FINE_LR
from vtmot_geometry import backward_warp
from vtmot_gpu_epoch_cache import (FIELDS, build_gpu_epoch_cache,
                                   expected_order_sha256)
from vtmot_keyframe import edge_metrics, install_mutual_guard, lncc, nmi
from vtmot_local_correction_ab import SPLIT_NAMES


ORDERS = {'S': None, 'A': 'before_sa', 'B': 'between_sa_ca',
          'C': 'after_ca'}
STAGES = ('coarse', 'fine')
EPOCHS = {'coarse': COARSE_EPOCHS, 'fine': 30}
LR = {'coarse': COARSE_LR, 'fine': FINE_LR}
ETA_MIN = {'coarse': 1e-6, 'fine': 5e-7}
EXPECTED_TRAIN_FRAMES = 38 * 200
EXPECTED_VALID_FRAMES = 4 * 200


def checkpoint_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def initialize(arm, args, device):
    """Identical non-DNS start, shared initial DNS parameters for A/B/C."""
    seed_all()
    base = GLUNet_model(
        evaluation=False, pyramid_type='VGG', cyclic_consistency=True,
        backbone_pretrained=False, coarse_attention=True,
        coarse_dns=arm != 'S', coarse_dns_order=ORDERS[arm] or 'before_sa',
        local_dcn_steps=0)
    load_base_weights(base, args.pretrained)
    payload = torch.load(args.attention_checkpoint, map_location='cpu',
                         weights_only=False)
    if not {'attention_state_dict', 'decoder4_state_dict'} <= payload.keys():
        raise ValueError('initial SA/CA checkpoint is missing trained modules')
    base.coarse_attention.load_state_dict(payload['attention_state_dict'], strict=True)
    base.decoder4.load_state_dict(payload['decoder4_state_dict'], strict=True)
    model = FineArm(base, use_fusion=False).to(device)
    check_no_dcn(model.base)
    if model.fusion is not None:
        raise AssertionError('position ablation must not add local fusion')
    return model


def gates(model):
    return {'attention': float(model.base.coarse_attention.gate.detach()),
            'dns': (float(model.base.coarse_dns.gate.detach())
                    if model.base.coarse_dns is not None else None)}


def active_parameters(model, stage):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if stage == 'coarse':
        model.base.train_coarse_encoder = True
        modules = [model.base.pyramid._modules['level_4'],
                   model.base.decoder4, model.base.coarse_attention]
        if model.base.coarse_dns is not None:
            modules.append(model.base.coarse_dns)
        active = [p for module in modules for p in module.parameters()]
    elif stage == 'fine':
        model.base.train_coarse_encoder = False
        active = [p for name in LOCAL_NAMES
                  for p in getattr(model.base, name).parameters()]
    else:
        raise ValueError(stage)
    if not active or len({id(p) for p in active}) != len(active):
        raise AssertionError('missing or duplicated optimizer parameters')
    for parameter in active:
        parameter.requires_grad_(True)
    if any(p.requires_grad for name in UP_NAMES
           for p in getattr(model.base, name).parameters()):
        raise AssertionError('pretrained flow upsampler became trainable')
    model.train()  # FineArm.train() leaves base and all BN modules in eval.
    if model.base.training or any(m.training for m in model.base.modules()
                                  if isinstance(m, nn.modules.batchnorm._BatchNorm)):
        raise AssertionError('frozen BatchNorm statistics changed mode')
    return active


def check_order_math():
    """A/B/C have identical learnable size and exact zero-DNS output."""
    seed_all()
    attention = CoarseSelfCrossAttention(channels=8, heads=2).eval()
    dns = CoarseDNS(channels=8, hidden=4).eval()
    # A nonzero trained attention gate exposes the B path; zero DNS still
    # recovers precisely the same SA/CA baseline in every placement.
    with torch.no_grad():
        attention.gate.fill_(-.03973483666777611)
        target = torch.randn(1, 8, 6, 6)
        source = torch.randn(1, 8, 6, 6)
        reference = attention(target, source)
        a = attention(*dns(target, source))
        b = attention(target, source, after_self=dns)
        c = dns(*reference)
        max_diff = max(float((x-y).abs().max()) for pair in (a, b, c)
                       for x, y in zip(pair, reference))
        if max_diff != 0. or float(dns.gate) != 0.:
            raise AssertionError(f'zero-DNS placement differs: {max_diff}')
        attention.gate.zero_()
        zero_attention = attention(target, source)
        if not (torch.equal(zero_attention[0], target) and
                torch.equal(zero_attention[1], source)):
            raise AssertionError('zero attention gate did not recover input')
    return {'zero_dns_max_abs_diff': max_diff,
            'zero_attention_recovers_input': True,
            'dns_parameters_each_arm': sum(p.numel() for p in dns.parameters()),
            'attention_parameters_each_arm': sum(p.numel() for p in attention.parameters())}


def check_initialization(args):
    install_mutual_guard()
    device = torch.device('cpu')
    models = {arm: initialize(arm, args, device) for arm in ORDERS}
    reference = models['S']
    common_names = ('coarse_attention', 'decoder4', 'decoder3', 'decoder2',
                    'decoder1', 'deconv4', 'deconv2', 'upfeat2')
    for arm in ('A', 'B', 'C'):
        model = models[arm]
        for name in common_names:
            left = getattr(reference.base, name).state_dict()
            right = getattr(model.base, name).state_dict()
            if any(not torch.equal(left[key], right[key]) for key in left):
                raise AssertionError(f'{arm} initial {name} differs')
        if float(model.base.coarse_dns.gate) != 0.:
            raise AssertionError(f'{arm} DNS gate is not zero')
    start_dns = models['A'].base.coarse_dns.state_dict()
    for arm in ('B', 'C'):
        if any(not torch.equal(v, models[arm].base.coarse_dns.state_dict()[k])
               for k, v in start_dns.items()):
            raise AssertionError(f'{arm} DNS initialization differs')
    counts = {arm: sum(p.numel() for p in model.parameters())
              for arm, model in models.items()}
    if len({counts[arm] for arm in ('A', 'B', 'C')}) != 1:
        raise AssertionError('A/B/C parameter counts differ')
    generator = torch.Generator().manual_seed(SEED + 1)
    target = torch.randn((1, 512, 16, 16), generator=generator)
    source = torch.randn((1, 512, 16, 16), generator=generator)
    with torch.inference_mode():
        reference_flow, reference_corr = reference.base.coarsest_resolution_flow(
            target, source, 256, 256, return_corr=True)
        full_differences = {}
        for arm in ('A', 'B', 'C'):
            flow, corr = models[arm].base.coarsest_resolution_flow(
                target, source, 256, 256, return_corr=True)
            full_differences[arm] = {
                'coarse_flow_max_abs': float((flow-reference_flow).abs().max()),
                'correlation_max_abs': float((corr-reference_corr).abs().max())}
            if any(value != 0. for value in full_differences[arm].values()):
                raise AssertionError(f'{arm} zero DNS changed full coarse output')
    return {'parameter_counts': counts, 'initial_gates': {
        arm: gates(model) for arm, model in models.items()},
        'zero_dns_full_coarse_differences': full_differences,
        'no_dns_initial_state_sha256': checkpoint_digest(reference),
        'dns_initial_state_sha256': hashlib.sha256(b''.join(
            value.detach().cpu().contiguous().numpy().tobytes()
            for value in start_dns.values())).hexdigest()}


def check_full_forward(args):
    if not torch.cuda.is_available():
        raise RuntimeError('full local-correlation check requires CUDA/CuPy')
    device = torch.device('cuda')
    generator = torch.Generator().manual_seed(SEED + 2)
    source = torch.randint(0, 256, (1, 3, 512, 512),
                           generator=generator, dtype=torch.uint8)
    target = torch.randint(0, 256, (1, 3, 512, 512),
                           generator=generator, dtype=torch.uint8)
    batch = {'source_image': source, 'target_image': target,
             'flow_map': torch.zeros(1, 2, 512, 512),
             'correspondence_mask': torch.ones(1, 512, 512, dtype=torch.bool)}
    initial = None
    differences = {}
    for arm in ORDERS:
        model = initialize(arm, args, device)
        with torch.inference_mode():
            output = model(batch, device)
        if initial is None:
            initial = {stage: output[stage].detach().clone()
                       for stage in ('coarse16', 'local32', 'local64', 'final512')}
        else:
            differences[arm] = {stage: float((output[stage]-initial[stage])
                                              .abs().max()) for stage in initial}
            if any(value != 0. for value in differences[arm].values()):
                raise AssertionError(f'{arm} zero DNS changed full flow')
        if arm == 'B':
            coarse = active_parameters(model, 'coarse')
            model.zero_grad(set_to_none=True)
            output = model(batch, device)
            loss, _, _ = objective(output, batch, device, hierarchical=True)
            loss.backward()
            dns_gradient = model.base.coarse_dns.gate.grad
            if (dns_gradient is None or not torch.isfinite(dns_gradient) or
                    float(dns_gradient.abs()) == 0.):
                raise AssertionError('B DNS gate has no coarse gradient')
            if not all(p.grad is None or torch.isfinite(p.grad).all()
                       for p in coarse):
                raise AssertionError('nonfinite coarse gradient')
            model.zero_grad(set_to_none=True)
            local = active_parameters(model, 'fine')
            output = model(batch, device)
            loss, _, _ = objective(output, batch, device, hierarchical=True)
            loss.backward()
            if not any(p.grad is not None and float(p.grad.abs().sum()) > 0
                       for p in local):
                raise AssertionError('fine local modules have no gradient')
            if any(p.grad is not None and torch.count_nonzero(p.grad)
                   for p in model.base.coarse_dns.parameters()):
                raise AssertionError('fine loss updated frozen DNS')
        del model, output
        torch.cuda.empty_cache()
    return {'zero_dns_full_stage_max_abs_differences': differences,
            'B_dns_gate_coarse_gradient': float(dns_gradient.detach()),
            'fine_local_gradient': 'finite and nonzero',
            'fine_coarse_gradient': 'zero'}


def check_gpu_epoch_cache(args, train_videos, device):
    """Verify every cached tensor, full shuffle order, and model output."""
    if device.type != 'cuda':
        raise RuntimeError('cache-check needs CUDA')
    model = initialize(args.arm or 'S', args, device).eval()
    frames = Frames(train_videos, phase=0)
    cached, stats = build_gpu_epoch_cache(
        frames, device, num_workers=args.data_workers, verify=True,
        reserve_gib=args.gpu_cache_reserve_gib)
    expected = expected_order_sha256(frames, 1, SEED, BATCH_SIZE)
    digest = hashlib.sha256()
    for batch in loader(cached, 1, num_workers=0):
        for name in batch['name']:
            digest.update((name + '\n').encode())
    if digest.hexdigest() != expected:
        raise AssertionError('GPU cache changed the full epoch sample order')
    cpu_batch = next(iter(loader(frames, 1, num_workers=0)))
    gpu_batch = next(iter(loader(cached, 1, num_workers=0)))
    if cpu_batch['name'] != gpu_batch['name']:
        raise AssertionError('first cached batch has different image pairs')
    for key in FIELDS:
        if not torch.equal(cpu_batch[key].to(device), gpu_batch[key]):
            raise AssertionError(f'first cached batch changed {key}')
    cpu_augmented = augment(cpu_batch, 1, enabled=True)
    gpu_augmented = augment(gpu_batch, 1, enabled=True)
    for key in FIELDS:
        if not torch.equal(cpu_augmented[key].to(device),
                           gpu_augmented[key]):
            raise AssertionError(f'GPU cache changed augmented {key}')
    with torch.inference_mode():
        cpu_output = model(cpu_augmented, device)
        gpu_output = model(gpu_augmented, device)
        stage_differences = {stage: float((cpu_output[stage]-gpu_output[stage])
                                          .abs().max())
                             for stage in ('coarse16', 'local32', 'local64',
                                           'final512')}
        if any(value != 0. for value in stage_differences.values()):
            raise AssertionError('GPU cache changed model output')
        cpu_loss = objective(cpu_output, cpu_augmented, device,
                             hierarchical=True)[0]
        gpu_loss = objective(gpu_output, gpu_augmented, device,
                             hierarchical=True)[0]
        if not torch.equal(cpu_loss, gpu_loss):
            raise AssertionError('GPU cache changed hierarchical loss')
    return {'cache': stats, 'full_epoch_order_sha256': expected,
            'first_batch_inputs_and_augmentation': 'exact',
            'all_four_flow_outputs_max_abs_diff': stage_differences,
            'hierarchical_loss_max_abs_diff': float((cpu_loss-gpu_loss).abs()),
            'gpu_allocated_mib': torch.cuda.memory_allocated()/1024**2,
            'test_accessed': False}


def checkpoint(path, args, model, optimizer, scheduler, arm, stage, epoch,
               steps, best_score, history):
    payload = {
        'recipe': 'vtmot_dns_position_v1', 'arm': arm, 'stage': stage,
        'pretrained_sha256': sha256_file(args.pretrained),
        'attention_checkpoint_sha256': sha256_file(args.attention_checkpoint),
        'split_sha256': sha256_file(args.split_file),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch, 'optimizer_steps': steps, 'best_val_final_epe_512px': best_score,
        'history': history, 'torch_rng_state': torch.get_rng_state(),
        'numpy_rng_state': np.random.get_state(),
        'python_rng_state': random.getstate(),
        'cuda_rng_state': (torch.cuda.get_rng_state()
                           if torch.cuda.is_available() else None),
        'test_accessed': False,
    }
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_best_coarse(args, arm, device):
    path = args.output / arm / 'best_coarse.pth'
    payload = torch.load(path, map_location='cpu', weights_only=False)
    expected = {'recipe': 'vtmot_dns_position_v1', 'arm': arm, 'stage': 'coarse',
                'pretrained_sha256': sha256_file(args.pretrained),
                'attention_checkpoint_sha256': sha256_file(args.attention_checkpoint),
                'split_sha256': sha256_file(args.split_file),
                'test_accessed': False}
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f'{arm} best coarse checkpoint {key} mismatch')
    model = initialize(arm, args, device)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    return model, payload


def validation(model, videos, device, num_workers=0):
    rows, summary = evaluate_vtmot(model, videos, device,
                                   num_workers=num_workers)
    if len(rows) != EXPECTED_VALID_FRAMES:
        raise AssertionError('validation must cover four complete sequences')
    return rows, summary, summary['all']['stage_epe_512px']['final512']


@torch.inference_mode()
def correlation_diagnostics(model, videos, device, num_workers=0):
    """Read-only selected-coarse CE/Top-1 under original 16-grid convention."""
    rows = []
    model.eval()
    options = {'num_workers': num_workers,
               'pin_memory': device.type == 'cuda'}
    if num_workers:
        options['prefetch_factor'] = 2
    for batch in DataLoader(Frames(videos), batch_size=1, shuffle=False,
                            **options):
        source, target, _, _, truth, valid, _ = prepare(batch, device)
        target_features = model.base.pyramid(target)[-3]
        source_features = model.base.pyramid(source)[-3]
        flow, corr = model.base.coarsest_resolution_flow(
            target_features, source_features, 256, 256, return_corr=True)
        _, sums = coarse_statistics(flow, corr, truth, valid, return_counts=True)
        epe_sum, n_epe, ce_sum, n_corr, hits = sums
        rows.append({'name': batch['name'][0], 'coarse_epe_256px': epe_sum/n_epe,
                     'coarse_pixels': n_epe, 'corr_ce': ce_sum/n_corr,
                     'corr_queries': n_corr, 'corr_hits': hits})
    denominator = sum(r['coarse_pixels'] for r in rows)
    corr_denominator = sum(r['corr_queries'] for r in rows)
    summary = {'coarse_epe_256px': sum(r['coarse_epe_256px']*r['coarse_pixels']
                                       for r in rows)/denominator,
               'corr_ce': sum(r['corr_ce']*r['corr_queries']
                              for r in rows)/corr_denominator,
               'corr_top1': sum(r['corr_hits'] for r in rows)/corr_denominator,
               'valid_coarse_pixels': denominator,
               'valid_corr_queries': corr_denominator}
    summary['by_sequence'] = {}
    for name in SPLIT_NAMES:
        chosen = [r for r in rows if r['name'].split('/')[0] == name]
        coarse_count = sum(r['coarse_pixels'] for r in chosen)
        query_count = sum(r['corr_queries'] for r in chosen)
        summary['by_sequence'][name] = {
            'frames': len(chosen), 'valid_coarse_pixels': coarse_count,
            'valid_corr_queries': query_count,
            'coarse_epe_256px': sum(r['coarse_epe_256px']*r['coarse_pixels']
                                   for r in chosen)/coarse_count,
            'corr_ce': sum(r['corr_ce']*r['corr_queries']
                           for r in chosen)/query_count,
            'corr_top1': sum(r['corr_hits'] for r in chosen)/query_count}
    return rows, summary


def image_evaluation(model, videos, device, num_workers=0):
    """Same full-res GT/mask for coarse16 and final512 registered images."""
    rows = []
    with frozen_inference(model):
        first = next(iter(Frames(videos)))
        model({key: value[None] if isinstance(value, torch.Tensor) else [value]
               for key, value in first.items()}, device)
        options = {'num_workers': num_workers,
                   'pin_memory': device.type == 'cuda'}
        if num_workers:
            options['prefetch_factor'] = 2
        for batch in DataLoader(Frames(videos), batch_size=1, shuffle=False,
                                **options):
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            output = model(batch, device)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            inference_ms = 1000.*(time.perf_counter()-start)
            truth = batch['flow_map'][0].permute(1, 2, 0).numpy()
            valid = batch['correspondence_mask'][0].numpy().astype(bool)
            ir = batch['source_image'][0].permute(1, 2, 0).numpy()
            vi = batch['target_image'][0].permute(1, 2, 0).numpy()
            row = {'sequence': batch['name'][0].split('/')[0],
                   'stem': batch['name'][0].split('/')[1],
                   'valid_pixels': int(valid.sum()),
                   'inference_ms': inference_ms}
            for stage in ('coarse16', 'final512'):
                # FineArm has already converted 256px coarse flow to 512px XY.
                flow = output[stage][0].permute(1, 2, 0).float().cpu().numpy()
                error = np.linalg.norm(flow-truth, axis=-1)
                registered, _ = backward_warp(ir, flow)
                retention, edge_f1 = edge_metrics(vi, registered, valid)
                prefix = stage + '_'
                row[prefix+'epe_512px'] = float(error[valid].mean())
                for threshold in (1, 3, 5):
                    row[prefix+f'pck_{threshold}_pixels'] = int(
                        ((error < threshold) & valid).sum())
                row[prefix+'nmi_64bin_2mi_entropy_sum'] = nmi(vi, registered, valid)
                row[prefix+'lncc17_squared_proxy'] = lncc(vi, registered, valid)
                row[prefix+'edge_information_retention_proxy'] = retention
                row[prefix+'edge_overlap_f1_3x3_proxy'] = edge_f1
            rows.append(row)
    return rows


def metric_summary(rows):
    def collect(chosen):
        pixels = sum(r['valid_pixels'] for r in chosen)
        result = {'frames': len(chosen), 'valid_pixels': pixels,
                  'inference_ms_mean': float(np.mean([r['inference_ms'] for r in chosen])),
                  'inference_ms_p95': float(np.percentile(
                      [r['inference_ms'] for r in chosen], 95))}
        for stage in ('coarse16', 'final512'):
            prefix = stage + '_'
            result[stage] = {
                'epe_512px': sum(r[prefix+'epe_512px']*r['valid_pixels']
                                 for r in chosen)/pixels,
                'pck_512px': {str(t): sum(r[prefix+f'pck_{t}_pixels']
                                           for r in chosen)/pixels for t in (1, 3, 5)},
                'nmi_64bin_2mi_entropy_sum': float(np.mean([
                    r[prefix+'nmi_64bin_2mi_entropy_sum'] for r in chosen])),
                'lncc17_squared_proxy': float(np.mean([
                    r[prefix+'lncc17_squared_proxy'] for r in chosen])),
                'edge_information_retention_proxy': float(np.mean([
                    r[prefix+'edge_information_retention_proxy'] for r in chosen])),
                'edge_overlap_f1_3x3_proxy': float(np.mean([
                    r[prefix+'edge_overlap_f1_3x3_proxy'] for r in chosen])),
            }
        return result
    return {'all': collect(rows), 'by_sequence': {
        name: collect([r for r in rows if r['sequence'] == name])
        for name in SPLIT_NAMES}}


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def train_stage(args, arm, stage, train_videos, val_videos, device):
    directory = args.output / arm
    result_path = directory / f'best_{stage}.pth'
    if result_path.exists():
        raise FileExistsError(f'will not overwrite prior experiment: {result_path}')
    directory.mkdir(parents=True, exist_ok=True)
    if stage == 'coarse':
        model = initialize(arm, args, device)
        parent_sha = None
    else:
        model, parent = load_best_coarse(args, arm, device)
        parent_sha = sha256_file(args.output / arm / 'best_coarse.pth')
    if args.gpu_epoch_cache and device.type != 'cuda':
        raise RuntimeError('--gpu-epoch-cache requires CUDA')
    gpu_label = (f'{torch.cuda.get_device_name()}, '
                 f'allocated_mib={torch.cuda.memory_allocated()/1024**2:.0f}'
                 if device.type == 'cuda' else 'none')
    print(f'{arm} {stage} runtime: device={device}, gpu={gpu_label}, '
          f'data_workers={args.data_workers}, '
          f'gpu_epoch_cache={args.gpu_epoch_cache}', flush=True)
    active = active_parameters(model, stage)
    locked = frozen_digest(model)
    optimizer = torch.optim.AdamW(active, lr=LR[stage])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS[stage], eta_min=ETA_MIN[stage])
    _, epoch_zero, best = validation(model, val_videos, device,
                                     args.data_workers)
    best_epoch = steps = 0
    history = []
    checkpoint(result_path, args, model, optimizer, scheduler, arm, stage,
               0, steps, best, history)
    for epoch in range(1, EPOCHS[stage]+1):
        train_start = time.perf_counter()
        cache_stats = None
        if args.gpu_epoch_cache:
            phase_frames = Frames(train_videos,
                                  phase=(epoch-1) % TRAIN_FRAME_STRIDE)
            print(f'{arm} {stage} epoch {epoch}: loading '
                  f'{len(phase_frames)} train frames into CUDA cache',
                  flush=True)
            cached_frames, cache_stats = build_gpu_epoch_cache(
                phase_frames, device, num_workers=args.data_workers,
                reserve_gib=args.gpu_cache_reserve_gib)
            print(f'{arm} {stage} epoch {epoch}: CUDA cache ready, '
                  f'{cache_stats["gib"]:.2f} GiB, '
                  f'{cache_stats["load_seconds"]:.1f}s, '
                  f'allocated={cache_stats["gpu_allocated_mib_after_load"]:.0f} MiB',
                  flush=True)
            update = train_epoch(model, optimizer, train_videos, device, epoch,
                                 cached_frames=cached_frames)
            if update['image_order_sha256'] != expected_order_sha256(
                    phase_frames, epoch, SEED, BATCH_SIZE):
                raise AssertionError('cached training changed sample order')
            del cached_frames
            torch.cuda.empty_cache()
        else:
            update = train_epoch(model, optimizer, train_videos, device, epoch,
                                 num_workers=args.data_workers)
        train_seconds = time.perf_counter()-train_start
        if update['steps'] != EXPECTED_TRAIN_FRAMES // TRAIN_FRAME_STRIDE // BATCH_SIZE:
            raise AssertionError('optimizer step count differs from fixed budget')
        steps += update['steps']
        validation_start = time.perf_counter()
        _, validation_summary, score = validation(
            model, val_videos, device, args.data_workers)
        validation_seconds = time.perf_counter()-validation_start
        lr_used = optimizer.param_groups[0]['lr']
        scheduler.step()
        record = {'epoch': epoch, 'cumulative_steps': steps,
                  'train': update, 'validation': validation_summary,
                  'lr_used': lr_used, 'gates': gates(model),
                  'train_wall_seconds': train_seconds,
                  'validation_wall_seconds': validation_seconds,
                  'gpu_cache': cache_stats}
        history.append(record)
        if frozen_digest(model) != locked:
            raise AssertionError(f'{arm} {stage}: frozen parameters/BN changed')
        if score < best:
            best, best_epoch = score, epoch
            checkpoint(result_path, args, model, optimizer, scheduler,
                       arm, stage, epoch, steps, best, history)
        save_json(directory / f'history_{stage}.json', history)
        cache_text = (f', cache={cache_stats["load_seconds"]:.1f}s'
                      if cache_stats else '')
        print(f'{arm} {stage} {epoch}/{EPOCHS[stage]}: '
              f'val final={score:.4f}, best={best:.4f}@{best_epoch}, '
              f'coarse={validation_summary["all"]["stage_epe_512px"]["coarse16"]:.4f}, '
              f'gates={gates(model)}, steps={steps}, '
              f'train={train_seconds:.1f}s{cache_text}, '
              f'val={validation_seconds:.1f}s',
              flush=True)
    selected = torch.load(result_path, map_location='cpu', weights_only=False)
    model.load_state_dict(selected['model_state_dict'], strict=True)
    image_rows = image_evaluation(model, val_videos, device,
                                  args.data_workers)
    image_summary = metric_summary(image_rows)
    save_csv(directory / f'per_frame_best_{stage}.csv', image_rows)
    report = {'recipe': 'vtmot_dns_position_v1', 'arm': arm,
              'order': ORDERS[arm], 'stage': stage,
              'initialization': 'original GLU-Net pretrained + RoadScene no-DNS '
                                'SA/CA attention/decoder4; fresh zero-gated DNS',
              'parent_best_coarse_sha256': parent_sha,
              'best_checkpoint_sha256': sha256_file(result_path),
              'selected_epoch': best_epoch, 'selected_steps': selected['optimizer_steps'],
              'completed_epochs': EPOCHS[stage], 'completed_steps': steps,
              'validation_selection': 'four-sequence valid-pixel-weighted final 512px EPE',
              'epoch_zero': epoch_zero, 'best_validation_final_epe_512px': best,
              'best_gates': gates(model),
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'trainable_count': sum(p.numel() for p in active),
              'best_image_metrics': image_summary,
              'inference_time_scope': 'model preprocessing and forward only; '
                                      'CUDA synchronized; excludes disk IO, '
                                      'GT loading, warp and metric computation',
              'loss': 'L_final + 0.1 L_32 + 0.2 L_64, masked Charbonnier',
              'max_epochs': EPOCHS[stage], 'batch_size': BATCH_SIZE,
              'seed': SEED, 'train_frame_stride': TRAIN_FRAME_STRIDE,
              'data_workers': args.data_workers,
              'gpu_epoch_cache': args.gpu_epoch_cache,
              'optimizer': 'AdamW', 'learning_rate': LR[stage],
              'scheduler': 'CosineAnnealingLR',
              'training_stability': {
                  'all_epoch_losses_finite': all(np.isfinite(
                      record['train']['train_charbonnier_512px']['loss'])
                      for record in history),
                  'maximum_preclip_gradient': max(
                      record['train']['preclip_gradient_max'] for record in history),
                  'epoch_zero_final_epe_512px': epoch_zero['all']
                      ['stage_epe_512px']['final512'],
                  'gate_trajectory': [record['gates'] for record in history]},
              'test_accessed': False}
    if stage == 'coarse':
        with frozen_inference(model):
            corr_rows, corr_summary = correlation_diagnostics(
                model, val_videos, device, args.data_workers)
        save_csv(directory / 'coarse_correlation_per_frame.csv', corr_rows)
        report['correlation'] = corr_summary
    save_json(directory / f'report_{stage}.json', report)
    return report


def final_report(args):
    reports = {}
    frames = {}
    for arm in ORDERS:
        directory = args.output / arm
        reports[arm] = {stage: json.loads((directory/f'report_{stage}.json')
                                          .read_text(encoding='utf-8'))
                        for stage in STAGES}
        with (directory/'per_frame_best_fine.csv').open(
                newline='', encoding='utf-8-sig') as stream:
            frames[arm] = list(csv.DictReader(stream))
        if (len(frames[arm]) != EXPECTED_VALID_FRAMES or
                reports[arm]['fine']['parent_best_coarse_sha256'] !=
                reports[arm]['coarse']['best_checkpoint_sha256']):
            raise AssertionError(f'{arm}: missing frames or wrong fine parent')
    paired = []
    for triplet in zip(*(frames[arm] for arm in ORDERS)):
        first = triplet[0]
        if any((r['sequence'], r['stem'], r['valid_pixels']) !=
               (first['sequence'], first['stem'], first['valid_pixels'])
               for r in triplet[1:]):
            raise AssertionError('arms have different images or GT masks')
        paired.append({'sequence': first['sequence'], 'stem': first['stem'],
                       'valid_pixels': first['valid_pixels'],
                       **{f'{arm}_final_epe_512px': row['final512_epe_512px']
                          for arm, row in zip(ORDERS, triplet)},
                       **{f'{arm}_coarse_epe_512px': row['coarse16_epe_512px']
                          for arm, row in zip(ORDERS, triplet)}})
    save_csv(args.output/'paired_per_frame.csv', paired)
    final_epe = {arm: reports[arm]['fine']['best_image_metrics']['all']
                 ['final512']['epe_512px'] for arm in ORDERS}
    reference = final_epe['S']
    ranking = sorted(final_epe, key=final_epe.get)
    # Fixed before inspecting any new result. Near ties need a matched second
    # seed, and this program does not turn the first run into a final choice.
    replication_margin_512px = .1
    near_tie = (final_epe[ranking[1]]-final_epe[ranking[0]] <
                replication_margin_512px)
    difficult = sorted(paired, key=lambda row: -float(row['S_final_epe_512px']))[:20]
    difficult_rows = [{**row, **{
        f'{arm}_minus_S_final_epe_512px':
            float(row[f'{arm}_final_epe_512px'])-float(row['S_final_epe_512px'])
        for arm in ('A', 'B', 'C')}} for row in difficult]
    improvement_counts = {arm: sum(
        float(row[f'{arm}_final_epe_512px']) < float(row['S_final_epe_512px'])
        for row in paired) for arm in ('A', 'B', 'C')}
    result = {'recipe': 'vtmot_dns_position_v1',
              'criterion': 'completed own coarse and fine training; '
                           'lowest full-validation final flow EPE',
              'final_epe_512px': final_epe,
              'dns_gain_vs_matched_sa_ca_512px': {
                  arm: reference-final_epe[arm] for arm in ('A', 'B', 'C')},
              'improved_frames_vs_matched_sa_ca': improvement_counts,
              'difficult_frames_defined_by_matched_sa_ca': difficult_rows,
              'provisional_ranking': ranking,
              'decision': {'primary_metric': 'valid-pixel-weighted final 512px EPE',
                           'replication_margin_512px': replication_margin_512px,
                           'needs_matched_second_seed_before_fixing_order': near_tie,
                           'selected_order': None if near_tie else ranking[0],
                           'note': 'coarse/correlation/image similarity metrics '
                                   'are diagnostics, not selection scores'},
              'arms': reports,
              'metric_definitions': {
                  'nmi_64bin_2mi_entropy_sum': 'RGB->gray 64 bins (uint8//4); '
                      '2*MI/(H(VI)+H(warped IR)); range 0..1',
                  'lncc17_squared_proxy': '17x17 local normalized cross '
                      'correlation squared, GT-valid-pixel mean',
                  'edge_information_retention_proxy': 'Sobel magnitude sum '
                      'min(VI,warped IR)/sum(VI), over GT-valid pixels',
                  'edge_overlap_f1_3x3_proxy': 'per-image valid-region top '
                      'quartile Sobel edges, symmetric F1 with 3x3 dilation '
                      '(1-pixel tolerance)',
                  'Q_F_AB': '未计算：无融合输出',
                  'paper_comparability': 'edge metrics are project proxies, '
                      'not paper R_edge; none of these values is claimed '
                      'comparable to a paper table'},
              'test_accessed': False}
    save_json(args.output/'comparison.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True,
                        choices=('check', 'cache-check', 'coarse', 'fine',
                                 'all', 'report'))
    parser.add_argument('--arm', choices=tuple(ORDERS))
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--split-file', type=Path)
    parser.add_argument('--pretrained', type=Path)
    parser.add_argument('--attention-checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--full-forward-check', action='store_true')
    parser.add_argument('--data-workers', type=int, default=0,
                        help='Parallel CPU workers for deterministic frame loading; '
                             '0 preserves the original loader')
    parser.add_argument('--gpu-epoch-cache', action='store_true',
                        help='Stage one training stride phase on CUDA each epoch')
    parser.add_argument('--gpu-cache-reserve-gib', type=float, default=4.0,
                        help='Free VRAM required after the epoch cache is allocated')
    args = parser.parse_args()
    if args.data_workers < 0:
        parser.error('--data-workers must be nonnegative')
    if args.gpu_cache_reserve_gib <= 0:
        parser.error('--gpu-cache-reserve-gib must be positive')
    math_check = check_order_math()
    if args.stage == 'check' and not all((args.pretrained, args.attention_checkpoint)):
        print(json.dumps({'order_math': math_check, 'test_accessed': False}, indent=2))
        return
    for key in ('pretrained', 'attention_checkpoint', 'split_file', 'output'):
        if getattr(args, key) is None:
            parser.error(f'--{key.replace("_", "-")} is required')
    if args.stage == 'check':
        print(json.dumps({'order_math': math_check,
                          'initialization': check_initialization(args),
                          'full_forward': (check_full_forward(args)
                                           if args.full_forward_check else None),
                          'test_accessed': False}, indent=2))
        return
    if args.stage == 'report':
        result = final_report(args)
        print(json.dumps({'final_epe_512px': result['final_epe_512px'],
                          'test_accessed': False}, indent=2))
        return
    if args.stage in STAGES and args.arm is None:
        parser.error('--arm is required for training')
    if args.data_root is None:
        parser.error('--data-root is required for training/cache-check')
    seed_all()
    install_mutual_guard()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_videos = VTMOTVideos(args.data_root, args.split_file, 'train')
    val_videos = VTMOTVideos(args.data_root, args.split_file, 'eval')
    if (len(train_videos.sequences) != 38 or
            sum(map(len, train_videos.sequences.values())) != EXPECTED_TRAIN_FRAMES or
            tuple(val_videos.sequences) != SPLIT_NAMES or
            sum(map(len, val_videos.sequences.values())) != EXPECTED_VALID_FRAMES):
        raise ValueError('unexpected VTMOT train/validation manifest')
    if args.stage == 'cache-check':
        print(json.dumps(check_gpu_epoch_cache(args, train_videos, device),
                         indent=2), flush=True)
        return
    if args.stage == 'all':
        existing = [args.output / arm / f'best_{stage}.pth'
                    for arm in ORDERS for stage in STAGES
                    if (args.output / arm / f'best_{stage}.pth').exists()]
        if existing:
            raise FileExistsError(
                'all-stage run needs a fresh output directory; existing '
                f'checkpoint: {existing[0]}')
        if args.gpu_epoch_cache:
            print('=== Checking CUDA epoch cache before S ===', flush=True)
            cache_check = check_gpu_epoch_cache(args, train_videos, device)
            save_json(args.output / 'gpu_cache_check.json', cache_check)
            print('CUDA epoch cache exact-value check passed', flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        for arm in ORDERS:
            print(f'=== {arm}: coarse then fine ===', flush=True)
            train_stage(args, arm, 'coarse', train_videos, val_videos, device)
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            train_stage(args, arm, 'fine', train_videos, val_videos, device)
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        result = final_report(args)
        print(json.dumps({'final_epe_512px': result['final_epe_512px'],
                          'test_accessed': False}, indent=2), flush=True)
        return
    result = train_stage(args, args.arm, args.stage, train_videos, val_videos, device)
    print(json.dumps({'arm': args.arm, 'stage': args.stage,
                      'selected_epoch': result['selected_epoch'],
                      'best_val_final_epe_512px': result['best_validation_final_epe_512px'],
                      'test_accessed': False}, indent=2), flush=True)


if __name__ == '__main__':
    main()
