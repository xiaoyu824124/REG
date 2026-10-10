"""Fine-only VTMOT adaptation from the selected coarse-adapted, no-DCN A1.

Only the original GLU-Net local decoder is trainable. The coarse encoder,
DNS, SA/CA, global correlation, flow upsamplers and every BN buffer stay fixed.
Train on VTMOT train; select on the four locked validation sequences only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from roadscene_coarse import sha256_file
from roadscene_fusion_hierarchical import objective
from roadscene_no_dcn import LOCAL_NAMES, UP_NAMES, check_no_dcn
from vtmot_a1_adapt import (BATCH_SIZE, Frames, TRAIN_FRAME_STRIDE,
                            evaluate_roadscene, evaluate_vtmot, frozen_digest,
                            install_mutual_guard, load_a1, save_csv, save_json,
                            seed_all, train_epoch)
from datasets.vtmot_video import VTMOTVideos


MAX_EPOCHS = 30
PATIENCE = 6
MIN_DELTA = .02
# Fixed by the RoadScene A1 development recipe before this VTMOT experiment.
LEARNING_RATE = 5e-6


def source_report(args, validation):
    report_path = args.coarse_adapt_checkpoint.with_name('report_val.json')
    frames_path = args.coarse_adapt_checkpoint.with_name('vtmot_val_per_frame.csv')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if (report.get('test_accessed') is not False or
            report.get('validation_frames') != 800 or
            report.get('selected_epoch') is None or
            set(report['per_sequence']) != set(validation.sequences) or
            report['weights']['best_adapted_sha256'] !=
            sha256_file(args.coarse_adapt_checkpoint)):
        raise ValueError('coarse adaptation report does not match locked validation')
    if not frames_path.is_file():
        raise FileNotFoundError(frames_path)
    return report, frames_path


def load_start(args, device):
    model = load_a1(args, device)
    payload = torch.load(args.coarse_adapt_checkpoint, map_location='cpu',
                         weights_only=False)
    expected = {'recipe': 'vtmot_a1_coarse_only_v1', 'stage': 'coarse',
                'source_a1_sha256': sha256_file(args.a1_checkpoint),
                'coarse_sha256': sha256_file(args.coarse_checkpoint),
                'pretrained_sha256': sha256_file(args.pretrained),
                'split_sha256': sha256_file(args.split_file),
                'test_accessed': False}
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f'coarse-adapted checkpoint {key} mismatch')
    model.load_state_dict(payload['model_state_dict'], strict=True)
    model.base.train_coarse_encoder = False
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    check_no_dcn(model.base)
    if model.fusion is not None:
        raise RuntimeError('A1 must have no additional local fusion module')
    return model, payload


def enable_local_only(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.base.train_coarse_encoder = False
    active = [p for name in LOCAL_NAMES
              for p in getattr(model.base, name).parameters()]
    if len({id(p) for p in active}) != len(active) or not active:
        raise RuntimeError('duplicate or missing local decoder parameters')
    for parameter in active:
        parameter.requires_grad_(True)
    if any(p.requires_grad for name in UP_NAMES
           for p in getattr(model.base, name).parameters()):
        raise RuntimeError('pretrained flow upsampler must stay frozen')
    model.train()  # FineArm.train() keeps the entire base in eval mode.
    if model.base.training or any(module.training for module in model.base.modules()
                                  if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)):
        raise RuntimeError('coarse and local BN statistics must stay frozen')
    if sum(p.requires_grad for p in model.parameters()) != len(active):
        raise RuntimeError('unexpected trainable parameters outside local decoder')
    return active


def save_checkpoint(path, args, model, optimizer, scheduler, epoch, steps,
                    score, history):
    payload = {'recipe': 'vtmot_a1_fine_only_v1', 'stage': 'fine',
               'source_adapted_sha256': sha256_file(args.coarse_adapt_checkpoint),
               'source_a1_sha256': sha256_file(args.a1_checkpoint),
               'coarse_sha256': sha256_file(args.coarse_checkpoint),
               'pretrained_sha256': sha256_file(args.pretrained),
               'split_sha256': sha256_file(args.split_file),
               'model_state_dict': model.state_dict(),
               'optimizer_state_dict': optimizer.state_dict(),
               'scheduler_state_dict': scheduler.state_dict(),
               'epoch': epoch, 'optimizer_steps': steps,
               'val_final_epe_512px': score, 'history': history,
               'torch_rng_state': torch.get_rng_state(),
               'numpy_rng_state': np.random.get_state(),
               'python_rng_state': random.getstate(),
               'cuda_rng_state': (torch.cuda.get_rng_state()
                                  if torch.cuda.is_available() else None),
               'test_accessed': False}
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def first_reference(frames_path, sequence, stem):
    with frames_path.open(newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            if row['sequence'] == sequence and row['stem'] == stem:
                return row
    raise ValueError(f'coarse-adapted validation frame missing: {sequence}/{stem}')


def code_check(args, model, train_videos, val_videos, frames_path, device):
    train_batch = next(iter(DataLoader(Frames(train_videos), batch_size=1)))
    val_name = next(iter(val_videos.sequences))
    val_stem = val_videos.sequences[val_name][0]
    with torch.inference_mode():
        before = model(train_batch, device)
    active = enable_local_only(model)
    frozen_before = frozen_digest(model)
    with torch.inference_mode():
        enabled = model(train_batch, device)
    initial_diff = float((before['final512']-enabled['final512']).abs().max())
    if initial_diff > 1e-5:
        raise RuntimeError(f'enabling local gradients changed initial flow: {initial_diff}')
    rows, _ = evaluate_vtmot(model, val_videos, device, frame_limit=1)
    reference = first_reference(frames_path, val_name, val_stem)
    first_diff = abs(rows[0]['final512_epe_512px'] -
                     float(reference['final512_epe_512px']))
    if first_diff > .01:
        raise RuntimeError(f'initial validation frame mismatch: {first_diff}')
    optimizer = torch.optim.AdamW(active, lr=LEARNING_RATE)
    optimizer.zero_grad(set_to_none=True)
    predicted = model(train_batch, device)
    loss, parts, _ = objective(predicted, train_batch, device, hierarchical=True)
    if not torch.isfinite(loss):
        raise RuntimeError('nonfinite fine-only code-check loss')
    loss.backward()
    preclip = torch.nn.utils.clip_grad_norm_(active, 1.)
    if not torch.isfinite(preclip) or preclip <= 0:
        raise RuntimeError('missing or nonfinite local gradient')
    optimizer.step()
    if frozen_digest(model) != frozen_before:
        raise RuntimeError('coarse, upsampler, or BN changed during local update')
    with torch.inference_mode():
        after = model(train_batch, device)
    coarse_diff = float((before['coarse16']-after['coarse16']).abs().max())
    if coarse_diff > 1e-5:
        raise RuntimeError(f'coarse prediction changed after local update: {coarse_diff}')
    save_json(args.output/'code_check.json', {
        'passed': True, 'initial_flow_max_abs_difference': initial_diff,
        'locked_first_frame_epe_difference_512px': first_diff,
        'coarse_flow_difference_after_step': coarse_diff,
        'loss': float(loss), 'loss_parts': {k: float(v) for k, v in parts.items()},
        'preclip_local_gradient_norm': float(preclip),
        'trainable_parameters': sum(p.numel() for p in active),
        'frozen_state_unchanged': True, 'test_accessed': False})


def paired_rows(frames_path, fine_rows):
    with frames_path.open(newline='', encoding='utf-8-sig') as stream:
        baseline = {(row['sequence'], row['stem']): row for row in csv.DictReader(stream)}
    if len(baseline) != 800 or len(fine_rows) != 800:
        raise RuntimeError('paired validation requires all 800 frames')
    rows = []
    for row in fine_rows:
        old = baseline[(row['sequence'], row['stem'])]
        if row['valid_pixels'] != int(old['valid_pixels']):
            raise RuntimeError('fine adaptation changed a validation GT mask')
        rows.append({'sequence': row['sequence'], 'stem': row['stem'],
                     'valid_pixels': row['valid_pixels'],
                     'coarse_adapted_epe_512px': float(old['final512_epe_512px']),
                     'fine_adapted_epe_512px': row['final512_epe_512px'],
                     'fine_minus_coarse_512px': (row['final512_epe_512px']-
                                                 float(old['final512_epe_512px']))})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('check', 'train'), required=True)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--split-file', required=True, type=Path)
    parser.add_argument('--pretrained', required=True, type=Path)
    parser.add_argument('--coarse-checkpoint', required=True, type=Path)
    parser.add_argument('--a1-checkpoint', required=True, type=Path,
                        help='original RoadScene best_A1.pth for lineage verification')
    parser.add_argument('--coarse-adapt-checkpoint', required=True, type=Path)
    parser.add_argument('--roadscene-root', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be a new, empty directory')
    if args.stage == 'train' and args.roadscene_root is None:
        parser.error('--roadscene-root is required for retention evaluation')
    args.output.mkdir(parents=True, exist_ok=True)
    seed_all()
    install_mutual_guard()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_videos = VTMOTVideos(args.data_root, args.split_file, 'train',
                               max_sequences=1 if args.stage == 'check' else 0,
                               max_frames=1 if args.stage == 'check' else 0)
    val_videos = VTMOTVideos(args.data_root, args.split_file, 'eval',
                             max_sequences=1 if args.stage == 'check' else 0,
                             max_frames=1 if args.stage == 'check' else 0)
    report, frames_path = source_report(args, VTMOTVideos(
        args.data_root, args.split_file, 'eval'))
    model, source_payload = load_start(args, device)
    if abs(source_payload['val_final_epe_512px']-
           report['adapted']['all']['stage_epe_512px']['final512']) > 1e-6:
        raise RuntimeError('coarse checkpoint and reported validation score differ')
    if args.stage == 'check':
        code_check(args, model, train_videos, val_videos, frames_path, device)
        print('fine-only code check passed; coarse/upscaler/BN frozen', flush=True)
        return
    active = enable_local_only(model)
    frozen_before = frozen_digest(model)
    optimizer = torch.optim.AdamW(active, lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=5e-7)
    _, initial = evaluate_vtmot(model, val_videos, device)
    baseline_epe = initial['all']['stage_epe_512px']['final512']
    if abs(baseline_epe-source_payload['val_final_epe_512px']) > .01:
        raise RuntimeError(f'epoch-0 validation mismatch: {baseline_epe} vs '
                           f'{source_payload["val_final_epe_512px"]}')
    best, bad, steps, history = baseline_epe, 0, 0, []
    best_path = args.output/'best_fine_adapted_a1.pth'
    save_checkpoint(best_path, args, model, optimizer, scheduler, 0, steps,
                    best, history)
    print(f'fine epoch 0: val final={baseline_epe:.4f}', flush=True)
    for epoch in range(1, MAX_EPOCHS+1):
        update = train_epoch(model, optimizer, train_videos, device, epoch)
        steps += update['steps']
        _, validation = evaluate_vtmot(model, val_videos, device)
        score = validation['all']['stage_epe_512px']['final512']
        scheduler.step()
        history.append({'epoch': epoch, 'cumulative_steps': steps,
                        'train': update, 'validation': validation,
                        'lr': optimizer.param_groups[0]['lr']})
        previous_best = best
        if score < best:
            best = score
            save_checkpoint(best_path, args, model, optimizer, scheduler,
                            epoch, steps, best, history)
        bad = 0 if score < previous_best-MIN_DELTA else bad+1
        if frozen_digest(model) != frozen_before:
            raise RuntimeError('frozen coarse, upsampler, or BN state changed')
        save_json(args.output/'history.json', history)
        print(f'fine adapt {epoch}/{MAX_EPOCHS}: val final={score:.4f}, '
              f'best={best:.4f}, patience={bad}/{PATIENCE}, steps={steps}',
              flush=True)
        if bad >= PATIENCE:
            break
    payload = torch.load(best_path, map_location='cpu', weights_only=False)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    model.eval()
    fine_rows, fine_summary = evaluate_vtmot(model, val_videos, device)
    save_csv(args.output/'vtmot_val_per_frame.csv', fine_rows)
    paired = paired_rows(frames_path, fine_rows)
    save_csv(args.output/'vtmot_val_paired.csv', paired)
    roadscene = evaluate_roadscene(args, model, device)
    per_sequence = {
        name: {'coarse_adapted': report['per_sequence'][name]['adapted_a1'],
               'fine_adapted': fine_summary['per_sequence'][name]
                              ['stage_epe_512px']['final512']}
        for name in val_videos.sequences}
    for value in per_sequence.values():
        value['change'] = value['fine_adapted']-value['coarse_adapted']
    result = {'recipe': 'VTMOT A1 fine-only adaptation',
              'source_adapted_sha256': sha256_file(args.coarse_adapt_checkpoint),
              'best_fine_sha256': sha256_file(best_path),
              'source_a1_sha256': sha256_file(args.a1_checkpoint),
              'split_sha256': sha256_file(args.split_file),
              'selected_epoch': payload['epoch'],
              'selected_steps': payload['optimizer_steps'],
              'stopped_at_epoch': history[-1]['epoch'] if history else 0,
              'selection': 'four-sequence valid-pixel-weighted final 512px EPE',
              'training': {'max_epochs': MAX_EPOCHS, 'patience': PATIENCE,
                           'min_delta_512px': MIN_DELTA, 'learning_rate': LEARNING_RATE,
                           'batch_size': BATCH_SIZE, 'train_frame_stride': TRAIN_FRAME_STRIDE,
                           'loss': 'L_final + 0.1 L_32 + 0.2 L_64, Charbonnier',
                           'trainable_modules': list(LOCAL_NAMES),
                           'flow_upsamplers_trainable': False},
              'coarse_adapted': report['adapted'], 'fine_adapted': fine_summary,
              'per_sequence': per_sequence,
              'improved_frames': sum(r['fine_minus_coarse_512px'] < 0 for r in paired),
              'roadscene_val': roadscene, 'test_accessed': False}
    save_json(args.output/'report_val.json', result)
    print(json.dumps({'selected_epoch': payload['epoch'],
                      'coarse_adapted_final_epe_512px': baseline_epe,
                      'fine_adapted_final_epe_512px': fine_summary['all']
                      ['stage_epe_512px']['final512'],
                      'improved_frames': result['improved_frames'],
                      'per_sequence': per_sequence,
                      'roadscene_val': roadscene}, indent=2), flush=True)


if __name__ == '__main__':
    main()
