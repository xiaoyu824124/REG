"""Single-frame VTMOT coarse adaptation of frozen no-DCN A1 local decoder.

The stage choice follows the read-only 800-frame diagnosis. Train on the 38
VTMOT train sequences only; select by full four-sequence validation EPE.
No keyframe logic, fusion, test sequences, DCN, or architectural change.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from datasets.roadscene import RoadScenePairs
from datasets.vtmot_video import VTMOTVideos
from roadscene_coarse import sha256_file
from roadscene_fusion_hierarchical import augment, objective
from roadscene_no_dcn import (COARSE_NAMES, LOCAL_NAMES, UP_NAMES,
                              check_no_dcn, evaluate_fine, summarize)
from vtmot_keyframe import load_a1, save_json
import models.our_models.GLUNet as glunet


SEED = 2026
MAX_EPOCHS = 30
PATIENCE = 6
MIN_DELTA = .02
TRAIN_FRAME_STRIDE = 5  # Rotate phases 0..4; all 7600 frames seen per five epochs.
LEARNING_RATE = 1e-5
BATCH_SIZE = 2


class Frames(Dataset):
    def __init__(self, videos, phase=None):
        self.videos = videos
        self.items = [(name, stem)
                      for name, stems in videos.sequences.items()
                      for stem in (stems[phase::TRAIN_FRAME_STRIDE]
                                   if phase is not None else stems)]
        if not self.items:
            raise ValueError('empty frame dataset')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        name, stem = self.items[index]
        frame = self.videos.input_frame(name, stem)
        flow, valid = self.videos.flow_truth(name, stem, frame['original_hw'])
        # Invalid GT coordinates are masked and zeroed for stable losses.
        flow = np.where(valid[..., None], flow, 0).astype(np.float32)
        return {'source_image': torch.from_numpy(frame['ir'].transpose(2, 0, 1).copy()),
                'target_image': torch.from_numpy(frame['vi'].transpose(2, 0, 1).copy()),
                'flow_map': torch.from_numpy(flow.transpose(2, 0, 1).copy()),
                'correspondence_mask': torch.from_numpy(valid.copy()),
                'name': f'{name}/{stem}'}


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def freeze_except_coarse(model):
    check_no_dcn(model.base)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.base.train_coarse_encoder = True
    active_modules = [model.base.pyramid._modules['level_4']]
    active_modules += [getattr(model.base, name) for name in COARSE_NAMES]
    parameters = [p for module in active_modules for p in module.parameters()]
    if len({id(p) for p in parameters}) != len(parameters):
        raise RuntimeError('duplicate coarse optimizer parameter')
    for parameter in parameters:
        parameter.requires_grad_(True)
    if any(p.requires_grad for name in LOCAL_NAMES + UP_NAMES
           for p in getattr(model.base, name).parameters()):
        raise RuntimeError('local decoder or pretrained upsampler is trainable')
    model.train()
    if model.base.training or any(module.training for module in model.base.modules()
                                  if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)):
        raise RuntimeError('coarse/local BatchNorm statistics must be frozen')
    return parameters


def frozen_digest(model):
    digest = hashlib.sha256()
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        if isinstance(value, torch.nn.Parameter) and value.requires_grad:
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def loader(dataset, epoch):
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                      generator=torch.Generator().manual_seed(SEED + epoch),
                      num_workers=0, pin_memory=torch.cuda.is_available())


def train_epoch(model, optimizer, train_videos, device, epoch, check=False):
    model.train()
    phase = (epoch - 1) % TRAIN_FRAME_STRIDE
    frames = Frames(train_videos, phase=None if check else phase)
    data = loader(frames, epoch)
    totals = {'loss': 0., 'local32': 0., 'local64': 0., 'final512': 0.}
    count, steps, grad_max = 0, 0, 0.
    order = hashlib.sha256()
    for batch in data:
        batch = augment(batch, epoch, enabled=True)
        for name in batch['name']:
            order.update((name + '\n').encode())
        optimizer.zero_grad(set_to_none=True)
        predicted = model(batch, device)
        loss, parts, valid_pixels = objective(predicted, batch, device,
                                               hierarchical=True)
        if not torch.isfinite(loss):
            raise RuntimeError(f'nonfinite train loss at {batch["name"]}')
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group['params']], 1.)
        if not torch.isfinite(gradient):
            raise RuntimeError(f'nonfinite coarse gradient at {batch["name"]}')
        grad_max = max(grad_max, float(gradient))
        optimizer.step()
        for key, value in [('loss', loss), *parts.items()]:
            totals[key] += float(value.detach()) * valid_pixels
        count += valid_pixels
        steps += 1
    return {'frames': len(frames), 'steps': steps, 'valid_pixels': count,
            'train_charbonnier_512px': {key: value/count for key, value in totals.items()},
            'preclip_gradient_max': grad_max,
            'image_order_sha256': order.hexdigest(),
            'frame_stride_phase': phase}


@contextmanager
def frozen_inference(model):
    """Evaluate with the same requires_grad=False path as deployed A1."""
    parameters = list(model.parameters())
    flags = [parameter.requires_grad for parameter in parameters]
    coarse_graph = model.base.train_coarse_encoder
    training = model.training
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        model.base.train_coarse_encoder = False
        model.eval()
        with torch.inference_mode():
            yield
    finally:
        for parameter, flag in zip(parameters, flags):
            parameter.requires_grad_(flag)
        model.base.train_coarse_encoder = coarse_graph
        model.train(training)


def evaluate_vtmot(model, videos, device, frame_limit=0):
    rows = []
    with frozen_inference(model):
        for batch in DataLoader(Frames(videos), batch_size=1, shuffle=False, num_workers=0):
            prediction = model(batch, device)
            truth = batch['flow_map'].to(device).float()
            valid = batch['correspondence_mask'].to(device).bool()
            number = int(valid.sum())
            row = {'sequence': batch['name'][0].split('/')[0],
                   'stem': batch['name'][0].split('/')[1], 'valid_pixels': number}
            for stage in ('coarse16', 'local32', 'local64', 'final512'):
                error = torch.linalg.vector_norm(prediction[stage]-truth, dim=1)
                row[f'{stage}_epe_512px'] = float(error[valid].mean())
            row['zero_epe_512px'] = float(torch.linalg.vector_norm(truth, dim=1)[valid].mean())
            rows.append(row)
            if frame_limit and len(rows) >= frame_limit:
                break
    return rows, summarize_vtmot(rows)


def summarize_vtmot(rows):
    def summary(chosen):
        count = sum(row['valid_pixels'] for row in chosen)
        return {'frames': len(chosen), 'valid_pixels': count,
                'stage_epe_512px': {
                    stage: sum(row[f'{stage}_epe_512px']*row['valid_pixels']
                               for row in chosen)/count
                    for stage in ('zero', 'coarse16', 'local32', 'local64', 'final512')}}
    names = sorted({row['sequence'] for row in rows})
    return {'all': summary(rows),
            'per_sequence': {name: summary([r for r in rows if r['sequence'] == name])
                             for name in names}}


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def diagnosis_frames_path(report_path):
    for name in ('diagnosis_per_frame.csv', 'per_frame.csv'):
        candidate = report_path.with_name(name)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'per-frame diagnosis next to {report_path}')


def checkpoint(path, model, optimizer, scheduler, args, epoch, steps, score, history):
    payload = {'recipe': 'vtmot_a1_coarse_only_v1', 'stage': 'coarse',
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
               'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
               'test_accessed': False}
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def install_mutual_guard():
    original = glunet.MutualMatching
    glunet.MutualMatching = lambda correlation: original(correlation.clamp_min(0))
    check = glunet.MutualMatching(torch.zeros(1, 1, 16, 16, 16, 16))
    if not torch.isfinite(check).all() or torch.count_nonzero(check):
        raise RuntimeError('all-zero MutualMatching normalization is unsafe')


def check_one_update(args, model, train_videos, val_videos, device):
    trainset = Frames(train_videos)
    batch = next(iter(DataLoader(trainset, batch_size=1)))
    model.eval()
    with torch.inference_mode():
        before = model(batch, device)
        repeat = model(batch, device)
    repeat_difference = float((before['final512']-repeat['final512']).abs().max())
    model.base.train_coarse_encoder = True
    with torch.inference_mode():
        flag_only = model(batch, device)
    flag_difference = float((before['final512']-flag_only['final512']).abs().max())
    model.base.train_coarse_encoder = False
    model.train()
    with torch.inference_mode():
        mode_only = model(batch, device)
    mode_difference = float((before['final512']-mode_only['final512']).abs().max())
    model.eval()
    active = freeze_except_coarse(model)
    frozen_before = frozen_digest(model)
    with torch.inference_mode():
        after_enable = model(batch, device)
    initial_difference = float((before['final512']-after_enable['final512']).abs().max())
    initial_mean_difference = float((before['final512']-after_enable['final512']).abs().mean())
    if initial_difference > .1 or initial_mean_difference > .02:
        raise RuntimeError(f'enabling coarse training altered A1 initialization: '
                           f'max_abs={initial_difference}, mean_abs={initial_mean_difference}, '
                           f'repeat={repeat_difference}, flag={flag_difference}, '
                           f'mode={mode_difference}')
    _, deployment_check = evaluate_vtmot(model, val_videos, device, frame_limit=1)
    with diagnosis_frames_path(args.diagnosis_report).open(
            newline='', encoding='utf-8-sig') as stream:
        locked_first = next(csv.DictReader(stream))
    deployment_epe = deployment_check['all']['stage_epe_512px']['final512']
    locked_epe = float(locked_first['final512_epe_512px'])
    if abs(deployment_epe-locked_epe) > .01:
        raise RuntimeError(f'epoch-0 deployment path differs from locked A1: '
                           f'{deployment_epe} vs {locked_epe}')
    optimizer = torch.optim.AdamW(active, lr=LEARNING_RATE)
    loss, parts, _ = objective(model(batch, device), batch, device, hierarchical=True)
    loss.backward()
    preclip = torch.nn.utils.clip_grad_norm_(active, 1.)
    if not torch.isfinite(preclip) or preclip <= 0:
        raise RuntimeError('missing or nonfinite coarse gradient')
    optimizer.step()
    if frozen_digest(model) != frozen_before:
        raise RuntimeError('frozen local layer, upsampler, or BN changed')
    loop_check = train_epoch(model, optimizer, train_videos, device, epoch=1,
                             check=True)
    if frozen_digest(model) != frozen_before:
        raise RuntimeError('training loop changed frozen local layer, upsampler, or BN')
    _, validation = evaluate_vtmot(model, val_videos, device, frame_limit=1)
    save_json(args.output/'code_check.json', {
        'passed': True, 'train_frames_checked': 1, 'val_frames_checked': 1,
        'initialization_max_abs_difference': initial_difference,
        'initialization_mean_abs_difference': initial_mean_difference,
        'baseline_repeat_max_abs_difference': repeat_difference,
        'coarse_graph_flag_max_abs_difference': flag_difference,
        'train_mode_max_abs_difference': mode_difference,
        'epoch0_deployment_epe_512px': deployment_epe,
        'locked_first_frame_epe_512px': locked_epe,
        'loss': float(loss),
        'parts': {k: float(v) for k, v in parts.items()},
        'preclip_gradient_norm': float(preclip), 'validation': validation,
        'training_loop_check': loop_check,
        'frozen_state_unchanged': True, 'test_accessed': False})


def verify_diagnosis(args, val_videos):
    report = json.loads(args.diagnosis_report.read_text(encoding='utf-8'))
    if (report['frames'] != 800 or report['split'] != 'eval' or
            report['a1_sha256'] != sha256_file(args.a1_checkpoint) or
            set(report['per_sequence']) != set(val_videos.sequences)):
        raise ValueError('diagnosis must be the locked 800-frame A1 evaluation')
    return report


def evaluate_roadscene(args, model, device):
    dataset = RoadScenePairs(args.roadscene_root, 'val')
    rows, summary = [], {}
    for label, candidate in (('original', load_a1(args, device)), ('adapted', model)):
        with frozen_inference(candidate):
            these = evaluate_fine(candidate, dataset, device, timed=True)
        rows.extend([{'model': label, **r} for r in these])
        summary[label] = summarize(these)
    save_csv(args.output/'roadscene_val_per_pair.csv', rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('check', 'train'), required=True)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--split-file', required=True, type=Path)
    parser.add_argument('--pretrained', required=True, type=Path)
    parser.add_argument('--coarse-checkpoint', required=True, type=Path)
    parser.add_argument('--a1-checkpoint', required=True, type=Path)
    parser.add_argument('--diagnosis-report', required=True, type=Path)
    parser.add_argument('--roadscene-root', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be a new, empty directory')
    if args.stage == 'train' and not args.roadscene_root:
        parser.error('--roadscene-root is required for RoadScene retention evaluation')
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
    original_report = verify_diagnosis(args, VTMOTVideos(
        args.data_root, args.split_file, 'eval'))
    model = load_a1(args, device)
    if args.stage == 'check':
        check_one_update(args, model, train_videos, val_videos, device)
        print('code check passed; local/upscaler/BN frozen', flush=True)
        return
    parameters = freeze_except_coarse(model)
    frozen_before = frozen_digest(model)
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)
    original_epe = original_report['all']['stages']['final512']['epe_512px']
    _, initial_validation = evaluate_vtmot(model, val_videos, device)
    baseline_epe = initial_validation['all']['stage_epe_512px']['final512']
    if abs(baseline_epe-original_epe) > .01:
        raise RuntimeError(f'epoch-0 A1 validation differs from locked result: '
                           f'{baseline_epe} vs {original_epe}')
    best, bad, steps, history = baseline_epe, 0, 0, []
    checkpoint(args.output/'best_adapted_a1.pth', model, optimizer, scheduler,
               args, 0, steps, best, history)
    for epoch in range(1, MAX_EPOCHS+1):
        update = train_epoch(model, optimizer, train_videos, device, epoch)
        steps += update['steps']
        rows, validation = evaluate_vtmot(model, val_videos, device)
        score = validation['all']['stage_epe_512px']['final512']
        scheduler.step()
        record = {'epoch': epoch, 'cumulative_steps': steps, 'train': update,
                  'validation': validation, 'lr': optimizer.param_groups[0]['lr']}
        history.append(record)
        old_best = best
        if score < best:
            best = score
            checkpoint(args.output/'best_adapted_a1.pth', model, optimizer,
                       scheduler, args, epoch, steps, best, history)
        bad = 0 if score < old_best-MIN_DELTA else bad+1
        if frozen_digest(model) != frozen_before:
            raise RuntimeError('frozen local/upscaler/BN state changed during adaptation')
        save_json(args.output/'history.json', history)
        print(f'coarse adapt {epoch}/{MAX_EPOCHS}: val final={score:.4f}, '
              f'best={best:.4f}, patience={bad}/{PATIENCE}, steps={steps}', flush=True)
        if bad >= PATIENCE:
            break
    payload = torch.load(args.output/'best_adapted_a1.pth',
                         map_location='cpu', weights_only=False)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    model.eval()
    adapted_rows, adapted = evaluate_vtmot(model, val_videos, device)
    save_csv(args.output/'vtmot_val_per_frame.csv', adapted_rows)
    with diagnosis_frames_path(args.diagnosis_report).open(
            newline='', encoding='utf-8-sig') as stream:
        locked_frames = {(r['sequence'], r['stem']): r for r in csv.DictReader(stream)}
    paired = []
    for row in adapted_rows:
        locked = locked_frames[(row['sequence'], row['stem'])]
        if row['valid_pixels'] != int(locked['valid_pixels']):
            raise RuntimeError('adapted validation mask differs from locked A1 mask')
        old = float(locked['final512_epe_512px'])
        new = row['final512_epe_512px']
        paired.append({'sequence': row['sequence'], 'stem': row['stem'],
                       'valid_pixels': row['valid_pixels'],
                       'zero_epe_512px': row['zero_epe_512px'],
                       'original_a1_epe_512px': old,
                       'adapted_a1_epe_512px': new,
                       'adapted_minus_original_512px': new-old})
    if len(paired) != 800 or len(locked_frames) != 800:
        raise RuntimeError('paired VTMOT validation must contain the locked 800 frames')
    save_csv(args.output/'vtmot_val_paired.csv', paired)
    roadscene = evaluate_roadscene(args, model, device)
    per_sequence = {}
    for name in val_videos.sequences:
        old = original_report['per_sequence'][name]['stages']
        new = adapted['per_sequence'][name]['stage_epe_512px']
        per_sequence[name] = {'zero': old['zero']['epe_512px'],
                              'original_a1': old['final512']['epe_512px'],
                              'adapted_a1': new['final512'],
                              'change_vs_original': new['final512']-old['final512']['epe_512px']}
    zero = original_report['all']['stages']['zero']['epe_512px']
    allowed = (adapted['all']['stage_epe_512px']['final512'] < zero and
               sum(x['change_vs_original'] <= -.1 for x in per_sequence.values()) >= 2)
    report = {'recipe': 'VTMOT A1 single-frame coarse-only adaptation',
              'weights': {'original_a1_sha256': sha256_file(args.a1_checkpoint),
                          'best_adapted_sha256': sha256_file(
                              args.output/'best_adapted_a1.pth'),
                          'coarse_sha256': sha256_file(args.coarse_checkpoint),
                          'pretrained_sha256': sha256_file(args.pretrained)},
              'train_sequences': len(train_videos.sequences),
              'train_frames_total': sum(map(len, train_videos.sequences.values())),
              'validation_sequences': len(val_videos.sequences),
              'validation_frames': len(adapted_rows),
              'selected_epoch': payload['epoch'], 'selected_steps': payload['optimizer_steps'],
              'selection': 'full four-sequence valid-pixel-weighted final 512px EPE',
              'training': {'max_epochs': MAX_EPOCHS, 'patience': PATIENCE,
                           'min_delta_512px': MIN_DELTA, 'batch_size': BATCH_SIZE,
                           'train_frame_stride': TRAIN_FRAME_STRIDE,
                           'stride_phase': '(epoch-1) mod 5',
                           'learning_rate': LEARNING_RATE,
                           'loss': 'L_final + 0.1 L_32 + 0.2 L_64, Charbonnier'},
              'original_a1': original_report['all']['stages']['final512'],
              'epoch0_a1_epe_512px': baseline_epe,
              'zero': original_report['all']['stages']['zero'],
              'adapted': adapted, 'per_sequence': per_sequence,
              'difficult_frames': {
                  'worst_adapted': sorted(paired, key=lambda r: r['adapted_a1_epe_512px'],
                                           reverse=True)[:10],
                  'largest_regressions': sorted(
                      paired, key=lambda r: r['adapted_minus_original_512px'],
                      reverse=True)[:10],
                  'largest_improvements': sorted(
                      paired, key=lambda r: r['adapted_minus_original_512px'])[:10]},
              'roadscene_val': roadscene,
              'allow_keyframe_followup': allowed,
              'keyframe_followup_rule': 'all-800 adapted final EPE below zero flow; '
                                        'at least two of four sequences improve >=0.1px vs A1',
              'transition_note': '0/122 accepted: 未实际测试到',
              'test_accessed': False}
    save_json(args.output/'report_val.json', report)
    print(json.dumps({'selected_epoch': payload['epoch'],
                      'zero_epe': zero, 'original_epe': baseline_epe,
                      'adapted_epe': adapted['all']['stage_epe_512px']['final512'],
                      'per_sequence': per_sequence, 'roadscene_val': roadscene,
                      'allow_keyframe_followup': allowed}, indent=2), flush=True)


if __name__ == '__main__':
    main()
