"""Matched no-DCN 2x2 RoadScene study: decoder fusion x hierarchical loss.

A0/A1 keep GLU-Net's local decoder. B0/B1 inject a zero-initialized
128/256 local signal *before* decoder1. All arms use the same coarse weights.
Only train and val are ever opened. Run --stage check locally, --stage all
on the training host. Checkpoints and reports never overwrite no_dcn_v1.
"""

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from datasets.roadscene import RoadScenePairs
from models.our_models.local_decoder_fusion import LocalDecoderFusion
from roadscene_no_dcn import (SEED, LOCAL_NAMES, UP_NAMES, base_model,
                              check_no_dcn, evaluate_fine, summarize, weighted,
                              write_csv, state_digest, state_snapshot_dict,
                              seed_all)
from roadscene_refinement_audit import pooled_truth
from roadscene_coarse import sha256_file


ARMS = ('A0', 'A1', 'B0', 'B1')
MAX_EPOCHS, PATIENCE, MIN_DELTA = 80, 12, .01
DEV_SIZE, DEV_EPOCHS = 24, 3
LR_CANDIDATES = (1e-6, 2e-6, 5e-6)
NEW_MODULE_LR = 1e-5
BASE_LOCAL_LR_FOR_DIAGNOSIS = 1e-5
WEIGHTS = {'local32': .1, 'local64': .2, 'final512': 1.}


class FineArm(nn.Module):
    def __init__(self, base, use_fusion):
        super().__init__()
        self.base = base
        self.fusion = LocalDecoderFusion() if use_fusion else None
        check_no_dcn(base)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()  # The frozen VGG and all BN running statistics stay fixed.
        return self

    def forward(self, batch, device):
        source, target, source256, target256, *_ = self.base.pre_process_data(
            batch['source_image'].to(device), batch['target_image'].to(device),
            device=device)
        capture_dcn = getattr(self, 'dcn_supervision', False)
        forward = self.base(target, source, target256, source256,
                            local_fusion=self.fusion,
                            return_dcn_trace=capture_dcn)
        flow256, flow512 = forward[:2]
        result = {'coarse16': F.interpolate(flow256[0], (512, 512),
                                          mode='bilinear', align_corners=False) * 2,
                'local32': F.interpolate(flow256[1], (512, 512),
                                         mode='bilinear', align_corners=False) * 2,
                'local64': F.interpolate(flow512[0], (512, 512),
                                         mode='bilinear', align_corners=False),
                'base128': F.interpolate(flow512[1], (512, 512),
                                         mode='bilinear', align_corners=False),
                'final512': F.interpolate(flow512[1], (512, 512),
                                          mode='bilinear', align_corners=False)}
        if capture_dcn:
            result['dcn_trace'] = forward[2]
        return result


def make_arm(args, arm, device):
    seed_all()
    base = base_model(args.pretrained, device, args.coarse_checkpoint)
    model = FineArm(base, arm.startswith('B')).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    local = [p for name in LOCAL_NAMES
             for p in getattr(model.base, name).parameters()]
    for p in local:
        p.requires_grad_(True)
    if model.fusion is not None:
        for p in model.fusion.parameters():
            p.requires_grad_(True)
    if any(p.requires_grad for name in UP_NAMES
           for p in getattr(base, name).parameters()):
        raise RuntimeError('Pretrained flow upsamplers must remain frozen')
    model.train()
    return model, local


def local_digest(model):
    selected = {name: value for name, value in model.base.state_dict().items()
                if any(name.startswith(prefix + '.') for prefix in LOCAL_NAMES)}
    return state_digest(state_snapshot_dict(selected))


def frozen_digest(model):
    selected = {name: value for name, value in model.base.named_parameters()
                if not any(name.startswith(prefix + '.') for prefix in LOCAL_NAMES)}
    selected.update({f'buffer:{name}': value
                     for name, value in model.base.named_buffers()})
    return state_digest(state_snapshot_dict(selected))


def mask_loss(prediction, truth, valid):
    return torch.sqrt((prediction - truth).square().sum(1) + .01)[valid].mean()


def objective(output, batch, device, hierarchical):
    truth = batch['flow_map'].to(device).float()
    valid = batch['correspondence_mask'].to(device).bool()
    if not valid.any():
        raise ValueError('Batch has no valid GT pixels')
    parts = {name: mask_loss(output[name], truth, valid)
             for name in WEIGHTS}
    loss = parts['final512']
    if hierarchical:
        loss = loss + WEIGHTS['local32'] * parts['local32']
        loss = loss + WEIGHTS['local64'] * parts['local64']
    return loss, parts, int(valid.sum())


def augment(batch, epoch, enabled):
    if not enabled:
        return batch
    # One deterministic geometry per pair; independent of arm and RNG history.
    result = dict(batch)
    for axis, direction in ((-1, 0), (-2, 1)):
        choices = [int(hashlib.sha256(
            f'{SEED}:{epoch}:{name}:{axis}'.encode()).hexdigest()[:2], 16) % 2
                   for name in batch['name']]
        if not any(choices):
            continue
        indices = torch.tensor(choices, dtype=torch.bool)
        for name in ('source_image', 'target_image', 'flow_map',
                     'correspondence_mask'):
            value = result[name].clone()
            value[indices] = torch.flip(value[indices], dims=(axis,))
            result[name] = value
        result['flow_map'][indices, direction] *= -1
    return result


def loader_for(dataset, epoch, code_check=False):
    return DataLoader(dataset, batch_size=1 if code_check else 2, shuffle=True,
                      num_workers=0,
                      generator=torch.Generator().manual_seed(SEED + epoch))


def train_epoch(model, optimizer, dataset, device, epoch, augmentation):
    model.train()
    seen = 0
    totals = {'loss': 0., **{name: 0. for name in WEIGHTS}}
    grad_norms = []
    order = hashlib.sha256()
    for raw in loader_for(dataset, epoch):
        batch = augment(raw, epoch, augmentation)
        for name in batch['name']:
            order.update((name + '\n').encode())
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch, device)
        loss, parts, valid_count = objective(
            prediction, batch, device, model.hierarchical)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite training loss, epoch={epoch}, pairs={batch["name"]}')
        loss.backward()
        all_parameters = [p for group in optimizer.param_groups
                          for p in group['params']]
        grad = torch.nn.utils.clip_grad_norm_(all_parameters, 1.)
        if not torch.isfinite(grad):
            raise RuntimeError(f'Nonfinite preclip gradient in {batch["name"]}')
        grad_norms.append(float(grad))
        optimizer.step()
        seen += valid_count
        totals['loss'] += float(loss.detach()) * valid_count
        for name, value in parts.items():
            totals[name] += float(value.detach()) * valid_count
    return {'steps': len(grad_norms), 'train_loss': totals['loss'] / seen,
            'train_stage_charbonnier_512px': {name: totals[name] / seen
                                              for name in WEIGHTS},
            'preclip_grad_mean': float(np.mean(grad_norms)),
            'preclip_grad_max': max(grad_norms),
            'image_order_sha256': order.hexdigest()}


def evaluate(model, dataset, device, timed=False):
    rows = evaluate_fine(model, dataset, device, timed=timed)
    return rows, summarize(rows)


def local_delta(model, initial):
    current = {name: value for name, value in model.base.named_parameters()
               if any(name.startswith(prefix + '.') for prefix in LOCAL_NAMES)}
    numerator = sum((value.detach().cpu() - initial[name]).square().sum().item()
                    for name, value in current.items())
    denominator = sum(value.square().sum().item() for value in initial.values())
    return (numerator / max(denominator, 1e-30)) ** .5


def diagnose(args, trainset, valset, device):
    model, local = make_arm(args, 'A0', device)
    model.hierarchical = False
    start = {name: value.detach().cpu().clone()
             for name, value in model.base.named_parameters()
             if any(name.startswith(prefix + '.') for prefix in LOCAL_NAMES)}
    frozen = frozen_digest(model)
    optimizer = torch.optim.AdamW(local, lr=BASE_LOCAL_LR_FOR_DIAGNOSIS)
    records = []
    epochs = 1 if args.code_check else 3
    for epoch in range(epochs + 1):
        train_rows, train_summary = evaluate(model, trainset, device)
        val_rows, val_summary = evaluate(model, valset, device)
        record = {'epoch': epoch, 'train': train_summary['stage_epe_512px'],
                  'val': val_summary['stage_epe_512px'],
                  'learning_rate': optimizer.param_groups[0]['lr'],
                  'local_parameter_relative_change': local_delta(model, start)}
        if epoch:
            record['update'] = update
        records.append(record)
        print(f'diagnose {epoch}/{epochs}: train={train_summary["stage_epe_512px"]["final512"]:.4f}, '
              f'val={val_summary["stage_epe_512px"]["final512"]:.4f}', flush=True)
        if epoch < epochs:
            update = train_epoch(model, optimizer, trainset, device, epoch + 1,
                                 augmentation=False)
    if frozen_digest(model) != frozen:
        raise RuntimeError('Diagnostic changed frozen coarse parameters or BN buffers')
    save_json(args.output / 'diagnosis_old_local.json',
              {'train_pairs': len(trainset), 'val_pairs': len(valset),
               'stage_flow_unit': '512x512 image pixels', 'records': records,
               'causal_claim': 'none: diagnostic alone cannot isolate LR/augmentation'})


def development(args, trainset, device):
    if args.code_check:
        indices = list(range(len(trainset)))
        fit = Subset(trainset, indices)
        dev = Subset(trainset, indices)
        rates = (LR_CANDIDATES[0],)
    else:
        rng = np.random.default_rng(SEED)
        permutation = rng.permutation(len(trainset)).tolist()
        fit = Subset(trainset, sorted(permutation[DEV_SIZE:]))
        dev = Subset(trainset, sorted(permutation[:DEV_SIZE]))
        rates = LR_CANDIDATES
    trials = []
    for lr in rates:
        model, local = make_arm(args, 'A0', device)
        model.hierarchical = False
        optimizer = torch.optim.AdamW(local, lr=lr)
        frozen = frozen_digest(model)
        for epoch in range(1, 2 if args.code_check else DEV_EPOCHS + 1):
            update = train_epoch(model, optimizer, fit, device, epoch,
                                 augmentation=True)
            _, summary = evaluate(model, dev, device)
            trials.append({'old_local_lr': lr, 'epoch': epoch,
                           'development_epe_512px':
                           summary['stage_epe_512px']['final512'],
                           'steps': update['steps']})
        if frozen_digest(model) != frozen:
            raise RuntimeError('Development stage changed frozen network')
    selected = min(trials, key=lambda x: (x['development_epe_512px'],
                                           x['old_local_lr']))
    recipe = {'old_local_lr': selected['old_local_lr'],
              'new_fusion_lr': NEW_MODULE_LR,
              'augmentation': 'deterministic paired horizontal and vertical flips; flip flow sign and mask',
              'development_selection': selected,
              'dev_pair_ids': [
                  (trainset.dataset.samples[trainset.indices[i]][2].stem
                   if isinstance(trainset, Subset) else trainset.samples[i][2].stem)
                  for i in dev.indices],
              'learning_rate_trials': trials,
              'hierarchical_weights': WEIGHTS,
              'max_epochs': MAX_EPOCHS, 'common_patience': PATIENCE,
              'min_delta_512px': MIN_DELTA,
              'locked_before_23_pair_model_selection': True}
    save_json(args.output / 'locked_recipe.json', recipe)
    return recipe


def save_json(path, payload):
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def checkpoint(args, arm, model, optimizer, scheduler, epoch, steps, score,
               recipe, history):
    payload = {'recipe': 'fusion_hierarchical_2x2', 'arm': arm,
               'coarse_sha256': sha256_file(args.coarse_checkpoint),
               'pretrained_sha256': sha256_file(args.pretrained),
               'model_state_dict': model.state_dict(),
               'optimizer_state_dict': optimizer.state_dict(),
               'scheduler_state_dict': scheduler.state_dict(),
               'epoch': epoch, 'optimizer_steps': steps,
               'val_final_epe_512px': score, 'locked_recipe': recipe,
               'history': history,
               'torch_rng_state': torch.get_rng_state(),
               'numpy_rng_state': np.random.get_state(),
               'python_rng_state': random.getstate(),
               'cuda_rng_state': torch.cuda.get_rng_state()
                   if torch.cuda.is_available() else None}
    temporary = args.output / f'best_{arm}.tmp'
    torch.save(payload, temporary)
    temporary.replace(args.output / f'best_{arm}.pth')


def coverage(args, valset, device):
    # The 32-grid entry is the *actual* deconv4 centre, not bilinear coarse flow.
    from roadscene_large_motion_dns import coarse_and_window
    coarse = base_model(args.pretrained, device, args.coarse_checkpoint)
    first = coarse_and_window(coarse, DataLoader(valset, batch_size=1), device)
    class Capture:
        def __init__(self):
            self.flow128 = None
        def __call__(self, target128, source128, target256, source256,
                     flow128, decoder_channels):
            self.flow128 = flow128.detach()
            return target128.new_zeros(target128.shape[0], decoder_channels,
                                        128, 128)
    rows = []
    with torch.no_grad():
        for batch in DataLoader(valset, batch_size=1):
            source, target, source256, target256, *_ = coarse.pre_process_data(
                batch['source_image'].to(device),
                batch['target_image'].to(device), device=device)
            capture = Capture()
            coarse(target, source, target256, source256, local_fusion=capture)
            truth = batch['flow_map'].to(device).float()
            valid = batch['correspondence_mask'].to(device).bool()
            row = {'name': batch['name'][0],
                   'first32_valid_queries': first[batch['name'][0]]
                       ['first_window_valid_queries'],
                   'first32_outside_queries': first[batch['name'][0]]
                       ['first_window_outside_queries']}
            for size, radius in ((128, 2), (256, 1)):
                prediction = F.interpolate(capture.flow128, (size, size),
                                           mode='bilinear', align_corners=False)
                gt, selected = pooled_truth(truth, valid, size)
                inside = ((prediction - gt).abs().amax(dim=1)
                          <= radius * 512 / size) & selected
                row[f'grid{size}_valid_queries'] = int(selected.sum())
                row[f'grid{size}_inside_queries'] = int(inside.sum())
            rows.append(row)
    save_json(args.output / 'coverage_val.json', {
        'definition': 'pooled GT mask >0.8; first32 actual deconv4, radius4; '
                      '128 radius2; 256 radius1; VI target to IR source',
        'per_image': rows,
        'summary': {label: sum(r[label] for r in rows)
                    for label in rows[0] if label != 'name'},
        'inside_fraction': {
            str(size): sum(r[f'grid{size}_inside_queries'] for r in rows) /
                       max(1, sum(r[f'grid{size}_valid_queries'] for r in rows))
            for size in (128, 256)},
        'first32_inside_fraction': 1 - sum(r['first32_outside_queries'] for r in rows) /
                                   max(1, sum(r['first32_valid_queries'] for r in rows)),
        'difficult': [r for r in rows if r['name'] in ('000002', '000014')]})


def train_four(args, trainset, valset, device, recipe):
    models, optimizers, schedulers, histories = {}, {}, {}, {}
    frozen, initial_local = {}, {}
    for arm in ARMS:
        model, local = make_arm(args, arm, device)
        model.hierarchical = arm.endswith('1')
        models[arm] = model
        initial_local[arm] = local_digest(model)
        frozen[arm] = frozen_digest(model)
        groups = [{'params': local, 'lr': recipe['old_local_lr'],
                   'name': 'old_local'}]
        if model.fusion is not None:
            groups.append({'params': model.fusion.parameters(),
                           'lr': recipe['new_fusion_lr'], 'name': 'fusion'})
        optimizers[arm] = torch.optim.AdamW(groups)
        schedulers[arm] = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizers[arm], T_max=MAX_EPOCHS, eta_min=1e-7)
        histories[arm] = []
    if len(set(initial_local.values())) != 1 or len(set(frozen.values())) != 1:
        raise RuntimeError('Four arms have unequal original-local/coarse initialization')
    one = next(iter(DataLoader(valset, batch_size=1)))
    with torch.no_grad():
        a = models['A0'](one, device)
        initialization = {'A0': {level: 0. for level in
                                  ('coarse16', 'local32', 'local64', 'final512')}}
        for arm in ('A1', 'B0', 'B1'):
            b = models[arm](one, device)
            differences = {level: float((a[level] - b[level]).abs().max())
                           for level in ('coarse16', 'local32', 'local64', 'final512')}
            if any(value > 1e-5 for value in differences.values()):
                raise RuntimeError(f'{arm} initialization mismatch: {differences}; '
                                   f'local_digest={local_digest(models[arm])} '
                                   f'vs {local_digest(models["A0"])}')
            initialization[arm] = differences
    if any(torch.count_nonzero(models[arm].fusion.to_decoder_input.weight) or
           torch.count_nonzero(models[arm].fusion.to_decoder_input.bias)
           for arm in ('B0', 'B1')):
        raise RuntimeError('Fusion injection must be exactly zero at initialization')
    save_json(args.output / 'initialization_check.json', initialization)
    best = {arm: float('inf') for arm in ARMS}
    bad = {arm: 0 for arm in ARMS}
    steps = {arm: 0 for arm in ARMS}
    for arm in ARMS:
        _, initial_summary = evaluate(models[arm], valset, device)
        best[arm] = initial_summary['stage_epe_512px']['final512']
        checkpoint(args, arm, models[arm], optimizers[arm], schedulers[arm],
                   0, 0, best[arm], recipe, histories[arm])
    for epoch in range(1, 2 if args.code_check else MAX_EPOCHS + 1):
        for arm in ARMS:
            model, optimizer = models[arm], optimizers[arm]
            train_record = train_epoch(model, optimizer, trainset, device,
                                       epoch, augmentation=True)
            steps[arm] += train_record['steps']
            val_rows, val_summary = evaluate(model, valset, device)
            score = val_summary['stage_epe_512px']['final512']
            schedulers[arm].step()
            record = {'epoch': epoch, 'optimizer_steps': steps[arm],
                      **train_record, 'val': val_summary,
                      'lr': {g['name']: g['lr'] for g in optimizer.param_groups}}
            histories[arm].append(record)
            old = best[arm]
            if score < best[arm]:
                best[arm] = score
                checkpoint(args, arm, model, optimizer, schedulers[arm],
                           epoch, steps[arm], score, recipe, histories[arm])
            bad[arm] = 0 if score < old - MIN_DELTA else bad[arm] + 1
            print(f'{arm} epoch {epoch}: val final={score:.4f}, '
                  f'best={best[arm]:.4f}, patience={bad[arm]}/{PATIENCE}',
                  flush=True)
        if len({histories[a][-1]['image_order_sha256'] for a in ARMS}) != 1:
            raise RuntimeError('Arms saw different batch orders')
        for arm in ARMS:
            if frozen_digest(models[arm]) != frozen[arm]:
                raise RuntimeError(f'{arm} changed frozen coarse/BN parameters')
            save_json(args.output / f'history_{arm}.json', histories[arm])
        if not args.code_check and all(value >= PATIENCE for value in bad.values()):
            break  # Synchronized stop: all four arms ran the same update count.
    if len(set(steps.values())) != 1:
        raise RuntimeError('Four arms ran unequal optimizer steps')
    return {'epochs_executed': epoch, 'steps_each': next(iter(steps.values())),
            'best_val_final_epe_512px': best}


def report(args, valset, device, training):
    rows_by_arm, summaries, metadata = {}, {}, {}
    for arm in ARMS:
        model, _ = make_arm(args, arm, device)
        payload = torch.load(args.output / f'best_{arm}.pth', map_location='cpu',
                             weights_only=False)
        if payload['coarse_sha256'] != sha256_file(args.coarse_checkpoint):
            raise ValueError('Mismatched fixed coarse checkpoint')
        model.load_state_dict(payload['model_state_dict'], strict=True)
        model.eval()
        rows, summary = evaluate(model, valset, device, timed=True)
        rows_by_arm[arm], summaries[arm] = rows, summary
        write_csv(args.output / f'per_image_{arm}.csv', rows)
        metadata[arm] = {'selected_epoch': payload['epoch'],
                         'selected_optimizer_steps': payload['optimizer_steps'],
                         'parameters': sum(p.numel() for p in model.parameters()),
                         'summary23': summary}
    paired = []
    maps = {arm: {row['name']: row for row in rows_by_arm[arm]} for arm in ARMS}
    if any(set(maps[arm]) != set(maps['A0']) for arm in ARMS):
        raise RuntimeError('Evaluation pairs differ')
    for name in sorted(maps['A0']):
        paired.append({'name': name, 'valid_pixels': maps['A0'][name]['valid_pixels'],
                       **{f'{arm}_final_epe_512px': maps[arm][name]['final512_epe_512px']
                          for arm in ARMS},
                       'B0_minus_A0': maps['B0'][name]['final512_epe_512px'] -
                           maps['A0'][name]['final512_epe_512px'],
                       'A1_minus_A0': maps['A1'][name]['final512_epe_512px'] -
                           maps['A0'][name]['final512_epe_512px'],
                       'B1_minus_B0': maps['B1'][name]['final512_epe_512px'] -
                           maps['B0'][name]['final512_epe_512px']})
    write_csv(args.output / 'per_image_paired.csv', paired)
    comparisons = {}
    for label, left, right in (('structure', 'B0', 'A0'),
                               ('hierarchical_original', 'A1', 'A0'),
                               ('hierarchical_fusion', 'B1', 'B0')):
        comparisons[label] = {
            'weighted_final_epe_change_512px':
                summaries[left]['stage_epe_512px']['final512'] -
                summaries[right]['stage_epe_512px']['final512'],
            'improved_pairs': sum(maps[left][name]['final512_epe_512px'] <
                                  maps[right][name]['final512_epe_512px']
                                  for name in maps[left])}
    result = {'protocol': {'direction': 'VI target to IR source',
                           'flow_unit': '512x512 image pixels',
                           'selection': 'full 23-pair valid-pixel-weighted final EPE',
                           'test_pairs_accessed': False,
                           'coarse_checkpoint_sha256': sha256_file(args.coarse_checkpoint),
                           'same_train_order_and_steps': True},
              'training': training, 'arms': metadata, 'comparisons': comparisons,
              'difficult': {arm: [r for r in rows_by_arm[arm]
                                  if r['name'] in ('000002', '000014')]
                            for arm in ARMS},
              'diagnostic21': {arm: summarize([r for r in rows_by_arm[arm]
                                             if r['name'] not in ('000002', '000014')])
                               for arm in ARMS}}
    save_json(args.output / 'report_val.json', result)
    print(json.dumps({'arms': metadata, 'comparisons': comparisons}, indent=2),
          flush=True)


def code_check(args, trainset, valset, device):
    models = {arm: make_arm(args, arm, device)[0] for arm in ARMS}
    batch = next(iter(DataLoader(valset, batch_size=1)))
    with torch.no_grad():
        outputs = {arm: model(batch, device) for arm, model in models.items()}
        repeated = models['A0'](batch, device)
    repeat_diffs = {level: float((repeated[level] - outputs['A0'][level]).abs().max())
                    for level in ('coarse16', 'local32', 'local64', 'final512')}
    print(f'baseline repeat max differences: {repeat_diffs}', flush=True)
    for arm in ARMS[1:]:
        arm_diffs = {}
        for level in ('coarse16', 'local32', 'local64', 'final512'):
            difference = float((outputs[arm][level] - outputs['A0'][level]).abs().max())
            arm_diffs[level] = difference
            if difference > max(1e-5, repeat_diffs[level] * 2, .01):
                raise RuntimeError(f'Initialization mismatch: {arm} {level}, '
                                   f'max_abs={difference}, '
                                   f'local_digest={local_digest(models[arm])} '
                                   f'vs {local_digest(models["A0"])}')
        print(f'{arm} initialization max differences: {arm_diffs}', flush=True)
    for arm, model in models.items():
        model.hierarchical = arm.endswith('1')
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=1e-6)
        training_batch = next(iter(DataLoader(trainset, batch_size=1)))
        training_batch = augment(training_batch, 1, True)
        prediction = model(training_batch, device)
        loss, _, _ = objective(prediction, training_batch, device,
                               model.hierarchical)
        loss.backward()
        if model.fusion is not None:
            head = model.fusion.to_decoder_input
            if (head.weight.grad is None or
                    not torch.isfinite(head.weight.grad).all() or
                    head.weight.grad.abs().sum() == 0):
                raise RuntimeError(f'{arm} fusion has no valid gradient')
        optimizer.step()
    # Geometry: VI at x=40 samples IR at x=41 for +4px on 128 grid.
    from models.our_models.local_recurrent_128_256 import sample_window
    impulse = torch.zeros(1, 1, 128, 128, device=device)
    impulse[0, 0, 40, 41] = 1
    flow = torch.zeros(1, 2, 128, 128, device=device)
    flow[:, 0] = 4
    windows, mask = sample_window(impulse, flow, radius=2)
    if float(windows[0, 0, 12, 40, 40]) != 1 or not mask[0, 12, 40, 40]:
        raise RuntimeError('VI->IR window direction or 512px scale failed')
    # A flip also flips the flow component and its validity mask.
    synthetic = {'name': ['sample'], 'source_image': torch.zeros(1, 3, 4, 4),
                 'target_image': torch.zeros(1, 3, 4, 4),
                 'flow_map': torch.zeros(1, 2, 4, 4),
                 'correspondence_mask': torch.ones(1, 4, 4, dtype=torch.bool)}
    synthetic['target_image'][0, :, 1, 1] = 1
    synthetic['source_image'][0, :, 1, 2] = 1
    synthetic['flow_map'][:, 0] = 1  # Target x=1 samples source x=2.
    synthetic['correspondence_mask'][:, :, -1] = False
    changed = augment(synthetic, 1, True)
    target_xy = torch.nonzero(changed['target_image'][0, 0])[0].tolist()
    source_xy = torch.nonzero(changed['source_image'][0, 0])[0].tolist()
    y, x = target_xy
    mapped = [y + int(changed['flow_map'][0, 1, y, x]),
              x + int(changed['flow_map'][0, 0, y, x])]
    if mapped != source_xy or not changed['correspondence_mask'][0, y, x]:
        raise RuntimeError('Paired geometry augmentation failed')
    save_json(args.output / 'code_check.json',
              {'passed': True, 'initialization_exact': True,
               'gradient_finite': True, 'flow_direction': 'VI to IR',
               'flow_unit': '512px', 'test_pairs_accessed': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True,
                        choices=('check', 'smoke', 'diagnose', 'dev', 'train', 'all'))
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--pretrained', required=True, type=Path)
    parser.add_argument('--coarse-checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.code_check = args.stage in ('check', 'smoke')
    if not (args.data_root.is_dir() and args.pretrained.is_file() and
            args.coarse_checkpoint.is_file()):
        parser.error('RoadScene, pretrained weights, or best coarse weight missing')
    if args.output.exists() and any(args.output.iterdir()) and args.stage in ('all', 'check', 'smoke'):
        parser.error('Use a new empty output directory for all/check/smoke')
    args.output.mkdir(parents=True, exist_ok=True)
    seed_all()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    glunet = __import__('models.our_models.GLUNet', fromlist=['MutualMatching'])
    original = glunet.MutualMatching
    glunet.MutualMatching = lambda corr: original(corr.clamp_min(0))
    try:
        zero = glunet.MutualMatching(torch.zeros(1, 1, 16, 16, 16, 16))
        if not torch.isfinite(zero).all() or torch.count_nonzero(zero):
            raise RuntimeError('All-zero MutualMatching is unsafe')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trainset = RoadScenePairs(args.data_root, 'train')
        valset = RoadScenePairs(args.data_root, 'val')
        if args.stage == 'check':
            code_check(args, Subset(trainset, [0]), Subset(valset, [0]), device)
            return
        if args.stage == 'smoke':
            trainset, valset = Subset(trainset, [0]), Subset(valset, [0])
            code_check(args, trainset, valset, device)
            diagnose(args, trainset, valset, device)
            recipe = development(args, trainset, device)
            coverage(args, valset, device)
            training = train_four(args, trainset, valset, device, recipe)
            report(args, valset, device, training)
            return
        if args.stage in ('diagnose', 'all'):
            diagnose(args, trainset, valset, device)
        if args.stage in ('dev', 'all'):
            development(args, trainset, device)
        if args.stage in ('train', 'all'):
            recipe_path = args.output / 'locked_recipe.json'
            if not recipe_path.is_file():
                parser.error('Run --stage dev first to lock the recipe')
            if any((args.output / f'best_{arm}.pth').exists() for arm in ARMS):
                parser.error('Existing selected weights found; use a new output directory')
            recipe = json.loads(recipe_path.read_text(encoding='utf-8'))
            coverage(args, valset, device)
            training = train_four(args, trainset, valset, device, recipe)
            report(args, valset, device, training)
    finally:
        glunet.MutualMatching = original


if __name__ == '__main__':
    main()
