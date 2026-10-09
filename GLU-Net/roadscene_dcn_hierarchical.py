"""Matched two-scale DCN x loss ablation against fusion_hierarchical_2x2 A0/A1.

The frozen no_dcn_v1 coarse checkpoint and old local initialization are shared.
Only RoadScene train and val are opened; the test split is never constructed.
D0/D1/D2 add one true deformable-convolution update at each 32/64 grid. Their
training budget, loader order, old-local LR, scheduler and validation metric
match the completed A0/A1 run. The historical checkpoints are read-only.
"""

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets.roadscene import RoadScenePairs
from models.our_models.local_dcn import LocalDeformableFlowUpdate
from models.our_models.mod import warp
from roadscene_coarse import sha256_file
from roadscene_fusion_hierarchical import (
    MAX_EPOCHS, WEIGHTS, augment, evaluate, loader_for, local_digest,
    make_arm, mask_loss, objective, save_json, train_epoch,
)
from roadscene_no_dcn import LOCAL_NAMES, UP_NAMES, seed_all, state_digest, state_snapshot_dict, summarize, write_csv
from roadscene_refinement_audit import pooled_truth


ARMS = ('D0', 'D1', 'D2')
RESIDUAL_WEIGHTS = {32: .1, 64: .2}
OFFSET_WEIGHT = .001


def reference(args):
    root = args.reference_dir
    required = [root / 'report_val.json', root / 'locked_recipe.json',
                root / 'best_A0.pth', root / 'best_A1.pth',
                root / 'per_image_A0.csv', root / 'per_image_A1.csv',
                root / 'history_A0.json', root / 'history_A1.json']
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'Missing matched no-DCN reference: {missing}')
    report = json.loads(required[0].read_text(encoding='utf-8'))
    recipe = json.loads(required[1].read_text(encoding='utf-8'))
    protocol = report['protocol']
    if (protocol['coarse_checkpoint_sha256'] != sha256_file(args.coarse_checkpoint)
            or protocol['direction'] != 'VI target to IR source'
            or protocol['flow_unit'] != '512x512 image pixels'
            or protocol['test_pairs_accessed']
            or not protocol['same_train_order_and_steps']):
        raise ValueError('Reference protocol or coarse weight does not match')
    if (report['arms']['A0']['summary23']['pairs'] != 23 or
            report['arms']['A1']['summary23']['pairs'] != 23 or
            report['training']['steps_each'] % report['training']['epochs_executed']):
        raise ValueError('Reference validation size or budget is unexpected')
    if (recipe['hierarchical_weights'] != WEIGHTS or
            recipe['max_epochs'] != MAX_EPOCHS or
            recipe['locked_before_23_pair_model_selection'] is not True):
        raise ValueError('Reference loss weights or locked recipe changed')
    return report, recipe


def make_dcn_arm(args, arm, device):
    model, old_local = make_arm(args, 'A0', device)
    # Attach after strict loading of the no-DCN coarse checkpoint. This leaves
    # every original weight, including the pretrained flow upsamplers, intact.
    model.base.local_dcn32 = LocalDeformableFlowUpdate()
    model.base.local_dcn64 = LocalDeformableFlowUpdate()
    model.base.local_dcn_steps = 1
    model.to(device)
    model.hierarchical = arm == 'D1'
    model.dcn_supervision = False
    dcn = list(model.base.local_dcn32.parameters()) + list(model.base.local_dcn64.parameters())
    if any(p.requires_grad for name in UP_NAMES
           for p in getattr(model.base, name).parameters()):
        raise RuntimeError('Pretrained flow upsampler became trainable')
    if any(torch.count_nonzero(layer.weight) or torch.count_nonzero(layer.bias)
           for module in (model.base.local_dcn32, model.base.local_dcn64)
           for layer in (module.offset, module.delta)):
        raise RuntimeError('DCN offset and residual heads must start at zero')
    model.train()
    return model, old_local, dcn


def fixed_digest(model):
    active = tuple(LOCAL_NAMES) + ('local_dcn32', 'local_dcn64')
    selected = {name: value for name, value in model.base.named_parameters()
                if not any(name.startswith(prefix + '.') for prefix in active)}
    selected.update({f'buffer:{name}': value
                     for name, value in model.base.named_buffers()})
    return state_digest(state_snapshot_dict(selected))


def geometry_check(device):
    # One grid pixel is 8 pixels at either 32/256 or 64/512. The 32-grid
    # update is converted to 512-pixel units by the FineArm output adapter.
    for grid, image_size in ((32, 256), (64, 512)):
        module = LocalDeformableFlowUpdate().to(device).eval()
        correlation = torch.zeros(1, 81, grid, grid, device=device)
        feature = torch.zeros(1, 1, grid, grid, device=device)
        flow = torch.zeros(1, 2, grid, grid, device=device)
        with torch.no_grad():
            unchanged, info = module(correlation, feature, feature, flow,
                                     image_size, image_size)
            if torch.count_nonzero(unchanged) or torch.count_nonzero(info['offset_grid']):
                raise RuntimeError(f'{grid}-grid DCN is not identity at initialization')
            module.delta.bias[0] = float(np.arctanh(.25))
            moved, _ = module(correlation, feature, feature, flow,
                              image_size, image_size)
            if not torch.allclose(moved[:, 0], torch.full_like(moved[:, 0], 8.), atol=1e-5):
                raise RuntimeError(f'{grid}-grid DCN residual has the wrong scale')
        impulse = torch.zeros(1, 1, grid, grid, device=device)
        impulse[0, 0, 10, 11] = 1.
        grid_flow = torch.zeros(1, 2, grid, grid, device=device)
        grid_flow[:, 0] = 1.
        sampled = warp(impulse, grid_flow)
        if abs(float(sampled[0, 0, 10, 10]) - 1.) > 1e-5:
            raise RuntimeError(f'{grid}-grid source sampling direction is wrong')
        invalid_flow = torch.zeros_like(flow)
        invalid_flow[:, 0] = image_size
        with torch.no_grad():
            _, invalid = module(correlation, feature, feature, invalid_flow,
                                image_size, image_size)
        if invalid['source_inside'].any():
            raise RuntimeError(f'{grid}-grid out-of-bounds source mask failed')


def dcn_specific_loss(output, batch, device):
    """Supervise the DCN's incremental correction without moving its input flow.

    Ground truth and both predictions are expressed in 512-image pixels. The
    auxiliary term only covers valid, source-inside, reachable correspondences;
    the final-flow loss still covers every valid pixel. Detaching the input
    flow makes this auxiliary term train the DCN update rather than giving the
    old local decoder a second, indirect objective.
    """
    truth = batch['flow_map'].to(device).float()
    valid = batch['correspondence_mask'].to(device).bool()
    if not valid.any():
        raise ValueError('Batch has no valid GT pixels')
    final = mask_loss(output['final512'], truth, valid)
    trace_by_grid = {entry['grid']: entry for entry in output['dcn_trace']}
    if set(trace_by_grid) != {32, 64} or len(output['dcn_trace']) != 2:
        raise RuntimeError('Expected exactly one 32-grid and one 64-grid DCN update')
    parts = {'final512': final}
    auxiliary = final.new_zeros(())
    for grid in (32, 64):
        entry = trace_by_grid[grid]
        factor = 2 if grid == 32 else 1
        before = entry[f'flow_before_{256 if grid == 32 else 512}px'].detach() * factor
        after = entry[f'flow_{256 if grid == 32 else 512}px'] * factor
        target, valid_grid = pooled_truth(truth, valid, grid)
        correction = target - before
        maximum = 4 * (512 / grid)  # LocalDeformableFlowUpdate.max_delta_grid.
        reachable = correction.abs().amax(dim=1) <= maximum
        chosen = valid_grid & entry['source_inside'] & reachable
        predicted = after - before
        residual_error = torch.sqrt((predicted - correction).square().sum(1) + .01)
        # Preserve a connected zero for the rare batch with no reachable cells.
        residual_loss = residual_error[chosen].mean() if chosen.any() else predicted.sum() * 0
        offset_penalty = entry['offset_grid'].abs().mean()
        auxiliary = auxiliary + RESIDUAL_WEIGHTS[grid] * residual_loss
        auxiliary = auxiliary + OFFSET_WEIGHT * offset_penalty
        parts[f'residual{grid}'] = residual_loss
        parts[f'offset{grid}'] = offset_penalty
        parts[f'eligible{grid}_fraction'] = chosen.float().sum() / valid_grid.sum().clamp_min(1)
    return final, auxiliary, parts, int(valid.sum())


def train_dcn_epoch(model, optimizer, dataset, device, epoch):
    model.train()
    model.dcn_supervision = True
    totals = {}
    seen, grad_norms = 0, []
    order = hashlib.sha256()
    try:
        for raw in loader_for(dataset, epoch):
            batch = augment(raw, epoch, enabled=True)
            for name in batch['name']:
                order.update((name + '\n').encode())
            optimizer.zero_grad(set_to_none=True)
            output = model(batch, device)
            final, auxiliary, parts, count = dcn_specific_loss(output, batch, device)
            loss = final + auxiliary
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite DCN-specific loss at epoch {epoch}, pairs={batch["name"]}')
            # The auxiliary residual target is detached from the input flow.
            # Route its gradients only into the two DCN modules, while the
            # common final-flow loss still updates the old local decoder.
            final.backward(retain_graph=True)
            dcn_parameters = list(model.base.local_dcn32.parameters()) + list(
                model.base.local_dcn64.parameters())
            torch.autograd.backward(auxiliary, inputs=dcn_parameters)
            parameters = [p for group in optimizer.param_groups for p in group['params']]
            grad = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            if not torch.isfinite(grad):
                raise RuntimeError(f'Nonfinite DCN gradient at epoch {epoch}, pairs={batch["name"]}')
            optimizer.step()
            seen += count
            grad_norms.append(float(grad))
            totals['loss'] = totals.get('loss', 0.) + float(loss.detach()) * count
            for name, value in parts.items():
                totals[name] = totals.get(name, 0.) + float(value.detach()) * count
    finally:
        model.dcn_supervision = False
    return {'steps': len(grad_norms), 'train_loss': totals['loss'] / seen,
            'train_dcn_loss_components': {name: value / seen for name, value in totals.items()
                                          if name != 'loss'},
            'preclip_grad_mean': float(np.mean(grad_norms)),
            'preclip_grad_max': max(grad_norms),
            'image_order_sha256': order.hexdigest()}


def initialization_check(args, trainset, valset, device):
    baseline, _ = make_arm(args, 'A0', device)
    models = {arm: make_dcn_arm(args, arm, device)[0] for arm in ARMS}
    frozen = {arm: fixed_digest(model) for arm, model in models.items()}
    if len(set(frozen.values())) != 1 or any(local_digest(model) != local_digest(baseline)
                                             for model in models.values()):
        raise RuntimeError('Old local/coarse initialization differs between arms')
    batch = next(iter(DataLoader(valset, batch_size=1)))
    baseline.eval()
    for model in models.values():
        model.eval()
    with torch.no_grad():
        original = baseline(batch, device)
        initial = {arm: model(batch, device) for arm, model in models.items()}
    levels = ('coarse16', 'local32', 'local64', 'final512')
    differences = {arm: {level: float((output[level] - original[level]).abs().max())
                         for level in levels} for arm, output in initial.items()}
    if any(value > 1e-5 for levels_by_arm in differences.values()
           for value in levels_by_arm.values()):
        raise RuntimeError(f'Two-scale DCN changed initial flow: {differences}')
    geometry_check(device)
    # A single backward pass verifies that both zero-initialized residual
    # heads receive an actionable gradient. Offset gradients appear later,
    # once the residual output head has moved away from zero.
    training_batch = next(iter(DataLoader(trainset, batch_size=1)))
    gradients = {}
    old_gradient_difference = 0.
    for arm, model in models.items():
        model.train()
        model.zero_grad(set_to_none=True)
        model.dcn_supervision = arm == 'D2'
        output = model(training_batch, device)
        if arm == 'D2':
            final, auxiliary, _, _ = dcn_specific_loss(output, training_batch, device)
            final.backward(retain_graph=True)
            old_local_parameters = [
                parameter
                for name in LOCAL_NAMES
                for parameter in getattr(model.base, name).parameters()
            ]
            old_local_grads = [parameter.grad.detach().clone()
                               if parameter.grad is not None else None
                               for parameter in old_local_parameters]
            torch.autograd.backward(auxiliary, inputs=list(model.base.local_dcn32.parameters()) +
                                    list(model.base.local_dcn64.parameters()))
            for parameter, before in zip(old_local_parameters, old_local_grads):
                after = parameter.grad
                if (before is None) != (after is None):
                    raise RuntimeError('DCN auxiliary changed which old local gradients exist')
                if before is not None:
                    old_gradient_difference = max(
                        old_gradient_difference,
                        float((after - before).abs().max()))
        else:
            loss, _, _ = objective(output, training_batch, device,
                                   model.hierarchical)
            loss.backward()
        gradients[arm] = {}
        for grid in (32, 64):
            grad = getattr(model.base, f'local_dcn{grid}').delta.weight.grad
            if grad is None or not torch.isfinite(grad).all() or not grad.abs().sum():
                raise RuntimeError(f'{arm} {grid}-grid DCN has no finite residual gradient')
            gradients[arm][str(grid)] = float(grad.norm())
        if fixed_digest(model) != frozen[arm]:
            raise RuntimeError('Backward pass changed frozen coarse weights or BN')
        model.dcn_supervision = False
    if old_gradient_difference != 0:
        raise RuntimeError('DCN-specific auxiliary loss leaked into old local parameters: '
                           f'{old_gradient_difference}')
    return {'maximum_initial_flow_difference': differences,
            'residual_gradient_norm': gradients,
            'old_local_gradient_difference_after_D2_aux': old_gradient_difference,
            'grid_to_image_pixels': {'32/256': 8, '64/512': 8},
            'direction': 'VI target to IR source', 'test_pairs_accessed': False}


def save_checkpoint(args, arm, model, optimizer, scheduler, epoch, steps,
                    score, history, recipe):
    payload = {'recipe': 'two_scale_dcn_hierarchical', 'arm': arm,
               'coarse_sha256': sha256_file(args.coarse_checkpoint),
               'pretrained_sha256': sha256_file(args.pretrained),
               'reference_report_sha256': sha256_file(args.reference_dir / 'report_val.json'),
               'model_state_dict': model.state_dict(),
               'optimizer_state_dict': optimizer.state_dict(),
               'scheduler_state_dict': scheduler.state_dict(),
               'epoch': epoch, 'optimizer_steps': steps,
               'val_final_epe_512px': score, 'locked_recipe': recipe,
               'history': list(history),
               'torch_rng_state': torch.get_rng_state(),
               'numpy_rng_state': np.random.get_state(),
               'python_rng_state': random.getstate(),
               'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
               'test_pairs_accessed': False}
    temporary = args.output / f'best_{arm}.tmp'
    torch.save(payload, temporary)
    temporary.replace(args.output / f'best_{arm}.pth')


def train(args, trainset, valset, device, baseline_report, recipe):
    budget_epochs = baseline_report['training']['epochs_executed']
    reference_history = {arm: json.loads(
        (args.reference_dir / f'history_{arm}.json').read_text(encoding='utf-8'))
        for arm in ('A0', 'A1')}
    if any(len(history) != budget_epochs for history in reference_history.values()):
        raise ValueError('No-DCN reference histories do not cover the stated budget')
    if args.smoke:
        budget_epochs = 1
    models, optimizers, schedulers, histories = {}, {}, {}, {}
    initial_local, frozen = {}, {}
    for arm in ARMS:
        model, local, dcn = make_dcn_arm(args, arm, device)
        models[arm] = model
        initial_local[arm] = local_digest(model)
        frozen[arm] = fixed_digest(model)
        optimizers[arm] = torch.optim.AdamW([
            {'params': local, 'lr': recipe['old_local_lr'], 'name': 'old_local'},
            {'params': dcn, 'lr': recipe['new_fusion_lr'], 'name': 'two_scale_dcn'}])
        schedulers[arm] = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizers[arm], T_max=MAX_EPOCHS, eta_min=1e-7)
        histories[arm] = []
    if len(set(initial_local.values())) != 1 or len(set(frozen.values())) != 1:
        raise RuntimeError('Two DCN arms did not share exact old-local/coarse initialization')
    best, steps = {}, {arm: 0 for arm in ARMS}
    for arm in ARMS:
        _, summary = evaluate(models[arm], valset, device)
        best[arm] = summary['stage_epe_512px']['final512']
        save_checkpoint(args, arm, models[arm], optimizers[arm], schedulers[arm],
                        0, 0, best[arm], histories[arm], recipe)
    for epoch in range(1, budget_epochs + 1):
        for arm in ARMS:
            model, optimizer = models[arm], optimizers[arm]
            update = (train_dcn_epoch(model, optimizer, trainset, device, epoch)
                      if arm == 'D2' else
                      train_epoch(model, optimizer, trainset, device, epoch,
                                  augmentation=True))
            steps[arm] += update['steps']
            _, summary = evaluate(model, valset, device)
            score = summary['stage_epe_512px']['final512']
            schedulers[arm].step()
            record = {'epoch': epoch, 'optimizer_steps': steps[arm], **update,
                      'val': summary,
                      'lr': {g['name']: g['lr'] for g in optimizer.param_groups}}
            histories[arm].append(record)
            if score < best[arm]:
                best[arm] = score
                save_checkpoint(args, arm, model, optimizer, schedulers[arm],
                                epoch, steps[arm], score, histories[arm], recipe)
            print(f'{arm} epoch {epoch}/{budget_epochs}: val final={score:.4f}, '
                  f'best={best[arm]:.4f}, steps={steps[arm]}', flush=True)
        order = {histories[arm][-1]['image_order_sha256'] for arm in ARMS}
        if not args.smoke:
            order.update(reference_history[arm][epoch - 1]['image_order_sha256']
                         for arm in ('A0', 'A1'))
        if len(order) != 1:
            raise RuntimeError('DCN and no-DCN arms saw different training batches')
        for arm in ARMS:
            if fixed_digest(models[arm]) != frozen[arm]:
                raise RuntimeError(f'{arm} changed frozen coarse weights or BN')
            save_json(args.output / f'history_{arm}.json', histories[arm])
    if (not args.smoke and
            (len(set(steps.values())) != 1 or
             steps['D0'] != baseline_report['training']['steps_each'])):
        raise RuntimeError('DCN and no-DCN training budgets differ')
    return {'epochs_executed': budget_epochs, 'steps_each': steps['D0'],
            'reference_epochs': baseline_report['training']['epochs_executed'],
            'reference_steps_each': baseline_report['training']['steps_each']}


def report(args, valset, device, baseline_report, training):
    rows_by_arm, summaries, metadata = {}, {}, {}
    for arm in ('A0', 'A1'):
        model, _ = make_arm(args, arm, device)
        payload = torch.load(args.reference_dir / f'best_{arm}.pth',
                             map_location='cpu', weights_only=False)
        if payload['coarse_sha256'] != sha256_file(args.coarse_checkpoint):
            raise ValueError(f'{arm} reference checkpoint has another coarse weight')
        model.load_state_dict(payload['model_state_dict'], strict=True)
        model.eval()
        rows, summary = evaluate(model, valset, device, timed=True)
        expected = baseline_report['arms'][arm]['summary23']['stage_epe_512px']['final512']
        observed = summary['stage_epe_512px']['final512']
        if abs(observed - expected) > 1e-3:
            raise RuntimeError(f'{arm} reference did not reproduce: {observed} vs {expected}')
        with (args.reference_dir / f'per_image_{arm}.csv').open(
                newline='', encoding='utf-8') as file:
            stored = {row['name']: row for row in csv.DictReader(file)}
        if set(stored) != {row['name'] for row in rows}:
            raise RuntimeError(f'{arm} reference image IDs changed')
        for row in rows:
            old = stored[row['name']]
            if (int(old['valid_pixels']) != row['valid_pixels'] or
                    abs(float(old['final512_epe_512px']) -
                        row['final512_epe_512px']) > 1e-3):
                raise RuntimeError(f'{arm} reference pair {row["name"]} did not reproduce')
        rows_by_arm[arm], summaries[arm] = rows, summary
        write_csv(args.output / f'per_image_{arm}_remeasured.csv', rows)
        metadata[arm] = {'selected_epoch': payload['epoch'],
                         'selected_optimizer_steps': payload['optimizer_steps'],
                         'parameters': sum(p.numel() for p in model.parameters()),
                         'summary23': summary,
                         'reference_final_epe_512px': expected}
    for arm in ARMS:
        model, _, _ = make_dcn_arm(args, arm, device)
        payload = torch.load(args.output / f'best_{arm}.pth', map_location='cpu',
                             weights_only=False)
        if payload['coarse_sha256'] != sha256_file(args.coarse_checkpoint):
            raise ValueError('Selected DCN checkpoint has a different coarse weight')
        model.load_state_dict(payload['model_state_dict'], strict=True)
        model.eval()
        rows, summary = evaluate(model, valset, device, timed=True)
        rows_by_arm[arm], summaries[arm] = rows, summary
        write_csv(args.output / f'per_image_{arm}.csv', rows)
        metadata[arm] = {'selected_epoch': payload['epoch'],
                         'selected_optimizer_steps': payload['optimizer_steps'],
                         'parameters': sum(p.numel() for p in model.parameters()),
                         'summary23': summary}
    maps = {arm: {row['name']: row for row in rows_by_arm[arm]}
            for arm in ('A0', 'A1', *ARMS)}
    names = set(maps['D0'])
    if any(set(items) != names for items in maps.values()):
        raise RuntimeError('Validation image pairs differ from the no-DCN reference')
    paired = []
    for name in sorted(names):
        counts = [int(maps[arm][name]['valid_pixels'])
                  for arm in ('A0', 'A1', *ARMS)]
        if len(set(counts)) != 1:
            raise RuntimeError(f'GT validity changed for pair {name}')
        values = {arm: float(maps[arm][name]['final512_epe_512px'])
                  for arm in ('A0', 'A1', *ARMS)}
        paired.append({'name': name, 'valid_pixels': counts[0],
                       **{f'{arm}_final_epe_512px': values[arm] for arm in ('A0', 'A1', *ARMS)},
                       'D0_minus_A0': values['D0'] - values['A0'],
                       'D1_minus_A1': values['D1'] - values['A1'],
                       'D1_minus_D0': values['D1'] - values['D0'],
                       'D2_minus_D0': values['D2'] - values['D0'],
                       'D2_minus_A1': values['D2'] - values['A1']})
    write_csv(args.output / 'per_image_paired.csv', paired)
    score = {arm: summaries[arm]['stage_epe_512px']['final512']
             for arm in summaries}
    comparisons = {}
    for label, left, right in (('dcn_final_loss', 'D0', 'A0'),
                               ('dcn_hierarchical_loss', 'D1', 'A1'),
                               ('loss_effect_with_dcn', 'D1', 'D0'),
                               ('dcn_specific_vs_final', 'D2', 'D0'),
                               ('dcn_specific_vs_hierarchical', 'D2', 'D1'),
                               ('dcn_specific_vs_best_no_dcn', 'D2', 'A1')):
        comparisons[label] = {
            'weighted_final_epe_change_512px': score[left] - score[right],
            'improved_pairs': sum(row[f'{left}_final_epe_512px'] <
                                  row[f'{right}_final_epe_512px'] for row in paired)}
    comparisons['dcn_x_loss_interaction_epe_512px'] = (
        (score['D1'] - score['D0']) - (score['A1'] - score['A0']))
    result = {
        'protocol': {'direction': 'VI target to IR source',
                     'flow_unit': '512x512 image pixels',
                     'selection': 'full 23-pair valid-pixel-weighted final EPE',
                     'test_pairs_accessed': False,
                     'coarse_checkpoint_sha256': sha256_file(args.coarse_checkpoint),
                     'reference_report_sha256': sha256_file(args.reference_dir / 'report_val.json'),
                     'equal_image_order_and_optimizer_steps': True},
        'training': training,
        'arms': metadata,
        'comparisons': comparisons,
        'diagnostic21': {arm: summarize([r for r in rows_by_arm[arm]
                                        if r['name'] not in ('000002', '000014')])
                         for arm in ('A0', 'A1', *ARMS)},
        'difficult': {arm: [r for r in rows_by_arm[arm]
                             if r['name'] in ('000002', '000014')]
                      for arm in ('A0', 'A1', *ARMS)}}
    save_json(args.output / 'report_val.json', result)
    print(json.dumps({'arms': {a: {'selected_epoch': metadata[a]['selected_epoch'],
                                   'final_epe_512px': score[a]} for a in ARMS},
                      'comparisons': comparisons}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('check', 'smoke', 'train', 'report'), required=True)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--pretrained', required=True, type=Path)
    parser.add_argument('--coarse-checkpoint', required=True, type=Path)
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.smoke = args.stage == 'smoke'
    if not (args.data_root.is_dir() and args.pretrained.is_file() and
            args.coarse_checkpoint.is_file()):
        parser.error('RoadScene data, pretrained weight or fixed coarse weight missing')
    baseline_report, recipe = reference(args)
    if args.stage in ('check', 'smoke', 'train') and args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a new, empty output directory for check/smoke/train')
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
        if len(trainset) != 176 or len(valset) != 23:
            raise ValueError('RoadScene train/val split differs from reference')
        if args.stage in ('check', 'smoke'):
            trainset, valset = Subset(trainset, [0]), Subset(valset, [0])
        if args.stage != 'report':
            check = initialization_check(args, trainset, valset, device)
            save_json(args.output / 'initialization_check.json', check)
            if args.stage == 'check':
                print(json.dumps(check, indent=2), flush=True)
                return
            training = train(args, trainset, valset, device, baseline_report, recipe)
            if args.stage == 'smoke':
                print(json.dumps(training, indent=2), flush=True)
                return
        else:
            training = {'epochs_executed': baseline_report['training']['epochs_executed'],
                        'steps_each': baseline_report['training']['steps_each']}
        report(args, valset, device, baseline_report, training)
    finally:
        glunet.MutualMatching = original


if __name__ == '__main__':
    main()
