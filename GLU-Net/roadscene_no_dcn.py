"""Independent RoadScene DNS+SA/CA coarse, then matched no-DCN fine ablation.

Run coarse first, then fine. Only train and val are opened. The historical
DCN best_fine checkpoint is deliberately not loaded by this recipe.
"""

import argparse
import csv
import hashlib
import importlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from datasets.roadscene import RoadScenePairs
from models.our_models.GLUNet import GLUNet_model
from models.our_models.local_no_dcn_multiscale import LocalNoDCNMultiScale
from models.our_models.local_recurrent_128_256 import sample_window
from roadscene_coarse import evaluate as evaluate_coarse, load_base_weights, sha256_file
from roadscene_refinement_audit import edge_quartile
import roadscene_staged as staged


SEED = 2026
COARSE_EPOCHS, FINE_EPOCHS = 200, 80
COARSE_PATIENCE, FINE_PATIENCE = 20, 12
COARSE_DELTA, FINE_DELTA = .01, .01
FINE_ARMS = ('original_local', 'multiscale_local')
COARSE_NAMES = ('coarse_dns', 'coarse_attention', 'decoder4')
LOCAL_NAMES = staged.LOCAL_MODULES
UP_NAMES = staged.FROZEN_FLOW_UPSAMPLERS
FIX = 'nonnegative_correlation_before_mutual_matching'


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def check_no_dcn(model):
    base = model.base if isinstance(model, FineModel) else model
    if base.local_dcn_steps != 0 or base.local_dcn32 is not None:
        raise RuntimeError('A true DCN module is present')
    forbidden = ('local_dcn32.', 'deform_conv', 'offset')
    bad = [key for key in model.state_dict() if any(word in key.lower() for word in forbidden)]
    if bad:
        raise RuntimeError(f'DCN or offset weights are present: {bad[:5]}')
    for module in model.modules():
        if 'deform' in module.__class__.__name__.lower():
            raise RuntimeError(f'Deformable module: {module.__class__.__name__}')


def base_model(pretrained, device, coarse_checkpoint=None):
    model = GLUNet_model(evaluation=False, pyramid_type='VGG',
                         cyclic_consistency=True, backbone_pretrained=False,
                         coarse_attention=True, coarse_dns=True,
                         local_dcn_steps=0)
    load_base_weights(model, pretrained)
    if coarse_checkpoint is not None:
        payload = torch.load(coarse_checkpoint, map_location='cpu', weights_only=False)
        if (payload.get('stage') != 'coarse' or payload.get('recipe') != 'no_dcn_v1' or
                payload.get('pretrained_sha256') != sha256_file(pretrained) or
                payload.get('matching_fix') != FIX):
            raise ValueError('Coarse checkpoint is not from this no-DCN recipe')
        if any('local_dcn' in key for key in payload['model_state_dict']):
            raise ValueError('Coarse checkpoint has DCN weights')
        model.load_state_dict(payload['model_state_dict'], strict=True)
    check_no_dcn(model)
    return model.to(device).eval()


class FineModel(nn.Module):
    def __init__(self, base, enhanced):
        super().__init__()
        self.base = base
        self.refiner = LocalNoDCNMultiScale() if enhanced else None
        self.base.train_coarse_encoder = False
        check_no_dcn(self)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()  # The original VGG and decoder BN statistics are fixed.
        return self

    def forward(self, batch, device):
        source, target, source256, target256, *_ = self.base.pre_process_data(
            batch['source_image'].to(device), batch['target_image'].to(device),
            device=device)
        features128, features256 = [], []

        def remember(store):
            def hook(_module, _inputs, value):
                if len(store) < 2:
                    store.append(value.detach())
            return hook

        handles = []
        if self.refiner is not None:
            handles = [self.base.pyramid._modules['level_2'].register_forward_hook(
                           remember(features128)),
                       self.base.pyramid._modules['level_1'].register_forward_hook(
                           remember(features256))]
        try:
            flow256, flow512 = self.base(target, source, target256, source256)
        finally:
            for handle in handles:
                handle.remove()
        base_final = F.interpolate(flow512[-1], (512, 512),
                                   mode='bilinear', align_corners=False)
        result = {'coarse16': F.interpolate(flow256[0], (512, 512),
                                            mode='bilinear', align_corners=False) * 2,
                  'local32': F.interpolate(flow256[1], (512, 512),
                                           mode='bilinear', align_corners=False) * 2,
                  'local64': F.interpolate(flow512[0], (512, 512),
                                           mode='bilinear', align_corners=False),
                  'base128': base_final, 'final512': base_final}
        if self.refiner is not None:
            if len(features128) != 2 or len(features256) != 2:
                raise RuntimeError('Expected target VI then source IR VGG features')
            extra = self.refiner(features128[0], features128[1],
                                 features256[0], features256[1], flow512[-1])
            result['update128'] = F.interpolate(extra['flow128'], (512, 512),
                                                 mode='bilinear', align_corners=False)
            result['final512'] = extra['final512']
            result['valid128'] = extra['valid128']
            result['valid256'] = extra['valid256']
        return result


def configure(model, stage):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if stage == 'coarse':
        model.train_coarse_encoder = True
        active = list(model.pyramid._modules['level_4'].parameters())
        active += [p for name in COARSE_NAMES for p in getattr(model, name).parameters()]
    else:
        model.base.train_coarse_encoder = False
        active = [p for name in LOCAL_NAMES for p in getattr(model.base, name).parameters()]
        if model.refiner is not None:
            active += list(model.refiner.parameters())
    for parameter in active:
        parameter.requires_grad_(True)
    if len({id(p) for p in active}) != len(active):
        raise RuntimeError('Overlapping optimizer parameters')
    base = model.base if stage == 'fine' else model
    if any(p.requires_grad for name in UP_NAMES for p in getattr(base, name).parameters()):
        raise RuntimeError('Pretrained flow upsampler became trainable')
    check_no_dcn(model)
    model.eval()
    return active


def geometry_check():
    result = {}
    for size, radius in ((128, 2), (256, 1)):
        source = torch.zeros(1, 1, size, size)
        source[0, 0, 40, 40 + size // 128] = 1
        flow = torch.zeros(1, 2, size, size)
        flow[:, 0] = 4  # 512px x flow; grid displacement is size/128.
        sampled, valid = sample_window(source, flow, radius)
        opposite, _ = sample_window(source, -flow, radius)
        middle = (2 * radius + 1) ** 2 // 2
        if (abs(float(sampled[0, 0, middle, 40, 40]) - 1) > 1e-5 or
                float(opposite[0, 0, middle, 40, 40]) != 0 or
                bool(valid[0, middle, 40, -1])):
            raise RuntimeError(f'VI->IR direction/scale/mask failed on {size} grid')
        result[str(size)] = {'flow_512px': [4, 0],
                             'grid_shift': [size // 128, 0],
                             'radius_grid': radius}
    return result


def save(path, payload):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def state_snapshot(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def state_digest(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(state[name].contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def validate_coarse(model, valset, device):
    model.eval()
    return evaluate_coarse(model, DataLoader(valset, batch_size=1), device)


@torch.no_grad()
def evaluate_fine(model, dataset, device, timed=False):
    model.eval()
    rows = []
    for batch in DataLoader(dataset, batch_size=1):
        if timed:
            for _ in range(2):
                model(batch, device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
        output = model(batch, device)
        if timed and device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000 if timed else None
        truth = batch['flow_map'].to(device).float()
        valid = batch['correspondence_mask'].to(device).bool()
        edge = edge_quartile(batch['target_image'].to(device), valid[0])
        large = valid[0] & (torch.linalg.vector_norm(truth[0], dim=0) >= 64)
        row = {'name': batch['name'][0], 'valid_pixels': int(valid.sum()),
               'time_ms': elapsed}
        for label in ('coarse16', 'local32', 'local64', 'base128',
                      'update128', 'final512'):
            if label not in output:
                continue
            error = torch.linalg.vector_norm(output[label][0] - truth[0], dim=0)
            row[f'{label}_epe_512px'] = float(error[valid[0]].mean())
            if label == 'final512':
                for threshold in (1, 3, 5):
                    row[f'fraction_below_{threshold}px'] = float(
                        ((error < threshold) & valid[0]).sum() / valid[0].sum())
                for region, mask in (('edge', edge), ('large64', large)):
                    row[f'{region}_pixels'] = int(mask.sum())
                    row[f'{region}_epe_512px'] = float(error[mask].mean()) if mask.any() else None
        rows.append(row)
    return rows


def weighted(rows, field, count='valid_pixels'):
    chosen = [r for r in rows if r.get(field) is not None]
    denominator = sum(r[count] for r in chosen)
    return sum(r[field] * r[count] for r in chosen) / denominator if denominator else None


def summarize(rows):
    return {'pairs': len(rows), 'valid_pixels': sum(r['valid_pixels'] for r in rows),
            'stage_epe_512px': {label: weighted(rows, f'{label}_epe_512px')
                                 for label in ('coarse16', 'local32', 'local64',
                                               'base128', 'update128', 'final512')},
            'fraction_below': {str(t): weighted(rows, f'fraction_below_{t}px')
                               for t in (1, 3, 5)},
            'edge_epe_512px': weighted(rows, 'edge_epe_512px', 'edge_pixels'),
            'large64_epe_512px': weighted(rows, 'large64_epe_512px', 'large64_pixels'),
            'inference_ms_mean': float(np.mean([r['time_ms'] for r in rows]))
                if rows and rows[0]['time_ms'] is not None else None}


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint(stage, model, optimizer, scheduler, epoch, steps, metric,
               history, args, arm=None):
    return {'recipe': 'no_dcn_v1', 'stage': stage, 'arm': arm,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch, 'optimizer_steps': steps, 'selection_metric': metric,
            'history': history, 'pretrained_sha256': sha256_file(args.pretrained),
            'matching_fix': FIX, 'test_pairs_accessed': False,
            'torch_rng_state': torch.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'python_rng_state': random.getstate(),
            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'code_sha256': sha256_file(Path(__file__)),
            'coarse_checkpoint_sha256': sha256_file(args.output/'best_coarse.pth')
                if stage == 'fine' else None}


def train_coarse(args, trainset, valset, device):
    seed_all()
    model = base_model(args.pretrained, device)
    parameters = configure(model, 'coarse')
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=.5, patience=6, min_lr=1e-7)
    epochs = 1 if args.code_check else COARSE_EPOCHS
    best = float('inf')
    bad, steps, history = 0, 0, []
    for epoch in range(1, epochs + 1):
        loader = DataLoader(trainset, batch_size=1 if args.code_check else 2,
                            shuffle=True, generator=torch.Generator().manual_seed(SEED+epoch))
        loss_sum = 0.
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = staged.coarse_training_loss(model, batch, device)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite coarse loss at epoch {epoch}, step {steps}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            loss_sum += float(loss.detach())
            steps += 1
        metrics = validate_coarse(model, valset, device)
        score = float(metrics['epe_256px'])
        scheduler.step(score)
        record = {'epoch': epoch, 'steps': steps, 'train_loss': loss_sum/len(loader),
                  'val_coarse_epe_256px': score, 'val_corr_top1': metrics.get('corr_top1'),
                  'lr': optimizer.param_groups[0]['lr']}
        history.append(record)
        previous_best = best
        if score < best:
            best = score
            save(args.output/'best_coarse.pth', checkpoint(
                'coarse', model, optimizer, scheduler, epoch, steps, best, history, args))
        if score < previous_best - COARSE_DELTA:
            bad = 0
        else:
            bad += 1
        (args.output/'history_coarse.json').write_text(json.dumps(history, indent=2))
        print(f'coarse {epoch}/{epochs}: val EPE={score:.4f}, best={best:.4f}, '
              f'patience={bad}/{COARSE_PATIENCE}', flush=True)
        if bad >= COARSE_PATIENCE:
            break
    check_no_dcn(model)
    return {'selected_val_coarse_epe_256px': best, 'stopped_epoch': epoch,
            'selected_checkpoint': str(args.output/'best_coarse.pth')}


def make_fine(arm, args, device):
    seed_all()
    base = base_model(args.pretrained, device, args.output/'best_coarse.pth')
    model = FineModel(base, enhanced=arm == 'multiscale_local').to(device)
    configure(model, 'fine')
    return model


def zero_check(candidate, batch, device):
    candidate.eval()
    with torch.no_grad():
        result = candidate(batch, device)
    difference = float((result['final512'] - result['base128']).abs().max())
    if difference != 0:
        raise RuntimeError(f'Zero residual did not reproduce baseline: {difference}')
    return difference


def train_fine_arm(arm, args, trainset, valset, device, initial_base_hash,
                   initial_local_hash):
    model = make_fine(arm, args, device)
    if state_digest(state_snapshot(model.base)) != initial_base_hash:
        raise RuntimeError(f'{arm} did not start from the same coarse/base state')
    base_state = {k: v for k, v in model.base.state_dict().items()
                  if any(k.startswith(name+'.') for name in LOCAL_NAMES)}
    if state_digest(state_snapshot_dict(base_state)) != initial_local_hash:
        raise RuntimeError(f'{arm} local initialization differs')
    if arm == 'multiscale_local':
        zero_check(model, next(iter(DataLoader(valset, batch_size=1))), device)
    def frozen_state():
        state = {k: v for k, v in model.base.named_parameters()
                 if not any(k.startswith(name+'.') for name in LOCAL_NAMES)}
        state.update({f'buffer:{k}': v for k, v in model.base.named_buffers()})
        return state_digest(state_snapshot_dict(state))
    frozen_before = frozen_state()
    local = [p for name in LOCAL_NAMES for p in getattr(model.base, name).parameters()]
    groups = [{'params': local, 'lr': 1e-5, 'name': 'original_local'}]
    if model.refiner is not None:
        groups.append({'params': model.refiner.parameters(), 'lr': 1e-4,
                       'name': 'multiscale_local'})
    optimizer = torch.optim.AdamW(groups)
    epochs = 1 if args.code_check else FINE_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)
    best, bad, steps, history = float('inf'), 0, 0, []
    initial = evaluate_fine(model, valset, device)
    initial_score = weighted(initial, 'final512_epe_512px')
    for epoch in range(1, epochs + 1):
        model.train()
        loader = DataLoader(trainset, batch_size=1 if args.code_check else 2,
                            shuffle=True, generator=torch.Generator().manual_seed(SEED+epoch))
        loss_sum = 0.
        order = hashlib.sha256()
        for batch in loader:
            for name in batch['name']:
                order.update((name+'\n').encode())
            optimizer.zero_grad(set_to_none=True)
            result = model(batch, device)
            truth = batch['flow_map'].to(device).float()
            valid = batch['correspondence_mask'].to(device).bool()
            loss = torch.sqrt((result['final512']-truth).square().sum(1)+.01)[valid].mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite fine loss in {arm}, epoch {epoch}')
            loss.backward()
            if arm == 'multiscale_local' and steps == 0:
                for head in (model.refiner.delta128, model.refiner.delta256):
                    if (head.weight.grad is None or
                            not torch.isfinite(head.weight.grad).all() or
                            head.weight.grad.abs().sum() == 0):
                        raise RuntimeError('New residual head lacks a finite gradient')
            torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group['params']], 1.)
            optimizer.step()
            loss_sum += float(loss.detach())
            steps += 1
        rows = evaluate_fine(model, valset, device)
        score = weighted(rows, 'final512_epe_512px')
        scheduler.step()
        history.append({'epoch': epoch, 'steps': steps, 'train_loss': loss_sum/len(loader),
                        'val_final_epe_512px': score,
                        'image_order_sha256': order.hexdigest(),
                        'lr': {g['name']: g['lr'] for g in optimizer.param_groups}})
        previous_best = best
        if score < best:
            best = score
            save(args.output/f'best_{arm}.pth', checkpoint(
                'fine', model, optimizer, scheduler, epoch, steps, best, history,
                args, arm=arm))
        if score < previous_best - FINE_DELTA:
            bad = 0
        else:
            bad += 1
        (args.output/f'history_{arm}.json').write_text(json.dumps(history, indent=2))
        print(f'{arm} {epoch}/{epochs}: val EPE={score:.4f}, best={best:.4f}, '
              f'patience={bad}/{FINE_PATIENCE}', flush=True)
        # Both arms execute the complete identical budget. Validation selects
        # the best checkpoint; arm-specific early stopping would break parity.
    frozen_after = frozen_state()
    if frozen_before != frozen_after:
        raise RuntimeError(f'{arm} changed frozen coarse or BN state')
    return {'initial_epe_512px': initial_score, 'selected_epe_512px': best,
            'executed_epochs': epoch, 'executed_steps': steps,
            'frozen_state_unchanged': True, 'history': history}


def state_snapshot_dict(state):
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def report_fine(args, valset, device, histories):
    records = {}
    for arm in FINE_ARMS:
        model = make_fine(arm, args, device)
        selected = torch.load(args.output/f'best_{arm}.pth', map_location='cpu',
                              weights_only=False)
        if selected['arm'] != arm or selected['recipe'] != 'no_dcn_v1':
            raise ValueError('Wrong fine checkpoint')
        model.load_state_dict(selected['model_state_dict'], strict=True)
        check_no_dcn(model)
        rows = evaluate_fine(model, valset, device, timed=True)
        write_csv(args.output/f'per_image_{arm}.csv', rows)
        records[arm] = {'selected_epoch': selected['epoch'],
                        'selected_steps': selected['optimizer_steps'],
                        'metrics23': summarize(rows), 'rows': rows,
                        'parameters': sum(p.numel() for p in model.parameters())}
    a = {r['name']: r for r in records['original_local']['rows']}
    b = {r['name']: r for r in records['multiscale_local']['rows']}
    if set(a) != set(b) or len(a) != (1 if args.code_check else 23):
        raise RuntimeError('Validation image pairs differ')
    paired = [{'name': name, 'valid_pixels': a[name]['valid_pixels'],
               'original_epe_512px': a[name]['final512_epe_512px'],
               'multiscale_epe_512px': b[name]['final512_epe_512px'],
               'multiscale_minus_original_512px':
                   b[name]['final512_epe_512px']-a[name]['final512_epe_512px']}
              for name in sorted(a)]
    write_csv(args.output/'per_image_paired.csv', paired)
    excluded = {'000002', '000014'}
    report = {'recipe': 'no_dcn_v1', 'code_check_only': args.code_check,
              'test_pairs_accessed': False,
              'flow_direction': 'VI target to IR source',
              'flow_units': 'all reported flows in 512x512 pixels',
              'geometry_check': geometry_check(),
              'coarse_checkpoint_sha256': sha256_file(args.output/'best_coarse.pth'),
              'historical_best_fine_loaded': False,
              'arms': {arm: {k: v for k, v in record.items() if k != 'rows'}
                       for arm, record in records.items()},
              'improved_pairs23': sum(r['multiscale_minus_original_512px'] < 0
                                      for r in paired),
              'diagnostic21': {arm: summarize([r for r in records[arm]['rows']
                                              if r['name'] not in excluded])
                               for arm in FINE_ARMS},
              'difficult2': {arm: [r for r in records[arm]['rows']
                                   if r['name'] in excluded] for arm in FINE_ARMS},
              'training': histories}
    (args.output/'report_fine.json').write_text(json.dumps(report, indent=2),
                                                encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('coarse', 'fine'), required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--pretrained', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--code-check', action='store_true')
    args = parser.parse_args()
    if not args.data_root.is_dir() or not args.pretrained.is_file():
        parser.error('RoadScene root or pretrained GLU-Net weights are missing')
    if args.stage == 'coarse' and args.output.exists() and any(args.output.iterdir()):
        parser.error('Coarse stage requires a new empty output directory')
    if args.stage == 'fine' and not (args.output/'best_coarse.pth').is_file():
        parser.error('Run coarse stage first in this output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    seed_all()
    geom = geometry_check()
    glunet = importlib.import_module('models.our_models.GLUNet')
    original = glunet.MutualMatching
    glunet.MutualMatching = lambda corr: original(corr.clamp_min(0))
    all_zero = glunet.MutualMatching(torch.zeros(1, 1, 16, 16, 16, 16))
    if not (torch.isfinite(all_zero).all() and torch.count_nonzero(all_zero) == 0):
        raise RuntimeError('All-zero MutualMatching is unsafe')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainset = RoadScenePairs(args.data_root, 'train')
    valset = RoadScenePairs(args.data_root, 'val')
    if args.code_check:
        trainset, valset = Subset(trainset, [0]), Subset(valset, [0])
    try:
        if args.stage == 'coarse':
            record = train_coarse(args, trainset, valset, device)
            (args.output/'report_coarse.json').write_text(json.dumps(
                {'recipe': 'no_dcn_v1', 'geometry_check': geom,
                 'code_check_only': args.code_check, 'test_pairs_accessed': False,
                 **record}, indent=2))
        else:
            seed_all()
            reference = make_fine('original_local', args, device)
            initial_base_hash = state_digest(state_snapshot(reference.base))
            local_state = {k: v for k, v in reference.base.state_dict().items()
                           if any(k.startswith(name+'.') for name in LOCAL_NAMES)}
            initial_local_hash = state_digest(state_snapshot_dict(local_state))
            del reference
            histories = {arm: train_fine_arm(arm, args, trainset, valset,
                                             device, initial_base_hash,
                                             initial_local_hash)
                         for arm in FINE_ARMS}
            if ([r['image_order_sha256'] for r in histories[FINE_ARMS[0]]['history']] !=
                    [r['image_order_sha256'] for r in histories[FINE_ARMS[1]]['history']] or
                    histories[FINE_ARMS[0]]['executed_steps'] !=
                    histories[FINE_ARMS[1]]['executed_steps']):
                raise RuntimeError('Fine arms used unequal image order or update steps')
            report = report_fine(args, valset, device, histories)
            print(json.dumps({'original_final_epe_512px':
                              report['arms']['original_local']['metrics23']
                              ['stage_epe_512px']['final512'],
                              'multiscale_final_epe_512px':
                              report['arms']['multiscale_local']['metrics23']
                              ['stage_epe_512px']['final512'],
                              'improved_pairs23': report['improved_pairs23']}, indent=2))
    finally:
        glunet.MutualMatching = original


if __name__ == '__main__':
    main()
