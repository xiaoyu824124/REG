"""Read-only train/val feasibility audit of a global similarity initialization.

No network parameter or checkpoint is changed. Candidate similarity transforms
are fitted from predictions only; train GT selects one fixed fitting rule.
Validation GT is used only for evaluation and for a separate labeled oracle.
"""

import argparse
import contextlib
import csv
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MAX_ANGLE_DEG = 50  # train GT maximum is 44.96 degrees; five-degree margin

from datasets.roadscene import RoadScenePairs
from roadscene_coarse import sha256_file
from roadscene_local_ablation import checked_payload, new_model
from roadscene_local256 import inputs
from roadscene_refinement_audit import pooled_truth


def geometry_maps(size, flow512):
    yy, xx = np.indices((size, size), dtype=np.float32)
    target = np.stack((xx, yy), axis=-1).reshape(-1, 2) * (512.0 / size)
    points = target + flow512.transpose(1, 2, 0).reshape(-1, 2)
    return target, points


def flow_from_similarity(matrix, size):
    target, _ = geometry_maps(size, np.zeros((2, size, size), np.float32))
    source = target @ matrix[:, :2].T + matrix[:, 2]
    return (source - target).reshape(size, size, 2).transpose(2, 0, 1)


def fit_similarity(source, destination, threshold, seed):
    if len(source) < 8:
        return None, 0
    cv2.setRNGSeed(int(seed))
    matrix, inliers = cv2.estimateAffinePartial2D(
        source.astype(np.float32), destination.astype(np.float32),
        method=cv2.RANSAC, ransacReprojThreshold=float(threshold),
        maxIters=2000, confidence=.99, refineIters=10)
    if matrix is None or inliers is None:
        return None, 0
    count = int(inliers.sum())
    scale = math.hypot(float(matrix[0, 0]), float(matrix[1, 0]))
    angle = abs(math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))))
    if count < 8 or not (.8 <= scale <= 1.2) or angle > MAX_ANGLE_DEG:
        return None, count
    return matrix.astype(np.float32), count


def least_squares_similarity(source, destination):
    """GT-only oracle: optimal squared-error similarity for the pooled points."""
    x, y = source.astype(np.float64), destination.astype(np.float64)
    x_mean, y_mean = x.mean(0), y.mean(0)
    xc, yc = x - x_mean, y - y_mean
    covariance = yc.T @ xc / len(x)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(2)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    scale = np.trace(np.diag(singular) @ correction) / np.mean((xc * xc).sum(1))
    result = np.zeros((2, 3), np.float32)
    result[:, :2] = (scale * rotation).astype(np.float32)
    result[:, 2] = (y_mean - scale * rotation @ x_mean).astype(np.float32)
    return result


def inference_cache(model, dataset, split, directory, device):
    directory.mkdir(parents=True, exist_ok=True)
    for number, batch in enumerate(DataLoader(dataset, batch_size=1, shuffle=False), 1):
        name = batch['name'][0]
        path = directory / f'{name}.npz'
        if path.exists():
            continue
        source, target, source256, target256 = inputs(model, batch, device)
        with torch.no_grad():
            target_feature = model.pyramid(target256)[-3]
            source_feature = model.pyramid(source256)[-3]
            target_feature, source_feature = model.enrich_coarse_features(
                target_feature, source_feature, target256, source256)
            predicted, corr = model.coarsest_resolution_flow(
                target_feature, source_feature, 256, 256, return_corr=True)
            gt = batch['flow_map'].to(device).float()
            valid = batch['correspondence_mask'].to(device).bool()
            truth16, selected16 = pooled_truth(gt, valid, 16)
            scores, indices = torch.topk(corr.flatten(2), 2, dim=1)
            margin = (scores[:, 0] - scores[:, 1])[0]
            top = scores[:, 0][0]
            best = indices[:, 0][0]
            # Query at (x,y) on grid16 maps to the predicted source bin.
            yy, xx = torch.meshgrid(torch.arange(16, device=device),
                                    torch.arange(16, device=device), indexing='ij')
            target_point = torch.stack((xx, yy), dim=-1).float() * 32
            source_point = torch.stack((best % 16, best // 16), dim=-1).float() * 32
            np.savez_compressed(
                path, coarse512=(predicted[0] * 2).cpu().numpy(),
                truth16=truth16[0].cpu().numpy(),
                valid16=selected16[0].cpu().numpy(),
                target=target_point.cpu().numpy().reshape(-1, 2),
                corr_source=source_point.cpu().numpy().reshape(-1, 2),
                margin=margin.cpu().numpy().reshape(-1),
                top=top.cpu().numpy().reshape(-1))
        if number % 25 == 0 or number == len(dataset):
            print(f'{split}: cached {number}/{len(dataset)} pairs', flush=True)


def candidate(cache, method, percentile, threshold, seed):
    target = cache['target']
    if method == 'corr':
        source = cache['corr_source']
    else:
        source = target + cache['coarse512'].transpose(1, 2, 0).reshape(-1, 2)
    margin = cache['margin']
    limit = np.percentile(margin, percentile)
    selected = np.isfinite(source).all(1) & np.isfinite(margin) & (margin > 0) & (margin >= limit)
    matrix, inliers = fit_similarity(target[selected], source[selected], threshold, seed)
    if matrix is None:
        return cache['coarse512'], False, int(selected.sum()), inliers
    return flow_from_similarity(matrix, 16), True, int(selected.sum()), inliers


def coarse_metrics(pred, truth, valid):
    error = np.linalg.norm(pred - truth, axis=0)
    return float(error[valid].sum()), int(valid.sum())


def rule_search(cache_dir):
    caches = [(p.stem, np.load(p)) for p in sorted(cache_dir.glob('*.npz'))]
    if len(caches) != 176:
        raise RuntimeError(f'Train cache must contain 176 pairs, got {len(caches)}')
    rules = []
    for method in ('corr', 'decoder'):
        for percentile in (0, 25, 50, 75):
            for threshold in (8, 16, 32):
                err, count, improve, succeed = 0., 0, 0, 0
                for name, cache in caches:
                    pred, fitted, _, _ = candidate(
                        cache, method, percentile, threshold, int(name) + 2026)
                    baseline_sum, pixels = coarse_metrics(
                        cache['coarse512'], cache['truth16'], cache['valid16'])
                    selected_sum, _ = coarse_metrics(pred, cache['truth16'], cache['valid16'])
                    err += selected_sum
                    count += pixels
                    improve += selected_sum < baseline_sum
                    succeed += fitted
                rules.append(dict(method=method, margin_percentile=percentile,
                                  ransac_threshold_512px=threshold,
                                  train_epe_512px=err / count,
                                  train_improved_pairs=improve,
                                  fitted_pairs=succeed))
    baseline_sum = sum(coarse_metrics(c['coarse512'], c['truth16'], c['valid16'])[0]
                       for _, c in caches)
    count = sum(int(c['valid16'].sum()) for _, c in caches)
    train_oracle_angles = []
    for _, cache in caches:
        selected = cache['valid16'].reshape(-1)
        target = cache['target'][selected]
        source = target + cache['truth16'].transpose(1, 2, 0).reshape(-1, 2)[selected]
        matrix = least_squares_similarity(target, source)
        train_oracle_angles.append(abs(math.degrees(math.atan2(
            float(matrix[1, 0]), float(matrix[0, 0])))))
    if max(train_oracle_angles) > MAX_ANGLE_DEG:
        raise RuntimeError('Train geometry exceeds the preselected angle guard')
    # A global similarity is considered train-feasible only if it improves the
    # weighted coarse EPE and at least half of all training pairs.
    eligible = [r for r in rules if r['train_epe_512px'] < baseline_sum / count
                and r['train_improved_pairs'] >= 88]
    chosen = min(eligible, key=lambda r: r['train_epe_512px']) if eligible else None
    for _, cache in caches:
        cache.close()
    return dict(train_baseline_epe_512px=baseline_sum / count,
                train_pairs=176, train_valid_queries=count,
                train_gt_oracle_max_abs_rotation_deg=max(train_oracle_angles),
                train_gt_oracle_pairs_over_30deg=sum(a > 30 for a in train_oracle_angles),
                max_deployable_angle_deg=MAX_ANGLE_DEG,
                selection_rule='minimum train weighted coarse EPE among rules improving >=88/176 pairs; otherwise no rule',
                chosen=chosen, rules=rules)


def full_coverage(pred16, truth, valid, size, radius):
    pred = F.interpolate(torch.from_numpy(pred16)[None].float(), (size, size),
                         mode='bilinear', align_corners=False)
    truth_grid, selected = pooled_truth(truth, valid, size)
    radius512 = radius * 512 / size
    delta = truth_grid - pred
    inside = (delta.abs().amax(1) <= radius512) & selected
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    source_x = xx[None] + truth_grid[:, 0] * size / 512
    source_y = yy[None] + truth_grid[:, 1] * size / 512
    source_inbounds = (source_x >= 0) & (source_x <= size - 1) & (source_y >= 0) & (source_y <= size - 1)
    strict = selected & source_inbounds
    return dict(valid_queries=int(selected.sum()), inside_queries=int(inside.sum()),
                strict_queries=int(strict.sum()),
                strict_inside=int((inside & strict).sum()),
                radius_grid=radius, radius_512px_per_axis=radius512)


def val_evaluation(dataset, cache_dir, chosen):
    rows = []
    for batch in DataLoader(dataset, batch_size=1, shuffle=False):
        name = batch['name'][0]
        with np.load(cache_dir / f'{name}.npz') as cache:
            baseline = cache['coarse512'].copy()
            truth16, valid16 = cache['truth16'], cache['valid16']
            if chosen:
                fitted, success, candidates, inliers = candidate(
                    cache, chosen['method'], chosen['margin_percentile'],
                    chosen['ransac_threshold_512px'], int(name) + 2026)
            else:
                fitted, success, candidates, inliers = baseline, False, 0, 0
            target = cache['target'][valid16.reshape(-1)]
            gt_source = target + truth16.transpose(1, 2, 0).reshape(-1, 2)[valid16.reshape(-1)]
            oracle = flow_from_similarity(
                least_squares_similarity(target, gt_source), 16)
        gt = batch['flow_map'].float()
        valid = batch['correspondence_mask'].bool()
        row = dict(name=name, fit_success=success, candidate_points=candidates,
                   ransac_inliers=inliers, valid_coarse_queries=int(valid16.sum()))
        for label, pred in (('original', baseline), ('predicted_fit', fitted),
                            ('oracle_gt_fit', oracle)):
            total, count = coarse_metrics(pred, truth16, valid16)
            row[f'{label}_coarse_epe_512px'] = total / count
            row[f'{label}_coarse_error_sum'] = total
            # 32/64 are GLU-Net's existing local windows (radius 4);
            # 128/256 are the independent recurrent candidate's windows.
            for size, radius in ((32, 4), (64, 4), (128, 2), (256, 1)):
                cov = full_coverage(pred, gt, valid, size, radius)
                for key, value in cov.items():
                    row[f'{label}_grid{size}_{key}'] = value
        rows.append(row)
    return rows


def exact_fine_coverage(model, dataset, device):
    rows = []
    for batch in DataLoader(dataset, batch_size=1, shuffle=False):
        source, target, source256, target256 = inputs(model, batch, device)
        with torch.no_grad():
            _, local = model(target, source, target256, source256)
        initial128 = local[-1]
        truth = batch['flow_map'].float().to(device)
        valid = batch['correspondence_mask'].bool().to(device)
        row = dict(name=batch['name'][0])
        for size, radius in ((128, 2), (256, 1)):
            prediction = F.interpolate(initial128, (size, size),
                                       mode='bilinear', align_corners=False)
            gt, selected = pooled_truth(truth, valid, size)
            radius512 = radius * 512 / size
            inside = ((gt - prediction).abs().amax(1) <= radius512) & selected
            yy, xx = torch.meshgrid(torch.arange(size, device=device),
                                    torch.arange(size, device=device), indexing='ij')
            sx = xx[None] + gt[:, 0] * size / 512
            sy = yy[None] + gt[:, 1] * size / 512
            strict = selected & (sx >= 0) & (sx <= size - 1) & (sy >= 0) & (sy <= size - 1)
            row[f'grid{size}_valid_queries'] = int(selected.sum())
            row[f'grid{size}_inside_queries'] = int(inside.sum())
            row[f'grid{size}_strict_queries'] = int(strict.sum())
            row[f'grid{size}_strict_inside'] = int((strict & inside).sum())
        rows.append(row)
    return rows


def summarize_coverage(rows, prefix, size):
    count = sum(r[f'{prefix}grid{size}_valid_queries'] for r in rows)
    inside = sum(r[f'{prefix}grid{size}_inside_queries'] for r in rows)
    strict_count = sum(r[f'{prefix}grid{size}_strict_queries'] for r in rows)
    strict_inside = sum(r[f'{prefix}grid{size}_strict_inside'] for r in rows)
    return dict(pairs=len(rows), valid_queries=count, inside_queries=inside,
                inside_fraction=inside / count, strict_queries=strict_count,
                strict_inside=strict_inside,
                strict_fraction=strict_inside / strict_count)


def summarize_epe(rows, label):
    total = sum(r[f'{label}_coarse_error_sum'] for r in rows)
    count = sum(r['valid_coarse_queries'] for r in rows)
    return dict(pairs=len(rows), valid_queries=count, weighted_epe_512px=total/count,
                improved_pairs_vs_original=sum(
                    r[f'{label}_coarse_epe_512px'] < r['original_coarse_epe_512px']
                    for r in rows))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--pretrained', type=Path, required=True)
    parser.add_argument('--coarse-checkpoint', type=Path, required=True)
    parser.add_argument('--fine-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    coarse_payload = checked_payload(args.coarse_checkpoint, args.pretrained, smoke=False)
    model = new_model(coarse_payload, device)
    fine_payload = torch.load(args.fine_checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(fine_payload['model_state_dict'], strict=True)
    model.eval()
    from models.our_models import GLUNet as glunet
    original_matching = glunet.MutualMatching
    glunet.MutualMatching = lambda corr: original_matching(corr.clamp_min(0))
    train = RoadScenePairs(args.data_root, 'train')
    val = RoadScenePairs(args.data_root, 'val')
    inference_cache(model, train, 'train', args.output/'cache_train', device)
    inference_cache(model, val, 'val', args.output/'cache_val', device)
    search = rule_search(args.output/'cache_train')
    (args.output/'train_rule_search.json').write_text(json.dumps(search, indent=2))
    print('Selected train-only rule:', search['chosen'], flush=True)
    rows = val_evaluation(val, args.output/'cache_val', search['chosen'])
    write_csv(args.output/'val_per_image.csv', rows)
    # Local CuPy kernels compile with torch's CUDA DLLs. This changes only
    # the compiler scratch location; no model code, state or results.
    exact = exact_fine_coverage(model, val, device)
    glunet.MutualMatching = original_matching
    write_csv(args.output/'val_exact_fine_coverage.csv', exact)
    groups = {'all23': lambda n: True,
              'other21': lambda n: n not in ('000002', '000014'),
              'hard2': lambda n: n in ('000002', '000014'),
              '000002': lambda n: n == '000002',
              '000014': lambda n: n == '000014'}
    summary = dict(test_pairs_accessed=False, model_weights_changed=False,
                   direction='VI target -> IR source',
                   base_coarse_sha256=sha256_file(args.coarse_checkpoint),
                   base_fine_sha256=sha256_file(args.fine_checkpoint),
                   train_rule=search['chosen'], groups={})
    for name, predicate in groups.items():
        subset = [r for r in rows if predicate(r['name'])]
        fine_subset = [r for r in exact if predicate(r['name'])]
        summary['groups'][name] = {
            'coarse': {label: summarize_epe(subset, label)
                       for label in ('original', 'predicted_fit', 'oracle_gt_fit')},
            'coarse_start_coverage': {
                label: {str(size): summarize_coverage(subset, label+'_', size)
                        for size in (32, 64, 128, 256)}
                for label in ('original', 'predicted_fit', 'oracle_gt_fit')},
            'existing_fine_start_coverage': {
                str(size): summarize_coverage(fine_subset, '', size)
                for size in (128, 256)},
        }
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({'train_rule': search['chosen'],
                      'hard2': summary['groups']['hard2'],
                      'other21': summary['groups']['other21']}, indent=2), flush=True)


if __name__ == '__main__':
    # Only the managed local sandbox needs a pre-existing CuPy scratch folder.
    # Normal server runs keep Python's original TemporaryDirectory behavior.
    if os.environ.get('REG_FIXED_TEMP'):
        scratch = Path(os.environ['REG_FIXED_TEMP'])
        if not scratch.is_dir():
            raise FileNotFoundError(scratch)
        @contextlib.contextmanager
        def fixed_temp(*_args, **_kwargs):
            yield str(scratch)
        tempfile.TemporaryDirectory = fixed_temp
    torch_lib = Path(torch.__file__).parent/'lib'
    if hasattr(os, 'add_dll_directory'):
        _dll_handle = os.add_dll_directory(str(torch_lib))
    main()
