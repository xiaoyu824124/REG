"""Read-only, full-validation diagnosis of frozen RoadScene A1 on VTMOT."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from datasets.vtmot_video import VTMOTVideos
from vtmot_geometry import backward_warp
from vtmot_a1_adapt import load_a1, save_json
from roadscene_coarse import sha256_file as sha256
import models.our_models.GLUNet as glunet


LEVELS = ('zero', 'coarse16', 'local32', 'local64', 'final128', 'final512')
WINDOWS = (32, 64, 128)


def inputs(frame, device):
    source = torch.from_numpy(frame['ir'].transpose(2, 0, 1).copy())[None].to(device)
    target = torch.from_numpy(frame['vi'].transpose(2, 0, 1).copy())[None].to(device)
    return source, target


@torch.inference_mode()
def stages_and_centres(model, frame, device):
    source_raw, target_raw = inputs(frame, device)
    source, target, source256, target256, *_ = model.base.pre_process_data(
        source_raw, target_raw, device=device)
    flow256, flow512 = model.base(target, source, target256, source256)
    coarse, local32 = flow256
    local64, final128 = flow512
    def full(flow, factor):
        return (F.interpolate(flow, (512, 512), mode='bilinear',
                              align_corners=False) * factor)[0].permute(1, 2, 0).cpu().numpy()
    predictions = {'coarse16': full(coarse, 2.),
                   'local32': full(local32, 2.),
                   'local64': full(local64, 1.),
                   'final128': full(final128, 1.),
                   'final512': full(final128, 1.)}
    # Exactly the warp centres used before each of the three local 9x9
    # correlations. All are converted to 512-image-pixel units.
    centres = {32: model.base.deconv4(coarse) * 2,
               64: F.interpolate(local32, (64, 64), mode='bilinear',
                                  align_corners=False) * 2,
               128: model.base.deconv2(local64)}
    centres = {size: value[0].permute(1, 2, 0).cpu().numpy()
               for size, value in centres.items()}
    native128 = final128[0].permute(1, 2, 0).cpu().numpy()
    return predictions, centres, native128


def vector_stats(pred, truth, valid):
    error = np.linalg.norm(pred-truth, axis=-1)
    magnitude = np.linalg.norm(pred, axis=-1)
    gt_magnitude = np.linalg.norm(truth, axis=-1)
    direction_mask = valid & (magnitude >= 1.) & (gt_magnitude >= 1.)
    cosine = np.sum(pred*truth, axis=-1) / np.maximum(magnitude*gt_magnitude, 1e-6)
    return {'epe_512px': float(np.mean(error[valid])),
            'pred_magnitude_mean_512px': float(np.mean(magnitude[valid])),
            'gt_magnitude_mean_512px': float(np.mean(gt_magnitude[valid])),
            'direction_pixels': int(direction_mask.sum()),
            'direction_cosine_mean': float(np.mean(cosine[direction_mask]))
                if direction_mask.any() else None,
            'direction_positive_fraction': float(np.mean(cosine[direction_mask] > 0))
                if direction_mask.any() else None}


def pooled_truth(truth, valid, size, device):
    flow = torch.from_numpy(truth.transpose(2, 0, 1).copy())[None].to(device)
    mask = torch.from_numpy(valid.copy())[None, None].to(device).float()
    weight = F.interpolate(mask, (size, size), mode='area')
    average = F.interpolate(flow*mask, (size, size), mode='area') / weight.clamp_min(1e-6)
    return (average[0].permute(1, 2, 0).cpu().numpy(),
            (weight[0, 0] > .8).cpu().numpy())


def coverage(centre, truth, valid, size, device):
    gt, selected = pooled_truth(truth, valid, size, device)
    # md=4 => 9x9 local search, radius four feature-grid cells.
    radius_512px = 4 * (512 / size)
    offset = gt-centre
    inside = selected & (np.max(np.abs(offset), axis=-1) <= radius_512px)
    return {'queries': int(selected.sum()), 'inside_queries': int(inside.sum()),
            'coverage': float(inside.sum()/selected.sum()) if selected.any() else None,
            'radius_grid_cells': 4, 'radius_512px_each_axis': radius_512px,
            'outside_epe_before_512px': float(np.linalg.norm(offset, axis=-1)[selected & ~inside].mean())
                if np.any(selected & ~inside) else None}


def photometric_gt_check(frame, aligned, gt_flow, valid):
    predicted, bounds = backward_warp(aligned, gt_flow)
    selected = valid & bounds
    target = frame['vi'].astype(np.float32)
    zero = aligned.astype(np.float32)
    registered = predicted.astype(np.float32)
    return {'gt_warp_mse_255': float(np.mean((registered[selected]-target[selected])**2)),
            'identity_warp_mse_255': float(np.mean((zero[selected]-target[selected])**2)),
            'gt_valid_pixels': int(valid.sum()),
            'gt_valid_fraction': float(valid.mean()),
            'photometric_pixels': int(selected.sum())}


def accumulate(rows):
    output = {'frames': len(rows),
              'valid_pixels': sum(int(r['valid_pixels']) for r in rows)}
    denominator = output['valid_pixels']
    output['stages'] = {}
    for level in LEVELS:
        output['stages'][level] = {
            'epe_512px': sum(r[f'{level}_epe_512px']*r['valid_pixels'] for r in rows)/denominator,
            'pred_magnitude_mean_512px': sum(
                r[f'{level}_pred_magnitude_mean_512px']*r['valid_pixels'] for r in rows)/denominator,
            'gt_magnitude_mean_512px': sum(
                r[f'{level}_gt_magnitude_mean_512px']*r['valid_pixels'] for r in rows)/denominator}
        direction_count = sum(r[f'{level}_direction_pixels'] for r in rows)
        output['stages'][level]['direction_pixels'] = direction_count
        for field in ('direction_cosine_mean', 'direction_positive_fraction'):
            output['stages'][level][field] = (
                sum(r[f'{level}_{field}']*r[f'{level}_direction_pixels'] for r in rows
                    if r[f'{level}_{field}'] is not None)/direction_count
                if direction_count else None)
    output['windows'] = {}
    for size in WINDOWS:
        queries = sum(r[f'window{size}_queries'] for r in rows)
        inside = sum(r[f'window{size}_inside_queries'] for r in rows)
        output['windows'][str(size)] = {'queries': queries, 'inside_queries': inside,
                                        'coverage': inside/queries if queries else None,
                                        'radius_512px_each_axis': 4*512/size}
    output['gt_check'] = {
        key: sum(r[key]*r['photometric_pixels'] for r in rows) /
             sum(r['photometric_pixels'] for r in rows)
        for key in ('gt_warp_mse_255', 'identity_warp_mse_255')}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--split-file', required=True, type=Path)
    parser.add_argument('--pretrained', required=True, type=Path)
    parser.add_argument('--coarse-checkpoint', required=True, type=Path)
    parser.add_argument('--a1-checkpoint', required=True, type=Path)
    parser.add_argument('--baseline-csv', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--max-sequences', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=0)
    args = parser.parse_args()
    if (args.output / 'report.json').exists():
        raise FileExistsError(f'diagnosis already exists: {args.output}')
    # A1 was trained and the locked 800-frame reference was evaluated with
    # this nonnegative-correlation guard around MutualMatching.
    original_mutual = glunet.MutualMatching
    glunet.MutualMatching = lambda corr: original_mutual(corr.clamp_min(0))
    all_zero = glunet.MutualMatching(torch.zeros(1, 1, 16, 16, 16, 16))
    if not torch.isfinite(all_zero).all() or torch.count_nonzero(all_zero):
        raise RuntimeError('all-zero MutualMatching query is unsafe')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = VTMOTVideos(args.data_root, args.split_file, 'eval',
                          max_sequences=args.max_sequences, max_frames=args.max_frames)
    model = load_a1(args, device)
    with args.baseline_csv.open(newline='', encoding='utf-8-sig') as stream:
        baseline = {(r['sequence'], r['stem']): r for r in csv.DictReader(stream)}
    rows = []
    for sequence, stems in dataset.sequences.items():
        for stem in stems:
            frame = dataset.input_frame(sequence, stem)
            predictions, centres, native128 = stages_and_centres(model, frame, device)
            truth, valid, aligned = dataset.evaluation_truth(sequence, stem,
                                                               frame['original_hw'])
            zero = np.zeros_like(truth)
            all_predictions = {'zero': zero, **predictions}
            row = {'sequence': sequence, 'stem': stem, 'valid_pixels': int(valid.sum())}
            for level, pred in all_predictions.items():
                row.update({f'{level}_{key}': value for key, value in
                            vector_stats(pred, truth, valid).items()})
            for size, centre in centres.items():
                row.update({f'window{size}_{key}': value for key, value in
                            coverage(centre, truth, valid, size, device).items()})
            native_gt, native_mask = pooled_truth(truth, valid, 128, device)
            row['final128_native_query_epe_512px'] = float(np.linalg.norm(
                native128-native_gt, axis=-1)[native_mask].mean())
            row.update(photometric_gt_check(frame, aligned, truth, valid))
            baseline_row = baseline[(sequence, stem)]
            if row['valid_pixels'] != int(baseline_row['gt_valid_pixels']):
                raise RuntimeError(f'GT valid pixel count differs from locked baseline: {sequence}/{stem}')
            if abs(row['final512_epe_512px']-float(baseline_row['epe_512px'])) > .002:
                raise RuntimeError(f'final A1 flow differs from locked baseline: {sequence}/{stem}')
            rows.append(row)
        print(f'{sequence}: {len(stems)} frames, '
              f'coarse={accumulate([r for r in rows if r["sequence"] == sequence])["stages"]["coarse16"]["epe_512px"]:.3f}, '
              f'final={accumulate([r for r in rows if r["sequence"] == sequence])["stages"]["final512"]["epe_512px"]:.3f}',
              flush=True)
    if not args.max_sequences and not args.max_frames and len(rows) != len(baseline):
        raise RuntimeError('diagnosis and locked baseline frame sets differ')
    output = {'split': 'eval', 'frames': len(rows), 'a1_sha256': sha256(args.a1_checkpoint),
              'pretrained_sha256': sha256(args.pretrained),
              'coarse_sha256': sha256(args.coarse_checkpoint),
              'flow_unit': '512x512 image pixels, XY, VI target to IR source',
              'full_512_mask': 'GT mapping inside 512x512 source; identical at every stage',
              'window_mask': 'GT valid area-pooled to grid; keep queries with >80% valid fraction',
              'native_128_note': 'The 128-grid prediction is the final flow upsampled to 512; '
                                 'native-query EPE has a different query denominator and is separate.',
              'all': accumulate(rows),
              'per_sequence': {name: accumulate([r for r in rows if r['sequence'] == name])
                               for name in dataset.sequences},
              'test_accessed': False}
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'per_frame.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(args.output / 'report.json', output)
    print(json.dumps({'all': output['all'], 'per_sequence': output['per_sequence']},
                     indent=2), flush=True)


if __name__ == '__main__':
    main()
