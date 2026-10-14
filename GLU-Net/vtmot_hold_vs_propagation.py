"""Frozen-keyframe VTMOT control: dual-modal propagation versus flow hold.

`check` uses synthetic flow fields only. `eval` reads the four validation
sequences; the dataset class has no test-split mode. The frozen keyframe list
is a *conditional* protocol, not a deployable online keyframe selector.

Both arms use exactly the same A1 prediction on each frozen keyframe. On a
non-keyframe, propagation composes current VI -> previous VI -> previous IR ->
current IR, while hold reuses the most recent A1 flow at the same pixel
coordinates. Neither arm uses GT to produce a flow or choose a keyframe.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from datasets.vtmot_video import VTMOTVideos
from vtmot_geometry import backward_warp, grid, propagate
from vtmot_keyframe import (install_mutual_guard, load_a1, predict_a1,
                            save_json, sha256, temporal_pair_fast,
                            temporal_quality, write_rows)
from vtmot_local_correction_ab import A1_SHA, SPLIT_NAMES, TRIGGER


RECIPE = 'vtmot_frozen_165_hold_vs_dual_modal_propagation_v1'
HEIGHT = WIDTH = 512
EXPECTED_FRAMES = 800
EXPECTED_KEYS = 165


def hold_keyframe_flow(key_flow, shape=(HEIGHT, WIDTH)):
    """Keep the last A1 VI->IR flow, checking its current IR source support."""
    if key_flow.shape != (*shape, 2) or not np.isfinite(key_flow).all():
        raise ValueError('keyframe flow must be finite 512px XY flow')
    coordinates = grid(shape) + key_flow
    valid = (np.isfinite(coordinates).all(axis=-1) &
             (coordinates[..., 0] >= 0) & (coordinates[..., 0] <= shape[1] - 1) &
             (coordinates[..., 1] >= 0) & (coordinates[..., 1] <= shape[0] - 1))
    # Do not replace invalid values or use this mask to change the GT denominator.
    return key_flow.copy(), valid


def load_schedule(path, split_file, sequences):
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('a1_sha256') != A1_SHA or payload.get('frames') != EXPECTED_FRAMES:
        raise ValueError('frozen schedule weight or frame count differs')
    if payload.get('split_sha256') and payload['split_sha256'] != sha256(split_file):
        raise ValueError('frozen schedule split hash differs')
    if set(payload['keyframes']) != set(sequences):
        raise ValueError('frozen schedule sequence names differ')
    schedule = {}
    for sequence, stems in sequences.items():
        selected = payload['keyframes'][sequence]
        if len(selected) != len(set(selected)) or not selected or selected[0] != stems[0]:
            raise ValueError(f'invalid keyframe start or duplicate: {sequence}')
        if not set(selected).issubset(stems):
            raise ValueError(f'unknown keyframe in {sequence}')
        schedule[sequence] = set(selected)
    if sum(map(len, schedule.values())) != EXPECTED_KEYS:
        raise ValueError(f'expected exactly {EXPECTED_KEYS} frozen keyframes')
    return payload, schedule


def synthetic_check():
    shape = (9, 11)
    key = np.zeros((*shape, 2), np.float32)
    key[..., 0] = 2
    key[..., 1] = -1
    held, valid = hold_keyframe_flow(key, shape)
    if not np.array_equal(held, key) or valid.sum() != (shape[0] - 1) * (shape[1] - 2):
        raise AssertionError('hold changed flow or current-source bounds')
    held[0, 0] = 99
    if key[0, 0, 0] != 2:
        raise AssertionError('held result aliases the stored keyframe flow')
    zero = np.zeros_like(key)
    shifted = np.zeros_like(key)
    shifted[..., 0] = 1
    composed, inside = propagate(shifted, key, zero)
    core = np.s_[2:7, 2:7]
    if not inside[core].all() or not np.array_equal(composed[core], shifted[core] + key[core]):
        raise AssertionError('VI->IR composition direction or units differ')
    # A source-invalid prediction still occupies its GT-valid evaluation pixel.
    gt_valid = np.ones(shape, bool)
    if (gt_valid & ~valid).sum() != shape[0] * shape[1] - valid.sum():
        raise AssertionError('prediction mask changed the GT denominator')
    return {'hold_is_last_keyframe_flow': 'passed',
            'current_ir_bounds': 'passed', 'propagation_xy_512px': 'passed',
            'fixed_gt_denominator': 'passed', 'test_accessed': False}


def row_metrics(frame, truth, gt_valid, flow, predicted_valid, keyframe, age,
                timing, input_ms, warp_ms, source_inside):
    if flow.shape != truth.shape or predicted_valid.shape != gt_valid.shape:
        raise ValueError('prediction and GT grid differ')
    if not np.isfinite(flow).all() or not gt_valid.any():
        raise ValueError('nonfinite flow or empty fixed GT mask')
    error = np.linalg.norm(flow - truth, axis=-1)
    return {'sequence': frame['sequence'], 'stem': frame['stem'],
            'keyframe': int(keyframe), 'age_since_keyframe': age,
            'gt_valid_pixels': int(gt_valid.sum()),
            'predicted_valid_pixels_on_gt': int((predicted_valid & gt_valid).sum()),
            'predicted_invalid_pixels_on_gt': int((~predicted_valid & gt_valid).sum()),
            'prediction_valid_fraction': float(predicted_valid.mean()),
            'source_outside_pixels_on_gt': int((~source_inside & gt_valid).sum()),
            'epe_512px': float(error[gt_valid].mean()),
            'frame_ms': float(sum(timing.values())),
            'input_ms': float(input_ms), 'warp_ms': float(warp_ms),
            'registration_ms': float(input_ms + sum(timing.values()) + warp_ms),
            **{f'{name}_ms': float(value) for name, value in timing.items()}}


def weighted_epe(rows):
    pixels = sum(row['gt_valid_pixels'] for row in rows)
    return (sum(row['epe_512px'] * row['gt_valid_pixels'] for row in rows) / pixels
            if pixels else None)


def summarize(rows):
    total = sum(row['gt_valid_pixels'] for row in rows)
    nonkeys = [row for row in rows if not row['keyframe']]
    names = sorted({row['sequence'] for row in rows})
    return {'frames': len(rows), 'keyframes': len(rows) - len(nonkeys),
            'gt_valid_pixels': total, 'pixel_weighted_epe_512px': weighted_epe(rows),
            'nonkey_pixel_weighted_epe_512px': weighted_epe(nonkeys),
            'prediction_invalid_on_gt_fraction':
                sum(row['predicted_invalid_pixels_on_gt'] for row in rows) / total,
            'total_compute_seconds': sum(row['frame_ms'] for row in rows) / 1000,
            'total_registration_seconds':
                sum(row['registration_ms'] for row in rows) / 1000,
            'input_seconds': sum(row['input_ms'] for row in rows) / 1000,
            'warp_seconds': sum(row['warp_ms'] for row in rows) / 1000,
            'compute_components_seconds': {
                field: sum(row.get(field, 0.) for row in rows) / 1000
                for field in ('a1_ms', 'temporal_ms', 'geometry_ms', 'hold_ms')},
            'by_sequence': {name: summarize_group([r for r in rows if r['sequence'] == name])
                            for name in names},
            'by_age': {str(age): summarize_group([r for r in rows
                                                 if r['age_since_keyframe'] == age])
                       for age in sorted({r['age_since_keyframe'] for r in rows})},
            'worst_nonkey_frames': [
                {'sequence': r['sequence'], 'stem': r['stem'],
                 'age_since_keyframe': r['age_since_keyframe'],
                 'epe_512px': r['epe_512px']}
                for r in sorted(nonkeys, key=lambda item: -item['epe_512px'])[:10]]}


def summarize_group(rows):
    pixels = sum(row['gt_valid_pixels'] for row in rows)
    nonkeys = [row for row in rows if not row['keyframe']]
    return {'frames': len(rows), 'nonkey_frames': len(nonkeys),
            'gt_valid_pixels': pixels, 'pixel_weighted_epe_512px': weighted_epe(rows),
            'nonkey_pixel_weighted_epe_512px': weighted_epe(nonkeys),
            'prediction_invalid_on_gt_fraction':
                sum(r['predicted_invalid_pixels_on_gt'] for r in rows) / pixels
                if pixels else None,
            'total_compute_seconds': sum(row['frame_ms'] for row in rows) / 1000,
            'total_registration_seconds':
                sum(row['registration_ms'] for row in rows) / 1000}


def evaluate(dataset, schedule, model, device):
    arms = {'dual_modal_propagation': [], 'hold_last_keyframe': []}
    paired = []
    wall_start = time.perf_counter()
    for sequence, stems in dataset.sequences.items():
        previous_frame = previous_flow = previous_valid = last_key_flow = None
        age = 0
        for index, stem in enumerate(stems):
            input_start = time.perf_counter()
            frame = dataset.input_frame(sequence, stem)
            input_ms = (time.perf_counter() - input_start) * 1000
            is_keyframe = stem in schedule[sequence]
            if index == 0 and not is_keyframe:
                raise AssertionError('each sequence must begin with A1')
            temporal_ms = geometry_ms = a1_ms = hold_ms = 0.
            quality = None
            if index and not is_keyframe:
                # Given a frozen list, neither arm needs temporal motion on a
                # keyframe. This is conditional execution cost, not the cost
                # of generating the list online.
                temporal_start = time.perf_counter()
                temporal = temporal_pair_fast(previous_frame, frame)
                temporal_ms = (time.perf_counter() - temporal_start) * 1000
                geometry_start = time.perf_counter()
                quality = temporal_quality(temporal, previous_flow, previous_valid,
                                           TRIGGER['fb_pixel_threshold'])
                geometry_ms = (time.perf_counter() - geometry_start) * 1000
            if is_keyframe:
                a1_start = time.perf_counter()
                fresh_flow, fresh_valid, _ = predict_a1(model, frame, device)
                a1_ms = (time.perf_counter() - a1_start) * 1000
                propagation_flow, propagation_valid = fresh_flow, fresh_valid
                hold_flow, hold_valid = fresh_flow, fresh_valid
                last_key_flow = fresh_flow.copy()
                age = 0
            else:
                propagation_flow, propagation_valid = quality['flow'], quality['valid']
                hold_start = time.perf_counter()
                hold_flow, hold_valid = hold_keyframe_flow(last_key_flow)
                hold_ms = (time.perf_counter() - hold_start) * 1000
                age += 1
            if is_keyframe and not np.array_equal(propagation_flow, hold_flow):
                raise AssertionError(f'keyframe flows differ: {sequence}/{stem}')
            warp_start = time.perf_counter()
            _, propagation_inside = backward_warp(frame['ir'], propagation_flow)
            propagation_warp_ms = (time.perf_counter() - warp_start) * 1000
            warp_start = time.perf_counter()
            _, hold_inside = backward_warp(frame['ir'], hold_flow)
            hold_warp_ms = (time.perf_counter() - warp_start) * 1000
            if not np.array_equal(hold_inside, hold_valid):
                raise AssertionError('held flow source-valid mask differs from image warp')
            # GT is read only after both predictions and the key decision.
            truth, gt_valid = dataset.flow_truth(sequence, stem, frame['original_hw'])
            timing = {'a1': a1_ms, 'temporal': temporal_ms, 'geometry': geometry_ms}
            a_row = row_metrics(frame, truth, gt_valid, propagation_flow, propagation_valid,
                                is_keyframe, age, timing, input_ms,
                                propagation_warp_ms, propagation_inside)
            b_row = row_metrics(frame, truth, gt_valid, hold_flow, hold_valid,
                                is_keyframe, age, {'a1': a1_ms, 'hold': hold_ms},
                                input_ms, hold_warp_ms, hold_inside)
            if a_row['gt_valid_pixels'] != b_row['gt_valid_pixels']:
                raise AssertionError('A/B GT denominators differ')
            arms['dual_modal_propagation'].append(a_row)
            arms['hold_last_keyframe'].append(b_row)
            paired.append({'sequence': sequence, 'stem': stem,
                           'keyframe': int(is_keyframe), 'age_since_keyframe': age,
                           'gt_valid_pixels': a_row['gt_valid_pixels'],
                           'propagation_epe_512px': a_row['epe_512px'],
                           'hold_epe_512px': b_row['epe_512px'],
                           'hold_minus_propagation_512px':
                               b_row['epe_512px'] - a_row['epe_512px'],
                           'propagation_invalid_on_gt':
                               a_row['predicted_invalid_pixels_on_gt'],
                           'hold_invalid_on_gt': b_row['predicted_invalid_pixels_on_gt']})
            previous_frame, previous_flow, previous_valid = (
                frame, propagation_flow, propagation_valid)
        print(f'{sequence}: {len(stems)} frames', flush=True)
    return arms, paired, time.perf_counter() - wall_start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('check', 'eval'))
    for name in ('data-root', 'split-file', 'pretrained', 'coarse-checkpoint',
                 'source-a1-checkpoint', 'source-coarse-adapt-checkpoint',
                 'a1-checkpoint', 'frozen-schedule', 'output'):
        parser.add_argument(f'--{name}', type=Path)
    args = parser.parse_args()
    checked = synthetic_check()
    if args.stage == 'check':
        print(json.dumps(checked, indent=2), flush=True)
        return
    for name in ('data_root', 'split_file', 'pretrained', 'coarse_checkpoint',
                 'source_a1_checkpoint', 'source_coarse_adapt_checkpoint',
                 'a1_checkpoint', 'frozen_schedule', 'output'):
        if getattr(args, name) is None:
            parser.error(f'--{name.replace("_", "-")} is required')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be a new, empty directory')
    if sha256(args.a1_checkpoint) != A1_SHA:
        raise ValueError('frozen A1 SHA-256 mismatch')
    install_mutual_guard()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_a1(args, device)
    dataset = VTMOTVideos(args.data_root, args.split_file, split='eval')
    if tuple(dataset.sequences) != SPLIT_NAMES or sum(map(len, dataset.sequences.values())) != EXPECTED_FRAMES:
        raise ValueError('expected only the four locked validation sequences / 800 frames')
    schedule_payload, schedule = load_schedule(args.frozen_schedule,
                                               args.split_file, dataset.sequences)
    # Warm the frozen A1 once, outside both arm timings.
    first = next(iter(dataset.sequences))
    predict_a1(model, dataset.input_frame(first, dataset.sequences[first][0]), device)
    arms, paired, wall_seconds = evaluate(dataset, schedule, model, device)
    summaries = {name: summarize(rows) for name, rows in arms.items()}
    reference = summaries['dual_modal_propagation']
    if (reference['gt_valid_pixels'] != schedule_payload['historical_gt_valid_pixels'] or
            abs(reference['pixel_weighted_epe_512px'] -
                schedule_payload['historical_fixed_fast_epe_512px']) > .01):
        raise RuntimeError('propagation arm does not reproduce the frozen reference')
    a, b = arms.values()
    if [(r['sequence'], r['stem'], r['keyframe'], r['gt_valid_pixels']) for r in a] != [
            (r['sequence'], r['stem'], r['keyframe'], r['gt_valid_pixels']) for r in b]:
        raise AssertionError('paired frame/schedule/GT mask mismatch')
    key_differences = [abs(r['hold_minus_propagation_512px']) for r in paired if r['keyframe']]
    if any(difference != 0. for difference in key_differences):
        raise AssertionError('A/B keyframe EPE differs')
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in arms.items():
        write_rows(args.output / f'{name}_per_frame.csv', rows)
    write_rows(args.output / 'paired_per_frame.csv', paired)
    nonkeys = [r for r in paired if not r['keyframe']]
    nonkey_pixels = sum(r['gt_valid_pixels'] for r in nonkeys)
    report = {'recipe': RECIPE, 'split': 'eval', 'test_accessed': False,
              'a1_sha256': A1_SHA, 'split_sha256': sha256(args.split_file),
              'frozen_schedule_sha256': sha256(args.frozen_schedule),
              'schedule_provenance': schedule_payload.get('source'),
              'schedule_creation_cost_seconds': None,
              'schedule_creation_cost_note':
                  'Historical schedule generation is excluded. This conditional '
                  'run cannot establish deployable online speedup.',
              'flow_direction': 'current VI target -> current IR source, XY',
              'flow_units': '512x512 image pixels',
              'mask': 'fixed per-frame GT valid mask; predicted-invalid pixels remain in EPE',
              'timing_scope': 'compute-only: common A1 call on frozen '
                              'keyframes; four 256-grid Farneback flows and '
                              'geometric quality on propagation non-keyframes; '
                              'or hold and current IR bounds on hold non-keyframes; '
                              'registration time also includes shared image '
                              'read/preprocessing and each arm image warp; '
                              'both exclude GT/metrics, schedule creation and '
                              'shared warm-up. Unlike the old online run, '
                              'keyframe temporal diagnostics are skipped.',
              'paired_wall_seconds_including_io_and_metrics': wall_seconds,
              'arms': summaries,
              'nonkey_hold_minus_propagation_weighted_epe_512px':
                  sum(r['hold_minus_propagation_512px'] * r['gt_valid_pixels']
                      for r in nonkeys) / nonkey_pixels,
              'nonkey_improved_by_hold': sum(r['hold_minus_propagation_512px'] < 0
                                             for r in nonkeys),
              'nonkey_worsened_by_hold': sum(r['hold_minus_propagation_512px'] > 0
                                             for r in nonkeys),
              'worst_hold_regressions': sorted(
                  nonkeys, key=lambda r: -r['hold_minus_propagation_512px'])[:10],
              'synthetic_check': checked}
    save_json(args.output / 'report.json', report)
    print(json.dumps({'frames': len(paired), 'keyframes': EXPECTED_KEYS,
                      'propagation_epe_512px': reference['pixel_weighted_epe_512px'],
                      'hold_epe_512px': summaries['hold_last_keyframe']['pixel_weighted_epe_512px'],
                      'propagation_compute_seconds': reference['total_compute_seconds'],
                      'hold_compute_seconds': summaries['hold_last_keyframe']['total_compute_seconds'],
                      'propagation_registration_seconds': reference['total_registration_seconds'],
                      'hold_registration_seconds':
                          summaries['hold_last_keyframe']['total_registration_seconds']},
                     indent=2), flush=True)


if __name__ == '__main__':
    main()
