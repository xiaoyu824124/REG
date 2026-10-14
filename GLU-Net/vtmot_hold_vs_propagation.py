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
from vtmot_geometry import backward_warp, grid, propagate, sample
from vtmot_keyframe import (edge_metrics, gray, install_mutual_guard, lncc,
                            load_a1, mse_ncc, nmi, predict_a1, save_ghost,
                            save_json, sha256, t_ssim, temporal_pair_fast,
                            temporal_quality, write_rows)
from vtmot_local_correction_ab import A1_SHA, SPLIT_NAMES, TRIGGER


RECIPE = 'vtmot_frozen_165_hold_vs_dual_modal_propagation_metrics_v2'
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
    image = np.tile(np.arange(32, dtype=np.uint8)[None, :, None], (32, 1, 3)) * 7
    frame = {'vi': image, 'ir': image}
    zero32 = np.zeros((32, 32, 2), np.float32)
    all32 = np.ones((32, 32), bool)
    spatial, residual = image_metrics(frame, zero32, all32, image, image, image)
    temporal32 = {'vi': zero32, 'vi_bounds': all32,
                  'vi_error': np.zeros((32, 32), np.float32)}
    common = temporal_metric_mask(all32, all32, temporal32)
    temporal_result = temporal_image_metrics(image, image, residual, residual,
                                             temporal32, common)
    if (not all(np.isfinite(value) for value in spatial.values()) or
            not common.all() or temporal_result['temporal_common_pixels'] != 1024 or
            temporal_result['motion_compensated_ir_mae_0to1'] != 0. or
            temporal_result['motion_compensated_alignment_residual_change_0to1'] != 0.):
        raise AssertionError('image/temporal diagnostic synthetic check failed')
    test_frame = {**frame, 'sequence': 'synthetic', 'stem': '000001'}
    row = row_metrics(test_frame, zero32, all32, zero32, all32, False, 1,
                      {'hold': 1.}, 2., 3., all32)
    row.update({f'pck_{radius}_pixels': 1024 for radius in (1, 3, 5)})
    row.update(spatial)
    row.update(region_metrics(np.zeros((32, 32), np.float32), all32,
                              temporal32, None))
    row.update(temporal_result)
    summary = summarize([row])
    if (summary['pck_512px']['1'] != 1. or
            summary['motion_compensated_alignment_residual_change_0to1'] != 0.):
        raise AssertionError('summary metric aggregation failed')
    return {'hold_is_last_keyframe_flow': 'passed',
            'current_ir_bounds': 'passed', 'propagation_xy_512px': 'passed',
            'fixed_gt_denominator': 'passed',
            'image_and_temporal_metrics': 'passed', 'test_accessed': False}


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


def image_metrics(frame, flow, gt_valid, visible_gt, registered_ir,
                  previous_registered):
    """Evaluation-only image measures; neither arm can read visible_gt."""
    registered_gt, _ = backward_warp(visible_gt, flow)
    retention, edge_f1 = edge_metrics(frame['vi'], registered_ir, gt_valid)
    same_mse, same_ncc = mse_ncc(frame['vi'], registered_gt, gt_valid)
    result = {'nmi_2mi_over_entropy_sum': nmi(frame['vi'], registered_ir, gt_valid),
              'edge_information_retention_proxy': retention,
              'edge_overlap_f1_3x3_proxy': edge_f1,
              'lncc17_squared_proxy': lncc(frame['vi'], registered_ir, gt_valid),
              'visible_gt_mse_0to1': same_mse,
              'visible_gt_ncc': same_ncc}
    if previous_registered is not None:
        result['raw_itf_255'] = float(np.mean(np.abs(
            gray(registered_ir).astype(np.float32) -
            gray(previous_registered).astype(np.float32))))
        result['raw_t_ssim'] = t_ssim(previous_registered, registered_ir)
    residual = (gray(registered_gt).astype(np.float32) -
                gray(frame['vi']).astype(np.float32))
    return result, residual


def temporal_metric_mask(gt_valid, previous_gt_valid, temporal):
    """Same GT/VI-motion support for both arms, independent of their output."""
    previous_at_current, _ = sample(previous_gt_valid.astype(np.float32),
                                    grid(gt_valid.shape) + temporal['vi'])
    return (gt_valid & (previous_at_current > .999) &
            temporal['vi_bounds'] &
            (temporal['vi_error'] <= TRIGGER['fb_pixel_threshold']))


def temporal_image_metrics(registered, previous_registered, residual,
                           previous_residual, temporal, common):
    """Motion-compensated image and alignment-residual change proxies."""
    pixels = int(common.sum())
    if not pixels:
        return {'temporal_common_pixels': 0,
                'motion_compensated_ir_mae_0to1': None,
                'motion_compensated_alignment_residual_change_0to1': None}
    previous_ir_at_current, _ = backward_warp(previous_registered, temporal['vi'])
    previous_residual_at_current, _ = sample(previous_residual, grid(common.shape) +
                                            temporal['vi'])
    ir_change = np.abs(gray(registered).astype(np.float32) -
                       gray(previous_ir_at_current).astype(np.float32))
    alignment_change = np.abs(residual - previous_residual_at_current)
    return {'temporal_common_pixels': pixels,
            'motion_compensated_ir_mae_0to1': float(ir_change[common].mean()/255.),
            'motion_compensated_alignment_residual_change_0to1':
                float(alignment_change[common].mean()/255.)}


def region_metrics(error, gt_valid, temporal, quality):
    if temporal is None:
        return {'moving_pixels': 0, 'moving_epe_512px': None,
                'temporal_low_confidence_pixels': 0,
                'temporal_low_confidence_epe_512px': None}
    moving = gt_valid & (np.linalg.norm(temporal['vi'], axis=-1) >= 3.)
    # The historical propagation confidence region is defined only on
    # non-keyframes. It is a shared diagnostic region, not true occlusion GT.
    low_confidence = (gt_valid & ~quality['valid'] if quality is not None
                      else np.zeros_like(gt_valid))
    return {'moving_pixels': int(moving.sum()),
            'moving_epe_512px': float(error[moving].mean()) if moving.any() else None,
            'temporal_low_confidence_pixels': int(low_confidence.sum()),
            'temporal_low_confidence_epe_512px':
                float(error[low_confidence].mean()) if low_confidence.any() else None}


def weighted_epe(rows):
    pixels = sum(row['gt_valid_pixels'] for row in rows)
    return (sum(row['epe_512px'] * row['gt_valid_pixels'] for row in rows) / pixels
            if pixels else None)


def weighted_metric(rows, value_key, count_key):
    applicable = [r for r in rows if r.get(value_key) is not None and r.get(count_key, 0)]
    pixels = sum(int(r[count_key]) for r in applicable)
    return (sum(float(r[value_key]) * int(r[count_key]) for r in applicable) / pixels
            if pixels else None)


def summarize(rows):
    total = sum(row['gt_valid_pixels'] for row in rows)
    nonkeys = [row for row in rows if not row['keyframe']]
    names = sorted({row['sequence'] for row in rows})
    result = summarize_group(rows)
    result.update({'keyframes': len(rows) - len(nonkeys),
                   'input_seconds': sum(row['input_ms'] for row in rows) / 1000,
                   'warp_seconds': sum(row['warp_ms'] for row in rows) / 1000,
                   'compute_components_seconds': {
                       field: sum(row.get(field, 0.) for row in rows) / 1000
                       for field in ('a1_ms', 'temporal_ms', 'geometry_ms', 'hold_ms')},
                   'by_sequence': {
                       name: summarize_group([r for r in rows if r['sequence'] == name])
                       for name in names},
                   'by_age': {
                       str(age): summarize_group([r for r in rows
                                                  if r['age_since_keyframe'] == age])
                       for age in sorted({r['age_since_keyframe'] for r in rows})},
                   'nonkey': summarize_group(nonkeys),
                   'keyframe_switch': summarize_group([
                       r for r in rows if r['keyframe'] and 'raw_itf_255' in r]),
                   'worst_nonkey_frames': [
                       {'sequence': r['sequence'], 'stem': r['stem'],
                        'age_since_keyframe': r['age_since_keyframe'],
                        'epe_512px': r['epe_512px']}
                       for r in sorted(nonkeys,
                                       key=lambda item: -item['epe_512px'])[:10]]})
    return result


def summarize_group(rows):
    pixels = sum(row['gt_valid_pixels'] for row in rows)
    nonkeys = [row for row in rows if not row['keyframe']]
    image_fields = ('nmi_2mi_over_entropy_sum', 'edge_information_retention_proxy',
                    'edge_overlap_f1_3x3_proxy', 'lncc17_squared_proxy',
                    'visible_gt_mse_0to1', 'visible_gt_ncc',
                    'raw_itf_255', 'raw_t_ssim')
    return {'frames': len(rows), 'nonkey_frames': len(nonkeys),
            'gt_valid_pixels': pixels, 'pixel_weighted_epe_512px': weighted_epe(rows),
            'nonkey_pixel_weighted_epe_512px': weighted_epe(nonkeys),
            'prediction_invalid_on_gt_fraction':
                sum(r['predicted_invalid_pixels_on_gt'] for r in rows) / pixels
                if pixels else None,
            'total_compute_seconds': sum(row['frame_ms'] for row in rows) / 1000,
            'total_registration_seconds':
                sum(row['registration_ms'] for row in rows) / 1000,
            'pck_512px': {str(radius):
                          sum(r[f'pck_{radius}_pixels'] for r in rows) / pixels
                          if pixels else None for radius in (1, 3, 5)},
            'frame_mean_image_metrics': {
                field: float(np.mean([r[field] for r in rows if field in r]))
                if any(field in r for r in rows) else None
                for field in image_fields},
            'moving_epe_512px': weighted_metric(rows, 'moving_epe_512px', 'moving_pixels'),
            'moving_pixels': sum(r['moving_pixels'] for r in rows),
            'temporal_low_confidence_epe_512px': weighted_metric(
                rows, 'temporal_low_confidence_epe_512px',
                'temporal_low_confidence_pixels'),
            'temporal_low_confidence_pixels': sum(
                r['temporal_low_confidence_pixels'] for r in rows),
            'motion_compensated_ir_mae_0to1': weighted_metric(
                rows, 'motion_compensated_ir_mae_0to1', 'temporal_common_pixels'),
            'motion_compensated_alignment_residual_change_0to1': weighted_metric(
                rows, 'motion_compensated_alignment_residual_change_0to1',
                'temporal_common_pixels'),
            'temporal_common_pixels': sum(r['temporal_common_pixels'] for r in rows)}


def evaluate(dataset, schedule, model, device, output):
    arms = {'dual_modal_propagation': [], 'hold_last_keyframe': []}
    paired = []
    wall_start = time.perf_counter()
    for sequence, stems in dataset.sequences.items():
        previous_frame = previous_flow = previous_valid = last_key_flow = None
        previous_gt_valid = None
        previous_registered = {name: None for name in arms}
        previous_residual = {name: None for name in arms}
        age = 0
        for index, stem in enumerate(stems):
            input_start = time.perf_counter()
            frame = dataset.input_frame(sequence, stem)
            input_ms = (time.perf_counter() - input_start) * 1000
            is_keyframe = stem in schedule[sequence]
            if index == 0 and not is_keyframe:
                raise AssertionError('each sequence must begin with A1')
            temporal_ms = geometry_ms = a1_ms = hold_ms = 0.
            temporal = quality = None
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
            propagation_registered, propagation_inside = backward_warp(
                frame['ir'], propagation_flow)
            propagation_warp_ms = (time.perf_counter() - warp_start) * 1000
            warp_start = time.perf_counter()
            hold_registered, hold_inside = backward_warp(frame['ir'], hold_flow)
            hold_warp_ms = (time.perf_counter() - warp_start) * 1000
            if not np.array_equal(hold_inside, hold_valid):
                raise AssertionError('held flow source-valid mask differs from image warp')
            # GT is read only after both predictions and the key decision.
            truth, gt_valid, visible_gt = dataset.evaluation_truth(
                sequence, stem, frame['original_hw'])
            # Motion used solely to score a keyframe transition is calculated
            # after both outputs and excluded from both inference timings.
            metric_temporal = temporal if temporal is not None else (
                temporal_pair_fast(previous_frame, frame) if index else None)
            common = (temporal_metric_mask(gt_valid, previous_gt_valid,
                                           metric_temporal)
                      if metric_temporal is not None else None)
            timing = {'a1': a1_ms, 'temporal': temporal_ms, 'geometry': geometry_ms}
            a_row = row_metrics(frame, truth, gt_valid, propagation_flow, propagation_valid,
                                is_keyframe, age, timing, input_ms,
                                propagation_warp_ms, propagation_inside)
            b_row = row_metrics(frame, truth, gt_valid, hold_flow, hold_valid,
                                is_keyframe, age, {'a1': a1_ms, 'hold': hold_ms},
                                input_ms, hold_warp_ms, hold_inside)
            if a_row['gt_valid_pixels'] != b_row['gt_valid_pixels']:
                raise AssertionError('A/B GT denominators differ')
            for name, row, flow, registered in (
                    ('dual_modal_propagation', a_row, propagation_flow,
                     propagation_registered),
                    ('hold_last_keyframe', b_row, hold_flow, hold_registered)):
                error = np.linalg.norm(flow - truth, axis=-1)
                row.update({f'pck_{radius}_pixels':
                            int(((error < radius) & gt_valid).sum())
                            for radius in (1, 3, 5)})
                row.update(region_metrics(error, gt_valid, metric_temporal, quality))
                image, residual = image_metrics(
                    frame, flow, gt_valid, visible_gt, registered,
                    previous_registered[name])
                row.update(image)
                if metric_temporal is not None:
                    row.update(temporal_image_metrics(
                        registered, previous_registered[name], residual,
                        previous_residual[name], metric_temporal, common))
                else:
                    row.update({'temporal_common_pixels': 0,
                                'motion_compensated_ir_mae_0to1': None,
                                'motion_compensated_alignment_residual_change_0to1':
                                    None})
                arms[name].append(row)
                previous_registered[name] = registered
                previous_residual[name] = residual
                if (sequence, stem) in {
                        ('photo-0319-18', '000186'),
                        ('photo-0319-25', '000119'),
                        ('wurenji-0302-02', '000089')}:
                    save_ghost(frame['vi'], registered,
                               output / name / 'ghosts' / f'{sequence}_{stem}.png')
            paired.append({'sequence': sequence, 'stem': stem,
                           'keyframe': int(is_keyframe), 'age_since_keyframe': age,
                           'gt_valid_pixels': a_row['gt_valid_pixels'],
                           'propagation_epe_512px': a_row['epe_512px'],
                           'hold_epe_512px': b_row['epe_512px'],
                           'hold_minus_propagation_512px':
                               b_row['epe_512px'] - a_row['epe_512px'],
                           'propagation_invalid_on_gt':
                               a_row['predicted_invalid_pixels_on_gt'],
                           'hold_invalid_on_gt': b_row['predicted_invalid_pixels_on_gt'],
                           'propagation_pck_5_pixels': a_row['pck_5_pixels'],
                           'hold_pck_5_pixels': b_row['pck_5_pixels'],
                           'propagation_edge_f1_proxy':
                               a_row['edge_overlap_f1_3x3_proxy'],
                           'hold_edge_f1_proxy': b_row['edge_overlap_f1_3x3_proxy'],
                           'propagation_temporal_alignment_change':
                               a_row['motion_compensated_alignment_residual_change_0to1'],
                           'hold_temporal_alignment_change':
                               b_row['motion_compensated_alignment_residual_change_0to1']})
            previous_gt_valid = gt_valid
            previous_frame, previous_flow, previous_valid = (
                frame, propagation_flow, propagation_valid)
        print(f'{sequence}: {len(stems)} frames', flush=True)
    return arms, paired, time.perf_counter() - wall_start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('check', 'eval'))
    for name in ('data-root', 'split-file', 'pretrained', 'coarse-checkpoint',
                 'source-a1-checkpoint', 'source-coarse-adapt-checkpoint',
                 'a1-checkpoint', 'frozen-schedule', 'prior-report', 'output'):
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
    arms, paired, wall_seconds = evaluate(dataset, schedule, model, device,
                                          args.output)
    summaries = {name: summarize(rows) for name, rows in arms.items()}
    reference = summaries['dual_modal_propagation']
    if (reference['gt_valid_pixels'] != schedule_payload['historical_gt_valid_pixels'] or
            abs(reference['pixel_weighted_epe_512px'] -
                schedule_payload['historical_fixed_fast_epe_512px']) > .01):
        raise RuntimeError('propagation arm does not reproduce the frozen reference')
    if args.prior_report is not None:
        prior = json.loads(args.prior_report.read_text(encoding='utf-8'))
        for name in summaries:
            old = prior['arms'][name]
            new = summaries[name]
            if (old['gt_valid_pixels'] != new['gt_valid_pixels'] or
                    abs(old['pixel_weighted_epe_512px'] -
                        new['pixel_weighted_epe_512px']) > 1e-4):
                raise RuntimeError(f'{name} EPE changed during metric-only rerun')
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
              'metric_definitions': {
                  'pck_512px': 'fraction of fixed GT-valid pixels with EPE below 1/3/5 px',
                  'moving': 'non-keyframe VI temporal flow magnitude >= 3 px at 512 grid',
                  'temporal_low_confidence':
                      'non-keyframe region rejected by the historical propagation '
                      'forward-backward and support mask; proxy, not occlusion GT',
                  'nmi_2mi_over_entropy_sum': '64-bin grayscale 2MI/(Hvi+Hir), range 0..1',
                  'edge_and_lncc': 'existing project proxies, not verified paper metrics',
                  'visible_gt': 'evaluation-only same-modality MSE/NCC after predicted warp',
                  'raw_itf_tssim': 'warped-IR frame difference without motion compensation; '
                                   'may reward freezing or blur',
                  'motion_compensated_ir_mae':
                      'current warped IR versus previous warped IR sampled at current '
                      'VI->previous VI coordinates; same GT/VI-FB mask for both arms',
                  'motion_compensated_alignment_residual_change':
                      'temporal change of (warped visible_gt minus visible_mis) '
                      'after VI-motion alignment; evaluation-only jitter proxy, '
                      'not a direct proof of registration stability'},
              'timing_scope': 'compute-only: common A1 call on frozen '
                              'keyframes; four 256-grid Farneback flows and '
                              'geometric quality on propagation non-keyframes; '
                              'or hold and current IR bounds on hold non-keyframes; '
                              'registration time also includes shared image '
                              'read/preprocessing and each arm image warp; '
                              'both exclude GT/metrics, schedule creation and '
                              'shared warm-up. Unlike the old online run, '
                              'keyframe temporal flow is computed only after '
                              'predictions for evaluation, excluded from both '
                              'arm timings.',
              'paired_wall_seconds_including_io_and_metrics': wall_seconds,
              'arms': summaries,
              'nonkey_hold_minus_propagation_weighted_epe_512px':
                  sum(r['hold_minus_propagation_512px'] * r['gt_valid_pixels']
                      for r in nonkeys) / nonkey_pixels,
              'nonkey_improved_by_hold': sum(r['hold_minus_propagation_512px'] < 0
                                             for r in nonkeys),
              'nonkey_worsened_by_hold': sum(r['hold_minus_propagation_512px'] > 0
                                             for r in nonkeys),
              'prior_report_sha256': (sha256(args.prior_report)
                                      if args.prior_report is not None else None),
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
