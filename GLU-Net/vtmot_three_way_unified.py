"""Frozen-A1, three-arm VTMOT validation with one temporal evaluation field.

The 165-keyframe list is fixed. A1 runs on every frame for arm A and on the
same keyframes for arms B/C. On non-keyframes B computes its own four Farneback
fields and geometric composition, while C holds the last keyframe flow. All
three arms run as separate complete passes, with identical A1 warm-up before
each. The evaluation motion is recomputed *after* each prediction and its
numerical identity across passes is checked by per-pair hashes. Its time is
evaluation overhead; it is never credited to propagation inference. GT and
visible_gt are also read only after prediction.
No training, checkpoint writing, or test-split access occurs here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from datasets.vtmot_video import VTMOTVideos
from vtmot_geometry import backward_warp
from vtmot_hold_vs_propagation import (
    EXPECTED_FRAMES, EXPECTED_KEYS, hold_keyframe_flow, image_metrics,
    load_schedule, region_metrics, row_metrics, summarize, synthetic_check,
    temporal_image_metrics, temporal_metric_mask,
)
from vtmot_keyframe import (
    install_mutual_guard, load_a1, predict_a1, save_json, sha256,
    temporal_pair_fast, temporal_quality, write_rows,
)
from vtmot_local_correction_ab import A1_SHA, SPLIT_NAMES, TRIGGER


ARMS = ('every_frame_a1', 'dual_modal_propagation', 'hold_last_keyframe')
RECIPE = 'vtmot_three_way_frozen_165_unified_v1'
GT_VALID_PIXELS = 204309555


def score_arm(frame, truth, gt_valid, visible_gt, flow, predicted_valid,
              registered, source_inside, keyframe, age, timing, input_ms,
              warp_ms, metric_temporal, common, quality, previous_registered,
              previous_residual):
    row = row_metrics(frame, truth, gt_valid, flow, predicted_valid,
                      keyframe, age, timing, input_ms, warp_ms, source_inside)
    error = np.linalg.norm(flow - truth, axis=-1)
    row.update({f'pck_{radius}_pixels': int(((error < radius) & gt_valid).sum())
                for radius in (1, 3, 5)})
    row.update(region_metrics(error, gt_valid, metric_temporal, quality))
    image, residual = image_metrics(frame, flow, gt_valid, visible_gt,
                                    registered, previous_registered)
    row.update(image)
    if metric_temporal is None:
        row.update({'temporal_common_pixels': 0,
                    'motion_compensated_ir_mae_0to1': None,
                    'motion_compensated_alignment_residual_change_0to1': None})
    else:
        row.update(temporal_image_metrics(registered, previous_registered,
                                          residual, previous_residual,
                                          metric_temporal, common))
    return row, residual


def motion_signature(temporal, common):
    digest = hashlib.sha256()
    for name in ('vi', 'vi_error', 'vi_bounds'):
        field = np.ascontiguousarray(temporal[name])
        digest.update(str((name, field.shape, field.dtype)).encode('ascii'))
        digest.update(field.tobytes())
    digest.update(np.ascontiguousarray(common).tobytes())
    return digest.hexdigest()


def evaluate_arm(dataset, schedule, model, device, arm, expected_signatures):
    rows = []
    signatures = {}
    accounting = {'input_seconds_once': 0., 'gt_read_seconds': 0.,
                  'evaluation_motion_seconds': 0., 'metric_seconds': 0.,
                  'registered_warp_seconds': 0.}
    started = time.perf_counter()
    for sequence, stems in dataset.sequences.items():
        previous_frame = previous_flow = previous_valid = None
        previous_gt_valid = last_key_flow = None
        previous_registered = previous_residual = None
        age = 0
        for index, stem in enumerate(stems):
            start = time.perf_counter()
            frame = dataset.input_frame(sequence, stem)
            input_ms = (time.perf_counter() - start) * 1000.
            accounting['input_seconds_once'] += input_ms / 1000.
            keyframe = stem in schedule[sequence]
            if index == 0 and not keyframe:
                raise AssertionError(f'first frame is not a keyframe: {sequence}')

            timing = {}
            quality = None
            if arm == ARMS[0] or keyframe:
                start = time.perf_counter()
                flow, valid, _ = predict_a1(model, frame, device)
                timing['a1'] = (time.perf_counter() - start) * 1000.
                if keyframe:
                    last_key_flow = flow.copy()
                age = 0
            elif arm == ARMS[1]:
                # This is actual inference work; the evaluation field below
                # cannot be substituted or credited as free computation.
                start = time.perf_counter()
                inference_temporal = temporal_pair_fast(previous_frame, frame)
                timing['temporal'] = (time.perf_counter() - start) * 1000.
                start = time.perf_counter()
                quality = temporal_quality(inference_temporal,
                                           previous_flow, previous_valid,
                                           TRIGGER['fb_pixel_threshold'])
                timing['geometry'] = (time.perf_counter() - start) * 1000.
                flow, valid = quality['flow'], quality['valid']
                age += 1
            else:
                start = time.perf_counter()
                flow, valid = hold_keyframe_flow(last_key_flow)
                timing['hold'] = (time.perf_counter() - start) * 1000.
                age += 1
            start = time.perf_counter()
            registered, source_inside = backward_warp(frame['ir'], flow)
            warp_ms = (time.perf_counter() - start) * 1000.
            accounting['registered_warp_seconds'] += warp_ms / 1000.
            if arm == ARMS[2] and not np.array_equal(source_inside, valid):
                raise AssertionError('held-flow bounds differ from image warp')

            # Evaluation begins only after this arm emitted its registered image.
            start = time.perf_counter()
            truth, gt_valid, visible_gt = dataset.evaluation_truth(
                sequence, stem, frame['original_hw'])
            accounting['gt_read_seconds'] += time.perf_counter() - start
            start = time.perf_counter()
            metric_temporal = (temporal_pair_fast(previous_frame, frame)
                               if index else None)
            accounting['evaluation_motion_seconds'] += time.perf_counter() - start
            start = time.perf_counter()
            common = (temporal_metric_mask(gt_valid, previous_gt_valid,
                                           metric_temporal)
                      if metric_temporal is not None else None)
            if metric_temporal is not None:
                identity = (sequence, stem)
                signature = motion_signature(metric_temporal, common)
                if expected_signatures is not None and (
                        expected_signatures.get(identity) != signature):
                    raise AssertionError(f'evaluation motion/mask changed: {identity}')
                signatures[identity] = signature
            row, residual = score_arm(
                frame, truth, gt_valid, visible_gt, flow, valid, registered,
                source_inside, keyframe, age, timing, input_ms, warp_ms,
                metric_temporal, common, quality, previous_registered,
                previous_residual)
            rows.append(row)
            accounting['metric_seconds'] += time.perf_counter() - start
            if metric_temporal is not None and not common.any():
                raise AssertionError(f'empty temporal common mask: {sequence}/{stem}')
            previous_registered, previous_residual = registered, residual
            previous_frame = frame
            previous_flow, previous_valid = flow, valid
            previous_gt_valid = gt_valid
        print(f'{arm}: evaluated {sequence}: {len(stems)} frames', flush=True)
    accounting['paired_wall_seconds_including_evaluation'] = (
        time.perf_counter() - started)
    return rows, signatures, accounting


def pair_rows(rows):
    reference = rows[ARMS[0]]
    for name in ARMS[1:]:
        other = rows[name]
        if len(reference) != len(other) or any(
                (a['sequence'], a['stem'], a['keyframe'], a['gt_valid_pixels'],
                 a['temporal_common_pixels']) !=
                (b['sequence'], b['stem'], b['keyframe'], b['gt_valid_pixels'],
                 b['temporal_common_pixels']) for a, b in zip(reference, other)):
            raise AssertionError(f'frame, schedule or common mask differs: {name}')
        if any(abs(a['epe_512px'] - b['epe_512px']) > 1e-3
               for a, b in zip(reference, other)
               if a['keyframe']):
            raise AssertionError(f'keyframe A1 EPE differs: {name}')
    paired = []
    for triplet in zip(*(rows[name] for name in ARMS)):
        first = triplet[0]
        paired.append({
            'sequence': first['sequence'], 'stem': first['stem'],
            'keyframe': first['keyframe'],
            'keyframe_switch': int(first['keyframe'] and first['stem'] != '000000'),
            'age_since_keyframe': first['age_since_keyframe'],
            'gt_valid_pixels': first['gt_valid_pixels'],
            'temporal_common_pixels': first['temporal_common_pixels'],
            **{f'{name}_{field}': row.get(field) for name, row in
               zip(ARMS, triplet) for field in (
                   'epe_512px', 'motion_compensated_ir_mae_0to1',
                   'motion_compensated_alignment_residual_change_0to1',
                   'registration_ms')},
        })
    return paired


def check_output(rows, paired, schedule_payload):
    if len(paired) != EXPECTED_FRAMES:
        raise AssertionError('expected 800 validation frames')
    if sum(r['keyframe'] for r in paired) != EXPECTED_KEYS:
        raise AssertionError('expected 165 frozen keyframes')
    adjacent = [r for r in paired if r['stem'] != '000000']
    if len(adjacent) != 796 or sum(r['keyframe_switch'] for r in paired) != 161:
        raise AssertionError('expected 796 adjacent pairs and 161 key switches')
    if any(r['temporal_common_pixels'] <= 0 for r in adjacent):
        raise AssertionError('an adjacent pair has no common temporal support')
    summaries = {name: summarize(rows[name]) for name in ARMS}
    for name, result in summaries.items():
        if (result['frames'] != EXPECTED_FRAMES or
                result['gt_valid_pixels'] != GT_VALID_PIXELS):
            raise AssertionError(f'{name}: frame or fixed GT mask mismatch')
        if result['temporal_common_pixels'] != summaries[ARMS[0]]['temporal_common_pixels']:
            raise AssertionError('temporal evaluation pixels differ')
    reference = summaries[ARMS[1]]
    if (reference['gt_valid_pixels'] !=
            schedule_payload['historical_gt_valid_pixels'] or
            abs(reference['pixel_weighted_epe_512px'] -
                schedule_payload['historical_fixed_fast_epe_512px']) > .01):
        raise AssertionError('propagation no longer matches frozen schedule')
    if abs(summaries[ARMS[0]]['pixel_weighted_epe_512px'] - 4.7993) > .02:
        raise AssertionError('every-frame A1 changed relative to locked validation')
    return summaries, adjacent


def synthetic_pipeline_check():
    """Exercise all three execution paths without reading VTMOT or weights."""
    height = width = 512
    ramp = np.tile(np.arange(width, dtype=np.uint8), (height, 1))
    image = np.repeat(ramp[..., None], 3, axis=2)

    class TwoFrameDataset:
        sequences = {'synthetic': ['000000', '000001']}

        def input_frame(self, sequence, stem):
            return {'sequence': sequence, 'stem': stem, 'ir': image,
                    'vi': image, 'original_hw': (height, width)}

        def evaluation_truth(self, sequence, stem, original_hw):
            calls['truth'] += 1
            return (np.zeros((height, width, 2), np.float32),
                    np.ones((height, width), bool), image)

    def fake_a1(model, frame, device):
        calls['predicted'] += 1
        return (np.zeros((height, width, 2), np.float32),
                np.ones((height, width), bool), 0.)

    def fake_temporal(previous, current):
        zeros = np.zeros((height, width, 2), np.float32)
        return {'vi': zeros, 'ir': zeros, 'vi_bounds': np.ones((height, width), bool),
                'ir_bounds': np.ones((height, width), bool),
                'vi_error': np.zeros((height, width), np.float32),
                'ir_error': np.zeros((height, width), np.float32)}

    def fake_quality(temporal, previous_flow, previous_valid, threshold):
        return {'flow': previous_flow.copy(), 'valid': previous_valid.copy()}

    calls = {'predicted': 0, 'truth': 0}
    rows = {}
    signatures = None
    with (patch(__name__ + '.predict_a1', fake_a1),
          patch(__name__ + '.temporal_pair_fast', fake_temporal),
          patch(__name__ + '.temporal_quality', fake_quality)):
        for name in ARMS:
            rows[name], found, _ = evaluate_arm(
                TwoFrameDataset(), {'synthetic': {'000000'}}, None, None,
                name, signatures)
            if signatures is None:
                signatures = found
    paired = pair_rows(rows)
    if calls != {'predicted': 4, 'truth': 6} or len(paired) != 2:
        raise AssertionError('synthetic three-arm frame accounting failed')
    if paired[1]['temporal_common_pixels'] != height * width:
        raise AssertionError('synthetic common mask is incomplete')
    for name in ARMS:
        if len(rows[name]) != 2 or any(r['epe_512px'] != 0 for r in rows[name]):
            raise AssertionError(f'{name}: synthetic EPE differs')
        if rows[name][1]['motion_compensated_alignment_residual_change_0to1'] != 0:
            raise AssertionError(f'{name}: synthetic temporal score differs')
    return 'passed: two frames, three separate arm passes, identical temporal fields'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('check', 'eval'))
    for name in ('data-root', 'split-file', 'pretrained', 'coarse-checkpoint',
                 'source-a1-checkpoint', 'source-coarse-adapt-checkpoint',
                 'a1-checkpoint', 'frozen-schedule', 'output'):
        parser.add_argument(f'--{name}', type=Path)
    args = parser.parse_args()
    checks = synthetic_check()
    if args.stage == 'check':
        print(json.dumps({'synthetic': checks,
                          'three_arm_pipeline': synthetic_pipeline_check(),
                          'arms': ARMS,
                          'expected_adjacent_pairs': 796,
                          'expected_keyframe_switches': 161}, indent=2))
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
    if (tuple(dataset.sequences) != SPLIT_NAMES or
            sum(map(len, dataset.sequences.values())) != EXPECTED_FRAMES):
        raise ValueError('only the four fixed validation sequences are allowed')
    schedule_payload, schedule = load_schedule(
        args.frozen_schedule, args.split_file, dataset.sequences)
    first = next(iter(dataset.sequences))
    rows = {}
    accounting = {}
    signatures = None
    for name in ARMS:
        # Same input, model, warm-up and synchronized A1 call for each pass.
        predict_a1(model, dataset.input_frame(first, dataset.sequences[first][0]),
                   device)
        rows[name], found, accounting[name] = evaluate_arm(
            dataset, schedule, model, device, name, signatures)
        if signatures is None:
            signatures = found
    paired = pair_rows(rows)
    summaries, adjacent = check_output(rows, paired, schedule_payload)
    candidate_pixels = sum(r['gt_valid_pixels'] for r in adjacent)
    common_pixels = sum(r['temporal_common_pixels'] for r in adjacent)
    coverage_by_sequence = {}
    for sequence in dataset.sequences:
        these = [r for r in adjacent if r['sequence'] == sequence]
        candidates = sum(r['gt_valid_pixels'] for r in these)
        common = sum(r['temporal_common_pixels'] for r in these)
        coverage_by_sequence[sequence] = {
            'adjacent_pairs': len(these),
            'keyframe_switches': sum(r['keyframe_switch'] for r in these),
            'candidate_gt_valid_pixels': candidates,
            'common_pixels': common,
            'common_pixel_fraction': common / candidates,
        }
    report = {
        'recipe': RECIPE, 'split': 'eval', 'test_accessed': False,
        'a1_sha256': A1_SHA, 'split_sha256': sha256(args.split_file),
        'frozen_schedule_sha256': sha256(args.frozen_schedule),
        'flow_direction': 'current VI target -> current IR source, XY',
        'flow_units': '512x512 image pixels',
        'gt_mask': 'fixed GT valid pixels; prediction invalidity never shrinks EPE denominator',
        'temporal_metric_definitions': {
            'evaluation_motion': 'independent 256-grid four-direction Farneback; '
                                 'resized to 512-pixel XY units; current VI -> previous VI',
            'common_mask': 'current GT valid AND previous GT valid sampled at current '
                           'VI->previous VI coordinates AND VI bounds AND VI forward/backward '
                           f'error <= {TRIGGER["fb_pixel_threshold"]} px',
            'motion_compensated_ir_mae_0to1': 'absolute grayscale difference of current '
                'warped IR and previous warped IR sampled at current VI->previous VI; '
                'divide by 255; common-pixel weighted mean',
            'motion_compensated_alignment_residual_change_0to1': 'absolute temporal '
                'change of (warped visible_gt minus visible_mis), sampling previous '
                'residual with current VI->previous VI; divide by 255; '
                'evaluation-only proxy, not proof that registration jitter vanished',
        },
        'temporal_coverage': {
            'adjacent_pairs': len(adjacent), 'first_frames_without_predecessor': 4,
            'keyframe_switches': sum(r['keyframe_switch'] for r in paired),
            'candidate_gt_valid_pixels': candidate_pixels,
            'common_pixels': common_pixels,
            'common_pixel_fraction': common_pixels / candidate_pixels,
            'excluded_pixels': candidate_pixels - common_pixels,
            'by_sequence': coverage_by_sequence,
        },
        'timing_scope': 'three separate complete arm passes, each with same A1 '
            'warm-up and synchronized GPU inference; per arm: image read/preprocess '
            '+ A1 where run + actual inference Farneback and geometry where run '
            '+ output warp. GT read, independent evaluation motion, common-mask '
            'construction and all metrics are reported separately; evaluation '
            'motion fields must hash identically across passes. Fixed keyframe-list '
            'generation is excluded.',
        'schedule_creation_cost_seconds': None,
        'online_deployment_speedup_claimed': False,
        'evaluation_overhead_by_arm': accounting,
        'arms': {},
        'synthetic_check': checks,
    }
    for name in ARMS:
        summary = summaries[name]
        summary['p95_registration_ms_per_frame'] = float(np.percentile(
            [r['registration_ms'] for r in rows[name]], 95))
        for sequence in dataset.sequences:
            summary['by_sequence'][sequence]['p95_registration_ms_per_frame'] = (
                float(np.percentile([
                    r['registration_ms'] for r in rows[name]
                    if r['sequence'] == sequence], 95)))
        summary['temporal_evaluable_pairs'] = sum(
            r['temporal_common_pixels'] > 0 for r in rows[name])
        report['arms'][name] = summary
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ARMS:
        write_rows(args.output / f'{name}_per_frame.csv', rows[name])
    write_rows(args.output / 'paired_per_frame.csv', paired)
    save_json(args.output / 'report.json', report)
    print(json.dumps({
        'frames': len(paired), 'adjacent_pairs': len(adjacent),
        'keyframe_switches': report['temporal_coverage']['keyframe_switches'],
        'temporal_common_fraction': report['temporal_coverage']['common_pixel_fraction'],
        'arms': {name: {
            'epe_512px': report['arms'][name]['pixel_weighted_epe_512px'],
            'alignment_residual_change': report['arms'][name][
                'motion_compensated_alignment_residual_change_0to1'],
            'ir_temporal_mae': report['arms'][name][
                'motion_compensated_ir_mae_0to1'],
            'registration_seconds': report['arms'][name]['total_registration_seconds'],
            'p95_registration_ms': report['arms'][name]['p95_registration_ms_per_frame'],
        } for name in ARMS}}, indent=2), flush=True)


if __name__ == '__main__':
    main()
