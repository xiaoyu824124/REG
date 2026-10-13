"""Frozen-A1 VTMOT control: dual-modal propagation versus sparse local correction.

`check` uses synthetic arrays only. `calibrate` reads train sequences and fixes a
forward/backward risk threshold. `eval` evaluates the two arms on the four
validation sequences; neither stage accepts the locked test split.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from datasets.vtmot_video import VTMOTVideos
from vtmot_geometry import backward_warp, grid, sample, self_test
from vtmot_keyframe import (frame_metrics, gray, install_mutual_guard, load_a1,
                            predict_a1, save_ghost, save_json, sha256, summarise,
                            temporal_fast_self_test, temporal_pair_fast,
                            temporal_quality, write_rows)


A1_SHA = 'af245ea28cd0fc33bff7aa9dd83cb1402c1108de7870c044a9960f791cbac435'
SPLIT_NAMES = ('photo-0306-02', 'photo-0319-18', 'photo-0319-25', 'wurenji-0302-02')
# Frozen from train_thresholds_fast.json of the original fixed_fast run.
TRIGGER = {'fb_pixel_threshold': 1.5,
           'minimum_valid_fraction': 0.9040260314941406,
           'maximum_fb_median_px': 0.04110261164605617,
           'fixed_interval': 5}
TILE = 64  # 512-pixel grid; matching itself runs on the 256-pixel grid.
MAX_TILES = 8
MAX_CORRECTION_512 = 6
MIN_STRUCTURAL_GAIN = .05


def risk_map(temporal, previous_registration, propagated_valid):
    """Current VI-grid risk from both FB errors and geometric support.

    IR FB error lives on the previous IR grid, so sample it after traversing
    current VI -> previous VI -> previous IR. It must not be added by index.
    """
    q = grid(temporal['vi'].shape[:2]) + temporal['vi']
    old_at, old_inside = sample(previous_registration, q)
    r = q + old_at
    ir_fb, ir_inside = sample(temporal['ir_error'], r)
    ir_bounds, _ = sample(temporal['ir_bounds'].astype(np.float32), r)
    vi_ok = temporal['vi_bounds'] & np.isfinite(temporal['vi_error'])
    support = propagated_valid & old_inside & ir_inside & vi_ok & (ir_bounds > .999)
    fb = np.maximum(np.clip(temporal['vi_error'], 0, 10), np.clip(ir_fb, 0, 10))
    fb[~np.isfinite(fb)] = 10
    return fb.astype(np.float32), support


def tile_scores(fb, support):
    if fb.shape != (512, 512) or support.shape != fb.shape:
        raise ValueError('confidence requires a 512x512 VI grid')
    scores = np.empty((8, 8), np.float32)
    for row in range(8):
        for col in range(8):
            sl = np.s_[row*TILE:(row+1)*TILE, col*TILE:(col+1)*TILE]
            good = support[sl]
            # A block with no valid correspondence is risky, but cannot be
            # locally matched; the matching gate handles it separately.
            scores[row, col] = (float(np.quantile(fb[sl][good], .9)) if good.any() else 10.) \
                               + 5. * (1. - float(good.mean()))
    return scores


def selected_tiles(scores, threshold):
    candidates = [(float(scores[row, col]), row, col)
                  for row in range(8) for col in range(8)
                  if scores[row, col] >= threshold]
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [(row, col) for _, row, col in candidates[:MAX_TILES]]


def structural_image(rgb):
    small = cv2.resize(gray(rgb), (256, 256), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    return cv2.GaussianBlur(magnitude, (3, 3), 0).astype(np.float32)


def match_tile(vi_structure, ir_structure, flow_256, support_256, row, col):
    """Small structural-search correction on one 32x32 tile at grid 256.

    Raw match scores are used only to choose a residual and demand a relative
    improvement over zero residual. They are not treated as probabilities.
    """
    y0, x0 = row*32, col*32
    yy, xx = np.mgrid[y0:y0+32, x0:x0+32].astype(np.float32)
    support = support_256[y0:y0+32, x0:x0+32].copy()
    if support.mean() < .55:
        return None
    target = vi_structure[y0:y0+32, x0:x0+32]
    target = (target-target.mean())/(target.std()+1e-3)
    if target.std() < .2:
        return None
    base = flow_256[y0:y0+32, x0:x0+32]
    costs = []
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            mx, my = xx+base[..., 0]+dx, yy+base[..., 1]+dy
            valid = support & (mx >= 0) & (mx <= 255) & (my >= 0) & (my <= 255)
            if valid.mean() < .55:
                continue
            warped = cv2.remap(ir_structure, mx.astype(np.float32),
                               my.astype(np.float32), cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT)
            source = warped[valid]
            source = (source-source.mean())/(source.std()+1e-3)
            cost = float(np.mean(np.abs(target[valid]-source)))
            costs.append((cost, dx, dy))
    zero = next((cost for cost, dx, dy in costs if dx == dy == 0), None)
    if zero is None or not costs:
        return None
    best = min(costs)
    gain = (zero-best[0])/max(zero, 1e-3)
    if (best[1] == best[2] == 0 or gain < MIN_STRUCTURAL_GAIN
            or math.hypot(best[1], best[2]) > MAX_CORRECTION_512/2):
        return None
    return best[1], best[2], gain


def correct_sparse(frame, flow, valid, fb, support, threshold):
    """Apply only a bounded, tapered residual on preselected VI-grid tiles."""
    scores = tile_scores(fb, support)
    chosen = selected_tiles(scores, threshold)
    result = flow.copy()
    if not chosen:
        return result, {'selected_tiles': 0, 'accepted_tiles': 0,
                        'corrected_fraction': 0., 'structural_gains': []}
    vi_structure, ir_structure = structural_image(frame['vi']), structural_image(frame['ir'])
    flow_256 = cv2.resize(flow, (256, 256), interpolation=cv2.INTER_LINEAR) * .5
    support_256 = cv2.resize(support.astype(np.uint8), (256, 256),
                             interpolation=cv2.INTER_NEAREST).astype(bool)
    taper = np.outer(np.hanning(66)[1:-1], np.hanning(66)[1:-1]).astype(np.float32)
    corrected = np.zeros((512, 512), bool)
    gains = []
    for row, col in chosen:
        matched = match_tile(vi_structure, ir_structure, flow_256, support_256, row, col)
        if matched is None:
            continue
        dx, dy, gain = matched
        sl = np.s_[row*TILE:(row+1)*TILE, col*TILE:(col+1)*TILE]
        weight = taper * support[sl] * valid[sl]
        result[sl] += (weight[..., None] * np.array([2*dx, 2*dy], np.float32))
        corrected[sl] = weight > .01
        gains.append(gain)
    if np.max(np.linalg.norm(result-flow, axis=-1)) > MAX_CORRECTION_512 + 1e-4:
        raise AssertionError('local correction exceeded the 512px displacement bound')
    return result, {'selected_tiles': len(chosen), 'accepted_tiles': len(gains),
                    'corrected_fraction': float(corrected.mean()),
                    'structural_gains': gains}


def roc_auc(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels, bool)
    positive = labels.sum()
    if positive == 0 or positive == len(labels):
        return None
    order = np.argsort(scores, kind='stable')
    ranks = np.empty(len(scores), np.float64)
    sorted_scores = scores[order]
    edges = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1, len(scores)]
    for start, stop in zip(edges[:-1], edges[1:]):
        ranks[order[start:stop]] = (start + 1 + stop)/2
    return float((ranks[labels].sum()-positive*(positive+1)/2)
                 / (positive*(len(labels)-positive)))


def train_samples(dataset, model, device):
    """Deterministic windows, chosen before looking at training GT."""
    for sequence, stems in dataset.sequences.items():
        for anchor in (20, 70, 120, 170):
            previous = dataset.input_frame(sequence, stems[anchor])
            old_flow, old_valid, _ = predict_a1(model, previous, device)
            for index in range(anchor+1, min(anchor+5, len(stems))):
                frame = dataset.input_frame(sequence, stems[index])
                temporal = temporal_pair_fast(previous, frame)
                quality = temporal_quality(temporal, old_flow, old_valid,
                                           TRIGGER['fb_pixel_threshold'])
                fb, support = risk_map(temporal, old_flow, quality['valid'])
                yield sequence, stems[index], frame, quality['flow'], quality['valid'], fb, support
                previous, old_flow, old_valid = frame, quality['flow'], quality['valid']


def calibrate(args, dataset, model, device):
    # Fix the sequence-level holdout before reading any GT. Adjacent frames
    # are correlated, so a tile-level random holdout would overstate evidence.
    sequence_names = sorted(dataset.sequences)
    audit_sequences = set(sequence_names[::5])
    fit_sequences = set(sequence_names) - audit_sequences
    if len(fit_sequences) != 30 or len(audit_sequences) != 8:
        raise ValueError('expected 30 fit and 8 audit training sequences')
    fit_scores, fit_labels, audit_scores, audit_labels = [], [], [], []
    for sequence, stem, frame, flow, valid, fb, support in train_samples(dataset, model, device):
        # GT enters only after propagation and risk have been emitted.
        truth, gt_valid = dataset.flow_truth(sequence, stem, frame['original_hw'])
        fresh, _, _ = predict_a1(model, frame, device)
        prop_error = np.linalg.norm(flow-truth, axis=-1)
        fresh_error = np.linalg.norm(fresh-truth, axis=-1)
        scores = tile_scores(fb, support)
        for row in range(8):
            for col in range(8):
                sl = np.s_[row*TILE:(row+1)*TILE, col*TILE:(col+1)*TILE]
                mask = gt_valid[sl]
                if mask.sum() < 100:
                    continue
                added = float(np.mean((prop_error[sl]-fresh_error[sl])[mask]))
                if sequence in audit_sequences:
                    audit_scores.append(float(scores[row, col]))
                    audit_labels.append(added > 1.)
                else:
                    fit_scores.append(float(scores[row, col]))
                    fit_labels.append(added > 1.)
    if not fit_scores or not audit_scores:
        raise RuntimeError('training yielded no valid risk tiles')
    threshold = float(np.quantile(fit_scores, .9))
    selected = np.asarray(audit_scores) >= threshold
    base_rate = float(np.mean(audit_labels))
    selected_rate = (float(np.mean(np.asarray(audit_labels)[selected]))
                     if selected.any() else None)
    auc = roc_auc(audit_scores, audit_labels)
    predictor_pass = (auc is not None and auc >= .6 and base_rate > 0
                      and selected_rate is not None
                      and selected_rate >= 1.3*base_rate)
    report = {'split': 'train', 'a1_sha256': A1_SHA,
              'split_sha256': sha256(args.split_file),
              'risk': 'max VI/IR forward-backward error mapped to current VI, '
                      'plus invalid-support fraction; no raw correlation probability',
              'label': 'tile mean EPE(propagated)-EPE(current A1) > 1px',
              'fit_sequences': sorted(fit_sequences),
              'audit_sequences': sorted(audit_sequences),
              'fit_tile_count': len(fit_scores),
              'audit_tile_count': len(audit_scores),
              'audit_selected_tile_count': int(selected.sum()),
              'audit_positive_fraction': base_rate,
              'audit_roc_auc': auc,
              'audit_selected_positive_fraction': selected_rate,
              'audit_enrichment': (selected_rate/base_rate
                                   if base_rate and selected_rate is not None else None),
              'risk_threshold': threshold, 'predictor_pass': predictor_pass,
              'sampling': '38 train sequences; sorted every fifth held out for audit '
                          '(30 fit / 8 audit); anchors 20/70/120/170; next four frames',
              'test_accessed': False}
    save_json(args.output / 'calibration_train.json', report)
    return report


def synthetic_check():
    rng = np.random.default_rng(2026)
    image = rng.integers(0, 256, (256, 256), dtype=np.uint8)
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    moved = cv2.warpAffine(rgb, np.float32([[1, 0, 2], [0, 1, -1]]),
                           (256, 256), flags=cv2.INTER_LINEAR)
    source = structural_image(cv2.resize(moved, (512, 512)))
    target = structural_image(cv2.resize(rgb, (512, 512)))
    match = match_tile(target, source, np.zeros((256, 256, 2), np.float32),
                       np.ones((256, 256), bool), 3, 3)
    if match is None or match[:2] != (2, -1):
        raise AssertionError(f'local correction direction/scale failed: {match}')
    scores = np.array([[0.]*8]*8, np.float32)
    scores[3, 3] = 10
    if selected_tiles(scores, 5) != [(3, 3)]:
        raise AssertionError('low-confidence tile selection failed')
    if abs(roc_auc([0, 1, 2, 3], [False, False, True, True])-1.) > 1e-9:
        raise AssertionError('confidence AUC failed')
    return {'geometry': self_test(), 'temporal_fast': temporal_fast_self_test(),
            'local_match_256_xy': [2, -1], 'correction_512_xy': [4, -2],
            'risk_rank': 'passed', 'test_accessed': False}


def region_epe(flow, truth, mask):
    return float(np.linalg.norm(flow-truth, axis=-1)[mask].mean()) if mask.any() else None


def region_summary(rows):
    def weighted(group, value, count):
        good = [r for r in group if r[count] and r[value] is not None]
        pixels = sum(r[count] for r in good)
        return {'pixels': pixels,
                'epe_512px': sum(r[value]*r[count] for r in good)/pixels if pixels else None}
    result = {}
    for name in (*SPLIT_NAMES, 'all'):
        group = rows if name == 'all' else [r for r in rows if r['sequence'] == name]
        result[name] = {
            'motion_3px_plus': weighted(group, 'moving_region_epe_512px',
                                        'moving_region_pixels'),
            'forward_backward_occlusion_proxy': weighted(
                group, 'occlusion_proxy_epe_512px', 'occlusion_proxy_pixels'),
            'by_age': {str(age): weighted([r for r in group if r['age_since_keyframe'] == age],
                                          'epe_512px', 'gt_valid_pixels')
                       for age in range(5)}}
    return result


def evaluate(args, dataset, model, device, calibration, schedule):
    arms = {'A_dual_propagation': [], 'B_sparse_correction': []}
    ghosts = {'photo-0319-18': {'000013', '000022', '000186'}}
    for sequence, stems in dataset.sequences.items():
        previous = previous_reg = previous_valid = None
        corrected_reg = corrected_valid = None
        previous_registered = {arm: None for arm in arms}
        since = 0
        for index, stem in enumerate(stems):
            frame = dataset.input_frame(sequence, stem)
            temporal_ms = 0.
            if index:
                temporal = temporal_pair_fast(previous, frame)
                temporal_ms = temporal['ms']
                start = time.perf_counter()
                quality_a = temporal_quality(temporal, previous_reg, previous_valid,
                                             TRIGGER['fb_pixel_threshold'])
                base_ms = 1000*(time.perf_counter()-start)
                reason = 'frozen_fixed_fast' if stem in schedule[sequence] else None
            else:
                reason, base_ms = 'sequence_start', 0.
            keyframe = reason is not None
            if keyframe:
                inference_start = time.perf_counter()
                flow_a, valid_a, a1_ms = predict_a1(model, frame, device)
                a1_wall_ms = 1000*(time.perf_counter()-inference_start)
                flow_b, valid_b = flow_a.copy(), valid_a.copy()
                corr_ms = extra_prop_ms = 0.
                detail = {'selected_tiles': 0, 'accepted_tiles': 0,
                          'corrected_fraction': 0., 'structural_gains': []}
                since = 0
                # On non-first keyframes the same temporal estimator was used
                # in both arms to make the already-frozen trigger decision.
                cost_a = temporal_ms+base_ms+a1_wall_ms
                cost_b = cost_a
            else:
                flow_a, valid_a, a1_ms, a1_wall_ms = quality_a['flow'], quality_a['valid'], 0., 0.
                start = time.perf_counter()
                quality_b = temporal_quality(temporal, corrected_reg, corrected_valid,
                                             TRIGGER['fb_pixel_threshold'])
                extra_prop_ms = 1000*(time.perf_counter()-start)
                start = time.perf_counter()
                fb, support = risk_map(temporal, corrected_reg, quality_b['valid'])
                flow_b, detail = correct_sparse(frame, quality_b['flow'], quality_b['valid'],
                                                fb, support, calibration['risk_threshold'])
                _, inside_b = backward_warp(frame['ir'], flow_b)
                valid_b = quality_b['valid'] & inside_b
                corr_ms = 1000*(time.perf_counter()-start)
                since += 1
                cost_a = temporal_ms+base_ms
                cost_b = temporal_ms+extra_prop_ms+corr_ms
            # Keep GT isolated until both methods have emitted outputs/timing.
            truth, gt_valid = dataset.flow_truth(sequence, stem, frame['original_hw'])
            if index:
                vi_motion = np.linalg.norm(temporal['vi'], axis=-1) >= 3.
                _, common_support = risk_map(temporal, previous_reg, quality_a['valid'])
                occlusion_proxy = ~common_support
            else:
                vi_motion = occlusion_proxy = np.zeros((512, 512), bool)
            for arm, flow, valid, cost in (("A_dual_propagation", flow_a, valid_a, cost_a),
                                          ("B_sparse_correction", flow_b, valid_b, cost_b)):
                metric, registered = frame_metrics(dataset, frame, flow, valid,
                                                   previous_registered[arm])
                if metric['gt_valid_pixels'] != int(gt_valid.sum()):
                    raise AssertionError('GT masks differ between region and main evaluator')
                row = {'sequence': sequence, 'stem': stem, 'frame_index': index,
                       'arm': arm, 'keyframe': int(keyframe),
                       'reason': reason or 'propagated', 'age_since_keyframe': since,
                       'frame_ms': cost, 'temporal_ms': temporal_ms,
                       'a1_ms': a1_ms, 'a1_wall_ms': a1_wall_ms,
                       'matching_ms': corr_ms if arm.startswith('B_') else 0.,
                       'moving_region_pixels': int((gt_valid & vi_motion).sum()),
                       'moving_region_epe_512px': region_epe(flow, truth, gt_valid & vi_motion),
                       'occlusion_proxy_pixels': int((gt_valid & occlusion_proxy).sum()),
                       'occlusion_proxy_epe_512px': region_epe(flow, truth,
                                                               gt_valid & occlusion_proxy),
                       'selected_tiles': detail['selected_tiles'] if arm.startswith('B_') else 0,
                       'accepted_tiles': detail['accepted_tiles'] if arm.startswith('B_') else 0,
                       'corrected_fraction': detail['corrected_fraction'] if arm.startswith('B_') else 0.,
                       **metric}
                arms[arm].append(row)
                previous_registered[arm] = registered
                if stem in ghosts.get(sequence, ()):
                    save_ghost(frame['vi'], registered,
                               args.output / arm / 'ghosts' / f'{sequence}_{stem}.png')
            previous, previous_reg, previous_valid = frame, flow_a, valid_a
            corrected_reg, corrected_valid = flow_b, valid_b
        print(f'{sequence}: A={np.mean([r["epe_512px"] for r in arms["A_dual_propagation"] if r["sequence"] == sequence]):.3f} '
              f'B={np.mean([r["epe_512px"] for r in arms["B_sparse_correction"] if r["sequence"] == sequence]):.3f}',
              flush=True)
    for arm, rows in arms.items():
        write_rows(args.output / arm / 'per_frame.csv', rows)
        save_json(args.output / arm / 'report.json', {'a1_sha256': A1_SHA,
                  'split_sha256': sha256(args.split_file), 'arm': arm,
                  'summary': summarise(rows)['all'],
                  'regions': region_summary(rows), 'test_accessed': False})
    a, b = arms.values()
    if [(r['sequence'], r['stem'], r['gt_valid_pixels'], r['keyframe']) for r in a] != \
       [(r['sequence'], r['stem'], r['gt_valid_pixels'], r['keyframe']) for r in b]:
        raise AssertionError('paired arms differ in frames, masks, or keyframe positions')
    baseline = json.loads(args.every_frame_report.read_text(encoding='utf-8'))
    if baseline['a1_sha256'] != A1_SHA or baseline['summary']['frames'] != 800:
        raise ValueError('every-frame reference uses different A1 or split')
    every = baseline['summary']
    s_a, s_b = summarise(a)['all'], summarise(b)['all']
    if every['gt_valid_pixels'] != s_a['gt_valid_pixels']:
        raise ValueError('every-frame reference uses a different GT mask')
    frozen = json.loads(args.frozen_schedule.read_text(encoding='utf-8'))
    if s_a['gt_valid_pixels'] != frozen['historical_gt_valid_pixels'] or \
       abs(s_a['pixel_weighted_epe_512px'] - frozen['historical_fixed_fast_epe_512px']) > .05:
        raise RuntimeError('A arm does not reproduce historical fixed_fast flow/mask; '
                           'do not interpret the correction comparison')
    paired = [{'sequence': ra['sequence'], 'stem': ra['stem'],
               'keyframe': ra['keyframe'], 'age_since_keyframe': ra['age_since_keyframe'],
               'a_epe_512px': ra['epe_512px'], 'b_epe_512px': rb['epe_512px'],
               'gain_a_minus_b_512px': ra['epe_512px']-rb['epe_512px']}
              for ra, rb in zip(a, b)]
    write_rows(args.output / 'paired_per_frame.csv', paired)
    # Predeclared practical gate: a correction must buy accuracy with a real
    # speed benefit relative to every-frame A1, not merely improve old flow.
    rule = {'maximum_epe_above_every_frame_px': .3,
            'maximum_compute_fraction_of_every_frame': .8,
            'minimum_improved_sequences_vs_propagation': 3}
    improved_sequences = sum(s_b['per_sequence'][n]['epe_512px'] <
                             s_a['per_sequence'][n]['epe_512px'] for n in SPLIT_NAMES)
    keep = (s_b['pixel_weighted_epe_512px'] <= every['pixel_weighted_epe_512px']+.3
            and s_b['total_compute_seconds'] <= every['total_compute_seconds']*.8
            and improved_sequences >= 3)
    save_json(args.output / 'comparison.json', {
        'a1_sha256': A1_SHA,
        'keyframe_schedule': 'historical fixed_fast keyframes frozen before this experiment',
        'schedule_sha256': sha256(args.frozen_schedule),
        'calibration': calibration, 'arms': {'A': s_a, 'B': s_b},
        'regions': {'A': region_summary(a), 'B': region_summary(b)},
        'every_frame_a1_reference': {'epe_512px': every['pixel_weighted_epe_512px'],
                                     'total_compute_seconds': every['total_compute_seconds']},
        'paired_improved_frames': sum(r['gain_a_minus_b_512px'] > 0 for r in paired),
        'improved_sequences': improved_sequences, 'rule': rule,
        'retain_correction': keep, 'test_accessed': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('check', 'calibrate', 'eval'), default='check')
    for name in ('data-root', 'split-file', 'pretrained', 'coarse-checkpoint',
                 'source-a1-checkpoint', 'source-coarse-adapt-checkpoint',
                 'a1-checkpoint', 'every-frame-report', 'frozen-schedule', 'output'):
        parser.add_argument(f'--{name}', type=Path)
    args = parser.parse_args()
    if args.stage == 'check':
        print(json.dumps(synthetic_check(), indent=2))
        return
    for name in ('data_root', 'split_file', 'pretrained', 'coarse_checkpoint',
                 'source_a1_checkpoint', 'source_coarse_adapt_checkpoint',
                 'a1_checkpoint', 'output'):
        if getattr(args, name) is None:
            parser.error(f'--{name.replace("_", "-")} is required')
    if sha256(args.a1_checkpoint) != A1_SHA:
        raise ValueError('frozen A1 checkpoint SHA-256 mismatch')
    if args.stage == 'eval' and (args.every_frame_report is None or args.frozen_schedule is None):
        parser.error('--every-frame-report and --frozen-schedule are required for eval')
    if args.output.exists() and any(args.output.iterdir()):
        if args.stage != 'eval' or set(p.name for p in args.output.iterdir()) != {'calibration_train.json'}:
            raise FileExistsError('use a new output directory; eval may reuse only its calibration file')
    split = 'train' if args.stage == 'calibrate' else 'eval'
    dataset = VTMOTVideos(args.data_root, args.split_file, split=split)
    if args.stage == 'calibrate' and len(dataset.sequences) != 38:
        raise ValueError('confidence calibration requires all 38 train sequences')
    if args.stage == 'eval' and (tuple(dataset.sequences) != SPLIT_NAMES or
                                 sum(map(len, dataset.sequences.values())) != 800):
        raise ValueError('formal evaluation requires four fixed validation sequences / 800 frames')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    install_mutual_guard()
    model = load_a1(args, device)
    first = next(iter(dataset.sequences))
    predict_a1(model, dataset.input_frame(first, dataset.sequences[first][0]), device)
    if args.stage == 'calibrate':
        report = calibrate(args, dataset, model, device)
        print(json.dumps(report, indent=2))
    else:
        path = args.output / 'calibration_train.json'
        if not path.is_file():
            raise FileNotFoundError('run --stage calibrate on all train sequences first')
        calibration = json.loads(path.read_text(encoding='utf-8'))
        if (calibration['a1_sha256'] != A1_SHA or
                calibration['split_sha256'] != sha256(args.split_file)):
            raise ValueError('confidence calibration provenance mismatch')
        if not calibration['predictor_pass']:
            raise RuntimeError('training confidence does not predict propagation error; stop branch')
        frozen = json.loads(args.frozen_schedule.read_text(encoding='utf-8'))
        if frozen['a1_sha256'] != A1_SHA or frozen['frames'] != 800 or \
           tuple(sorted(frozen['keyframes'])) != tuple(sorted(SPLIT_NAMES)):
            raise ValueError('frozen keyframe schedule provenance mismatch')
        schedule = {name: set(stems) for name, stems in frozen['keyframes'].items()}
        if any('000000' not in stems for stems in schedule.values()):
            raise ValueError('every sequence must start with a frozen keyframe')
        evaluate(args, dataset, model, device, calibration, schedule)


if __name__ == '__main__':
    main()
