"""A1-only VTMOT keyframe registration; no fusion and no test-split access.

Stages: check -> baseline -> calibrate -> compare -> transition. The baseline
and train-only calibration files are protocol gates, not model-selection data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from datasets.vtmot_video import VTMOTVideos, direction_check
from roadscene_fusion_hierarchical import make_arm
from vtmot_geometry import backward_warp, grid, propagate, sample, self_test


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                    encoding='utf-8')


def load_a1(args, device):
    recipe = SimpleNamespace(pretrained=args.pretrained,
                             coarse_checkpoint=args.coarse_checkpoint)
    model, _ = make_arm(recipe, 'A1', device)
    payload = torch.load(args.a1_checkpoint, map_location='cpu', weights_only=False)
    if payload.get('arm') != 'A1':
        raise ValueError('checkpoint is not the selected A1 arm')
    if payload.get('pretrained_sha256') != sha256(args.pretrained):
        raise ValueError('A1 and base pretrained weights do not match')
    if payload.get('coarse_sha256') != sha256(args.coarse_checkpoint):
        raise ValueError('A1 and fixed coarse checkpoint do not match')
    model.load_state_dict(payload['model_state_dict'], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError('A1 must be completely frozen')
    return model


@torch.inference_mode()
def predict_a1(model, frame, device):
    source = torch.from_numpy(frame['ir'].transpose(2, 0, 1).copy())[None].to(device)
    target = torch.from_numpy(frame['vi'].transpose(2, 0, 1).copy())[None].to(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    prediction = model({'source_image': source, 'target_image': target}, device)['final512']
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = 1000 * (time.perf_counter() - start)
    xy = prediction[0].permute(1, 2, 0).float().cpu().numpy().copy()
    if xy.shape != (512, 512, 2) or not np.isfinite(xy).all():
        raise RuntimeError(f'A1 emitted invalid flow: {xy.shape}')
    _, bounds = backward_warp(frame['ir'], xy)
    return xy, bounds, elapsed


def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def farneback(first, second):
    """Output first-grid pixels -> corresponding second-grid pixels, XY."""
    return cv2.calcOpticalFlowFarneback(gray(first), gray(second), None,
                                        pyr_scale=.5, levels=4, winsize=21,
                                        iterations=5, poly_n=7, poly_sigma=1.5,
                                        flags=0).astype(np.float32)


def temporal_pair(previous, current):
    start = time.perf_counter()
    vi_cp = farneback(current['vi'], previous['vi'])
    vi_pc = farneback(previous['vi'], current['vi'])
    ir_pc = farneback(previous['ir'], current['ir'])
    ir_cp = farneback(current['ir'], previous['ir'])
    vi_cycle, vi_bounds = sample(vi_pc, grid(vi_cp.shape[:2]) + vi_cp)
    ir_cycle, ir_bounds = sample(ir_cp, grid(ir_pc.shape[:2]) + ir_pc)
    vi_error = np.linalg.norm(vi_cp + vi_cycle, axis=-1)
    ir_error = np.linalg.norm(ir_pc + ir_cycle, axis=-1)
    elapsed = 1000 * (time.perf_counter() - start)
    return {'vi': vi_cp, 'ir': ir_pc, 'vi_error': vi_error,
            'ir_error': ir_error, 'vi_bounds': vi_bounds,
            'ir_bounds': ir_bounds, 'ms': elapsed}


def temporal_self_test():
    rng = np.random.default_rng(2026)
    pattern = rng.integers(0, 256, (80, 96), dtype=np.uint8)
    first = cv2.cvtColor(pattern, cv2.COLOR_GRAY2RGB)
    second = cv2.warpAffine(first, np.array([[1, 0, 3], [0, 1, -2]], np.float32),
                            (96, 80))
    forward = farneback(first, second)
    backward = farneback(second, first)
    core = (slice(15, 65), slice(15, 81))
    median_f = np.median(forward[core].reshape(-1, 2), axis=0)
    median_b = np.median(backward[core].reshape(-1, 2), axis=0)
    if np.max(np.abs(median_f - [3, -2])) > .5 or np.max(np.abs(median_b - [-3, 2])) > .5:
        raise AssertionError(f'Farneback direction failed: {median_f}, {median_b}')
    return {'forward_xy': median_f.tolist(), 'backward_xy': median_b.tolist()}


def temporal_quality(temporal, previous_registration, previous_valid,
                     fb_threshold):
    vi_ok = temporal['vi_bounds'] & (temporal['vi_error'] <= fb_threshold)
    ir_ok = temporal['ir_bounds'] & (temporal['ir_error'] <= fb_threshold)
    predicted, valid = propagate(temporal['vi'], previous_registration,
                                 temporal['ir'], vi_ok, previous_valid, ir_ok)
    # At current VI p, q is previous VI, r is previous IR. Compare the two
    # modality-specific temporal displacements at corresponding locations.
    q = grid(temporal['vi'].shape[:2]) + temporal['vi']
    old_reg_at, _ = sample(previous_registration, q)
    r = q + old_reg_at
    ir_at, _ = sample(temporal['ir'], r)
    motion_difference = np.linalg.norm(temporal['vi'] + ir_at, axis=-1)
    valid_fraction = float(valid.mean())
    values = (temporal['vi_error'][vi_ok], temporal['ir_error'][ir_ok])
    fb_values = np.concatenate([value for value in values if value.size]) if any(
        value.size for value in values) else np.array([np.inf])
    return {'flow': predicted, 'valid': valid,
            'valid_fraction': valid_fraction,
            'fb_median_px': float(np.median(fb_values)),
            'motion_median_px': float(np.median(motion_difference[valid]))
            if valid.any() else float('inf')}


def mse_ncc(first, second, valid):
    x = first[valid].astype(np.float64) / 255.
    y = second[valid].astype(np.float64) / 255.
    if not len(x):
        return float('nan'), float('nan')
    mse = np.mean((x-y)**2)
    x, y = x.ravel(), y.ravel()
    ncc = np.mean((x-x.mean())*(y-y.mean())) / np.sqrt(
        (np.mean((x-x.mean())**2)+1e-5) * (np.mean((y-y.mean())**2)+1e-6))
    return float(mse), float(ncc)


def sobel(image):
    intensity = gray(image).astype(np.float32)
    gx = cv2.Sobel(intensity, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(intensity, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def nmi(first, second, valid):
    x = (gray(first)[valid] // 4).astype(np.int32)
    y = (gray(second)[valid] // 4).astype(np.int32)
    if not len(x):
        return float('nan')
    joint = np.bincount(x * 64 + y, minlength=4096).reshape(64, 64).astype(np.float64)
    joint /= joint.sum()
    px, py = joint.sum(1), joint.sum(0)
    hxy = -np.sum(joint[joint > 0] * np.log(joint[joint > 0]))
    hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
    hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
    return float((hx+hy-hxy) / max(.5*(hx+hy), 1e-12))


def lncc(first, second, valid, window=17):
    a = gray(first).astype(np.float64) / 255.
    b = gray(second).astype(np.float64) / 255.
    box = lambda x: cv2.boxFilter(x, -1, (window, window), normalize=False,
                                   borderType=cv2.BORDER_CONSTANT)
    count = window * window
    sa, sb = box(a), box(b)
    ma, mb = sa/count, sb/count
    cross = box(a*b) - mb*sa - ma*sb + ma*mb*count
    va = box(a*a) - 2*ma*sa + ma*ma*count
    vb = box(b*b) - 2*mb*sb + mb*mb*count
    return float(np.mean((cross*cross/(va*vb+1e-5))[valid]))


def edge_metrics(first, second, valid):
    a, b = sobel(first), sobel(second)
    retention = float(np.minimum(a[valid], b[valid]).sum() /
                      max(1e-12, a[valid].sum()))
    ta, tb = np.quantile(a[valid], .75), np.quantile(b[valid], .75)
    binary_a, binary_b = (a >= ta) & valid, (b >= tb) & valid
    dilated_b = cv2.dilate(binary_b.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    dilated_a = cv2.dilate(binary_a.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    precision = float((binary_b & dilated_a).sum() / max(1, binary_b.sum()))
    recall = float((binary_a & dilated_b).sum() / max(1, binary_a.sum()))
    overlap = 2*precision*recall/max(precision+recall, 1e-12)
    return retention, overlap


def t_ssim(first, second):
    a, b = gray(first).astype(np.float32), gray(second).astype(np.float32)
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    va = cv2.GaussianBlur(a*a, (11, 11), 1.5) - mu_a*mu_a
    vb = cv2.GaussianBlur(b*b, (11, 11), 1.5) - mu_b*mu_b
    cab = cv2.GaussianBlur(a*b, (11, 11), 1.5) - mu_a*mu_b
    return float(np.mean(((2*mu_a*mu_b+(.01*255)**2) *
                          (2*cab+(.03*255)**2)) /
                         ((mu_a*mu_a+mu_b*mu_b+(.01*255)**2) *
                          (va+vb+(.03*255)**2))))


def frame_metrics(dataset, frame, flow, predicted_valid, previous_registered=None):
    # Evaluation GT and visible_gt are loaded only after the method has decided.
    truth, valid, visible_gt = dataset.evaluation_truth(
        frame['sequence'], frame['stem'], frame['original_hw'])
    err = np.linalg.norm(flow-truth, axis=-1)
    warped_ir, ir_inside = backward_warp(frame['ir'], flow)
    warped_gt, _ = backward_warp(visible_gt, flow)
    # Use the same GT mask for all arms. Out-of-bounds predictions retain the
    # remap border value and are not allowed to hide difficult pixels.
    spatial = valid
    edge_retention, edge_overlap = edge_metrics(frame['vi'], warped_ir, spatial)
    same_mse, same_ncc = mse_ncc(frame['vi'], warped_gt, spatial)
    result = {'epe_512px': float(err[valid].mean()),
              'zero_epe_512px': float(np.linalg.norm(truth, axis=-1)[valid].mean()),
              'gt_valid_pixels': int(valid.sum()),
              'gt_valid_fraction': float(valid.mean()),
              'prediction_valid_fraction': float(predicted_valid.mean()),
              'nmi': nmi(frame['vi'], warped_ir, spatial),
              'edge_information_retention': edge_retention,
              'edge_overlap_f1_3x3': edge_overlap,
              'lncc17_squared': lncc(frame['vi'], warped_ir, spatial),
              'visible_gt_mse': same_mse, 'visible_gt_ncc': same_ncc}
    if previous_registered is not None:
        result['itf_255'] = float(np.mean(np.abs(
            gray(warped_ir).astype(np.float32) -
            gray(previous_registered).astype(np.float32))))
        result['t_ssim'] = t_ssim(previous_registered, warped_ir)
    return result, warped_ir


def summarise(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['sequence'], []).append(row)
    metric_keys = ('epe_512px', 'zero_epe_512px', 'gt_valid_fraction',
                   'prediction_valid_fraction', 'nmi', 'edge_information_retention',
                   'edge_overlap_f1_3x3', 'lncc17_squared', 'itf_255', 't_ssim',
                   'visible_gt_mse', 'visible_gt_ncc', 'frame_ms')
    result = {}
    for name, group in grouped.items():
        result[name] = {key: float(np.mean([item[key] for item in group if key in item]))
                        for key in metric_keys if any(key in item for item in group)}
        result[name].update({'frames': len(group),
                             'keyframe_fraction': sum(r['keyframe'] for r in group)/len(group),
                             'switch_epe_512px': float(np.mean([
                                 r['epe_512px'] for r in group
                                 if r['keyframe'] and r['frame_index'] > 0]))
                             if any(r['keyframe'] and r['frame_index'] > 0 for r in group)
                             else None,
                             'propagated_epe_512px': float(np.mean([
                                 r['epe_512px'] for r in group if not r['keyframe']]))
                             if any(not r['keyframe'] for r in group) else None,
                             'keyframes': [{'stem': r['stem'], 'reason': r['reason']}
                                           for r in group if r['keyframe']],
                             'total_compute_seconds': sum(r['frame_ms'] for r in group)/1000,
                             'worst_frames': [{'stem': r['stem'], 'epe_512px': r['epe_512px']}
                                              for r in sorted(group, key=lambda x: -x['epe_512px'])[:5]]})
    total_pixels = sum(r['gt_valid_pixels'] for r in rows)
    result['all'] = {
        'frames': len(rows), 'gt_valid_pixels': total_pixels,
        'pixel_weighted_epe_512px': sum(r['epe_512px']*r['gt_valid_pixels'] for r in rows)/total_pixels,
        'pixel_weighted_zero_epe_512px': sum(r['zero_epe_512px']*r['gt_valid_pixels'] for r in rows)/total_pixels,
        'frame_mean_epe_512px': float(np.mean([r['epe_512px'] for r in rows])),
        'keyframe_fraction': sum(r['keyframe'] for r in rows)/len(rows),
        'total_compute_seconds': sum(r['frame_ms'] for r in rows)/1000,
        'per_sequence': {k: v for k, v in result.items() if k != 'all'}}
    result['all']['frame_mean_metrics'] = {
        key: float(np.mean([item[key] for item in rows if key in item]))
        for key in metric_keys if any(key in item for item in rows)}
    return result


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def paired_report(current_rows, reference_path, output, reference_name):
    reference = {(r['sequence'], r['stem']): r for r in read_rows(reference_path)}
    paired = []
    for row in current_rows:
        key = row['sequence'], row['stem']
        if key not in reference:
            raise ValueError(f'missing reference frame {key}')
        original = reference[key]
        if int(row['gt_valid_pixels']) != int(original['gt_valid_pixels']):
            raise ValueError(f'GT mask pixel count changed at {key}')
        paired.append({'sequence': key[0], 'stem': key[1],
                       'reference_epe_512px': float(original['epe_512px']),
                       'candidate_epe_512px': row['epe_512px'],
                       'epe_gain_512px': float(original['epe_512px'])-row['epe_512px'],
                       'keyframe': row['keyframe'], 'reason': row['reason'],
                       'reference_frame_ms': float(original['frame_ms']),
                       'candidate_frame_ms': row['frame_ms']})
    if len(paired) != len(reference):
        raise ValueError('reference and current method have different frame counts')
    write_rows(output / f'paired_vs_{reference_name}.csv', paired)
    switch = [r for r in paired if r['keyframe'] and r['reason'] != 'sequence_start']
    return {'reference': reference_name, 'frames': len(paired),
            'improved_frames': sum(r['epe_gain_512px'] > 0 for r in paired),
            'mean_epe_gain_512px': float(np.mean([r['epe_gain_512px'] for r in paired])),
            'switch_frames': len(switch),
            'switch_mean_epe_gain_512px': float(np.mean([r['epe_gain_512px'] for r in switch]))
            if switch else None,
            'worst_regressions': sorted(paired, key=lambda r: r['epe_gain_512px'])[:10]}


def save_ghost(image_vi, warped_ir, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    a = gray(image_vi)
    b = gray(warped_ir)
    overlay = np.stack((a, b, (a.astype(np.uint16)+b)//2), axis=-1).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def thresholds_from_training(args, dataset, model, device):
    samples = {'valid_fraction': [], 'fb_median_px': [], 'motion_median_px': []}
    for sequence, stems in dataset.sequences.items():
        positions = np.linspace(1, len(stems)-1,
                                min(args.calibration_pairs_per_sequence, len(stems)-1),
                                dtype=int)
        for index in sorted(set(positions.tolist())):
            prev = dataset.input_frame(sequence, stems[index-1])
            curr = dataset.input_frame(sequence, stems[index])
            prior, valid, _ = predict_a1(model, prev, device)
            temporal = temporal_pair(prev, curr)
            quality = temporal_quality(temporal, prior, valid,
                                       args.fb_pixel_threshold)
            for key in samples:
                if np.isfinite(quality[key]):
                    samples[key].append(quality[key])
    if not all(samples.values()):
        raise RuntimeError('training sequences yielded no finite temporal confidence')
    return {'fb_pixel_threshold': args.fb_pixel_threshold,
            'minimum_valid_fraction': float(np.quantile(samples['valid_fraction'], .1)),
            'maximum_fb_median_px': float(np.quantile(samples['fb_median_px'], .9)),
            'maximum_motion_median_px': float(np.quantile(samples['motion_median_px'], .9)),
            'transition_max_flow_difference_px': args.transition_max_difference,
            'fixed_interval': args.fixed_interval,
            'adaptive_max_interval': args.adaptive_max_interval,
            'calibration_counts': {key: len(value) for key, value in samples.items()},
            'threshold_source': 'VTMOT train input/prediction only; GT not read'}


def trigger_reason(quality, config, frames_since_keyframe):
    if quality['valid_fraction'] < config['minimum_valid_fraction']:
        return 'low_temporal_coverage'
    if quality['fb_median_px'] > config['maximum_fb_median_px']:
        return 'forward_backward_error'
    if quality['motion_median_px'] > config['maximum_motion_median_px']:
        return 'cross_modal_motion_disagreement'
    if frames_since_keyframe >= config['adaptive_max_interval'] - 1:
        return 'maximum_interval'
    return None


def run_method(args, dataset, model, device, method, config=None):
    rows = []
    worst = []
    first_switches = []
    for sequence, stems in dataset.sequences.items():
        previous = previous_flow = previous_valid = previous_registered = None
        since_keyframe = 0
        for index, stem in enumerate(stems):
            frame = dataset.input_frame(sequence, stem)
            frame_start = time.perf_counter()
            temporal = None
            quality = None
            if index == 0:
                reason = 'sequence_start'
                timing = 0.
            elif method == 'baseline':
                reason = 'every_frame'
                timing = 0.
            else:
                temporal = temporal_pair(previous, frame)
                quality = temporal_quality(temporal, previous_flow, previous_valid,
                                           config['fb_pixel_threshold'])
                timing = temporal['ms']
                if quality['valid_fraction'] < config['minimum_valid_fraction']:
                    reason = 'low_temporal_coverage'
                elif quality['fb_median_px'] > config['maximum_fb_median_px']:
                    reason = 'forward_backward_error'
                elif method == 'fixed' and since_keyframe >= config['fixed_interval'] - 1:
                    reason = 'fixed_interval'
                elif method in ('adaptive', 'transition'):
                    reason = trigger_reason(quality, config, since_keyframe)
                else:
                    reason = None
            is_keyframe = reason is not None
            if is_keyframe:
                fresh_flow, valid, infer_ms = predict_a1(model, frame, device)
                timing += infer_ms
                flow = fresh_flow
                if method == 'transition' and index > 0 and quality is not None:
                    close = np.linalg.norm(quality['flow']-fresh_flow, axis=-1)
                    both = quality['valid'] & valid
                    close_enough = both & (close <= config['transition_max_flow_difference_px'])
                    # Image-only confidence: do not defer a visibly better A1 estimate.
                    old_warp, _ = backward_warp(frame['ir'], quality['flow'])
                    new_warp, _ = backward_warp(frame['ir'], fresh_flow)
                    old_edge = edge_metrics(frame['vi'], old_warp, both)[1] if both.any() else 0.
                    new_edge = edge_metrics(frame['vi'], new_warp, both)[1] if both.any() else 0.
                    if (both.mean() >= config['minimum_valid_fraction'] and
                            close_enough.sum() / max(1, both.sum()) >= .9 and
                            old_edge >= new_edge - .01):
                        flow = .5*quality['flow'] + .5*fresh_flow
                        reason += '+one_frame_transition'
                # Always reset propagation history to new A1, even if this
                # frame's displayed result was blended for one frame.
                state_flow, state_valid = fresh_flow, valid
                since_keyframe = 0
            else:
                flow, valid = quality['flow'], quality['valid']
                state_flow, state_valid = flow, valid
                since_keyframe += 1
            if not np.isfinite(flow).all():
                raise RuntimeError(f'nonfinite flow at {sequence}/{stem}')
            timing = 1000 * (time.perf_counter() - frame_start)
            metrics, registered = frame_metrics(dataset, frame, flow, valid,
                                                previous_registered)
            row = {'sequence': sequence, 'stem': stem, 'frame_index': index,
                   'method': method, 'keyframe': int(is_keyframe),
                   'reason': reason or 'propagated', 'frame_ms': timing,
                   'temporal_ms': temporal['ms'] if temporal else 0.,
                   'a1_ms': infer_ms if is_keyframe else 0.,
                   'temporal_valid_fraction': quality['valid_fraction'] if quality else 1.,
                   'temporal_fb_median_px': quality['fb_median_px'] if quality else 0.,
                   'temporal_motion_median_px': quality['motion_median_px'] if quality else 0.,
                   **metrics}
            rows.append(row)
            visual = (row['epe_512px'], sequence, stem, frame['vi'].copy(),
                      registered.copy())
            worst.append(visual)
            worst = sorted(worst, key=lambda x: -x[0])[:3]
            if is_keyframe and index > 0 and len(first_switches) < 3:
                first_switches.append(visual)
            previous, previous_flow, previous_valid = frame, state_flow, state_valid
            previous_registered = registered
        print(f'{method}: {sequence}, {len(stems)} frames, '
              f'EPE={np.mean([r["epe_512px"] for r in rows if r["sequence"] == sequence]):.3f}',
              flush=True)
    for label, collection in (('worst', worst), ('switch', first_switches)):
        for _, sequence, stem, vi, warped in collection:
            save_ghost(vi, warped, args.output / method / 'ghosts' /
                       f'{label}_{sequence}_{stem}.png')
    return rows


def load_config(args):
    path = args.output / 'train_thresholds.json'
    if not path.is_file():
        raise FileNotFoundError(f'run --stage calibrate first: {path}')
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload['a1_sha256'] != sha256(args.a1_checkpoint):
        raise ValueError('calibration was created with another A1 weight')
    return payload['thresholds']


def require_baseline(args):
    path = args.output / 'baseline' / 'report.json'
    if not path.is_file():
        raise FileNotFoundError(f'run --stage baseline before this stage: {path}')
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload['a1_sha256'] != sha256(args.a1_checkpoint):
        raise ValueError('baseline was created with another A1 weight')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True,
                        choices=('check', 'baseline', 'calibrate', 'compare', 'transition'))
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--split-file', required=True, type=Path)
    parser.add_argument('--pretrained', required=True, type=Path)
    parser.add_argument('--coarse-checkpoint', required=True, type=Path)
    parser.add_argument('--a1-checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--max-sequences', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--calibration-pairs-per-sequence', type=int, default=8)
    parser.add_argument('--fb-pixel-threshold', type=float, default=1.5)
    parser.add_argument('--fixed-interval', type=int, default=5)
    parser.add_argument('--adaptive-max-interval', type=int, default=8)
    parser.add_argument('--transition-max-difference', type=float, default=2.)
    args = parser.parse_args()
    if min(args.calibration_pairs_per_sequence, args.fixed_interval,
           args.adaptive_max_interval) < 1:
        parser.error('intervals and calibration count must be positive')
    if args.stage in ('compare', 'transition', 'calibrate'):
        require_baseline(args)
    if args.stage == 'calibrate' and (args.max_sequences or args.max_frames):
        parser.error('formal calibration must sample every train sequence; leave subset limits at zero')
    split = 'train' if args.stage == 'calibrate' else 'eval'
    dataset = VTMOTVideos(args.data_root, args.split_file, split=split,
                          max_sequences=args.max_sequences, max_frames=args.max_frames)
    if args.stage == 'check':
        checks = {'geometry': self_test(), 'dataset':
                  direction_check(dataset, next(iter(dataset.sequences)),
                                  next(iter(dataset.sequences.values()))[0]),
                  'temporal_direction': temporal_self_test(),
                  'sequences': list(dataset.sequences), 'frames':
                  sum(map(len, dataset.sequences.values())), 'test_accessed': False}
        print(json.dumps(checks, indent=2))
        save_json(args.output / 'check.json', checks)
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_a1(args, device)
    if args.stage == 'calibrate':
        path = args.output / 'train_thresholds.json'
        if path.exists():
            raise FileExistsError(f'train thresholds already locked: {path}')
        config = thresholds_from_training(args, dataset, model, device)
        save_json(path, {'a1_sha256': sha256(args.a1_checkpoint),
                         'pretrained_sha256': sha256(args.pretrained),
                         'split': 'train', 'thresholds': config})
        print(json.dumps(config, indent=2))
        return
    config = load_config(args) if args.stage != 'baseline' else None
    methods = {'baseline': ('baseline',), 'compare': ('fixed', 'adaptive'),
               'transition': ('transition',)}[args.stage]
    evaluated = {}
    for method in methods:
        out = args.output / method
        if (out / 'report.json').exists():
            raise FileExistsError(f'existing report will not be overwritten: {out}')
        wall_start = time.perf_counter()
        rows = run_method(args, dataset, model, device, method, config)
        evaluated[method] = rows
        report = {'method': method, 'split': 'eval', 'a1_sha256': sha256(args.a1_checkpoint),
                  'coarse_sha256': sha256(args.coarse_checkpoint),
                  'pretrained_sha256': sha256(args.pretrained),
                  'preprocess': 'common center crop plus bilinear resize to 512x512',
                  'direction': 'current visible_mis target -> current infrared source; XY',
                  'gt_mask': 'mapped GT source inside 512x512; identical for all methods',
                  'timing': 'A1 GPU synchronized; temporal 4x Farneback; per-frame compute, I/O and metrics excluded',
                  'a1_parameters': sum(p.numel() for p in model.parameters()),
                  'subset_limits': {'max_sequences': args.max_sequences,
                                    'max_frames': args.max_frames},
                  'secondary_metrics': 'registration-only proxies on visible_mis vs warped IR; '
                                       'ITF/T-SSIM on warped IR video, not fusion output',
                  'summary': summarise(rows)['all'], 'test_accessed': False}
        report['wall_seconds_including_io_metrics'] = time.perf_counter() - wall_start
        if method in ('fixed', 'adaptive'):
            report['paired_vs_baseline'] = paired_report(
                rows, args.output / 'baseline' / 'per_frame.csv', out, 'baseline')
        elif method == 'transition':
            reference = args.output / 'adaptive' / 'per_frame.csv'
            if not reference.is_file():
                raise FileNotFoundError('run --stage compare before transition')
            report['paired_vs_adaptive'] = paired_report(rows, reference, out, 'adaptive')
        write_rows(out / 'per_frame.csv', rows)
        save_json(out / 'report.json', report)
        print(json.dumps({'method': method, 'summary':
                          {k: v for k, v in report['summary'].items()
                           if k != 'per_sequence'}}, indent=2))
    if args.stage == 'compare':
        paired = paired_report(evaluated['adaptive'],
                               args.output / 'fixed' / 'per_frame.csv',
                               args.output, 'fixed')
        baseline = json.loads((args.output / 'baseline' / 'report.json').read_text(
            encoding='utf-8'))['summary']
        options = {method: summarise(rows)['all'] for method, rows in evaluated.items()}
        # Rule fixed before inspecting VTMOT validation outcomes: a propagation
        # method may cost at most 0.5 px EPE, must save >=20% compute, and may
        # call A1 on no more than half the frames. Among eligible arms use EPE.
        eligible = [name for name, value in options.items()
                    if value['pixel_weighted_epe_512px'] <= baseline['pixel_weighted_epe_512px'] + .5
                    and value['total_compute_seconds'] <= baseline['total_compute_seconds'] * .8
                    and value['keyframe_fraction'] <= .5]
        selected = min(eligible, key=lambda name: options[name]['pixel_weighted_epe_512px']) \
            if eligible else 'baseline'
        save_json(args.output / 'selection_val.json', {
            'rule': 'EPE <= per-frame A1 + 0.5px; compute <= 80% of A1; '
                    'keyframe fraction <= 50%; minimize EPE among eligible',
            'selected': selected, 'eligible': eligible,
            'selected_beats_zero_flow': (
                (options[selected]['pixel_weighted_epe_512px'] if selected in options
                 else baseline['pixel_weighted_epe_512px']) <
                baseline['pixel_weighted_zero_epe_512px']),
            'adaptive_vs_fixed': paired, 'test_accessed': False})


if __name__ == '__main__':
    main()
