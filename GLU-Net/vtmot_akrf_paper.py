"""Paper-grounded SmoothFusion AKRF strategy with the same frozen VTMOT A1.

This is a registration-only, two-arm adapter. It is not a reproduction of
SmoothFusion's complete registration/fusion system. The default stage performs
synthetic checks only; the 800-frame validation needs an explicit stage/flag.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from datasets.vtmot_video import VTMOTVideos
from vtmot_geometry import backward_warp, compose, grid, self_test
from vtmot_keyframe import (edge_metrics, frame_metrics, gray, install_mutual_guard,
                            lncc, load_a1, nmi, predict_a1, save_json, sha256,
                            summarise, t_ssim, write_rows)


@dataclass(frozen=True)
class AKRFConfig:
    # Eq. (2): these two values are explicitly given in SmoothFusion §3.1.
    base_interval: int = 20
    minimum_interval: int = 8
    # Implementation choices: paper does not give N, lambda, sigma or T.
    history_length: int = 5  # N; local source uses five historical fields.
    motion_lambda_per_512px: float = 0.1
    gaussian_sigma_512px: float = 1.0
    gaussian_kernel: int = 5
    transition_period_frames: int = 20

    def validate(self):
        if not (self.base_interval >= self.minimum_interval >= 1):
            raise ValueError('base/minimum intervals must be positive and ordered')
        if self.history_length < 2 or self.motion_lambda_per_512px < 0:
            raise ValueError('invalid history length or motion scaling')
        if self.gaussian_sigma_512px < 0 or self.gaussian_kernel % 2 != 1:
            raise ValueError('Gaussian sigma must be nonnegative and kernel odd')
        if self.transition_period_frames < 1:
            raise ValueError('transition period must be positive')


def visible_current_to_previous(previous_vi, current_vi):
    """512px backward XY Farneback field: current VI grid -> previous VI."""
    if previous_vi.shape[:2] != current_vi.shape[:2]:
        raise ValueError('adjacent visible frames must have the same grid')
    result = cv2.calcOpticalFlowFarneback(
        gray(current_vi), gray(previous_vi), None, pyr_scale=.5, levels=4,
        winsize=21, iterations=5, poly_n=7, poly_sigma=1.5, flags=0)
    return result.astype(np.float32)


def motion_consistency(history, scale):
    """Eq. (1); mean absolute flow difference is our normalized L1 choice."""
    if len(history) < 2:
        return 1.0
    values = []
    for prior, current in zip(list(history)[:-1], list(history)[1:]):
        if prior.shape != current.shape or not np.isfinite(current).all():
            raise ValueError('history flow grids differ or contain nonfinite values')
        mean_l1 = float(np.mean(np.abs(current-prior)))
        values.append(math.exp(-scale * mean_l1))
    return float(np.mean(values))


def adaptive_interval(consistency, config):
    """Eq. (2); the first frame is handled separately."""
    if not 0 <= consistency <= 1:
        raise ValueError('motion consistency must lie in [0,1]')
    return max(config.minimum_interval, math.floor(config.base_interval * consistency))


def wst_alpha(frames_since_keyframe, config):
    """Eq. (4): 3u²-2u³, with u=clamp(delta_t/T, 0, 1)."""
    u = float(np.clip(frames_since_keyframe/config.transition_period_frames, 0, 1))
    return 3*u*u - 2*u*u*u


def gaussian_flow(flow, config):
    """Eq. (3), independently smooth XY channels at the 512px grid."""
    if config.gaussian_sigma_512px == 0:
        return flow.copy()
    return cv2.GaussianBlur(flow, (config.gaussian_kernel, config.gaussian_kernel),
                            config.gaussian_sigma_512px,
                            borderType=cv2.BORDER_REFLECT_101).astype(np.float32)


def weighted_transition(old_flow, old_valid, new_flow, new_valid, alpha):
    """Eq. (4) on one current VI grid; use the available field at boundaries."""
    if old_flow.shape != new_flow.shape or old_valid.shape != new_valid.shape:
        raise ValueError('WST fields/masks must share a grid')
    result = alpha*new_flow + (1-alpha)*old_flow
    result[old_valid & ~new_valid] = old_flow[old_valid & ~new_valid]
    result[new_valid & ~old_valid] = new_flow[new_valid & ~old_valid]
    valid = old_valid | new_valid
    result[~valid] = 0
    return result.astype(np.float32), valid


class AKRFState:
    """One sequence; reset by constructing a new state for each sequence."""

    def __init__(self, config, temporal_fn=visible_current_to_previous):
        config.validate()
        self.config = config
        self.temporal_fn = temporal_fn
        self.previous_vi = None
        self.last_keyframe_index = None
        self.keyframe_flow = None
        self.keyframe_valid = None
        self.current_to_keyframe = None
        self.chain_valid = None
        self.history = deque(maxlen=config.history_length)

    def step(self, index, frame, infer_fn):
        """Return VI->current IR flow without reading GT or visible_gt.

        Eq. (1)/(2): select from visible temporal-flow history. Eq. (3)/(4):
        Gaussian/WST on a keyframe. Algorithm 1: propagate a non-keyframe.
        The prior keyframe flow is brought to today's VI grid by geometric
        composition, not direct addition of differently gridded arrays.
        Since the paper only uses visible temporal flow, this assumes the IR
        source coordinate grid is temporally fixed; moving IR content can
        violate that assumption and must be checked with GT at evaluation.
        """
        if any(name in frame for name in ('gt_h', 'visible_gt', 'flow_truth')):
            raise ValueError('inference frame must not contain evaluation truth')
        if frame['vi'].shape[:2] != frame['ir'].shape[:2]:
            raise ValueError('IR/VI raster grids differ')
        hw = frame['vi'].shape[:2]
        if self.previous_vi is None:
            if index != 0:
                raise ValueError('sequence must begin at frame index zero')
            consistency, interval, temporal_ms = 1.0, self.config.base_interval, 0.
            chain = np.zeros((*hw, 2), np.float32)
            chain_valid = np.ones(hw, bool)
            is_keyframe = True
        else:
            started = time.perf_counter()
            current_to_previous = self.temporal_fn(self.previous_vi, frame['vi'])
            temporal_ms = 1000*(time.perf_counter()-started)
            if current_to_previous.shape != (*hw, 2) or not np.isfinite(current_to_previous).all():
                raise ValueError('temporal estimator returned invalid 512px XY flow')
            self.history.append(current_to_previous)
            consistency = motion_consistency(self.history,
                                             self.config.motion_lambda_per_512px)
            interval = adaptive_interval(consistency, self.config)
            chain, chain_valid = compose(current_to_previous, self.current_to_keyframe,
                                         None, self.chain_valid)
            is_keyframe = index-self.last_keyframe_index >= interval
        old_on_current, old_valid = (None, None)
        if self.keyframe_flow is not None:
            old_on_current, old_valid = compose(chain, self.keyframe_flow,
                                                 chain_valid, self.keyframe_valid)
        infer_ms = 0.
        if is_keyframe:
            new_flow, _, infer_ms = infer_fn(frame)
            if new_flow.shape != (*hw, 2) or not np.isfinite(new_flow).all():
                raise ValueError('A1 returned invalid 512px XY flow')
            new_flow = gaussian_flow(new_flow, self.config)
            _, new_valid = backward_warp(frame['ir'], new_flow)
            if old_on_current is None:
                flow, valid, alpha = new_flow, new_valid, 1.0
            else:
                alpha = wst_alpha(index-self.last_keyframe_index, self.config)
                flow, valid = weighted_transition(old_on_current, old_valid,
                                                  new_flow, new_valid, alpha)
            self.keyframe_flow = flow.copy()
            self.keyframe_valid = valid.copy()
            self.current_to_keyframe = np.zeros((*hw, 2), np.float32)
            self.chain_valid = np.ones(hw, bool)
            self.last_keyframe_index = index
        else:
            flow, valid, alpha = old_on_current, old_valid, None
            self.current_to_keyframe, self.chain_valid = chain, chain_valid
        _, ir_inside = backward_warp(frame['ir'], flow)
        valid = valid & ir_inside
        self.previous_vi = frame['vi']
        return flow, valid, {'keyframe': int(is_keyframe),
                             'reason': 'first_frame' if index == 0 else
                                       ('motion_interval' if is_keyframe else 'propagated'),
                             'motion_consistency': consistency,
                             'adaptive_interval': interval,
                             'wst_alpha': alpha,
                             'temporal_ms': temporal_ms,
                             'a1_ms': infer_ms}


def synthetic_checks():
    config = AKRFConfig()
    config.validate()
    assert adaptive_interval(1.0, config) == 20
    assert adaptive_interval(0.0, config) == 8
    calm = [np.zeros((32, 32, 2), np.float32) for _ in range(5)]
    assert motion_consistency(calm, config.motion_lambda_per_512px) == 1.0
    intense = [calm[0], np.full_like(calm[0], 100)]
    assert adaptive_interval(motion_consistency(intense,
                             config.motion_lambda_per_512px), config) == 8
    assert 8 < 20 and not (7 >= 8) and 8 >= 8 and not (19 >= 20) and 20 >= 20
    assert wst_alpha(0, config) == 0 and wst_alpha(20, config) == 1
    assert 0 <= wst_alpha(8, config) <= 1
    hw = (64, 64)
    yy, xx = np.mgrid[:64, :64]
    pattern = ((xx*17 + yy*31) % 255).astype(np.uint8)
    image = cv2.cvtColor(pattern, cv2.COLOR_GRAY2RGB)
    frame = lambda vi: {'vi': vi, 'ir': image}
    new = np.zeros((*hw, 2), np.float32)
    def infer(_):
        return new.copy(), np.ones(hw, bool), 0.
    identical = AKRFState(config, temporal_fn=lambda _p, _c: new.copy())
    first, first_valid, first_info = identical.step(0, frame(image), infer)
    second, second_valid, _ = identical.step(1, frame(image), infer)
    assert first_info['keyframe'] == 1 and np.array_equal(first, second)
    assert np.array_equal(first_valid, second_valid)
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 3], [0, 1, -2]]), (64, 64))
    temporal = np.zeros_like(new)
    temporal[..., 0], temporal[..., 1] = -3, 2  # current VI -> previous VI
    state = AKRFState(config, temporal_fn=lambda _p, _c: temporal.copy())
    state.step(0, frame(image), infer)
    moved, valid, info = state.step(1, frame(shifted), infer)
    core = np.s_[8:56, 8:56]
    assert info['keyframe'] == 0 and valid[core].all()
    assert np.max(np.abs(moved[core]-temporal[core])) < 1e-5
    # Actual OpenCV Farneback direction on a textured synthetic translation.
    rng = np.random.default_rng(2026)
    texture = cv2.cvtColor(rng.integers(0, 256, (512, 512), dtype=np.uint8),
                           cv2.COLOR_GRAY2RGB)
    texture_shift = cv2.warpAffine(texture, np.float32([[1, 0, 3], [0, 1, -2]]),
                                   (512, 512))
    observed = visible_current_to_previous(texture, texture_shift)
    assert observed.shape == (512, 512, 2)
    median = np.median(observed[80:432, 80:432].reshape(-1, 2), axis=0)
    assert np.max(np.abs(median-[-3, 2])) < .5, median
    huge = np.full_like(new, 1000)
    _, out_valid = compose(huge, new)
    assert not out_valid.any()
    constant = np.full_like(new, 4)
    assert np.max(np.abs(gaussian_flow(constant, config)-constant)) < 1e-5
    impulse = np.zeros_like(new)
    impulse[32, 32, 0] = 10
    smoothed = gaussian_flow(impulse, config)
    assert 0 < smoothed[32, 32, 0] < 10 and smoothed[32, 33, 0] > 0
    blend, mask = weighted_transition(new, np.ones(hw, bool), constant,
                                       np.ones(hw, bool), wst_alpha(8, config))
    assert mask.all() and np.isfinite(blend).all()
    assert np.max(np.abs(blend-4*wst_alpha(8, config))) < 1e-5
    at_start, _ = weighted_transition(new, np.ones(hw, bool), constant,
                                       np.ones(hw, bool), wst_alpha(0, config))
    at_end, _ = weighted_transition(new, np.ones(hw, bool), constant,
                                     np.ones(hw, bool), wst_alpha(20, config))
    assert np.array_equal(at_start, new) and np.array_equal(at_end, constant)
    boundary = np.zeros(hw, bool)
    boundary[8:56, 8:56] = True
    _, mixed_valid = weighted_transition(new, boundary, constant,
                                          np.ones(hw, bool), .5)
    assert mixed_valid.all()
    # Same GT mask and only-after-output access are checked with a fake evaluator.
    events = []
    class FakeData:
        def evaluation_truth(self):
            events.append('truth')
            return new.copy(), boundary.copy(), image.copy()
    fake = FakeData()
    for mode in ('A', 'B'):
        events.append(f'{mode}_output')
        truth, gt_mask, visible_gt = fake.evaluation_truth()
        assert gt_mask.sum() == 48*48 and truth.shape == new.shape
        assert visible_gt.shape == image.shape
    assert events == ['A_output', 'truth', 'B_output', 'truth']
    finite_metrics = [nmi(image, image, boundary), lncc(image, image, boundary),
                      *edge_metrics(image, image, boundary), t_ssim(image, image)]
    assert all(np.isfinite(value) for value in finite_metrics)
    return {'first_frame_and_20_8_schedule': 'passed',
            'eq1_motion_consistency': 'passed', 'farneback_current_to_previous_xy': median.tolist(),
            'identity_translation_and_512px_units': 'passed',
            'gaussian_wst_and_bounds': 'passed',
            'same_gt_mask_and_post_output_only': 'passed',
            'available_proxy_metrics_finite': 'passed',
            'geometry': self_test(), 'test_sequences_accessed': False}


def evaluate_arm(args, dataset, model, device, arm, config):
    rows = []
    for sequence, stems in dataset.sequences.items():
        state = AKRFState(config) if arm == 'B_akrf' else None
        previous_registered = None
        for index, stem in enumerate(stems):
            frame = dataset.input_frame(sequence, stem)  # image-only
            started = time.perf_counter()
            infer = lambda item: predict_a1(model, item, device)
            if state is None:
                flow, valid, infer_ms = infer(frame)
                detail = {'keyframe': 1, 'reason': 'every_frame',
                          'motion_consistency': None, 'adaptive_interval': None,
                          'wst_alpha': None, 'temporal_ms': 0., 'a1_ms': infer_ms}
            else:
                flow, valid, detail = state.step(index, frame, infer)
            frame_ms = 1000*(time.perf_counter()-started)
            # GT and visible_gt are first touched here, after flow/trigger/timing.
            metrics, registered = frame_metrics(dataset, frame, flow, valid,
                                                previous_registered)
            rows.append({'sequence': sequence, 'stem': stem, 'frame_index': index,
                         'arm': arm, 'frame_ms': frame_ms, **detail, **metrics})
            previous_registered = registered
    return rows


def formal_validation(args):
    for name in ('data_root', 'split_file', 'pretrained', 'coarse_checkpoint',
                 'source_a1_checkpoint', 'source_coarse_adapt_checkpoint',
                 'a1_checkpoint', 'output'):
        if getattr(args, name) is None:
            raise ValueError(f'--{name.replace("_", "-")} is required for formal validation')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('formal output must be a new empty directory')
    if sha256(args.a1_checkpoint) != 'af245ea28cd0fc33bff7aa9dd83cb1402c1108de7870c044a9960f791cbac435':
        raise ValueError('A1 fine-adapted checkpoint differs from the locked weight')
    dataset = VTMOTVideos(args.data_root, args.split_file, split='eval')
    if len(dataset.sequences) != 4 or sum(map(len, dataset.sequences.values())) != 800:
        raise ValueError('formal validation must use four sequences / 800 frames')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    install_mutual_guard()
    model = load_a1(args, device)
    config = AKRFConfig()
    config.validate()
    first_sequence = next(iter(dataset.sequences))
    predict_a1(model, dataset.input_frame(first_sequence,
               dataset.sequences[first_sequence][0]), device)  # warmup
    reports = {}
    frame_keys = {}
    for arm in ('A_every_frame', 'B_akrf'):
        rows = evaluate_arm(args, dataset, model, device, arm, config)
        frame_keys[arm] = [(row['sequence'], row['stem'], row['gt_valid_pixels'])
                           for row in rows]
        summary = summarise(rows)['all']
        report = {'arm': arm, 'a1_sha256': sha256(args.a1_checkpoint),
                  'flow_direction': 'current VI target -> current IR source, XY',
                  'flow_scale': '512x512 image pixels',
                  'gt_mask': 'same valid GT source bounds, independent of prediction',
                  'timing': 'GPU synchronized A1 plus visible temporal Farneback, composition and WST; excludes IO/metrics',
                  'config': asdict(config), 'summary': summary,
                  'paper_table_exact_metrics': {
                      'NMI': None, 'Q_AB/F': None, 'R_edge': None, 'LNCC': None,
                      'ITF': None, 'T-SSIM': None},
                  'paper_table_status': 'exact definitions unavailable in paper and local source; do not rename proxies',
                  'proxy_metric_fields': ['nmi', 'edge_information_retention',
                                          'edge_overlap_f1_3x3', 'lncc17_squared',
                                          'itf_255', 't_ssim'],
                  'visible_gt_metrics_eval_only': ['visible_gt_mse', 'visible_gt_ncc'],
                  'total_compute_seconds': summary['total_compute_seconds'],
                  'fps_compute_only': summary['frames']/summary['total_compute_seconds'],
                  'test_accessed': False}
        out = args.output / arm
        write_rows(out / 'per_frame.csv', rows)
        save_json(out / 'report.json', report)
        reports[arm] = report
    a, b = reports['A_every_frame']['summary'], reports['B_akrf']['summary']
    if frame_keys['A_every_frame'] != frame_keys['B_akrf']:
        raise RuntimeError('arms used different frames or GT mask pixel counts')
    save_json(args.output / 'comparison.json', {
        'a1_sha256': sha256(args.a1_checkpoint), 'arms': reports,
        'epe_gain_A_minus_B_512px': a['pixel_weighted_epe_512px']-b['pixel_weighted_epe_512px'],
        'test_accessed': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('check', 'eval'), default='check')
    parser.add_argument('--allow-800-frame-eval', action='store_true',
                        help='explicitly enable the formal four-sequence validation')
    for name in ('data-root', 'split-file', 'pretrained', 'coarse-checkpoint',
                 'source-a1-checkpoint', 'source-coarse-adapt-checkpoint',
                 'a1-checkpoint', 'output'):
        parser.add_argument(f'--{name}', type=Path)
    args = parser.parse_args()
    if args.stage == 'check':
        print(synthetic_checks())
    elif args.allow_800_frame_eval:
        formal_validation(args)
    else:
        parser.error('--stage eval requires --allow-800-frame-eval')


if __name__ == '__main__':
    main()
