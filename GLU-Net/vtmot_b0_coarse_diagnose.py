"""Read-only B0 coarse-to-local geometry audit; never trains or opens test.

check: synthetic geometry and checkpoint/protocol audit.
smoke: first two TRAIN frames only.
eval: exactly the locked four validation sequences / 800 frames, on request.
P1 zeros only the input of deconv4. Both probes retain identical weights.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import vtmot_dns_position_ablation as prior
from datasets.vtmot_video import VTMOTVideos
from roadscene_coarse import sha256_file
from roadscene_fusion_hierarchical import objective
from roadscene_refinement_audit import edge_quartile, pooled_truth
from vtmot_geometry import backward_warp, grid
from vtmot_keyframe import edge_metrics, install_mutual_guard, lncc, mse_ncc, nmi

STAGES = ('raw16', 'bilinear32', 'deconv32', 'decoded32', 'decoded64',
          'decoded128', 'final512')
WINDOWS = {'32': 4, '64': 4, '128': 4}
BINS = ((0, 8), (8, 16), (16, 32), (32, 64), (64, float('inf')))


def check_geometry():
    """Use a known VI->IR translation; 32-grid radius 4 means 64 full pixels."""
    truth = torch.zeros(1, 2, 512, 512)
    truth[:, 0] = 32
    truth[:, 1] = -16
    valid = torch.ones(1, 512, 512, dtype=torch.bool)
    native, selected = pooled_truth(truth, valid, 32)
    raw = torch.empty(1, 2, 16, 16)
    raw[:, 0] = 16
    raw[:, 1] = -8
    bilinear = F.interpolate(raw, (32, 32), mode='bilinear', align_corners=False)
    assert selected.all() and torch.allclose(native, bilinear * 2)
    zero = torch.zeros_like(bilinear)
    inside = (native - zero).abs().amax(1) <= 4 * 512 / 32
    assert inside.all()
    far = native.clone()
    far[:, 0] = 80
    assert not ((far-zero).abs().amax(1) <= 64).any()
    mapped = grid((512, 512)) + np.array([32, -16], dtype=np.float32)
    assert np.allclose(mapped[100, 100], [132, 84])
    return {'direction': 'VI target -> IR source, XY', 'raw16_unit': '256px',
            'reference32_unit': '256px', 'reference32_512px_xy': [32, -16],
            'first_window_radius_grid_px': 4,
            'first_window_radius_512px_per_axis': 64,
            'known_translation_and_window': 'passed'}


def load_fixed(args, device):
    comparison = json.loads(args.comparison.read_text(encoding='utf-8'))
    fine = comparison['arms']['B0']['fine']
    coarse = comparison['arms']['B0']['coarse']
    best_fine = args.comparison.parent/'B0'/'best_fine.pth'
    best_coarse = args.comparison.parent/'B0'/'best_coarse.pth'
    for path, report in ((best_fine, fine), (best_coarse, coarse)):
        actual = sha256_file(path)
        if actual != report['best_checkpoint_sha256']:
            raise ValueError(f'{path} SHA-256 mismatch: {actual}')
    if fine['parent_sha256'] != sha256_file(best_coarse):
        raise ValueError('fine checkpoint does not descend from B0 best coarse')
    if (fine['arm'], fine['stage'], fine['recipe'], fine['test_accessed']) != (
            'B0', 'fine', 'vtmot_dns_adaptive_scale_v1', False):
        raise ValueError('wrong fixed B0 experiment')
    payload = torch.load(best_fine, map_location='cpu', weights_only=False)
    expected = {'recipe': 'vtmot_dns_adaptive_scale_v1', 'arm': 'B0',
                'stage': 'fine', 'test_accessed': False,
                'parent_sha256': sha256_file(best_coarse),
                'split_sha256': sha256_file(args.split_file),
                'pretrained_sha256': sha256_file(args.pretrained),
                'attention_checkpoint_sha256': sha256_file(args.attention_checkpoint)}
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f'best fine {key} mismatch')
    model = prior.initialize('B', args, device)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    model.base.train_coarse_encoder = False
    model.eval()
    if model.base.coarse_dns.radii != (1, 2) or model.base.coarse_dns_order != 'between_sa_ca':
        raise AssertionError('fixed B0 is not SA -> DNS(1,2) -> CA')
    if model.base.local_dcn32 is not None or model.base.local_dcn64 is not None:
        raise AssertionError('fixed B0 unexpectedly has DCN')
    if model.base.div != 1 or model.base.iterative_refinement:
        raise AssertionError('diagnostic assumes original div=1 and no legacy iteration')
    audit = {'checkpoint': str(best_fine), 'fine_sha256': sha256_file(best_fine),
             'coarse_sha256': sha256_file(best_coarse),
             'selected_fine_epoch': fine['selected_epoch'],
             'reported_full_valid_epe_512px': fine['best_validation_final_epe_512px'],
             'reported_coarse16_epe_512px': coarse['best_image_metrics']['all']['coarse16']['epe_512px'],
             'reported_zero_epe_512px': fine['epoch_zero']['all']['stage_epe_512px']['zero'],
             'all_parameters_frozen': all(not p.requires_grad for p in model.parameters()),
             'normalization_eval': all(not m.training for m in model.modules()
                                       if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)),
             'test_accessed': False}
    return model, audit


@torch.no_grad()
def infer(model, frame, device, zero_deconv_input=False):
    """Return actual native tensors, with a one-call deconv4 input substitution."""
    base = model.base
    source = torch.from_numpy(frame['ir']).permute(2, 0, 1)[None].to(device)
    target = torch.from_numpy(frame['vi']).permute(2, 0, 1)[None].to(device)
    source, target, source256, target256, *_ = base.pre_process_data(source, target, device)
    captured = {}

    def pre4(_module, inputs):
        captured['raw16'] = inputs[0].detach().clone()
        replacement = torch.zeros_like(inputs[0]) if zero_deconv_input else inputs[0]
        captured['deconv_input16'] = replacement.detach().clone()
        return (replacement,)

    def post4(_module, _inputs, output):
        captured['deconv32'] = output.detach().clone()

    def post2(_module, inputs, output):
        captured['decoded64'] = inputs[0].detach().clone()
        captured['warp128_center'] = output.detach().clone()

    handles = [base.deconv4.register_forward_pre_hook(pre4),
               base.deconv4.register_forward_hook(post4),
               base.deconv2.register_forward_hook(post2)]
    base.capture_local64_pre_flow = True
    try:
        flow256, flow512 = base(target, source, target256, source256)
        captured['decoded32'] = flow256[1].detach().clone()
        captured['decoded128'] = flow512[1].detach().clone()
        captured['warp64_center'] = base.local64_pre_flow.detach().clone()
    finally:
        for handle in handles:
            handle.remove()
        base.capture_local64_pre_flow = False
        base.local64_pre_flow = None
    if not torch.equal(flow256[0], captured['raw16']):
        raise AssertionError('coarse output changed before deconv4')
    captured['bilinear32'] = F.interpolate(captured['raw16'], (32, 32),
                                           mode='bilinear', align_corners=False)
    captured['bilinear_input32'] = F.interpolate(captured['deconv_input16'], (32, 32),
                                                 mode='bilinear', align_corners=False)
    captured['warp32_center'] = captured['deconv32'].clone()
    captured['final512'] = F.interpolate(captured['decoded128'], (512, 512),
                                         mode='bilinear', align_corners=False)
    return captured


def full_flows(c):
    def up(name, unit):
        return F.interpolate(c[name], (512, 512), mode='bilinear',
                             align_corners=False) * unit
    return {'raw16': up('raw16', 2), 'bilinear32': up('bilinear32', 2),
            'deconv32': up('deconv32', 2), 'decoded32': up('decoded32', 2),
            'decoded64': up('decoded64', 1), 'decoded128': up('decoded128', 1),
            'final512': c['final512']}


def search(c, truth, valid, size, before, after):
    native_gt, queries = pooled_truth(truth, valid, size)
    center = c[before] * (2 if size == 32 else 1)
    prediction = c[after] * (2 if size == 32 else 1)
    if center.shape[-2:] != (size, size):
        raise AssertionError('search center grid mismatch')
    delta = native_gt-center
    inside = queries & (delta.abs().amax(1) <= WINDOWS[str(size)] * 512 / size)
    outside = queries & ~inside
    yy, xx = torch.meshgrid(torch.arange(size, device=truth.device),
                            torch.arange(size, device=truth.device), indexing='ij')
    xy = torch.stack((xx, yy))[None] + center * (size/512)
    center_inbounds = (xy[:, 0] >= 0) & (xy[:, 0] <= size-1) & (
        xy[:, 1] >= 0) & (xy[:, 1] <= size-1)
    def group(mask):
        count = int(mask.sum())
        return {'queries': count,
                'center_epe_sum': float(torch.linalg.vector_norm(delta, dim=1)[mask].sum()),
                'after_epe_sum': float(torch.linalg.vector_norm(native_gt-prediction, dim=1)[mask].sum())}
    return {'grid': size, 'radius_grid_pixels': WINDOWS[str(size)],
            'radius_512px_per_axis': WINDOWS[str(size)] * 512/size,
            'valid_query_rule': 'area GT valid fraction >0.8; pooled GT XY',
            'valid_queries': int(queries.sum()), 'center_inbounds_queries': int((queries & center_inbounds).sum()),
            'inside': group(inside), 'outside': group(outside)}, inside, outside, queries


def summarize(rows):
    result = {'frames': len(rows), 'valid_pixels': sum(r['valid_pixels'] for r in rows)}
    pixels = result['valid_pixels']
    result['zero_flow_epe_512px'] = sum(r['zero_epe_sum'] for r in rows)/pixels
    result['stages_full512'] = {name: {
        'epe': sum(r[f'{name}_epe_sum'] for r in rows)/pixels,
        'pck': {str(t): sum(r[f'{name}_pck{t}_hits'] for r in rows)/pixels for t in (1, 3, 5)}}
        for name in STAGES}
    result['windows'] = {str(size): {
        'valid_queries': sum(r[f'window{size}_queries'] for r in rows),
        'inside_fraction': sum(r[f'window{size}_inside'] for r in rows)/max(1, sum(r[f'window{size}_queries'] for r in rows)),
        'center_inbounds_fraction': sum(r[f'window{size}_inbounds'] for r in rows)/max(1, sum(r[f'window{size}_queries'] for r in rows)),
        'inside_center_epe': sum(r[f'window{size}_inside_center_epe_sum'] for r in rows)/max(1, sum(r[f'window{size}_inside'] for r in rows)),
        'inside_after_epe': sum(r[f'window{size}_inside_after_epe_sum'] for r in rows)/max(1, sum(r[f'window{size}_inside'] for r in rows)),
        'outside_center_epe': sum(r[f'window{size}_outside_center_epe_sum'] for r in rows)/max(1, sum(r[f'window{size}_outside'] for r in rows)),
        'outside_after_epe': sum(r[f'window{size}_outside_after_epe_sum'] for r in rows)/max(1, sum(r[f'window{size}_outside'] for r in rows))}
        for size in (32, 64, 128)}
    result['deconv_minus_bilinear_mean_512px'] = sum(r['deconv_shift_sum'] for r in rows)/max(1, sum(r['deconv_shift_count'] for r in rows))
    result['deconv_minus_bilinear_frame_p95_mean_512px'] = float(
        np.mean([r['deconv_shift_p95_512px'] for r in rows]))
    result['deconv_output_mean_magnitude_512px'] = float(
        np.mean([r['deconv32_mean_magnitude_512px'] for r in rows]))
    result['deconv_input16_max_abs_256px'] = max(r['deconv_input16_max_abs_256px']
                                                for r in rows)
    result['native_grid'] = {name: {
        'queries': sum(r[f'native_{name}_queries'] for r in rows),
        'epe_512px_at_native_queries': sum(r[f'native_{name}_epe_sum'] for r in rows)/max(
            1, sum(r[f'native_{name}_queries'] for r in rows))}
        for name in ('raw16', 'bilinear32', 'deconv32', 'decoded32',
                     'decoded64', 'decoded128')}
    result['edge_top_quartile'] = {name: {
        'pixels': sum(r[f'{name}_edge_pixels'] for r in rows),
        'epe_512px': sum(r[f'{name}_edge_epe_sum'] for r in rows)/max(
            1, sum(r[f'{name}_edge_pixels'] for r in rows))}
        for name in STAGES}
    result['gt_displacement_bins'] = {}
    for low, high in BINS:
        label = f'{low}_{high}'.replace('inf', 'plus')
        count = sum(r[f'motion_{label}_pixels'] for r in rows)
        result['gt_displacement_bins'][label] = {
            'pixels': count,
            'stage_epe_512px': {name: sum(
                r[f'{name}_motion_{label}_epe_sum'] for r in rows)/max(1, count)
                for name in STAGES}}
    result['first_window_assigned_full_pixels'] = {}
    for side in ('inside', 'outside', 'invalid_query'):
        count = sum(r[f'first32_{side}_pixels'] for r in rows)
        result['first_window_assigned_full_pixels'][side] = {
            'pixels': count,
            'stage_epe_512px': {name: sum(r[f'{name}_first32_{side}_epe_sum']
                                           for r in rows)/max(1, count) for name in STAGES}}
    for key in ('nmi', 'lncc17_squared', 'edge_f1_3x3_proxy',
                'edge_retention_proxy', 'visible_gt_mse_rgb01', 'visible_gt_ncc_rgb01'):
        result[key] = float(np.mean([r[key] for r in rows]))
    result['worst_frames'] = [{'sequence': r['sequence'], 'stem': r['stem'],
                               'final_epe_512px': r['final512_epe_sum']/r['valid_pixels']}
                              for r in sorted(rows, key=lambda x: -x['final512_epe_sum']/x['valid_pixels'])[:5]]
    return result


def evaluate_one(dataset, model, frame, device, mode):
    captured = infer(model, frame, device, zero_deconv_input=(mode == 'P1'))
    # GT and aligned visible are opened only after both flow hypotheses exist.
    return captured


def frame_rows(dataset, model, frame, device):
    predictions = {mode: evaluate_one(dataset, model, frame, device, mode)
                   for mode in ('P0', 'P1')}
    truth_np, valid_np, visible_gt = dataset.evaluation_truth(
        frame['sequence'], frame['stem'], frame['original_hw'])
    truth = torch.from_numpy(truth_np).permute(2, 0, 1)[None].to(device)
    valid = torch.from_numpy(valid_np)[None].to(device)
    edge = edge_quartile(torch.from_numpy(frame['vi']).permute(2, 0, 1)[None].to(device), valid[0])
    displacement = torch.linalg.vector_norm(truth, dim=1)
    rows = []
    for mode, c in predictions.items():
        full = full_flows(c)
        row = {'mode': mode, 'sequence': frame['sequence'], 'stem': frame['stem'],
               'valid_pixels': int(valid.sum()), 'gt_valid_fraction': float(valid.float().mean()),
               'zero_epe_sum': float(displacement[valid].sum())}
        row['deconv_input16_max_abs_256px'] = float(c['deconv_input16'].abs().max())
        row['deconv32_mean_magnitude_512px'] = float(
            torch.linalg.vector_norm(c['deconv32']*2, dim=1).mean())
        for name, flow in full.items():
            err = torch.linalg.vector_norm(flow-truth, dim=1)
            row[f'{name}_epe_sum'] = float(err[valid].sum())
            for threshold in (1, 3, 5):
                row[f'{name}_pck{threshold}_hits'] = int(((err < threshold)&valid).sum())
            row[f'{name}_edge_epe_sum'] = float(err[edge[None]].sum())
            row[f'{name}_edge_pixels'] = int(edge.sum())
            for low, high in BINS:
                selected = valid & (displacement >= low) & (displacement < high)
                label = f'{low}_{high}'.replace('inf', 'plus')
                row[f'{name}_motion_{label}_epe_sum'] = float(err[selected].sum())
                row[f'motion_{label}_pixels'] = int(selected.sum())
        for size, before, after in ((32, 'warp32_center', 'decoded32'),
                                    (64, 'warp64_center', 'decoded64'),
                                    (128, 'warp128_center', 'decoded128')):
            window, inside, outside, queries = search(c, truth, valid, size, before, after)
            row[f'window{size}_queries'] = window['valid_queries']
            row[f'window{size}_inside'] = window['inside']['queries']
            row[f'window{size}_outside'] = window['outside']['queries']
            row[f'window{size}_inbounds'] = window['center_inbounds_queries']
            for side in ('inside', 'outside'):
                row[f'window{size}_{side}_center_epe_sum'] = window[side]['center_epe_sum']
                row[f'window{size}_{side}_after_epe_sum'] = window[side]['after_epe_sum']
            # Native-grid measurements stay distinct from full-pixel metrics.
            row[f'native{size}_queries'] = window['valid_queries']
            row[f'native{size}_decoded_epe_sum'] = (window['inside']['after_epe_sum'] +
                                                   window['outside']['after_epe_sum'])
            if size == 32:
                first_inside, first_outside, first_queries = inside, outside, queries
        assigned_inside = F.interpolate(first_inside[:, None].float(), (512, 512),
                                        mode='nearest')[:, 0].bool() & valid
        assigned_outside = F.interpolate(first_outside[:, None].float(), (512, 512),
                                         mode='nearest')[:, 0].bool() & valid
        assigned_invalid = valid & ~assigned_inside & ~assigned_outside
        for side, selected in (('inside', assigned_inside),
                               ('outside', assigned_outside),
                               ('invalid_query', assigned_invalid)):
            row[f'first32_{side}_pixels'] = int(selected.sum())
            for name, flow in full.items():
                error = torch.linalg.vector_norm(flow-truth, dim=1)
                row[f'{name}_first32_{side}_epe_sum'] = float(error[selected].sum())
        for name, size, unit in (('raw16', 16, 2), ('deconv32', 32, 2),
                                 ('decoded32', 32, 2), ('decoded64', 64, 1),
                                 ('decoded128', 128, 1)):
            native_gt, native_valid = pooled_truth(truth, valid, size)
            native_err = torch.linalg.vector_norm(c[name]*unit-native_gt, dim=1)
            row[f'native_{name}_queries'] = int(native_valid.sum())
            row[f'native_{name}_epe_sum'] = float(native_err[native_valid].sum())
        native_gt, native_valid = pooled_truth(truth, valid, 32)
        native_err = torch.linalg.vector_norm(c['bilinear32']*2-native_gt, dim=1)
        row['native_bilinear32_queries'] = int(native_valid.sum())
        row['native_bilinear32_epe_sum'] = float(native_err[native_valid].sum())
        offset = torch.linalg.vector_norm((c['deconv32']-c['bilinear_input32'])*2, dim=1)
        row['deconv_shift_sum'] = float(offset[native_valid].sum())
        row['deconv_shift_count'] = int(native_valid.sum())
        row['deconv_shift_p95_512px'] = float(torch.quantile(offset[native_valid], .95))
        output_np = full['final512'][0].permute(1, 2, 0).cpu().numpy()
        warped_ir, _ = backward_warp(frame['ir'], output_np)
        warped_visible_gt, _ = backward_warp(visible_gt, output_np)
        retention, f1 = edge_metrics(frame['vi'], warped_ir, valid_np)
        mse, ncc = mse_ncc(frame['vi'], warped_visible_gt, valid_np)
        row.update({'nmi': nmi(frame['vi'], warped_ir, valid_np),
                    'lncc17_squared': lncc(frame['vi'], warped_ir, valid_np),
                    'edge_f1_3x3_proxy': f1, 'edge_retention_proxy': retention,
                    'visible_gt_mse_rgb01': mse, 'visible_gt_ncc_rgb01': ncc})
        rows.append(row)
    return rows


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def gradient_smoke(model, frame, dataset, device):
    """One read-only backward pass proves frozen locals transmit coarse gradients."""
    active = prior.active_parameters(model, 'coarse')
    if not active or model.base.training:
        raise AssertionError('coarse trainable selection / BN eval mismatch')
    truth, valid, _ = dataset.evaluation_truth(
        frame['sequence'], frame['stem'], frame['original_hw'])
    batch = {
        'source_image': torch.from_numpy(frame['ir']).permute(2, 0, 1)[None],
        'target_image': torch.from_numpy(frame['vi']).permute(2, 0, 1)[None],
        'flow_map': torch.from_numpy(truth).permute(2, 0, 1)[None],
        'correspondence_mask': torch.from_numpy(valid)[None],
    }
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        output = model(batch, device)
        loss, parts, _ = objective(output, batch, device, hierarchical=True)
        loss.backward()
    coarse = model.base.decoder4
    local = model.base.decoder3
    coarse_norm = sum(float(p.grad.abs().sum()) for p in coarse.parameters()
                      if p.grad is not None)
    local_grad_count = sum(p.grad is not None for p in local.parameters())
    if not np.isfinite(coarse_norm) or coarse_norm <= 0 or local_grad_count:
        raise AssertionError('frozen local path failed to transmit coarse gradient')
    result = {'loss': float(loss.detach()),
              'parts': {k: float(v.detach()) for k, v in parts.items()},
              'coarse_decoder4_gradient_l1': coarse_norm,
              'frozen_decoder3_parameters_with_grad': local_grad_count,
              'BN_eval': all(not m.training for m in model.modules()
                             if isinstance(m, torch.nn.modules.batchnorm._BatchNorm))}
    model.zero_grad(set_to_none=True)
    for p in model.parameters():
        p.requires_grad_(False)
    model.base.train_coarse_encoder = False
    model.eval()
    with torch.no_grad():
        direct = model(batch, device)
        hooked = infer(model, frame, device, zero_deconv_input=False)
        stage_diff = {
            'coarse16': float((direct['coarse16'] - full_flows(hooked)['raw16']).abs().max()),
            'local32': float((direct['local32'] - full_flows(hooked)['decoded32']).abs().max()),
            'final512': float((direct['final512'] - hooked['final512']).abs().max()),
        }
    if any(value != 0 for value in stage_diff.values()):
        raise AssertionError(f'diagnostic hooks changed B0 inference: {stage_diff}')
    result['hook_vs_original_max_abs_flow_difference'] = stage_diff
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('check', 'smoke', 'eval'), required=True)
    parser.add_argument('--comparison', type=Path, default=Path('roadscene_runs/vtmot_dns_adaptive_scale_v1/comparison.json'))
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--split-file', type=Path, default=Path('experiments/vtmot_a1_adapt/split.json'))
    parser.add_argument('--pretrained', type=Path, default=Path('pre_trained_models/GLUNet_DPED_CityScape_ADE.pth'))
    parser.add_argument('--attention-checkpoint', type=Path, default=Path('roadscene_runs/sa_ca/best_attention.pth'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--allow-800-frame-eval', action='store_true')
    args = parser.parse_args()
    geometry = check_geometry()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    install_mutual_guard()
    model, checkpoint = load_fixed(args, device)
    if args.stage == 'check':
        print(json.dumps({'geometry': geometry, 'checkpoint': checkpoint,
                          'full_model_forward': 'not run'}, indent=2))
        return
    if device.type != 'cuda':
        raise RuntimeError('full local correlation needs CUDA/CuPy')
    if args.data_root is None or args.output is None:
        parser.error('smoke/eval require --data-root and --output')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('output must be a new empty directory')
    if args.stage == 'eval' and not args.allow_800_frame_eval:
        parser.error('explicit --allow-800-frame-eval required')
    split = 'train' if args.stage == 'smoke' else 'eval'
    dataset = VTMOTVideos(args.data_root, args.split_file, split,
                          max_sequences=1 if args.stage == 'smoke' else 0,
                          max_frames=2 if args.stage == 'smoke' else 0)
    if args.stage == 'eval' and (tuple(dataset.sequences) != prior.SPLIT_NAMES or
                                  sum(map(len, dataset.sequences.values())) != 800):
        raise ValueError('full eval requires fixed four validation sequences and 800 frames')
    gradient_result = None
    if args.stage == 'smoke':
        first_sequence = next(iter(dataset.sequences))
        first_frame = dataset.input_frame(first_sequence, dataset.sequences[first_sequence][0])
        gradient_result = gradient_smoke(model, first_frame, dataset, device)
    rows = []
    for sequence, stems in dataset.sequences.items():
        for stem in stems:
            frame = dataset.input_frame(sequence, stem)
            rows.extend(frame_rows(dataset, model, frame, device))
        print(f'diagnosed {sequence}: {len(stems)} frames', flush=True)
    pairs = []
    by_frame = defaultdict(dict)
    for row in rows:
        by_frame[(row['sequence'], row['stem'])][row['mode']] = row
    for (sequence, stem), probes in by_frame.items():
        left, right = probes['P0'], probes['P1']
        if left['valid_pixels'] != right['valid_pixels']:
            raise AssertionError('P0/P1 GT mask changed')
        pairs.append({'sequence': sequence, 'stem': stem,
                      'valid_pixels': left['valid_pixels'],
                      'P0_final_epe_512px': left['final512_epe_sum']/left['valid_pixels'],
                      'P1_final_epe_512px': right['final512_epe_sum']/right['valid_pixels'],
                      'P1_minus_P0_final_epe_512px': (
                          right['final512_epe_sum']-left['final512_epe_sum'])/left['valid_pixels'],
                      'P0_window32_inside_fraction': left['window32_inside']/left['window32_queries'],
                      'P1_window32_inside_fraction': right['window32_inside']/right['window32_queries']})
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['mode']].append(row)
    result = {'stage': args.stage, 'geometry': geometry, 'checkpoint': checkpoint,
              'flow_direction': 'VI target -> IR source',
              'interpolation': 'F.interpolate align_corners=False; model warp grid_sample align_corners=True',
              'window_query_rule': 'area-pooled GT, mask fraction >0.8; 9x9 local correlation radius 4',
              'full_pixel_rule': 'same original 512px GT valid mask for P0/P1, regardless of predicted bounds',
              'image_metrics': 'NMI 64-bin 2MI/(H1+H2), squared LNCC 17x17; edge metrics are proxies; visible_gt is evaluation only',
              'native_grid_rule': 'GT pooled by area at each native grid; separately counted from 512px raster metrics',
              'gradient_smoke': gradient_result,
              'probes': {mode: summarize(group) for mode, group in grouped.items()},
              'by_sequence': {sequence: {mode: summarize([r for r in grouped[mode] if r['sequence'] == sequence])
                                        for mode in grouped} for sequence in dataset.sequences},
              'paired_final_improved_P1': sum(r['P1_minus_P0_final_epe_512px'] < 0 for r in pairs),
              'paired_final_worsened_P1': sum(r['P1_minus_P0_final_epe_512px'] > 0 for r in pairs),
              'test_accessed': False}
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output/'per_frame.csv', rows)
    write_csv(args.output/'paired_per_frame.csv', pairs)
    (args.output/'diagnosis.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
        encoding='utf-8')
    print(json.dumps({'stage': args.stage, 'frames_per_probe': len(grouped['P0']),
                      'P0_final_epe': result['probes']['P0']['stages_full512']['final512']['epe'],
                      'P1_final_epe': result['probes']['P1']['stages_full512']['final512']['epe'],
                      'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
