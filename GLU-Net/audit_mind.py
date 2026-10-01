"""Checks fixed MIND, masks/CMR, coarse gate equality, gradients and CRFT units.

Uses synthetic data + the FIRST VALIDATION pair only; never touches test data.
Training steps in this audit verify gradients, not registration performance.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from datasets.roadscene import RoadScenePairs
from models.our_models.mind import FixedMIND2D
from roadscene_coarse import make_model, prepare, predict_coarse, coarse_statistics
from roadscene_metrics import flow_metrics, aggregate_flow


@torch.no_grad()
def audit_descriptor(device):
    torch.manual_seed(2026)
    descriptor = FixedMIND2D().to(device)
    target = torch.rand(1, 1, 512, 512, device=device) * 0.6 + 0.2
    target = F.avg_pool2d(F.pad(target, (2, 2, 2, 2), mode="replicate"), 5, 1)
    source = torch.roll(target, shifts=(-32, 32), dims=(-2, -1))
    target_m = descriptor(target)
    source_m = descriptor(source)
    # A translated patch must have the SAME descriptor away from boundaries.
    translated_mae = (target_m[..., 34:478, 2:478] - source_m[..., 2:446, 34:510]).abs().max()
    assert translated_mae < 1e-5
    affine_diff = (target_m - descriptor(0.7 * target + 0.1)).abs().max()
    assert affine_diff < 1e-4
    inversion_diff = (target_m - descriptor(1 - target)).abs().max()
    assert inversion_diff < 1e-4
    constant = descriptor(torch.ones_like(target))
    torch.testing.assert_close(constant, torch.ones_like(constant))
    torch.testing.assert_close(target_m.amax(dim=1), torch.ones_like(target[:, 0]))
    assert torch.isfinite(target_m).all() and target_m.amin() >= 0 and target_m.amax() <= 1
    # Pool DESCRIPTORS (not images) 512->16. The 32px shift becomes one bin.
    target_16 = F.interpolate(target_m, (16, 16), mode="area")
    source_16 = F.interpolate(source_m, (16, 16), mode="area")
    scale_diff = (target_16[..., 2:14, 1:14] - source_16[..., 1:13, 2:15]).abs().max()
    assert scale_diff < 1e-5
    # Actual model path recomputes MIND on identical 256px quantized images.
    target_256 = F.interpolate(target * 255, (256, 256), mode="area").byte().float() / 255
    source_256 = F.interpolate(source * 255, (256, 256), mode="area").byte().float() / 255
    recomputed = F.interpolate(descriptor(target_256), (16, 16), mode="area")
    recomputed_source = F.interpolate(descriptor(source_256), (16, 16), mode="area")
    model_scale_diff = (recomputed[..., 2:14, 1:14] - recomputed_source[..., 1:13, 2:15]).abs().max()
    assert model_scale_diff < 1e-5
    return {"translation_max_diff": float(translated_mae), "affine_intensity_max_diff": float(affine_diff),
            "intensity_inversion_max_diff": float(inversion_diff), "pooled_alignment_max_diff": float(scale_diff),
            "actual_256_path_alignment_max_diff": float(model_scale_diff),
            "recompute_vs_pool_mean_abs_diff_observation": float((recomputed - target_16).abs().mean()),
            "shape": list(target_m.shape), "range": [float(target_m.min()), float(target_m.max())]}


def audit_metrics():
    truth = torch.zeros(2, 2, 2)
    prediction = truth.clone()
    prediction[0] = torch.tensor([[0., 2.], [4., 100.]])
    mask = torch.tensor([[True, True], [True, False]])
    first = flow_metrics(prediction, truth, mask)
    assert first["aepe_512px"] == 2.0 and first["valid_full"] == 3
    second = flow_metrics(torch.ones_like(truth), truth, torch.ones_like(mask))
    result = aggregate_flow([first, second])
    # Strict < 2: exactly 2px fails at PAIR level. Pixel success differs.
    assert result["cmr_pair_percent"]["2"] == 50.0
    assert abs(result["pixel_success_percent_auxiliary"]["2"] - 500 / 7) < 1e-8
    return result


def audit_models(args, device, batch):
    source, target, _, _, truth, valid, _ = prepare(batch, device)
    checks = {}
    for attention in (False, True):
        torch.manual_seed(2026)
        reference = make_model(args.pretrained, attention, device)
        torch.manual_seed(2026)
        branch = make_model(args.pretrained, attention, device, mind="b")
        # Ensure SAME trained/initialized SA weights independent of constructor RNG.
        if attention:
            branch.coarse_attention.load_state_dict(reference.coarse_attention.state_dict())
        checkpoint = args.attention_checkpoint if attention else args.baseline_checkpoint
        if checkpoint:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            for model in (reference, branch):
                model.decoder4.load_state_dict(payload["decoder4_state_dict"])
                if attention:
                    model.coarse_attention.load_state_dict(payload["attention_state_dict"])
        with torch.no_grad():
            before_f, before_c = predict_coarse(reference, target, source)
            after_f, after_c = predict_coarse(branch, target, source)
        diff = {"flow_max_diff": float((before_f - after_f).abs().max()),
                "correlation_max_diff": float((before_c - after_c).abs().max())}
        assert max(diff.values()) == 0
        checks[f"b_{'attention' if attention else 'baseline'}_zero_gate"] = diff
        if args.full_gate_check:
            with torch.no_grad():
                inputs = reference.pre_process_data(batch["source_image"], batch["target_image"], device=device)
                _, before_full = reference(inputs[1], inputs[0], inputs[3], inputs[2])
                _, after_full = branch(inputs[1], inputs[0], inputs[3], inputs[2])
                full_diff = float((before_full[-1] - after_full[-1]).abs().max())
                assert full_diff == 0
                diff["full_flow_max_diff"] = full_diff
            del before_full, after_full, inputs
        diff["reference_checkpoint"] = str(checkpoint) if checkpoint else "base pretrained"
        del reference, branch, before_f, before_c, after_f, after_c
    for route in ("a", "b"):
        model = make_model(args.pretrained, False, device, train_decoder=True, mind=route)
        with torch.no_grad():
            gated = make_model(args.pretrained, True, device, mind=route)
            before, before_corr = predict_coarse(model, target, source)
            after, after_corr = predict_coarse(gated, target, source)
            difference = {"flow_max_diff": float((before - after).abs().max()),
                          "corr_max_diff": float((before_corr - after_corr).abs().max())}
            assert max(difference.values()) == 0
            checks[f"{route}_sa_zero_gate"] = difference
            for key, value in model.coarse_mind.state_dict().items():
                torch.testing.assert_close(value, gated.coarse_mind.state_dict()[key], atol=0, rtol=0)
            del gated, before, after, before_corr, after_corr
        predicted, corr = predict_coarse(model, target, source)
        epe, ce, _ = coarse_statistics(predicted, corr, truth, valid)
        (epe + 2 * ce).backward()
        parameter = model.coarse_mind.input_projection.weight if route == "a" else model.coarse_mind.gate
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
        assert all(parameter.grad is None for parameter in model.pyramid.parameters())
        checks[f"{route}_gradient"] = {"gradient_abs_sum": float(parameter.grad.abs().sum()),
                                       "frozen_encoder_has_no_parameter_grad": True}
        del model, predicted, corr, epe, ce
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--crft-root", type=Path)
    parser.add_argument("--crft-forward-check", action="store_true", help="slow 512px random CRFT forward; numerical check is always at 64px")
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--attention-checkpoint", type=Path)
    parser.add_argument("--full-gate-check", action="store_true", help="requires working CuPy/CUDA")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    loader = DataLoader(Subset(RoadScenePairs(args.data_root, "val"), [0]), batch_size=1)
    batch = next(iter(loader))
    report = {"descriptor": audit_descriptor(device), "metrics": audit_metrics(),
              "models": audit_models(args, device, batch), "device": str(device),
              "note": "synthetic checks and one validation pair; not trained MIND results"}
    descriptor = FixedMIND2D().to(device)
    source, target, source_rgb, target_rgb, *_ = prepare(batch, device)
    with torch.no_grad():
        report["real_pair_descriptor"] = {side: {"shape": list(value.shape),
            "min": float(value.min()), "max": float(value.max()), "mean": float(value.mean())}
            for side, rgb in (("IR", source_rgb), ("visible", target_rgb))
            for value in [descriptor(rgb)]}
    if args.crft_root:
        from crft_adapter import (audit_crft_coordinates, import_official, official_config,
                                  predict_crft, enable_chunked_attention, resize_crft_input)
        report["crft_coordinates"] = audit_crft_coordinates(args.crft_root)
        import cv2
        import numpy as np
        for side in ("source_image", "target_image"):
            image = batch[side][0].permute(1, 2, 0).numpy()
            official = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)
            prepared = resize_crft_input(batch[side].to(device), 64)[0].permute(1, 2, 0).cpu().numpy()
            max_difference = float(np.abs(official.astype(float) - prepared).max())
            assert max_difference == 0
            report.setdefault("crft_resize_matches_opencv", {})[side] = max_difference
        config, _ = official_config(args.crft_root)
        core = import_official(args.crft_root)(config).to(device).eval()
        with torch.no_grad():
            small_target = F.interpolate(batch["target_image"].to(device).float(), (64, 64), mode="area")
            small_source = F.interpolate(batch["source_image"].to(device).float(), (64, 64), mode="area")
            original, _ = predict_crft(core, small_target, small_source, input_size=64)
            adapter = enable_chunked_attention(core)
            chunked, _ = predict_crft(core, small_target, small_source)
            equivalence = float((original - chunked).abs().max())
            torch.testing.assert_close(original, chunked, atol=1e-4, rtol=1e-5)
            report["crft_chunk_equivalence_64px_random_weights"] = {
                "max_flow_abs_diff": equivalence, "adapter": adapter}
            # Position embedding caches in official CRFT are shape-specific.
            for module in core.modules():
                if type(module).__name__ == "SwinPosEmbMLP":
                    module.pos_embed = None
            del small_target, small_source, original, chunked
        if args.crft_forward_check:
            with torch.no_grad():
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                started = time.perf_counter()
                print("CRFT 512px chunked forward code check started (random weights)", flush=True)
                predicted, coarse = predict_crft(core, batch["target_image"].to(device),
                                                batch["source_image"].to(device), input_size=512)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                assert predicted.shape == (1, 2, 512, 512) and torch.isfinite(predicted).all()
                report["crft_random_forward_code_check"] = {"shape": list(predicted.shape),
                    "coarse_shape": list(coarse.shape), "all_finite": True,
                    "seconds_first_forward_not_benchmark": time.perf_counter() - started,
                    "parameters": sum(parameter.numel() for parameter in core.parameters()),
                    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
                    "note": "RANDOM weights: shape/runtime check only, no accuracy metrics or trained CRFT cost"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
