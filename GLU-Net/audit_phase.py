"""Code audit for fixed phase maps and the zero-gate SA/CA identity."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.roadscene import RoadScenePairs
from models.our_models.phase_congruency import FixedPhaseCongruency2D
from roadscene_coarse import make_model, predict_coarse, prepare
from roadscene_compare import load_glu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--attention-checkpoint", type=Path, required=True)
    parser.add_argument("--full-model", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2026)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    descriptor = FixedPhaseCongruency2D().to(device).eval()
    with torch.no_grad():
        flat = descriptor(torch.full((1, 1, 256, 256), .5, device=device))
        gray = torch.rand(1, 1, 256, 256, device=device)
        normal = descriptor(gray)
        inverted = descriptor(1 - gray)
        shifted = descriptor(torch.roll(gray, (8, -8), dims=(-2, -1)))
        aligned = torch.roll(normal, (8, -8), dims=(-2, -1))
    if not torch.isfinite(normal).all() or normal.min() < 0 or normal.max() > 1:
        raise AssertionError("Phase maps are nonfinite or outside [0,1]")
    if flat.abs().max() > 1e-5:
        raise AssertionError("Flat image has nonzero phase edges")
    inversion_diff = float((normal - inverted).abs().max())
    if inversion_diff > 1e-3:
        raise AssertionError(f"Contrast inversion changed phase map: {inversion_diff}")
    # Ignore the FFT/reflection-pad boundary, where a rolled image changes
    # the periodic continuation. The interior should follow the image shift.
    shift_mae = float((shifted - aligned)[..., 48:-48, 48:-48].abs().mean())
    if shift_mae > .01:
        raise AssertionError(f"Phase map lost synthetic spatial alignment: {shift_mae}")
    attention, _ = load_glu(args.pretrained, args.attention_checkpoint,
                            "attention", device)
    guided = make_model(args.pretrained, True, device, phase=True)
    guided.decoder4.load_state_dict(attention.decoder4.state_dict())
    guided.coarse_attention.load_state_dict(attention.coarse_attention.state_dict())
    batch = next(iter(DataLoader(RoadScenePairs(args.data_root, "val"), batch_size=1)))
    source_in, target_in, source_rgb, target_rgb, *_ = prepare(batch, device)
    with torch.no_grad():
        map_source = descriptor(source_rgb)
        map_target = descriptor(target_rgb)
        map_source_coarse = F.interpolate(map_source, size=(16, 16), mode="area")
        map_target_coarse = F.interpolate(map_target, size=(16, 16), mode="area")
        reference_flow, reference_corr = predict_coarse(attention, target_in, source_in)
        guided_flow, guided_corr = predict_coarse(guided, target_in, source_in)
    flow_diff = float((guided_flow - reference_flow).abs().max())
    corr_diff = float((guided_corr - reference_corr).abs().max())
    if flow_diff != 0 or corr_diff != 0:
        raise AssertionError(f"Zero gate changed coarse output: {flow_diff}, {corr_diff}")
    report = {"pair": batch["name"][0], "phase_shape": list(map_source.shape),
              "coarse_phase_shape": list(map_source_coarse.shape),
              "coarse_source_range": [float(map_source_coarse.min()), float(map_source_coarse.max())],
              "coarse_target_range": [float(map_target_coarse.min()), float(map_target_coarse.max())],
              "phase_source_range": [float(map_source.min()), float(map_source.max())],
              "phase_target_range": [float(map_target.min()), float(map_target.max())],
              "flat_max": float(flat.abs().max()), "inversion_max_abs_diff": inversion_diff,
              "translated_interior_mae": shift_mae,
              "zero_gate_coarse_flow_max_abs_diff": flow_diff,
              "zero_gate_corr_max_abs_diff": corr_diff,
              "full_model_checked": args.full_model}
    if args.full_model:
        with torch.no_grad():
            full_source = batch["source_image"].to(device)
            full_target = batch["target_image"].to(device)
            ref_args = attention.pre_process_data(full_source, full_target, device=device)
            new_args = guided.pre_process_data(full_source, full_target, device=device)
            _, ref_flows = attention(ref_args[1], ref_args[0], ref_args[3], ref_args[2])
            _, new_flows = guided(new_args[1], new_args[0], new_args[3], new_args[2])
            _, repeat_flows = attention(ref_args[1], ref_args[0], ref_args[3], ref_args[2])
            full_diff = float((ref_flows[-1] - new_flows[-1]).abs().max())
            repeat_diff = float((ref_flows[-1] - repeat_flows[-1]).abs().max())
            report["baseline_repeat_full_flow_max_abs_diff"] = repeat_diff
            # CuPy local correlation may differ at float32 roundoff after a
            # different CUDA kernel schedule; coarse outputs above are exact.
            if full_diff > max(1e-5, 2 * repeat_diff):
                raise AssertionError(f"Zero gate changed full flow: {full_diff}; "
                                     f"baseline repeat {repeat_diff}")
            report["zero_gate_full_flow_max_abs_diff"] = full_diff
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
