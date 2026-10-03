"""Read-only 32-grid DCN residual audit for staged RoadScene validation.

Compares best_coarse and best_fine on the same validation pairs. Does not
train, read the test split, or modify any checkpoint.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.ops import DeformConv2d

from datasets.roadscene import RoadScenePairs
from models.our_models.GLUNet import GLUNet_model
from roadscene_coarse import sha256_file
from roadscene_refinement_audit import pooled_truth

LEVELS = ("coarse16", "local32", "local64", "final128")


def offset_axis_check():
    layer = DeformConv2d(1, 1, 3, padding=1, bias=False)
    with torch.no_grad():
        layer.weight.zero_()
        layer.weight[0, 0, 1, 1] = 1
    impulse = torch.zeros(1, 1, 5, 5)
    impulse[0, 0, 2, 2] = 1
    offset = torch.zeros(1, 18, 5, 5)
    offset[:, 8] = 1  # Centre kernel point, first offset component.
    y_peak = torch.nonzero(layer(impulse, offset)[0, 0] > .5).tolist()
    offset.zero_()
    offset[:, 9] = 1
    x_peak = torch.nonzero(layer(impulse, offset)[0, 0] > .5).tolist()
    if y_peak != [[1, 2]] or x_peak != [[2, 1]]:
        raise RuntimeError(f"Unexpected deform offset axis order: {y_peak}, {x_peak}")
    return {"positive_y_offset_impulse_output_yx": y_peak,
            "positive_x_offset_impulse_output_yx": x_peak,
            "offset_channel_pairs": "(y, x) in 32-grid pixels",
            "flow_delta_channels": "(x, y) in 32-grid pixels"}


def load_pair(args, device):
    coarse = torch.load(args.coarse_checkpoint, map_location="cpu",
                        weights_only=False)
    fine = torch.load(args.fine_checkpoint, map_location="cpu",
                      weights_only=False)
    base_hash = sha256_file(args.pretrained)
    for stage, payload in (("coarse", coarse), ("fine", fine)):
        if (payload.get("stage") != stage or
                payload.get("arm") != "dns_attention" or
                payload.get("base_sha256") != base_hash):
            raise ValueError(f"Incompatible {stage} checkpoint")
        if payload.get("code_check_only") and not args.smoke:
            raise ValueError("Code-check checkpoint cannot be used as formal evidence")
    if (fine.get("parent_sha256") != sha256_file(args.coarse_checkpoint) or
            coarse.get("epoch") != 37 and not args.smoke or
            fine.get("epoch") != 1 and not args.smoke):
        raise ValueError("Fine checkpoint must descend from selected coarse best")
    def construct(payload):
        model = GLUNet_model(evaluation=False, pyramid_type="VGG",
                            cyclic_consistency=True, backbone_pretrained=False,
                            coarse_attention=True, coarse_dns=True,
                            local_dcn_steps=1)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        return model.to(device).eval()
    unchanged = {}
    changed = {}
    for name, left in coarse["model_state_dict"].items():
        right = fine["model_state_dict"][name]
        diff = float((left.float() - right.float()).abs().max())
        (changed if name.startswith("local_dcn32.") else unchanged)[name] = diff
    if any(value != 0 for value in unchanged.values()):
        raise RuntimeError("Fine checkpoint changed non-DCN weights despite epoch-1 DCN-only training")
    if not any(value > 0 for value in changed.values()):
        raise RuntimeError("Fine checkpoint did not update the DCN")
    metadata = {"coarse_epoch": coarse["epoch"], "fine_epoch": fine["epoch"],
                "base_sha256": base_hash,
                "coarse_sha256": sha256_file(args.coarse_checkpoint),
                "fine_sha256": sha256_file(args.fine_checkpoint),
                "non_dcn_max_weight_change": max(unchanged.values()),
                "dcn_max_weight_change": max(changed.values())}
    return construct(coarse), construct(fine), metadata


def percentile(values, p):
    return float(np.quantile(np.concatenate(values), p)) if values else None


def grouped_metrics(records):
    count = sum(item["count"] for item in records)
    if not count:
        return {"queries": 0}
    return {"queries": count,
            "pre_dcn_epe_512px": sum(item["before_sum"] for item in records) / count,
            "post_dcn_epe_512px": sum(item["after_sum"] for item in records) / count,
            "improved_fraction": sum(item["improved"] for item in records) / count,
            "delta_toward_gt_fraction": sum(item["aligned"] for item in records) / count,
            "delta_magnitude_512px_p50": percentile(
                [item["pred_norm"] for item in records if item["count"]], .5),
            "delta_magnitude_512px_p90": percentile(
                [item["pred_norm"] for item in records if item["count"]], .9),
            "needed_magnitude_512px_p50": percentile(
                [item["desired_norm"] for item in records if item["count"]], .5),
            "needed_magnitude_512px_p90": percentile(
                [item["desired_norm"] for item in records if item["count"]], .9),
            "cosine_mean_when_both_nonzero":
                sum(item["cosine_sum"] for item in records) /
                max(1, sum(item["cosine_count"] for item in records)),
            "cosine_valid_queries": sum(item["cosine_count"] for item in records)}


def one_group(pred, needed, before_error, after_error, mask):
    predicted = pred[:, mask]
    required = needed[:, mask]
    before = before_error[mask]
    after = after_error[mask]
    pred_norm = torch.linalg.vector_norm(predicted, dim=0)
    need_norm = torch.linalg.vector_norm(required, dim=0)
    dot = (predicted * required).sum(dim=0)
    defined = (pred_norm > 1e-5) & (need_norm > 1e-5)
    cosine = dot[defined] / (pred_norm[defined] * need_norm[defined])
    return {"count": int(mask.sum()), "before_sum": float(before.sum()),
            "after_sum": float(after.sum()),
            "improved": int((after < before).sum()),
            "aligned": int((dot > 0).sum()),
            "pred_norm": pred_norm.cpu().numpy(),
            "desired_norm": need_norm.cpu().numpy(),
            "cosine_sum": float(cosine.sum()),
            "cosine_count": int(defined.sum())}


@torch.no_grad()
def audit(args, coarse_model, fine_model, device):
    dataset = RoadScenePairs(args.data_root, "val")
    if args.smoke:
        dataset = Subset(dataset, range(min(args.max_pairs, len(dataset))))
    elif len(dataset) != 23:
        raise ValueError("Formal audit requires all 23 validation pairs")
    group_records = {key: [] for key in
                     ("all", "inside_first_window", "outside_first_window",
                      "gt_displacement_64+", "large_and_outside")}
    offset_records = {key: {"y": [], "x": []} for key in group_records}
    offset_y, offset_x, delta_x, delta_y = [], [], [], []
    rows = []
    largest_pre_difference = 0.
    for batch in DataLoader(dataset, batch_size=1, shuffle=False):
        raw_source = batch["source_image"].to(device)
        raw_target = batch["target_image"].to(device)
        truth = batch["flow_map"].to(device).float()
        valid = batch["correspondence_mask"].to(device).bool()
        predictions = []
        for model in (coarse_model, fine_model):
            source, target, source256, target256, *_ = model.pre_process_data(
                raw_source, raw_target, device=device)
            flows256, flows512, trace = model(
                target, source, target256, source256, return_dcn_trace=True)
            if len(trace) != 1:
                raise RuntimeError("Expected exactly one 32-grid DCN update")
            delta_grid = trace[0]["delta_grid"]
            post = trace[0]["flow_256px"]
            pre = post - delta_grid * 8  # 256-image px / 32-grid px.
            final = F.interpolate(flows512[-1], (512, 512),
                                  mode="bilinear", align_corners=False)
            predictions.append((flows256, flows512, pre, post, trace[0], final))
        base_flow, base_flow512, base_pre, base_post, base_trace, base_final = predictions[0]
        fine_flow, fine_flow512, fine_pre, fine_post, fine_trace, fine_final = predictions[1]
        pre_difference = float((base_pre - fine_pre).abs().max())
        largest_pre_difference = max(largest_pre_difference, pre_difference)
        if pre_difference > .01:
            raise RuntimeError("Coarse and fine differ before DCN; paired residual attribution is invalid")
        truth32, selected = pooled_truth(truth, valid, 32)
        first_warp_centre = fine_model.deconv4(fine_flow[0]) * 2
        inside = ((truth32 - first_warp_centre).abs().amax(dim=1) <= 64) & selected
        outside = selected & ~inside
        large = (torch.linalg.vector_norm(truth32, dim=1) >= 64) & selected
        before = fine_pre * 2
        after = fine_post * 2
        predicted_delta = (after - before)[0]
        needed_delta = (truth32 - before)[0]
        error_before = torch.linalg.vector_norm(needed_delta, dim=0)
        error_after = torch.linalg.vector_norm((truth32 - after)[0], dim=0)
        chosen = {"all": selected, "inside_first_window": inside,
                  "outside_first_window": outside, "gt_displacement_64+": large,
                  "large_and_outside": large & outside}
        for key, mask in chosen.items():
            group_records[key].append(one_group(predicted_delta, needed_delta,
                                                 error_before, error_after, mask[0]))
            if mask.any():
                offset_records[key]["y"].append(
                    fine_trace["offset_grid"][0, 0::2, mask[0]]
                    .cpu().numpy().reshape(-1))
                offset_records[key]["x"].append(
                    fine_trace["offset_grid"][0, 1::2, mask[0]]
                    .cpu().numpy().reshape(-1))
        offset = fine_trace["offset_grid"][0]
        grid_delta = fine_trace["delta_grid"][0]
        mask = selected[0]
        # torchvision's 18 offset channels are nine (y, x) pairs.
        offset_y.append(offset[0::2, mask].cpu().numpy().reshape(-1))
        offset_x.append(offset[1::2, mask].cpu().numpy().reshape(-1))
        delta_x.append(grid_delta[0, mask].cpu().numpy())
        delta_y.append(grid_delta[1, mask].cpu().numpy())
        coarse_error = torch.linalg.vector_norm(base_final - truth, dim=1)
        fine_error = torch.linalg.vector_norm(fine_final - truth, dim=1)
        row = {"name": batch["name"][0], "valid_full": int(valid.sum()),
               "valid_32_queries": int(mask.sum()),
               "outside_32_queries": int(outside.sum()),
               "large_32_queries": int(large.sum()),
               "pre_dcn_epe_512px": float(error_before[mask].mean()),
               "post_dcn_epe_512px": float(error_after[mask].mean()),
               "coarse_checkpoint_final_epe_512px": float(coarse_error[valid].mean()),
               "fine_checkpoint_final_epe_512px": float(fine_error[valid].mean()),
               "source_inside_fraction": float(fine_trace["source_inside"][0, mask].float().mean())}
        for label, (base_level, fine_level) in zip(
                LEVELS, zip((*base_flow, *base_flow512),
                            (*fine_flow, *fine_flow512))):
            scale = 2 if label in ("coarse16", "local32") else 1
            base_full = F.interpolate(base_level, (512, 512),
                                      mode="bilinear", align_corners=False) * scale
            fine_full = F.interpolate(fine_level, (512, 512),
                                      mode="bilinear", align_corners=False) * scale
            base_epe = float(torch.linalg.vector_norm(
                base_full - truth, dim=1)[valid].mean())
            fine_epe = float(torch.linalg.vector_norm(
                fine_full - truth, dim=1)[valid].mean())
            row[f"{label}_coarse_checkpoint_epe_512px"] = base_epe
            row[f"{label}_fine_checkpoint_epe_512px"] = fine_epe
            row[f"{label}_delta_epe_512px"] = fine_epe - base_epe
        for key in ("inside_first_window", "outside_first_window", "gt_displacement_64+"):
            record = group_records[key][-1]
            row[f"{key}_queries"] = record["count"]
            row[f"{key}_pre_epe_512px"] = (
                record["before_sum"] / record["count"] if record["count"] else None)
            row[f"{key}_post_epe_512px"] = (
                record["after_sum"] / record["count"] if record["count"] else None)
        all_record = group_records["all"][-1]
        row["delta_toward_gt_fraction"] = all_record["aligned"] / all_record["count"]
        row["dcn_improved_query_fraction"] = all_record["improved"] / all_record["count"]
        row["delta_magnitude_512px_p90"] = float(np.quantile(
            all_record["pred_norm"], .9))
        row["needed_magnitude_512px_p90"] = float(np.quantile(
            all_record["desired_norm"], .9))
        rows.append(row)
    def quantiles(v):
        values = np.concatenate(v)
        return {"signed_mean": float(values.mean()),
                "abs_p50": float(np.quantile(np.abs(values), .5)),
                "abs_p90": float(np.quantile(np.abs(values), .9)),
                "abs_p99": float(np.quantile(np.abs(values), .99)),
                "abs_max": float(np.max(np.abs(values)))}
    oy, ox = np.concatenate(offset_y), np.concatenate(offset_x)
    stage_comparison = {}
    for label in LEVELS:
        baseline_key = f"{label}_coarse_checkpoint_epe_512px"
        fine_key = f"{label}_fine_checkpoint_epe_512px"
        delta_key = f"{label}_delta_epe_512px"
        pixels = sum(row["valid_full"] for row in rows)
        stage_comparison[label] = {
            "coarse_checkpoint_epe_512px": sum(
                row[baseline_key] * row["valid_full"] for row in rows) / pixels,
            "fine_checkpoint_epe_512px": sum(
                row[fine_key] * row["valid_full"] for row in rows) / pixels,
            "improved_pairs": sum(row[delta_key] < -0.001 for row in rows),
            "worsened_pairs": sum(row[delta_key] > 0.001 for row in rows)}
    return {"samples": len(rows), "largest_paired_pre_dcn_difference_256px":
            largest_pre_difference,
            "stage_comparison": stage_comparison,
            "groups": {key: grouped_metrics(records)
                       for key, records in group_records.items()},
            "offset_32_grid_pixels": {
                "y": quantiles(offset_y), "x": quantiles(offset_x),
                "y_near_bound_fraction": float(np.mean(np.abs(oy) >= 1.9)),
                "x_near_bound_fraction": float(np.mean(np.abs(ox) >= 1.9)),
                "limit_per_axis": 2.0},
            "offset_by_group_32_grid_pixels": {
                key: ({"y": quantiles(record["y"]),
                       "x": quantiles(record["x"])} if record["y"] else None)
                for key, record in offset_records.items()},
            "predicted_delta_32_grid_pixels": {
                "x": quantiles(delta_x), "y": quantiles(delta_y),
                "limit_per_axis": 4.0},
            "per_image": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--coarse-checkpoint", type=Path, required=True)
    parser.add_argument("--fine-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=1)
    args = parser.parse_args()
    if args.max_pairs < 1:
        parser.error("--max-pairs must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coarse, fine, provenance = load_pair(args, device)
    result = audit(args, coarse, fine, device)
    payload = {"split": "val", "test_pairs_accessed": False,
               "code_check_only": args.smoke, "source": provenance,
               "offset_axis_check": offset_axis_check(), **result}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "dcn_residual_val.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    with (args.output / "dcn_residual_per_image_val.csv").open(
            "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(result["per_image"][0]))
        writer.writeheader()
        writer.writerows(result["per_image"])
    print(json.dumps({"split": "val", "samples": result["samples"],
                      "stage_comparison": result["stage_comparison"],
                      "groups": result["groups"],
                      "offset_32_grid_pixels": result["offset_32_grid_pixels"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
