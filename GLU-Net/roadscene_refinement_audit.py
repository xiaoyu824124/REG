"""Train/val-only diagnosis of the selected GLU-Net+SA/CA refinement path.

All four predicted flows are converted to 512x512 pixel units before their
per-pixel EPE is compared. Search coverage is measured separately at the
actual 32, 64, and 128 grid queries, before each 9x9 local correlation.
This script does not train, tune, read test, or alter the model.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.roadscene import RoadScenePairs
from roadscene_coarse import sha256_file
from roadscene_compare import load_glu


BIN_EDGES = (0, 8, 16, 32, 64, float("inf"))
BIN_NAMES = ("0-8", "8-16", "16-32", "32-64", "64+")
STAGES = ("coarse16", "local32", "local64", "final128")


def pooled_truth(flow, valid, size):
    weight = F.interpolate(valid[:, None].float(), size=(size, size), mode="area")
    mean = F.interpolate(flow * valid[:, None], size=(size, size), mode="area")
    return mean / weight.clamp_min(1e-6), weight[:, 0] > 0.8


def edge_quartile(target, valid):
    rgb = target.float() / 255
    gray = (rgb * rgb.new_tensor((.299, .587, .114))[None, :, None, None]).sum(
        dim=1, keepdim=True)
    kx = gray.new_tensor(((-1, 0, 1), (-2, 0, 2), (-1, 0, 1)))[None, None] / 8
    ky = kx.transpose(-1, -2)
    strength = torch.sqrt(F.conv2d(gray, kx, padding=1).square() +
                          F.conv2d(gray, ky, padding=1).square())[0, 0]
    threshold = torch.quantile(strength[valid], .75)
    return valid & (strength >= threshold)


def sums_for_mask(errors, mask):
    count = int(mask.sum())
    return {"pixels": count,
            **{stage: float(values[mask].sum()) for stage, values in errors.items()},
            **{f"{before}_to_{after}_improved": int(((errors[after] < errors[before]) & mask).sum())
               for before, after in zip(STAGES[:-1], STAGES[1:])},
            **{f"{before}_to_{after}_worsened": int(((errors[after] > errors[before]) & mask).sum())
               for before, after in zip(STAGES[:-1], STAGES[1:])}}


def search_stage(pre, post, flow, valid, size):
    truth, selected = pooled_truth(flow, valid, size)
    cell = 512 / size
    delta = truth - pre
    inside = (delta.abs().amax(dim=1) <= 4 * cell) & selected
    before_epe = torch.linalg.vector_norm(delta, dim=1)
    after_epe = torch.linalg.vector_norm(truth - post, dim=1)
    def group(mask):
        count = int(mask.sum())
        return {"queries": count,
                "before_epe_sum": float(before_epe[mask].sum()),
                "after_epe_sum": float(after_epe[mask].sum()),
                "improved": int(((after_epe < before_epe) & mask).sum()),
                "worsened": int(((after_epe > before_epe) & mask).sum())}
    displacement = torch.linalg.vector_norm(truth, dim=1)
    by_displacement = {}
    for index, label in enumerate(BIN_NAMES):
        in_bin = selected & (displacement >= BIN_EDGES[index]) & (
            displacement < BIN_EDGES[index + 1])
        by_displacement[label] = {"queries": int(in_bin.sum()),
                                  "inside": int((inside & in_bin).sum()),
                                  "outside": int((~inside & in_bin).sum())}
    return {"grid": size, "window_radius_feature_pixels": 4,
            "window_radius_512px_per_axis": 4 * cell,
            "valid_queries": int(selected.sum()),
            "inside": group(inside), "outside": group(selected & ~inside),
            "by_gt_displacement_512px": by_displacement}, inside, selected


def save_error_panel(path, target, valid, errors):
    gray = target.float().mean(dim=0).clamp(0, 255).byte().cpu().numpy()
    target_rgb = np.repeat(gray[..., None], 3, axis=-1)
    panels = [target_rgb]
    for stage in STAGES:
        error = errors[stage]
        intensity = (error / 80).clamp(0, 1)
        red = (intensity * 255).byte()
        green = ((1 - intensity) * 255).byte()
        heat = torch.stack((red, green, torch.zeros_like(red)), dim=-1)
        heat[~valid] = 0
        panels.append(heat.cpu().numpy())
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(panels, axis=1)).save(path)


@torch.no_grad()
def evaluate(model, loader, device, preview_dir):
    rows = []
    for batch in loader:
        name = batch["name"][0]
        raw_source = batch["source_image"].to(device)
        raw_target = batch["target_image"].to(device)
        flow_gt = batch["flow_map"].to(device).float()
        valid = batch["correspondence_mask"].to(device).bool()
        if flow_gt.shape[-2:] != (512, 512):
            raise ValueError("RoadScene refinement audit requires 512x512 GT")
        source, target, source_256, target_256, *_ = model.pre_process_data(
            raw_source, raw_target, device=device)
        flows_256, flows_512 = model(target, source, target_256, source_256)
        coarse, local32 = flows_256
        local64, final = flows_512
        # flow4/flow3 are in 256px image units; flow2/flow1 are in 512px.
        full = {"coarse16": F.interpolate(coarse, (512, 512),
                                          mode="bilinear", align_corners=False) * 2,
                "local32": F.interpolate(local32, (512, 512),
                                         mode="bilinear", align_corners=False) * 2,
                "local64": F.interpolate(local64, (512, 512),
                                         mode="bilinear", align_corners=False),
                "final128": F.interpolate(final, (512, 512),
                                          mode="bilinear", align_corners=False)}
        errors = {stage: torch.linalg.vector_norm(pred[0] - flow_gt[0], dim=0)
                  for stage, pred in full.items()}
        displacement = torch.linalg.vector_norm(flow_gt[0], dim=0)
        edge = edge_quartile(raw_target, valid[0])
        groups = {"all": sums_for_mask(errors, valid[0]),
                  "visible_edge_top25pct": sums_for_mask(errors, edge),
                  "other_valid": sums_for_mask(errors, valid[0] & ~edge)}
        for index, label in enumerate(BIN_NAMES):
            selected = valid[0] & (displacement >= BIN_EDGES[index]) & (
                displacement < BIN_EDGES[index + 1])
            groups[f"gt_displacement_{label}"] = sums_for_mask(errors, selected)
        # These are the exact flow estimates used as warp centres before each
        # local correlation, expressed in 512px units at that feature grid.
        pre32 = model.deconv4(coarse) * 2
        pre32_bilinear = F.interpolate(coarse, size=(32, 32),
                                       mode="bilinear", align_corners=False) * 2
        pre64 = F.interpolate(local32, size=(64, 64), mode="bilinear",
                              align_corners=False) * 2
        pre128 = model.deconv2(local64)
        pre128_bilinear = F.interpolate(local64, size=(128, 128),
                                        mode="bilinear", align_corners=False)
        search32, inside32, selected32 = search_stage(pre32, local32 * 2,
                                                      flow_gt, valid, 32)
        search64, _, _ = search_stage(pre64, local64, flow_gt, valid, 64)
        search128, _, _ = search_stage(pre128, final, flow_gt, valid, 128)
        candidate32, _, _ = search_stage(pre32_bilinear, local32 * 2,
                                         flow_gt, valid, 32)
        candidate128, _, _ = search_stage(pre128_bilinear, final,
                                          flow_gt, valid, 128)
        # Nearest assignment gives a full-resolution view of which pixels
        # belong to a coarse 32-grid query outside its nominal search window.
        assigned_outside = F.interpolate((selected32 & ~inside32)[:, None].float(),
                                         size=(512, 512), mode="nearest")[0, 0].bool()
        assigned_inside = F.interpolate(inside32[:, None].float(),
                                        size=(512, 512), mode="nearest")[0, 0].bool()
        outside_full = valid[0] & assigned_outside
        groups["outside_first_window_assigned_pixels"] = sums_for_mask(errors, outside_full)
        groups["inside_first_window_assigned_pixels"] = sums_for_mask(
            errors, valid[0] & assigned_inside)
        groups["invalid_first_grid_assigned_pixels"] = sums_for_mask(
            errors, valid[0] & ~assigned_outside & ~assigned_inside)
        row = {"name": name, "valid_full": int(valid.sum()),
               "gt_displacement_median_512px": float(displacement[valid[0]].median()),
               "gt_displacement_p90_512px": float(torch.quantile(
                   displacement[valid[0]], .9)),
               "stage_epe_512px": {stage: groups["all"][stage] / groups["all"]["pixels"]
                                    for stage in STAGES},
               "search": {"local32": search32, "local64": search64,
                          "final128": search128},
               "bilinear_center_counterfactual": {"local32": candidate32,
                                                   "final128": candidate128},
               "groups": groups}
        rows.append(row)
        if preview_dir is not None and name in {"000009", "000014", "000021"}:
            save_error_panel(preview_dir / f"{name}.png", raw_target[0],
                             valid[0], errors)
    return rows


def combine_groups(rows, group):
    parts = [row["groups"][group] for row in rows]
    count = sum(part["pixels"] for part in parts)
    return {"pixels": count,
            "epe_512px": {stage: sum(part[stage] for part in parts) / count
                           if count else None for stage in STAGES},
            "transitions": {
                f"{before}_to_{after}": {
                    "improved_pixels": sum(part[f"{before}_to_{after}_improved"]
                                           for part in parts),
                    "worsened_pixels": sum(part[f"{before}_to_{after}_worsened"]
                                           for part in parts)}
                for before, after in zip(STAGES[:-1], STAGES[1:])}}


def combine_search(rows, stage, key="search"):
    records = [row[key][stage] for row in rows]
    count = sum(record["valid_queries"] for record in records)
    output = {"valid_queries": count,
              "window_radius_512px_per_axis": records[0]["window_radius_512px_per_axis"]}
    for key in ("inside", "outside"):
        group = {field: sum(record[key][field] for record in records)
                 for field in ("queries", "before_epe_sum", "after_epe_sum",
                               "improved", "worsened")}
        group["fraction_of_valid_queries"] = group["queries"] / count
        group["before_epe"] = group["before_epe_sum"] / group["queries"] if group["queries"] else None
        group["after_epe"] = group["after_epe_sum"] / group["queries"] if group["queries"] else None
        output[key] = group
    output["by_gt_displacement_512px"] = {}
    for label in BIN_NAMES:
        parts = [record["by_gt_displacement_512px"][label] for record in records]
        bins = {field: sum(part[field] for part in parts)
                for field in ("queries", "inside", "outside")}
        bins["outside_fraction"] = (bins["outside"] / bins["queries"]
                                    if bins["queries"] else None)
        output["by_gt_displacement_512px"][label] = bins
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--attention-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_glu(args.pretrained, args.attention_checkpoint,
                                 "attention", device)
    dataset = RoadScenePairs(args.data_root, "val")
    rows = evaluate(model, DataLoader(dataset, batch_size=1, shuffle=False),
                    device, args.output / "heatmaps")
    summary = {"pairs": len(rows),
               "groups": {group: combine_groups(rows, group) for group in (
                   "all", "visible_edge_top25pct", "other_valid",
                   *(f"gt_displacement_{name}" for name in BIN_NAMES),
                   "outside_first_window_assigned_pixels",
                   "inside_first_window_assigned_pixels",
                   "invalid_first_grid_assigned_pixels")},
               "search": {stage: combine_search(rows, stage) for stage in (
                   "local32", "local64", "final128")},
               "bilinear_center_counterfactual": {
                   stage: combine_search(rows, stage, "bilinear_center_counterfactual")
                   for stage in ("local32", "final128")},
               "final_pair_aepe_below_5": sum(row["stage_epe_512px"]["final128"] < 5
                                               for row in rows)}
    report = {"split": "val", "dataset": str(args.data_root),
              "pretrained_sha256": sha256_file(args.pretrained),
              "attention_checkpoint_sha256": sha256_file(args.attention_checkpoint),
              "selected_checkpoint_epoch": checkpoint.get("epoch"),
              "protocol": {
                  "direction": "visible target -> warped IR source",
                  "stage_epe": "same native512 GT and valid mask; all predicted flows interpolated to512; 16/32-grid flows multiplied by2, 64/128 unchanged",
                  "local_search": "9x9 radius4 in each feature-grid axis; nominal 512px radii 64 at32, 32 at64, 16 at128",
                  "grid_gt": "area mean of valid GT, grid query retained when valid fraction>0.8",
                  "inside": "max absolute x/y residual before local correlation <= nominal axis radius",
                  "bilinear_center_counterfactual": "search coverage if deconv4/deconv2 were replaced by align_corners=False bilinear interpolation; the full model was NOT rerun with these replacements",
                  "detail_proxy": "top quartile of visible-image Sobel magnitude among valid pixels per pair",
                  "selection": "analysis only; no training or test access"},
              "summary": summary, "per_image": rows}
    (args.output / "diagnostics.json").write_text(json.dumps(report, indent=2),
                                                    encoding="utf-8")
    flat = []
    for row in rows:
        record = {"name": row["name"], "valid_full": row["valid_full"],
                  "gt_displacement_p90_512px": row["gt_displacement_p90_512px"]}
        record.update({f"epe_{stage}_512px": row["stage_epe_512px"][stage]
                       for stage in STAGES})
        for stage in ("local32", "local64", "final128"):
            search = row["search"][stage]
            record[f"{stage}_outside_fraction"] = search["outside"]["queries"] / search["valid_queries"]
        for stage in ("local32", "final128"):
            search = row["bilinear_center_counterfactual"][stage]
            record[f"{stage}_bilinear_center_outside_fraction"] = (
                search["outside"]["queries"] / search["valid_queries"])
        for label in (*BIN_NAMES, "visible_edge_top25pct", "other_valid"):
            group = (f"gt_displacement_{label}" if label in BIN_NAMES else label)
            part = row["groups"][group]
            record[f"{group}_pixels"] = part["pixels"]
            record[f"{group}_coarse_epe_512px"] = (
                part["coarse16"] / part["pixels"] if part["pixels"] else "")
            record[f"{group}_final_epe_512px"] = (
                part["final128"] / part["pixels"] if part["pixels"] else "")
        record["last_stage_worsened_pixel_fraction"] = (
            row["groups"]["all"]["local64_to_final128_worsened"] /
            row["groups"]["all"]["pixels"])
        flat.append(record)
    with (args.output / "per_image.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps({"pairs": summary["pairs"],
                      "all_stage_epe_512px": summary["groups"]["all"]["epe_512px"],
                      "search_outside_fraction": {
                          key: value["outside"]["fraction_of_valid_queries"]
                          for key, value in summary["search"].items()},
                      "bilinear_center_outside_fraction": {
                          key: value["outside"]["fraction_of_valid_queries"]
                          for key, value in summary["bilinear_center_counterfactual"].items()},
                      "final_pair_aepe_below_5": summary["final_pair_aepe_below_5"]},
                     indent=2))


if __name__ == "__main__":
    main()
