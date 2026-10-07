"""Read-only RoadScene validation diagnosis: official 23 plus a 21-pair view.

The 21-pair view excludes two named failure cases only from the report. It
never changes the RoadScene split, checkpoint, selection metric or GT mask.
The locked test split is not accepted by this script.
"""

import argparse
import csv
import importlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.roadscene import RoadScenePairs
from roadscene_coarse import sha256_file
from roadscene_local_ablation import checked_payload, new_model
from roadscene_metrics import aggregate_flow, flow_metrics
from roadscene_refinement_audit import (
    STAGES, combine_groups, combine_search, evaluate as evaluate_stages)
from roadscene_staged_mutual_fix import FIX


EXCLUDED = frozenset(("000014", "000002"))
THRESHOLDS = (1, 3, 5)
SEARCH_STAGES = ("local32", "local64", "final128")


def load_selected(pretrained, coarse_path, fine_path, device):
    coarse = checked_payload(coarse_path, pretrained, smoke=False)
    fine = torch.load(fine_path, map_location="cpu", weights_only=False)
    if (fine.get("stage") != "fine" or fine.get("arm") != "dns_attention" or
            fine.get("parent_sha256") != sha256_file(coarse_path) or
            fine.get("base_sha256") != sha256_file(pretrained) or
            fine.get("matching_fix") != FIX or fine.get("code_check_only") or
            fine.get("epoch") != 6):
        raise ValueError("Expected the selected fresh_mutual_fix fine checkpoint")
    model = new_model(coarse, device)
    model.load_state_dict(fine["model_state_dict"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def final_rows(model, dataset, device):
    rows = []
    for batch in DataLoader(dataset, batch_size=1, shuffle=False):
        source, target, source256, target256, *_ = model.pre_process_data(
            batch["source_image"].to(device),
            batch["target_image"].to(device), device=device)
        _, local = model(target, source, target256, source256)
        prediction = F.interpolate(local[-1], (512, 512), mode="bilinear",
                                   align_corners=False)
        truth = batch["flow_map"].to(device).float()
        valid = batch["correspondence_mask"].to(device).bool()
        rows.append({"name": batch["name"][0],
                     **flow_metrics(prediction[0], truth[0], valid[0])})
    return rows


def group_epe(row, group, stage="final128"):
    record = row["groups"][group]
    return record[stage] / record["pixels"] if record["pixels"] else None


def image_row(detail, final):
    if (detail["name"] != final["name"] or
            detail["valid_full"] != final["valid_full"]):
        raise ValueError("Detail and final evaluator image or GT mask differs")
    row = {"name": final["name"], "in_diagnostic_21":
           final["name"] not in EXCLUDED,
           "valid_pixels": final["valid_full"],
           "final_aepe_512px": final["aepe_512px"],
           "gt_displacement_p90_512px": detail["gt_displacement_p90_512px"]}
    for stage in STAGES:
        row[f"{stage}_epe_512px"] = detail["stage_epe_512px"][stage]
    for threshold in THRESHOLDS:
        row[f"pixel_fraction_epe_below_{threshold}px"] = (
            final["pixel_hits"][str(threshold)] / final["valid_full"])
    for name, group in (
            ("edge_top25pct", "visible_edge_top25pct"),
            ("non_edge", "other_valid"),
            ("first32_inside", "inside_first_window_assigned_pixels"),
            ("first32_outside", "outside_first_window_assigned_pixels")):
        row[f"{name}_pixels"] = detail["groups"][group]["pixels"]
        row[f"{name}_final_epe_512px"] = group_epe(detail, group)
    for stage in SEARCH_STAGES:
        record = detail["search"][stage]
        row[f"{stage}_valid_queries"] = record["valid_queries"]
        row[f"{stage}_outside_queries"] = record["outside"]["queries"]
        row[f"{stage}_outside_fraction"] = (
            record["outside"]["queries"] / record["valid_queries"])
    for stage in ("local32", "final128"):
        candidate = detail["bilinear_center_counterfactual"][stage]
        row[f"{stage}_bilinear_center_outside_fraction_counterfactual"] = (
            candidate["outside"]["queries"] / candidate["valid_queries"])
    return row


def summarize(final_rows_selected, detail_rows_selected):
    flow = aggregate_flow(final_rows_selected)
    all_pixels = combine_groups(detail_rows_selected, "all")
    edge = combine_groups(detail_rows_selected, "visible_edge_top25pct")
    non_edge = combine_groups(detail_rows_selected, "other_valid")
    inside = combine_groups(
        detail_rows_selected, "inside_first_window_assigned_pixels")
    outside = combine_groups(
        detail_rows_selected, "outside_first_window_assigned_pixels")
    if (all_pixels["pixels"] != flow["valid_pixels_full"] or
            abs(all_pixels["epe_512px"]["final128"] -
                flow["final_flow_epe_512px"]) > 1e-4):
        raise ValueError("Stage and final evaluators disagree on valid pixels or EPE")
    return {
        "pairs": flow["samples"],
        "valid_pixels": flow["valid_pixels_full"],
        "final_epe_valid_pixel_weighted_512px": flow["final_flow_epe_512px"],
        "pair_mean_aepe_512px": flow["aepe_pair_mean_512px"],
        "pixel_fraction_epe_below": {
            str(t): flow["pixel_success_percent_auxiliary"][str(t)] / 100
            for t in THRESHOLDS},
        "stage_epe_valid_pixel_weighted_512px": all_pixels["epe_512px"],
        "stage_transition_pixel_counts": all_pixels["transitions"],
        "edge_definition": "per-pair top quartile of visible Sobel magnitude among valid pixels",
        "edge_top25pct": edge,
        "other_valid": non_edge,
        "first32_inside_assigned_pixels": inside,
        "first32_outside_assigned_pixels": outside,
        "actual_local_search": {
            stage: combine_search(detail_rows_selected, stage)
            for stage in SEARCH_STAGES},
        "bilinear_center_counterfactual_not_model_output": {
            stage: combine_search(detail_rows_selected, stage,
                                  "bilinear_center_counterfactual")
            for stage in ("local32", "final128")},
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--coarse-checkpoint", type=Path, required=True)
    parser.add_argument("--fine-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be a new, empty directory")
    dataset = RoadScenePairs(args.data_root, "val")
    if len(dataset) != 23:
        raise ValueError("Expected the unchanged 23-pair validation split")
    names = [flow_path.stem for _, _, flow_path in dataset.samples]
    if not EXCLUDED.issubset(names) or len(set(names)) != 23:
        raise ValueError("Validation pair IDs do not match the 23-pair protocol")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    glunet_module = importlib.import_module("models.our_models.GLUNet")
    original_matching = glunet_module.MutualMatching
    glunet_module.MutualMatching = (
        lambda correlation: original_matching(correlation.clamp_min(0)))
    try:
        model = load_selected(args.pretrained, args.coarse_checkpoint,
                              args.fine_checkpoint, device)
        with torch.no_grad():
            details = evaluate_stages(
                model, DataLoader(dataset, batch_size=1, shuffle=False),
                device, preview_dir=None)
            finals = final_rows(model, dataset, device)
        by_detail = {row["name"]: row for row in details}
        by_final = {row["name"]: row for row in finals}
        if by_detail.keys() != by_final.keys() or len(by_final) != 23:
            raise ValueError("The two evaluator passes used different pairs")
        rows = [image_row(by_detail[name], by_final[name]) for name in names]
        subset = [name for name in names if name not in EXCLUDED]
        official = summarize([by_final[name] for name in names],
                             [by_detail[name] for name in names])
        diagnostic = summarize([by_final[name] for name in subset],
                               [by_detail[name] for name in subset])
        prior_path = args.fine_checkpoint.parent / "report_fine.json"
        prior = json.loads(prior_path.read_text(encoding="utf-8"))["result"]["common"]
        if (official["valid_pixels"] != prior["valid_pixels_full"] or
                abs(official["final_epe_valid_pixel_weighted_512px"] -
                    prior["final_flow_epe_512px"]) > 1e-4):
            raise ValueError("Official 23-pair rerun differs from selected fine report")
        args.output.mkdir(parents=True, exist_ok=True)
        write_csv(args.output / "per_image_all23.csv", rows)
        write_csv(args.output / "per_image_diagnostic21.csv",
                  [row for row in rows if row["in_diagnostic_21"]])
        report = {
            "experiment": "read_only_validation_subset_diagnosis",
            "split": "val", "test_pairs_accessed": False,
            "official_validation_unchanged": True,
            "exclusion_applies_only_to_diagnostic_view": sorted(EXCLUDED),
            "dataset": str(args.data_root),
            "pretrained_sha256": sha256_file(args.pretrained),
            "coarse_checkpoint_sha256": sha256_file(args.coarse_checkpoint),
            "fine_checkpoint_sha256": sha256_file(args.fine_checkpoint),
            "matching_fix": FIX,
            "flow_direction": "visible target -> infrared source",
            "flow_unit": "512x512 image pixels at every reported stage",
            "mask": "same finite GT and in-bounds source-coordinate mask in both views",
            "window": "actual 9x9 local correlation, radius 4 grid cells; query-valid GT fraction >0.8",
            "window_ratio_unit": "valid local grid queries, not image pixels",
            "bilinear_center_counterfactual": "search coverage if only the pre-correlation deconv4/deconv2 centre were bilinear; model predictions were not rerun",
            "official_23": official,
            "diagnostic_21": diagnostic,
        }
        (args.output / "report_val_views.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        compact = {key: {
            "pairs": view["pairs"],
            "valid_pixels": view["valid_pixels"],
            "final_epe_512px": view["final_epe_valid_pixel_weighted_512px"],
            "stage_epe_512px": view["stage_epe_valid_pixel_weighted_512px"],
            "pixel_fraction_epe_below": view["pixel_fraction_epe_below"],
            "edge_final_epe_512px": view["edge_top25pct"]["epe_512px"]["final128"],
            "local_outside_fraction": {stage: view["actual_local_search"][stage]
                                       ["outside"]["fraction_of_valid_queries"]
                                       for stage in SEARCH_STAGES},
            "bilinear_center_outside_fraction_counterfactual": {
                stage: view["bilinear_center_counterfactual_not_model_output"]
                    [stage]["outside"]["fraction_of_valid_queries"]
                for stage in ("local32", "final128")},
        } for key, view in (("official_23", official),
                            ("diagnostic_21", diagnostic))}
        print(json.dumps(compact, indent=2), flush=True)
    finally:
        glunet_module.MutualMatching = original_matching


if __name__ == "__main__":
    main()
