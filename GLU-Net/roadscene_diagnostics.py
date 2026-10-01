"""Paired RoadScene train/val diagnostics for fixed GLU-Net checkpoints.

This tool deliberately rejects the test split. It never trains or selects a model.
Displacement bins and final-flow EPE use pixels at the original 512px scale.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.roadscene import RoadScenePairs
from roadscene_coarse import (correspondence_targets, make_model, prepare,
                              predict_coarse, sha256_file)


BIN_EDGES = (0, 8, 16, 32, 64, float("inf"))
BIN_NAMES = ("0-8", "8-16", "16-32", "32-64", "64+")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def binned_errors(epe, displacement, valid):
    result = {}
    for index, name in enumerate(BIN_NAMES):
        selected = valid & (displacement >= BIN_EDGES[index]) & (
            displacement < BIN_EDGES[index + 1])
        result[name] = {"pixels": int(selected.sum().item()),
                        "epe_sum": float(epe[selected].sum().item())}
    return result


def load_variant(base_path, checkpoint_path, attention, device, dns=False):
    model = make_model(base_path, attention, device, dns=dns)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.decoder4.load_state_dict(payload["decoder4_state_dict"])
    if attention:
        model.coarse_attention.load_state_dict(payload["attention_state_dict"])
    if dns:
        model.coarse_dns.load_state_dict(payload["dns_state_dict"])
    return model.eval(), payload


@torch.no_grad()
def evaluate_variant(model, loader, device, coarse_only):
    rows = {}
    warmed_up = False
    for batch in loader:
        name = batch["name"][0]
        source_in, target_in, _, _, truth, valid, _ = prepare(batch, device)
        if not warmed_up:
            predict_coarse(model, target_in, source_in)
            if not coarse_only:
                inputs = model.pre_process_data(batch["source_image"],
                                                batch["target_image"], device=device)
                model(inputs[1], inputs[0], inputs[3], inputs[2])
            synchronize(device)
            warmed_up = True
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        synchronize(device)
        started = time.perf_counter()
        predicted, corr = predict_coarse(model, target_in, source_in)
        synchronize(device)
        coarse_ms = (time.perf_counter() - started) * 1000

        coarse_error = torch.linalg.vector_norm(predicted - truth, dim=1)[0]
        coarse_disp = torch.linalg.vector_norm(truth, dim=1)[0] * 2.0
        coarse_valid = valid[0]
        indices, corr_valid = correspondence_targets(truth, valid)
        scores = corr.permute(0, 2, 3, 1)
        valid_scores = scores[corr_valid]
        valid_indices = indices[corr_valid]
        hits = valid_scores.argmax(dim=-1) == valid_indices
        true_scores = valid_scores.gather(1, valid_indices[:, None])
        true_rank = (valid_scores > true_scores).sum(dim=1) + 1
        if not coarse_valid.any() or not corr_valid.any():
            raise ValueError(f"No valid coarse correspondences in {name}")
        row = {
            "name": name,
            "valid_coarse": int(coarse_valid.sum().item()),
            "valid_corr_queries": int(corr_valid.sum().item()),
            "coarse_epe_256px": float(coarse_error[coarse_valid].mean().item()),
            "corr_top1": float(hits.float().mean().item()),
            "corr_top5": float((true_rank <= 5).float().mean().item()),
            "corr_gt_rank_mean": float(true_rank.float().mean().item()),
            "corr_ce": float(F.cross_entropy(valid_scores / 0.1,
                                               valid_indices).item()),
            "coarse_ms": coarse_ms,
            "coarse_bins_512px": binned_errors(coarse_error * 2.0, coarse_disp,
                                                 coarse_valid),
        }
        if not coarse_only:
            source, target, source_256, target_256, *_ = model.pre_process_data(
                batch["source_image"], batch["target_image"], device=device)
            synchronize(device)
            started = time.perf_counter()
            _, flows = model(target, source, target_256, source_256)
            synchronize(device)
            full_ms = (time.perf_counter() - started) * 1000
            gt = batch["flow_map"].to(device).float()
            full_valid = batch["correspondence_mask"].to(device).bool()[0]
            full_pred = F.interpolate(flows[-1], size=gt.shape[-2:],
                                      mode="bilinear", align_corners=False)
            full_error = torch.linalg.vector_norm(full_pred - gt, dim=1)[0]
            full_disp = torch.linalg.vector_norm(gt, dim=1)[0]
            if not full_valid.any():
                raise ValueError(f"No valid full-resolution pixels in {name}")
            valid_disp = full_disp[full_valid]
            row.update({
                "valid_full": int(full_valid.sum().item()),
                "gt_displacement_median_512px": float(valid_disp.median().item()),
                "gt_displacement_p90_512px": float(torch.quantile(valid_disp, 0.9).item()),
                "final_epe_512px": float(full_error[full_valid].mean().item()),
                "full_ms": full_ms,
                "full_bins_512px": binned_errors(full_error, full_disp, full_valid),
            })
        if device.type == "cuda":
            row["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        rows[name] = row
    return rows


def summarize_bins(rows, key):
    combined = {}
    for name in BIN_NAMES:
        pixels = sum(row[key][name]["pixels"] for row in rows.values())
        error_sum = sum(row[key][name]["epe_sum"] for row in rows.values())
        combined[name] = {"pixels": pixels,
                          "epe": error_sum / pixels if pixels else None}
    return combined


def summarize_pair(baseline, alternative, coarse_only, alternative_name="attention",
                   reference_name="baseline"):
    names = sorted(baseline)
    if names != sorted(alternative):
        raise ValueError("Models evaluated different RoadScene pairs")
    per_image = []
    for name in names:
        before, after = baseline[name], alternative[name]
        record = {"name": name,
                  "valid_coarse": before["valid_coarse"],
                  "valid_corr_queries": before["valid_corr_queries"],
                  "valid_full": before.get("valid_full"),
                  "gt_displacement_median_512px": before.get("gt_displacement_median_512px"),
                  "gt_displacement_p90_512px": before.get("gt_displacement_p90_512px")}
        for key in ("coarse_epe_256px", "corr_top1", "corr_top5",
                    "corr_gt_rank_mean", "corr_ce", "coarse_ms",
                    "final_epe_512px", "full_ms", "peak_allocated_mib"):
            if key in before:
                record[f"{reference_name}_{key}"] = before[key]
                record[f"{alternative_name}_{key}"] = after[key]
                record[f"delta_{key}"] = after[key] - before[key]
        for bin_name in BIN_NAMES:
            for key, label in (("coarse_bins_512px", "coarse_epe_512px"),
                               ("full_bins_512px", "final_epe_512px")):
                if key not in before:
                    continue
                count = before[key][bin_name]["pixels"]
                record[f"bin_{bin_name}_pixels_{label}"] = count
                if count:
                    base_epe = before[key][bin_name]["epe_sum"] / count
                    alternative_epe = after[key][bin_name]["epe_sum"] / count
                    record[f"bin_{bin_name}_{reference_name}_{label}"] = base_epe
                    record[f"bin_{bin_name}_{alternative_name}_{label}"] = alternative_epe
                    record[f"bin_{bin_name}_delta_{label}"] = alternative_epe - base_epe
        per_image.append(record)
    rank_key = "delta_coarse_epe_256px" if coarse_only else "delta_final_epe_512px"
    worst = sorted(per_image, key=lambda row: row[rank_key], reverse=True)
    return {
        "per_image": per_image,
        "degraded_pairs": [row for row in worst if row[rank_key] > 0],
        "largest_improvements": list(reversed(worst[-5:])),
        "displacement_bins": {
            f"{reference_name}_coarse_epe_512px": summarize_bins(baseline, "coarse_bins_512px"),
            f"{alternative_name}_coarse_epe_512px": summarize_bins(alternative, "coarse_bins_512px"),
            **({} if coarse_only else {
                f"{reference_name}_final_epe_512px": summarize_bins(baseline, "full_bins_512px"),
                f"{alternative_name}_final_epe_512px": summarize_bins(alternative, "full_bins_512px"),
            }),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--attention-checkpoint", type=Path, required=True)
    parser.add_argument("--dns-checkpoint", type=Path)
    parser.add_argument("--dns-contrastive-checkpoint", type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coarse-only", action="store_true")
    args = parser.parse_args()
    if bool(args.dns_checkpoint) != bool(args.dns_contrastive_checkpoint):
        parser.error("supply both DNS checkpoints to compare DNS and contrastive loss")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(RoadScenePairs(args.data_root, args.split), batch_size=1,
                        shuffle=False, num_workers=0)
    variants = {}
    payloads = {}
    selected = [("baseline", args.baseline_checkpoint, False, False),
                ("attention", args.attention_checkpoint, True, False)]
    if args.dns_checkpoint:
        selected.extend([("dns", args.dns_checkpoint, False, True),
                         ("dns_contrastive", args.dns_contrastive_checkpoint,
                          False, True)])
    for name, checkpoint, attention, dns in selected:
        model, payload = load_variant(args.pretrained, checkpoint, attention,
                                      device, dns=dns)
        variants[name] = evaluate_variant(model, loader, device, args.coarse_only)
        payloads[name] = {"path": str(checkpoint), "sha256": sha256_file(checkpoint),
                          "epoch": payload.get("epoch")}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = summarize_pair(variants["baseline"], variants["attention"], args.coarse_only)
    if args.dns_checkpoint:
        report["comparisons"] = {
            "baseline_vs_dns": summarize_pair(variants["baseline"], variants["dns"],
                                               args.coarse_only, "dns"),
            "baseline_vs_dns_contrastive": summarize_pair(
                variants["baseline"], variants["dns_contrastive"],
                args.coarse_only, "dns_contrastive"),
            "dns_vs_dns_contrastive": summarize_pair(
                variants["dns"], variants["dns_contrastive"],
                args.coarse_only, "dns_contrastive", "dns"),
        }
    report.update({"dataset": str(args.data_root), "split": args.split,
                   "coarse_only": args.coarse_only,
                   "pretrained_sha256": sha256_file(args.pretrained),
                   "checkpoints": payloads,
                   "bin_definition": "GT displacement norm in 512x512 pixel units; valid pixels only",
                   "note": "Analysis only: no training, model selection, or test access."})
    (args.output / "diagnostics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    rows = report["per_image"]
    with (args.output / "per_image.csv").open("w", newline="", encoding="utf-8") as file:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if args.dns_checkpoint:
        for name, comparison in report["comparisons"].items():
            rows = comparison["per_image"]
            with (args.output / f"per_image_{name}.csv").open(
                    "w", newline="", encoding="utf-8") as file:
                fieldnames = list(dict.fromkeys(key for row in rows for key in row))
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    print(json.dumps({"pairs": len(rows), "degraded_pairs": len(report["degraded_pairs"]),
                      "displacement_bins": report["displacement_bins"]}, indent=2))


if __name__ == "__main__":
    main()
