"""Common 512px train/val evaluator for GLU, SA/CA, MIND, DNS and CRFT.

Test is locked except for a CRFT-only final run bound to a successful validation
report with identical weight/source/evaluator hashes. Existing GLU test results
are not recomputed or used here. Missing CRFT weights are reported, never replaced
with a randomly initialized accuracy result.
"""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.roadscene import RoadScenePairs
from roadscene_coarse import (coarse_statistics, make_model, prepare, predict_coarse, sha256_file)
from roadscene_diagnostics import binned_errors, summarize_bins
from roadscene_metrics import CMR_THRESHOLDS, aggregate_flow, flow_metrics


PROTOCOL = {"version": 1, "direction": "visible(target) -> warped IR(source)",
    "resolution": [512, 512], "flow_unit": "512x512 image pixels",
    "mask": "finite GT and GT-mapped source coordinate inside image; no prediction-dependent exclusions",
    "aepe": "mean valid-pixel EPE per pair; all pairs included in arithmetic pair mean",
    "cmr": "100 * count(pair AEPE < threshold) / count(all pairs)",
    "cmr_thresholds_512px": list(CMR_THRESHOLDS),
    "legacy_epe": "sum valid EPE / sum valid pixels (unchanged project convention)",
    "batch_size": 1, "warmup_forwards": 3, "timed_repeats_per_pair": 5,
    "timing": "median of 5 synchronized forwards including on-device model preprocessing and output interpolation; excludes disk IO, host transfer, GT and metrics",
    "precision": "float32, no autocast", "crft_native_input_size": 64,
    "preprocessing": "same native512 uint8 RGB pairs; GLU ImageNet/256 byte area resize; CRFT official64 uint8 bilinear resize + internal per-channel mean/std; CRFT flow resized/scaled to512"}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def evaluator_hashes():
    root = Path(__file__).parent
    names = ("roadscene_compare.py", "roadscene_metrics.py", "roadscene_coarse.py",
             "datasets/roadscene.py", "models/our_models/GLUNet.py", "crft_adapter.py",
             "models/our_models/mind.py", "models/our_models/phase_congruency.py",
             "models/our_models/coarse_attention.py",
             "models/our_models/coarse_dns.py", "models/feature_backbones/VGG_features.py",
             "roadscene_diagnostics.py")
    return {name: sha256_file(root / name) for name in names}


def load_glu(pretrained, checkpoint, arm, device):
    attention = arm in {"attention", "mind_a_attention", "mind_b_attention"}
    mind = "a" if arm.startswith("mind_a") else "b" if arm.startswith("mind_b") else None
    dns = arm in {"dns", "dns_contrastive"}
    phase = arm == "phase_attention"
    if phase:
        attention = True
    model = make_model(pretrained, attention, device, dns=dns, mind=mind,
                       phase=phase)
    meta = {"base_sha256": sha256_file(pretrained)}
    if checkpoint:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("arm", arm) != arm:
            raise ValueError(f"Checkpoint arm mismatch: {checkpoint}")
        if payload.get("base_sha256", meta["base_sha256"]) != meta["base_sha256"]:
            raise ValueError("Checkpoint was trained from a different base weight")
        model.decoder4.load_state_dict(payload["decoder4_state_dict"])
        if attention:
            model.coarse_attention.load_state_dict(payload["attention_state_dict"])
        if mind:
            model.coarse_mind.load_state_dict(payload["mind_state_dict"])
        if phase:
            model.coarse_phase.load_state_dict(payload["phase_state_dict"])
        if dns:
            model.coarse_dns.load_state_dict(payload["dns_state_dict"])
        meta.update({"path": str(checkpoint), "sha256": sha256_file(checkpoint),
                     "epoch": payload.get("epoch"),
                     "budget": payload.get("budget"),
                     "code_check_only": payload.get("budget", {}).get("code_check", False)})
    else:
        if mind or dns or phase:
            raise ValueError("Accuracy comparison requires a trained MIND/DNS/phase checkpoint")
        meta["initialization"] = "base pretrained, no RoadScene decoder fine-tuning"
    return model.eval(), meta


def _predict_full(model, source, target, crft):
    if crft:
        from crft_adapter import predict_crft
        return predict_crft(model, target, source)
    source_in, target_in, source_256, target_256, *_ = model.pre_process_data(source, target,
                                                                          device=source.device)
    _, flows = model(target_in, source_in, target_256, source_256)
    return F.interpolate(flows[-1], size=target.shape[-2:], mode="bilinear",
                         align_corners=False), None


def save_preview(folder, name, source, target, predicted, truth, valid):
    folder.mkdir(parents=True, exist_ok=True)
    y, x = torch.meshgrid(torch.arange(512, device=source.device),
                          torch.arange(512, device=source.device), indexing="ij")
    grid = torch.stack(((x + predicted[0]) / 511 * 2 - 1,
                        (y + predicted[1]) / 511 * 2 - 1), dim=-1)[None]
    warped = F.grid_sample(source.float()[None] / 255, grid, align_corners=True)[0]
    original = source.float() / 255
    target = target.float() / 255
    errors = torch.linalg.vector_norm(predicted - truth, dim=0)
    heat = (errors / 40).clamp(0, 1)
    heat_rgb = torch.stack((heat, 1 - heat, torch.zeros_like(heat)))
    heat_rgb[:, ~valid] = 0
    panel = torch.cat((target, original, warped, heat_rgb), dim=2)
    Image.fromarray((panel.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(
        folder / f"{name}.png")
    np.savez_compressed(folder / f"{name}.npz", final_flow=predicted.cpu().numpy(),
                        gt=truth.cpu().numpy(), valid=valid.cpu().numpy())


@torch.no_grad()
def evaluate_common(model, loader, device, crft=False, previews=None, preview_names=()):
    rows = []
    for index, batch in enumerate(loader):
        source = batch["source_image"].to(device)
        target = batch["target_image"].to(device)
        if source.shape[-2:] != (512, 512) or target.shape[-2:] != (512, 512):
            raise ValueError("Protocol requires native 512x512 inputs and GT")
        if index == 0:
            for _ in range(PROTOCOL["warmup_forwards"]):
                _predict_full(model, source, target, crft)
            synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        timings = []
        predicted = coarse_crft = None
        for _ in range(PROTOCOL["timed_repeats_per_pair"]):
            del predicted, coarse_crft
            synchronize(device)
            started = time.perf_counter()
            predicted, coarse_crft = _predict_full(model, source, target, crft)
            synchronize(device)
            timings.append((time.perf_counter() - started) * 1000)
        peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
        truth = batch["flow_map"].to(device).float()
        valid = batch["correspondence_mask"].to(device).bool()
        row = {"name": batch["name"][0], **flow_metrics(predicted[0], truth[0], valid[0]),
               "inference_ms": float(np.median(timings)), "peak_allocated_mib": peak,
               "full_bins_512px": binned_errors(torch.linalg.vector_norm(predicted[0] - truth[0], dim=0),
                  torch.linalg.vector_norm(truth[0], dim=0), valid[0])}
        source_in, target_in, _, _, coarse_gt, coarse_valid, _ = prepare(batch, device)
        if crft:
            # Official coarse flow is on 1/8 grid and in that grid's pixels.
            coarse_flow = F.interpolate(coarse_crft, (16, 16), mode="area")
            coarse_flow = coarse_flow * coarse_flow.new_tensor((256 / coarse_crft.shape[3],
                256 / coarse_crft.shape[2]))[None, :, None, None]
            coarse_error = torch.linalg.vector_norm(coarse_flow - coarse_gt, dim=1)
            row.update({"coarse_epe_256px": float(coarse_error[coarse_valid].mean()),
                        "valid_coarse": int(coarse_valid.sum()), "corr_top1": None,
                        "corr_note": "native CRFT 8x8 matches at64px are not GLU 16x16 bins; not compared"})
        else:
            coarse_flow, corr = predict_coarse(model, target_in, source_in)
            values, counts = coarse_statistics(coarse_flow, corr, coarse_gt, coarse_valid, return_counts=True)
            row.update({"coarse_epe_256px": float(values[0]), "corr_top1": float(values[2]),
                        "valid_coarse": counts[1], "valid_corr_queries": counts[3],
                        "corr_hits": counts[4]})
        rows.append(row)
        if previews is not None and (not preview_names or row["name"] in preview_names):
            save_preview(previews, row["name"], source[0], target[0], predicted[0], truth[0], valid[0])
        del predicted, coarse_crft, truth, valid, source, target, source_in, target_in
        del coarse_gt, coarse_valid, coarse_flow
        if not crft:
            del corr
    summary = aggregate_flow(rows)
    coarse_count = sum(row["valid_coarse"] for row in rows)
    summary.update({"coarse_epe_256px": sum(row["coarse_epe_256px"] * row["valid_coarse"] for row in rows) / coarse_count,
                    "valid_pixels_coarse": coarse_count,
                    "inference_ms_mean_pair_median": float(np.mean([row["inference_ms"] for row in rows])),
                    "peak_allocated_mib": max(row["peak_allocated_mib"] for row in rows) if device.type == "cuda" else None,
                    "corr_top1": (sum(row["corr_hits"] for row in rows) / sum(row["valid_corr_queries"] for row in rows)) if not crft else None,
                    "displacement_bins_512px": summarize_bins({row["name"]: row for row in rows}, "full_bins_512px")})
    return summary, rows


def save_comparison(output, results):
    flattened = []
    by_arm = {arm: {row["name"]: row for row in data["per_image"]}
              for arm, data in results.items() if "per_image" in data}
    if not by_arm:
        return
    names = next(iter(by_arm.values())).keys()
    for name in names:
        row = {"name": name}
        for arm, records in by_arm.items():
            record = records[name]
            for key in ("aepe_512px", "coarse_epe_256px", "corr_top1", "valid_full", "inference_ms", "peak_allocated_mib"):
                row[f"{arm}_{key}"] = record.get(key)
            if "attention" in by_arm:
                row[f"{arm}_minus_attention_aepe"] = record["aepe_512px"] - by_arm["attention"][name]["aepe_512px"]
        flattened.append(row)
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)
    contrasts = (("baseline", "attention"), ("baseline", "mind_a"),
                 ("baseline", "mind_b"), ("attention", "mind_a"),
                 ("attention", "mind_b"), ("attention", "mind_a_attention"),
                 ("attention", "mind_b_attention"),
                 ("attention", "phase_attention"),
                 ("mind_b_attention", "phase_attention"),
                 ("mind_a", "mind_a_attention"),
                 ("mind_b", "mind_b_attention"), ("baseline", "dns"),
                 ("dns", "dns_contrastive"), ("attention", "dns_contrastive"),
                 ("attention", "crft"))
    differences = {}
    for reference, alternative in contrasts:
        if reference not in by_arm or alternative not in by_arm:
            continue
        before, after = by_arm[reference], by_arm[alternative]
        if before.keys() != after.keys():
            raise ValueError("Methods were evaluated on different image pairs")
        paired = []
        for name in before:
            if before[name]["valid_full"] != after[name]["valid_full"]:
                raise ValueError("Methods used different GT masks")
            paired.append({"name": name,
                "delta_final_aepe_512px": after[name]["aepe_512px"] - before[name]["aepe_512px"],
                "delta_coarse_epe_256px": after[name]["coarse_epe_256px"] - before[name]["coarse_epe_256px"],
                "delta_inference_ms": after[name]["inference_ms"] - before[name]["inference_ms"]})
        ranked = sorted(paired, key=lambda row: row["delta_final_aepe_512px"], reverse=True)
        differences[f"{reference}_vs_{alternative}"] = {
            "improved_pairs": sum(row["delta_final_aepe_512px"] < 0 for row in paired),
            "degraded_pairs": [row for row in ranked if row["delta_final_aepe_512px"] > 0],
            "coarse_improved_final_degraded": [row for row in ranked if row["delta_coarse_epe_256px"] < 0 < row["delta_final_aepe_512px"]],
            "largest_improvements": list(reversed(ranked[-3:])), "per_image": paired,
            "interpretation": "A/B without attention versus SA/CA is descriptive, not a single-factor attention/MIND effect" if reference == "attention" and alternative in {"mind_a", "mind_b"} else "matched structural contrast"}
    (output / "paired_differences.json").write_text(json.dumps(differences, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--attention-checkpoint", type=Path)
    parser.add_argument("--experiment-dir", type=Path, help="selected MIND or DNS arms in best_<arm>.pth")
    parser.add_argument("--crft-root", type=Path)
    parser.add_argument("--crft-checkpoint", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--final-crft-test", action="store_true")
    parser.add_argument("--validated-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview-ids", nargs="*", default=("000013", "000002", "000004"),
                        help="matched cases chosen from prior train/val analysis, never test")
    args = parser.parse_args()
    if args.split == "test" and not (args.final_crft_test and args.validated_report and args.crft_checkpoint):
        parser.error("Test locked: only CRFT final test after validated-report is permitted")
    if args.final_crft_test and args.split != "test":
        parser.error("final-crft-test requires test split")
    if args.experiment_dir and args.split != "test":
        args.baseline_checkpoint = args.baseline_checkpoint or args.experiment_dir / "best_baseline.pth"
        args.attention_checkpoint = args.attention_checkpoint or args.experiment_dir / "best_attention.pth"
        if not args.baseline_checkpoint.is_file() or not args.attention_checkpoint.is_file():
            parser.error("Matched experiment requires its baseline and attention control checkpoints")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2026)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    arms = [] if args.split == "test" else [("baseline", args.baseline_checkpoint),
                                            ("attention", args.attention_checkpoint)]
    if args.experiment_dir and args.split != "test":
        for arm in ("mind_a", "mind_a_attention", "mind_b", "mind_b_attention",
                    "phase_attention", "dns", "dns_contrastive"):
            path = args.experiment_dir / f"best_{arm}.pth"
            if path.is_file():
                arms.append((arm, path))
    loader = DataLoader(RoadScenePairs(args.data_root, args.split), batch_size=1, shuffle=False)
    data_manifest = [{"name": paths[2].stem,
                      "files": {str(path.relative_to(args.data_root)): sha256_file(path)
                                for path in paths}} for paths in loader.dataset.samples]
    pair_manifest = hashlib.sha256(json.dumps(data_manifest, sort_keys=True).encode()).hexdigest()
    for arm, checkpoint in arms:
        model, meta = load_glu(args.pretrained, checkpoint, arm, device)
        if meta.get("code_check_only"):
            raise ValueError("Smoke-check weights cannot be used as experiment results")
        if args.experiment_dir:
            budget = meta.get("budget")
            if not budget:
                raise ValueError("Matched experiment checkpoint is missing training budget metadata")
            if arm != "baseline" and budget != results["baseline"]["checkpoint"]["budget"]:
                raise ValueError("Arms have different training budgets; retrain matched controls")
        summary, rows = evaluate_common(model, loader, device,
                                         previews=args.output / "previews" / arm,
                                         preview_names=args.preview_ids)
        results[arm] = {"summary": summary, "per_image": rows, "checkpoint": meta}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"{arm}: {json.dumps(summary)}", flush=True)
    if args.crft_checkpoint:
        if not args.crft_root:
            parser.error("crft-checkpoint requires crft-root")
        from crft_adapter import load_crft
        model, meta = load_crft(args.crft_root, args.crft_checkpoint, device)
        if args.split == "test":
            validation = json.loads(args.validated_report.read_text(encoding="utf-8"))
            previous = validation.get("results", {}).get("crft", {})
            if validation.get("split") != "val" or previous.get("checkpoint") != meta or validation.get("protocol") != PROTOCOL or validation.get("evaluator_hashes") != evaluator_hashes():
                raise ValueError("CRFT final protocol differs from successful validation report")
            if not previous.get("summary", {}).get("samples") == 23:
                raise ValueError("CRFT must first evaluate all 23 validation pairs")
        summary, rows = evaluate_common(model, loader, device, crft=True,
            previews=args.output / "previews" / "crft" if args.split != "test" else None,
            preview_names=args.preview_ids)
        results["crft"] = {"summary": summary, "per_image": rows, "checkpoint": meta}
        del model
    else:
        results["crft"] = {"status": "not_evaluated", "reason": "CRFT RoadScene trained checkpoint unavailable"}
    report = {"split": args.split, "dataset": str(args.data_root), "pair_manifest_sha256": pair_manifest,
              "data_manifest": data_manifest,
              "protocol": PROTOCOL, "evaluator_hashes": evaluator_hashes(), "results": results,
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
              "note": "No test-based selection. Frozen GLU test results kept separately; new results use this protocol."}
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_comparison(args.output, results)


if __name__ == "__main__":
    main()
