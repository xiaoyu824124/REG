"""Matched RoadScene validation of SA/CA, DNS, and DNS -> SA -> CA.

All three arms start from the same pretrained GLU-Net, train for the same
updates and validate with one evaluator. No arm continues from a trained
SA/CA checkpoint. The selected existing SA/CA checkpoint is used only for
a zero-gate identity check. Test is never opened by this script.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets.roadscene import RoadScenePairs
from roadscene_coarse import make_model, predict_coarse, prepare, sha256_file
from roadscene_compare import evaluate_common, load_glu
from roadscene_large_motion_dns import (SEED, coarse_and_window,
                                        combine_diagnostics, train_arm)


ARMS = ("attention", "dns", "dns_attention")


@torch.no_grad()
def check_dns_zero_gate(pretrained, selected_attention, val_loader, device):
    reference, meta = load_glu(pretrained, selected_attention, "attention", device)
    combined = make_model(pretrained, attention=True, dns=True, device=device)
    payload = torch.load(selected_attention, map_location="cpu", weights_only=False)
    combined.decoder4.load_state_dict(payload["decoder4_state_dict"])
    combined.coarse_attention.load_state_dict(payload["attention_state_dict"])
    if float(combined.coarse_dns.gate) != 0:
        raise RuntimeError("DNS gate must initialize to exactly zero")
    batch = next(iter(val_loader))
    source, target, *_ = prepare(batch, device)
    base_flow, base_corr = predict_coarse(reference, target, source)
    new_flow, new_corr = predict_coarse(combined, target, source)
    diff = {"coarse_flow_max_abs_diff": float((base_flow - new_flow).abs().max()),
            "correlation_max_abs_diff": float((base_corr - new_corr).abs().max()),
            "selected_attention_sha256": meta["sha256"]}
    if max(diff["coarse_flow_max_abs_diff"],
           diff["correlation_max_abs_diff"]) > 1e-6:
        raise RuntimeError(f"Zero-gate combined model changed SA/CA: {diff}")
    # Full path is unchanged after the same coarse flow; confirm end-to-end.
    source_full, target_full, source_256, target_256, *_ = reference.pre_process_data(
        batch["source_image"], batch["target_image"], device=device)
    _, reference_full = reference(target_full, source_full, target_256, source_256)
    _, combined_full = combined(target_full, source_full, target_256, source_256)
    diff["final_flow_max_abs_diff"] = float((reference_full[-1] - combined_full[-1]).abs().max())
    if diff["final_flow_max_abs_diff"] > 1e-6:
        raise RuntimeError(f"Zero-gate combined model changed final SA/CA flow: {diff}")
    del reference, combined
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return diff


def evaluate_arm(arm, args, loader, device):
    model, meta = load_glu(args.pretrained, args.output / f"best_{arm}.pth",
                           arm, device)
    summary, rows = evaluate_common(model, loader, device, previews=None)
    details = coarse_and_window(model, loader, device)
    by_name = {row["name"]: row for row in rows}
    if by_name.keys() != details.keys():
        raise RuntimeError("Final and window analysis used different pairs")
    for name, item in details.items():
        by_name[name].update(item)
    result = {"summary": summary, "diagnostics": combine_diagnostics(details),
              "per_image": rows, "checkpoint": meta}
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--existing-attention-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--code-check", action="store_true",
                        help="one epoch on two train/one val pairs; not an accuracy result")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch size and lr must be positive")
    if args.output.exists() and any(args.output.glob("best_*.pth")):
        parser.error("Output already contains checkpoints; choose a new directory")
    if args.code_check:
        args.epochs = 1
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    full_train = RoadScenePairs(args.data_root, "train")
    full_val = RoadScenePairs(args.data_root, "val")
    train_dataset = Subset(full_train, range(2)) if args.code_check else full_train
    val_dataset = Subset(full_val, range(1)) if args.code_check else full_val
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    check = check_dns_zero_gate(args.pretrained, args.existing_attention_checkpoint,
                                val_loader, device)
    report = {"split": "val", "code_check_only": args.code_check,
              "train_pairs": len(train_dataset), "val_pairs": len(val_dataset),
              "arms": ARMS, "base_sha256": sha256_file(args.pretrained),
              "existing_attention_sha256": sha256_file(args.existing_attention_checkpoint),
              "epochs": args.epochs, "batch_size": args.batch_size,
              "lr": args.lr, "seed": SEED,
              "train_steps_per_epoch": (len(train_dataset) + args.batch_size - 1) // args.batch_size,
              "initialization_check": check,
              "design": {
                  "dns": "shared 2D two-ring centre-free neighbour self-similarity on VGG 16x16 features, projected to 512 channels, zero-gated residual",
                  "combined_order": "VGG -> DNS residual -> existing self attention -> existing cross attention -> Global Correlation",
                  "arms_are_independent": "all start from the same pretrained base; no continuation from trained SA/CA; no contrastive loss or sampling/weighting change",
                  "training": "same train split, shuffled image order, batch/steps, AdamW, coarse EPE + 2x correlation CE",
                  "within_arm_checkpoint": "minimum validation coarse EPE, matching selected existing SA/CA protocol",
                  "between_arm_decision": "final 512px EPE, actual first-window miss rate, >=64px GT region EPE and per-pair stability; coarse EPE alone insufficient",
                  "test": "never accessed"},
              "history": {}, "costs": {}, "results": {}}
    for arm in ARMS:
        history, cost = train_arm(arm, args, train_dataset, val_loader,
                                  device, high=[], low=[])
        report["history"][arm], report["costs"][arm] = history, cost
        (args.output / "progress.json").write_text(json.dumps(report, indent=2),
                                                     encoding="utf-8")
    expected_order = [row["image_order_sha256"] for row in report["history"]["attention"]]
    for arm in ARMS[1:]:
        if [row["image_order_sha256"] for row in report["history"][arm]] != expected_order:
            raise RuntimeError("Matched DNS arms saw different train image order")
    common_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    for arm in ARMS:
        report["results"][arm] = evaluate_arm(arm, args, common_loader, device)
        summary = report["results"][arm]["summary"]
        diag = report["results"][arm]["diagnostics"]
        print(f"{arm}: coarse={summary['coarse_epe_256px']:.4f}, "
              f"final={summary['final_flow_epe_512px']:.4f}, "
              f"first-window-outside={diag['first_window_outside_fraction']:.4f}",
              flush=True)
    indexed = {arm: {row["name"]: row for row in report["results"][arm]["per_image"]}
               for arm in ARMS}
    pairs = []
    for name in indexed["attention"]:
        row = {"name": name, "valid_full": indexed["attention"][name]["valid_full"]}
        for arm in ARMS:
            item = indexed[arm][name]
            if item["valid_full"] != row["valid_full"]:
                raise RuntimeError("Valid GT masks differ across arms")
            large = item["full_bins_512px"]["64+"]
            row[f"{arm}_coarse_256px"] = item["coarse_epe_256px"]
            row[f"{arm}_coarse_full_512px"] = item["coarse_full_epe_sum_512px"] / item["valid_full"]
            row[f"{arm}_first_window_outside"] = (
                item["first_window_outside_queries"] / item["first_window_valid_queries"])
            row[f"{arm}_large_final_epe_512px"] = (
                large["epe_sum"] / large["pixels"] if large["pixels"] else "")
            row[f"{arm}_final_epe_512px"] = item["aepe_512px"]
            row[f"{arm}_inference_ms"] = item["inference_ms"]
        pairs.append(row)
    report["paired_stability"] = {arm: {
        "improved_vs_attention": sum(row[f"{arm}_final_epe_512px"] < row["attention_final_epe_512px"]
                                     for row in pairs),
        "worsened_vs_attention": sum(row[f"{arm}_final_epe_512px"] > row["attention_final_epe_512px"]
                                     for row in pairs)} for arm in ARMS[1:]}
    with (args.output / "per_image.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2),
                                                  encoding="utf-8")
    print(json.dumps({"code_check_only": args.code_check,
                      "paired_stability": report["paired_stability"]}, indent=2))


if __name__ == "__main__":
    main()
