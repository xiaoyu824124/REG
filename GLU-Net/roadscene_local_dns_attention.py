"""Matched A/B 256-grid fine-only RoadScene experiment.

The selected fresh_mutual_fix fine checkpoint is immutable. Train split updates
only the new branch. Fixed 21-pair validation view selects checkpoints; all 23
pairs and the two excluded cases are still reported. Test is never loaded.
"""

import argparse
import csv
import importlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from datasets.roadscene import RoadScenePairs
from models.our_models.local_256_refiner import Local256Residual
from models.our_models.local_dns_attention_fusion import (
    LocalDNSAttentionFusion, LocalScaleAttention)
from roadscene_coarse import sha256_file
from roadscene_local256 import (GROUPS, THRESHOLDS, base_change, evaluate,
                                inputs, make_model, snapshot_base)
from roadscene_local_ablation import save_atomic
from roadscene_staged_mutual_fix import FIX
import roadscene_staged as staged


EXCLUDED = frozenset(("000002", "000014"))
STEPS = 528
EVAL_EVERY = 176
SEED = staged.SEED
ARMS = ("local256", "dns_local_saca_fusion")
ABLATIONS = {
    "without_dns": {"use_dns": False},
    "without_local_saca": {"use_attention": False},
    "without_cross_scale": {"use_cross_scale": False},
}


class Candidate(nn.Module):
    def __init__(self, base, arm, flags=None):
        super().__init__()
        self.base = base.eval()
        self.base.train_coarse_encoder = False
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        self.arm = arm
        self.flags = flags or {}
        self.refiner = (Local256Residual() if arm == "local256" else
                        LocalDNSAttentionFusion(**self.flags))

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def pre_process_data(self, *args, **kwargs):
        return self.base.pre_process_data(*args, **kwargs)

    def run(self, target, source, target256, source256):
        level1, level2 = [], []

        def capture(store):
            def remember(_module, _inputs, output):
                if len(store) < 2:
                    store.append(output.detach())
            return remember

        handles = [self.base.pyramid._modules["level_1"].register_forward_hook(
            capture(level1))]
        if self.arm != "local256" and self.flags.get("use_cross_scale", True):
            handles.append(self.base.pyramid._modules["level_2"].register_forward_hook(
                capture(level2)))
        try:
            with torch.no_grad():
                coarse, local = self.base(target, source, target256, source256)
        finally:
            for handle in handles:
                handle.remove()
        if (len(level1) != 2 or
                any(value.shape[1:] != (64, 256, 256) for value in level1)):
            raise RuntimeError("VGG level_1 must be VI then IR at 256 grid")
        if self.arm == "local256":
            final, grid, delta = self.refiner(level1[0], level1[1], local[-1])
            valid = None
        else:
            if not level2:
                # Cross-scale ablation retains the 128-grid interface with no
                # information path from it; supply zeros only in that ablation.
                level2 = [torch.zeros(target.shape[0], 128, 128, 128,
                                      device=target.device, dtype=level1[0].dtype)] * 2
            elif (len(level2) != 2 or
                  any(value.shape[1:] != (128, 128, 128) for value in level2)):
                raise RuntimeError("VGG level_2 must be VI then IR at 128 grid")
            final, grid, delta, valid = self.refiner(
                level2[0], level2[1], level1[0], level1[1], local[-1])
        return {"coarse": coarse, "local": local, "final512": final,
                "grid256": grid, "residual256": delta, "warp_valid256": valid}

    def forward(self, target, source, target256, source256):
        result = self.run(target, source, target256, source256)
        return result["coarse"], [*result["local"], result["final512"]]


def selected_indices(dataset):
    names = [sample[2].stem for sample in dataset.samples]
    if len(names) != 23 or len(set(names)) != 23 or not EXCLUDED.issubset(names):
        raise ValueError("Expected original 23-pair RoadScene validation")
    return [index for index, name in enumerate(names) if name not in EXCLUDED]


@torch.no_grad()
def geometry_check():
    result = {}
    for size in (128, 256):
        source = torch.zeros(1, 1, size, size)
        shift = size // 128
        source[0, 0, 40, 40 + shift] = 1
        positive = torch.zeros(1, 2, size, size)
        positive[:, 0] = 4  # +4 image px = +1/+2 grid px at 128/256.
        samples, valid = LocalScaleAttention.sample_neighbours(source, positive)
        opposite, _ = LocalScaleAttention.sample_neighbours(source, -positive)
        correct = float(samples[0, 0, 4, 40, 40])
        wrong = float(opposite[0, 0, 4, 40, 40])
        outside = bool(valid[0, 4, 40, size - 1])
        if abs(correct - 1) > 1e-5 or wrong != 0 or outside:
            raise RuntimeError(f"VI->IR warp geometry failed at grid {size}")
        result[str(size)] = {"flow_512px": 4, "source_offset_grid": shift,
                             "correct_center": correct,
                             "opposite_center": wrong,
                             "border_center_valid": outside}
    return result


@torch.no_grad()
def validation_epe(model, val21, device):
    model.eval()
    error_sum, count = 0., 0
    for batch in DataLoader(val21, batch_size=1, shuffle=False):
        source, target, source256, target256 = inputs(model, batch, device)
        prediction = model.run(target, source, target256, source256)["final512"]
        truth = batch["flow_map"].to(device).float()
        valid = batch["correspondence_mask"].to(device).bool()
        error = torch.linalg.vector_norm(prediction - truth, dim=1)
        error_sum += float(error[valid].sum())
        count += int(valid.sum())
    return error_sum / count


@torch.no_grad()
def exact_initialization(model, val, device):
    largest, invalid = 0., [0, 0]
    for batch in DataLoader(val, batch_size=1, shuffle=False):
        source, target, source256, target256 = inputs(model, batch, device)
        result = model.run(target, source, target256, source256)
        baseline = F.interpolate(result["local"][-1], (512, 512),
                                 mode="bilinear", align_corners=False)
        largest = max(largest, float((result["final512"] - baseline).abs().max()))
        if result["warp_valid256"] is not None:
            mask = result["warp_valid256"]
            invalid[0] += int((~mask).sum())
            invalid[1] += mask.numel()
            if torch.count_nonzero(result["residual256"][~mask[:, None].expand_as(
                    result["residual256"])]) != 0:
                raise RuntimeError("Invalid warp predicted a nonzero residual")
    if largest != 0:
        raise RuntimeError(f"Zero head failed to reproduce frozen flow: {largest}")
    return {"max_abs_flow_difference_512px": largest,
            "invalid_warp_fraction": invalid[0] / invalid[1] if invalid[1] else None}


def group_summary(rows):
    result = {"pairs": len(rows), "valid_pixels": sum(row["valid_pixels"] for row in rows),
              "improved_pairs_vs_base": sum(row["new_epe_512px"] <
                                             row["baseline_epe_512px"] for row in rows)}
    for arm in ("baseline", "new"):
        result[f"{arm}_final_epe_512px"] = (
            sum(row[f"{arm}_epe_512px"] * row["valid_pixels"] for row in rows) /
            result["valid_pixels"])
        result[f"{arm}_mean_pair_epe_512px"] = float(np.mean([
            row[f"{arm}_epe_512px"] for row in rows]))
        result[f"{arm}_mean_inference_ms"] = float(np.mean([
            row[f"{arm}_ms"] for row in rows]))
        result[f"{arm}_pixel_fraction_below"] = {
            str(t): sum(row[f"{arm}_fraction_epe_below_{t}px"] *
                        row["valid_pixels"] for row in rows) /
                    result["valid_pixels"] for t in THRESHOLDS}
    result["regions"] = {}
    for group in GROUPS[1:]:
        pixels = sum(row[f"{group}_pixels"] for row in rows)
        result["regions"][group] = {
            "pixels": pixels,
            **{arm + "_epe_512px": sum(
                (row[f"{group}_{arm}_epe_512px"] or 0) *
                row[f"{group}_pixels"] for row in rows) / pixels if pixels else None
               for arm in ("baseline", "new")}}
    return result


def stable_gain(a_rows, b_rows):
    """Fixed before the server run; only val21 decides extra ablations."""
    a = {row["name"]: row for row in a_rows if row["name"] not in EXCLUDED}
    b = {row["name"]: row for row in b_rows if row["name"] not in EXCLUDED}
    if len(a) != 21 or set(a) != set(b):
        raise ValueError("A and B must be paired on the same fixed 21 images")
    aa, bb = group_summary(list(a.values())), group_summary(list(b.values()))
    improved = sum(b[name]["new_epe_512px"] < a[name]["new_epe_512px"]
                   for name in a)
    gain = aa["new_final_epe_512px"] - bb["new_final_epe_512px"]
    regions = {}
    for name in ("edge_top_quartile", "large_motion_64plus", "first32_outside"):
        before = aa["regions"][name]["new_epe_512px"]
        after = bb["regions"][name]["new_epe_512px"]
        regions[name] = None if before is None or after is None else after - before
    decision = {"val21_gain_b_vs_a_512px": gain,
                "val21_pairs_improved_b_vs_a": improved,
                "region_change_b_minus_a_512px": regions,
                "rule": "gain >= 0.1px; >=15/21 pairs improve; edge, large-motion and first32-outside EPE each increase <=0.1px"}
    decision["run_ablations"] = (
        gain >= .1 and improved >= 15 and
        all(value is not None and value <= .1 for value in regions.values()))
    return decision


def train_arm(model, trainset, val21, device, output, steps, code_check):
    optimizer = torch.optim.AdamW(model.refiner.parameters(), lr=1e-4)
    frozen = snapshot_base(model)
    best = validation_epe(model, val21, device)
    history = [{"step": 0, "val21_final_epe_512px": best}]
    selected_step = 0

    def save():
        save_atomic({"arm": model.arm, "flags": model.flags,
                     "refiner_state_dict": model.refiner.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict(),
                     "step": selected_step, "selection_val21_epe_512px": best,
                     "history": history, "test_pairs_accessed": False},
                    output / f"best_{model.arm}.pth")
    save()
    total = 0
    for epoch in range(1, (steps + len(trainset) - 1) // len(trainset) + 1):
        loader = DataLoader(trainset, batch_size=1, shuffle=True,
                            generator=torch.Generator().manual_seed(SEED + epoch),
                            num_workers=0, pin_memory=device.type == "cuda")
        for batch in loader:
            if total >= steps:
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss, _ = staged.dense_training_loss(model, batch, device,
                                                 coarse_auxiliary=False)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss in {model.arm} at step {total}")
            loss.backward()
            if total == 0 and (model.refiner.head.weight.grad is None or
                               model.refiner.head.weight.grad.abs().sum() == 0):
                raise RuntimeError("New residual head lacks gradient")
            if any(parameter.grad is not None for parameter in model.base.parameters()):
                raise RuntimeError("Frozen global/local network received a gradient")
            torch.nn.utils.clip_grad_norm_(model.refiner.parameters(), 1.)
            optimizer.step()
            total += 1
            if total % EVAL_EVERY == 0 or total == steps:
                value = validation_epe(model, val21, device)
                history.append({"step": total, "val21_final_epe_512px": value,
                                "train_loss_last": float(loss.detach())})
                if value < best:
                    best, selected_step = value, total
                    save()
                print(f"{model.arm} step {total}/{steps}: val21={value:.4f}, "
                      f"best={best:.4f}", flush=True)
                (output / f"history_{model.arm}.json").write_text(
                    json.dumps(history, indent=2), encoding="utf-8")
    change = base_change(model, frozen)
    if change != 0:
        raise RuntimeError(f"Frozen parameter or BN changed: {change}")
    chosen = torch.load(output / f"best_{model.arm}.pth", map_location="cpu",
                        weights_only=False)
    model.refiner.load_state_dict(chosen["refiner_state_dict"])
    model.eval()
    return {"selected_step": selected_step, "steps_executed": total,
            "best_val21_epe_512px": best, "frozen_max_abs_change": change,
            "history": history}


def write_rows(path, rows):
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
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--code-check", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be a new empty directory")
    if args.steps < 1:
        parser.error("--steps must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geometry = geometry_check()
    trainset = RoadScenePairs(args.data_root, "train")
    valset = RoadScenePairs(args.data_root, "val")
    if len(trainset) != 176:
        raise ValueError("Expected original 176-pair train split")
    val21 = Subset(valset, selected_indices(valset))
    if args.code_check:
        trainset = Subset(trainset, range(1))
        val21 = Subset(valset, [selected_indices(valset)[0]])
    glunet = importlib.import_module("models.our_models.GLUNet")
    original = glunet.MutualMatching
    glunet.MutualMatching = lambda correlation: original(correlation.clamp_min(0))
    try:
        report = {"protocol": "frozen fresh_mutual_fix best_fine, train176, fixed val21 checkpoint selection",
                  "code_check_only": args.code_check,
                  "test_pairs_accessed": False, "steps_per_arm": args.steps,
                  "batch_size": 1, "evaluation_interval_steps": EVAL_EVERY,
                  "train_order_seed_per_epoch": "2026 + epoch",
                  "loss": "masked final 512px Charbonnier, no coarse loss",
                  "direction": "visible target -> infrared source",
                  "flow_scale": "512px units; sample grid128 at flow/4, grid256 at flow/2",
                  "pretrained_sha256": sha256_file(args.pretrained),
                  "coarse_sha256": sha256_file(args.coarse_checkpoint),
                  "fine_sha256": sha256_file(args.fine_checkpoint),
                  "matching_fix": FIX, "selection_excluded_ids": sorted(EXCLUDED),
                  "synthetic_geometry": geometry,
                  "arms": {}}
        rows_by_arm = {}
        for arm in ARMS:
            torch.manual_seed(SEED)
            base, _ = make_model(args.coarse_checkpoint, args.fine_checkpoint,
                                 args.pretrained, device)
            model = Candidate(base.base, arm).to(device).eval()
            initialization = exact_initialization(model, valset if not args.code_check
                                                   else val21, device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            training = train_arm(model, trainset, val21, device, args.output,
                                 1 if args.code_check else args.steps,
                                 args.code_check)
            peak_train = (torch.cuda.max_memory_allocated(device) / 2**20
                          if device.type == "cuda" else None)
            rows, _ = evaluate(model, valset if not args.code_check else val21,
                               device, args.output / arm / "edge_ghosts",
                               args.code_check)
            write_rows(args.output / f"per_image_{arm}.csv", rows)
            rows_by_arm[arm] = rows
            selected = [row for row in rows if row["name"] not in EXCLUDED]
            excluded = [row for row in rows if row["name"] in EXCLUDED]
            report["arms"][arm] = {
                "initialization": initialization, "training": training,
                "parameters_total": sum(p.numel() for p in model.parameters()),
                "parameters_new": sum(p.numel() for p in model.refiner.parameters()),
                "peak_train_allocated_mib": peak_train}
            if args.code_check:
                report["arms"][arm]["smoke_val1"] = group_summary(rows)
            else:
                report["arms"][arm].update({
                    "val21": group_summary(selected),
                    "val23": group_summary(rows),
                    "excluded2": group_summary(excluded)})
            (args.output / "report_val.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps({"arm": arm,
                              "smoke_val1" if args.code_check else "val21":
                                  report["arms"][arm]["smoke_val1" if args.code_check
                                                      else "val21"],
                              **({} if args.code_check else
                                 {"val23": report["arms"][arm]["val23"]})},
                             indent=2), flush=True)
            del model, base
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if not args.code_check:
            report["decision"] = stable_gain(
                rows_by_arm["local256"], rows_by_arm["dns_local_saca_fusion"])
            (args.output / "report_val.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8")
            if report["decision"]["run_ablations"]:
                for ablation, flags in ABLATIONS.items():
                    torch.manual_seed(SEED)
                    base, _ = make_model(args.coarse_checkpoint,
                                         args.fine_checkpoint,
                                         args.pretrained, device)
                    model = Candidate(base.base, ablation, flags).to(device).eval()
                    initialization = exact_initialization(model, valset, device)
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    training = train_arm(model, trainset, val21, device,
                                         args.output, args.steps, False)
                    peak = (torch.cuda.max_memory_allocated(device) / 2**20
                            if device.type == "cuda" else None)
                    rows, _ = evaluate(model, valset, device,
                                       args.output / ablation / "edge_ghosts",
                                       False)
                    write_rows(args.output / f"per_image_{ablation}.csv", rows)
                    report["arms"][ablation] = {
                        "flags": flags, "initialization": initialization,
                        "training": training,
                        "parameters_new": sum(p.numel() for p in model.refiner.parameters()),
                        "peak_train_allocated_mib": peak,
                        "val21": group_summary([row for row in rows
                                                if row["name"] not in EXCLUDED]),
                        "val23": group_summary(rows),
                        "excluded2": group_summary([row for row in rows
                                                    if row["name"] in EXCLUDED])}
                    (args.output / "report_val.json").write_text(
                        json.dumps(report, indent=2), encoding="utf-8")
                    del model, base
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
    finally:
        glunet.MutualMatching = original


if __name__ == "__main__":
    main()
