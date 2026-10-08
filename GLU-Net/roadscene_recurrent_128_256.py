"""Frozen-best-fine 128/256 local recurrent RoadScene experiment.

Within each modality: cross-scale gathering -> existing 2D DNS residual ->
bounded self attention. Then target VI samples source IR using current flow,
followed by local cross attention, correlation/difference and a shared update.
Only the new branch trains. Validation IDs 000002/000014 are reported but not
used for exploratory checkpoint selection. The locked test split is not read.
"""

import argparse
import csv
import hashlib
import importlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from datasets.roadscene import RoadScenePairs
from models.our_models.local_recurrent_128_256 import (
    LocalRecurrent128256, sample_window)
from roadscene_coarse import sha256_file
from roadscene_local256 import (base_change, evaluate, inputs, make_model,
                                snapshot_base)
from roadscene_local_dns_attention import (EXCLUDED, group_summary,
                                           selected_indices, write_rows)
from roadscene_local_ablation import save_atomic
from roadscene_refinement_audit import edge_quartile, pooled_truth, search_stage
from roadscene_staged_mutual_fix import FIX
import roadscene_staged as staged


SEED = staged.SEED
EPOCHS = 20
BATCH_SIZE = 1
LR = 1e-4
MIN_LR = 1e-5
RADIUS = {128: 2, 256: 1}
ADOPTION = {"minimum_val21_epe_gain_512px": .1,
            "minimum_val21_improved_pairs": 15}


class FrozenFineWithRecurrent(nn.Module):
    def __init__(self, base, rounds, use_dns=True):
        super().__init__()
        self.base = base.eval()
        self.base.train_coarse_encoder = False
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        self.refiner = LocalRecurrent128256(rounds=rounds, use_dns=use_dns)
        self.rounds = rounds
        self.use_dns = use_dns

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()  # fixes every original BatchNorm buffer
        return self

    def pre_process_data(self, *args, **kwargs):
        return self.base.pre_process_data(*args, **kwargs)

    def run(self, target, source, target256, source256):
        first, second = [], []

        def capture(store):
            def remember(_module, _inputs, output):
                if len(store) < 2:
                    store.append(output.detach())
            return remember

        handles = [
            self.base.pyramid._modules["level_1"].register_forward_hook(
                capture(first)),
            self.base.pyramid._modules["level_2"].register_forward_hook(
                capture(second))]
        try:
            with torch.no_grad():
                coarse, local = self.base(target, source, target256, source256)
        finally:
            for handle in handles:
                handle.remove()
        if (len(first) != 2 or len(second) != 2 or
                any(value.shape[1:] != (64, 256, 256) for value in first) or
                any(value.shape[1:] != (128, 128, 128) for value in second)):
            raise RuntimeError("VGG full-image order/shape must be VI then IR")
        refined = self.refiner(second[0], second[1], first[0], first[1],
                               local[-1])
        return {"coarse": coarse, "local": local,
                "initial512": refined["flows512"][0],
                "rounds512": refined["flows512"][1:],
                "final512": refined["flows512"][-1],
                "round_diagnostics": refined["round_diagnostics"]}

    def forward(self, target, source, target256, source256):
        result = self.run(target, source, target256, source256)
        return result["coarse"], [*result["local"], result["final512"]]


def geometry_check():
    result = {}
    for size, radius in RADIUS.items():
        source = torch.zeros(1, 1, size, size)
        shift = size // 128
        source[0, 0, 40, 40 + shift] = 1
        flow = torch.zeros(1, 2, size, size)
        flow[:, 0] = 4
        correct, valid = sample_window(source, flow, radius)
        opposite, _ = sample_window(source, -flow, radius)
        center = (2 * radius + 1) ** 2 // 2
        peak = float(correct[0, 0, center, 40, 40])
        wrong = float(opposite[0, 0, center, 40, 40])
        border = bool(valid[0, center, 40, size - 1])
        if abs(peak - 1) > 1e-5 or wrong != 0 or border:
            raise RuntimeError(f"VI->IR warp direction or scale failed at {size}")
        result[str(size)] = {"flow_512px": [4, 0],
                             "source_offset_grid": [shift, 0],
                             "correct_center": peak,
                             "opposite_center": wrong,
                             "border_center_valid": border,
                             "window_radius_grid": radius,
                             "window_radius_512px_per_axis":
                                 radius * 512 / size}
    return result


@torch.no_grad()
def initial_coverage(base, valset, device):
    """GT query must be in the square source window about the initial flow."""
    rows = []
    for batch in DataLoader(valset, batch_size=1, shuffle=False):
        source, target, source256, target256 = inputs(base, batch, device)
        _, local = base.base(target, source, target256, source256)
        initial128 = local[-1]
        initial256 = F.interpolate(initial128, (256, 256),
                                   mode="bilinear", align_corners=False)
        gt = batch["flow_map"].to(device).float()
        valid = batch["correspondence_mask"].to(device).bool()
        row = {"name": batch["name"][0]}
        for size, prediction in ((128, initial128), (256, initial256)):
            truth, selected = pooled_truth(gt, valid, size)
            radius_image = RADIUS[size] * 512 / size
            error = truth - prediction
            inside = ((error.abs().amax(dim=1) <= radius_image) & selected)
            count = int(selected.sum())
            inside_count = int(inside.sum())
            outside = selected & ~inside
            row[f"grid{size}_valid_queries"] = count
            row[f"grid{size}_inside_queries"] = inside_count
            row[f"grid{size}_outside_queries"] = count - inside_count
            row[f"grid{size}_inside_fraction"] = inside_count / count if count else None
            row[f"grid{size}_outside_epe_512px"] = (
                float(torch.linalg.vector_norm(error, dim=1)[outside].mean())
                if outside.any() else None)
        rows.append(row)
    summary = {}
    for label, selected in (("all23", rows),
                            ("diagnostic21", [r for r in rows if r["name"]
                                              not in EXCLUDED]),
                            ("excluded2", [r for r in rows if r["name"]
                                           in EXCLUDED])):
        summary[label] = {}
        for size in RADIUS:
            count = sum(row[f"grid{size}_valid_queries"] for row in selected)
            inside = sum(row[f"grid{size}_inside_queries"] for row in selected)
            summary[label][f"grid{size}"] = {
                "valid_queries": count, "inside_queries": inside,
                "outside_queries": count - inside,
                "inside_fraction": inside / count if count else None,
                "radius_512px_per_axis": RADIUS[size] * 512 / size}
    return rows, summary


@torch.no_grad()
def zero_check(model, dataset, device):
    maximum = 0.
    for batch in DataLoader(dataset, batch_size=1, shuffle=False):
        source, target, source256, target256 = inputs(model, batch, device)
        result = model.run(target, source, target256, source256)
        baseline = F.interpolate(result["local"][-1], (512, 512),
                                 mode="bilinear", align_corners=False)
        for output in (result["initial512"], *result["rounds512"]):
            maximum = max(maximum, float((output - baseline).abs().max()))
        for info in result["round_diagnostics"]:
            invalid = ~info["update_valid"]
            if torch.count_nonzero(info["delta512"][:, :, invalid[0]]) != 0:
                raise RuntimeError("Invalid source window received a residual")
    if maximum != 0:
        raise RuntimeError(f"Zero head did not reproduce frozen baseline: {maximum}")
    return maximum


def state_sha256(module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def validation_epe(model, dataset, device):
    model.eval()
    total, count = 0., 0
    for batch in DataLoader(dataset, batch_size=1, shuffle=False):
        source, target, source256, target256 = inputs(model, batch, device)
        prediction = model.run(target, source, target256, source256)["final512"]
        truth = batch["flow_map"].to(device).float()
        valid = batch["correspondence_mask"].to(device).bool()
        errors = torch.linalg.vector_norm(prediction - truth, dim=1)
        total += float(errors[valid].sum())
        count += int(valid.sum())
    return total / count


def iteration_loss(model, batch, device):
    source, target, source256, target256 = inputs(model, batch, device)
    result = model.run(target, source, target256, source256)
    truth = batch["flow_map"].to(device).float()
    valid = batch["correspondence_mask"].to(device).bool()
    if not valid.any():
        raise ValueError("Training batch has no valid GT pixels")
    weights = ([1.] if model.rounds == 1 else [.9, 1.])
    losses = []
    for prediction in result["rounds512"]:
        error = torch.sqrt((prediction - truth).square().sum(dim=1) + .01)
        losses.append(error[valid].mean())
    return sum(weight * loss for weight, loss in zip(weights, losses)) / sum(weights)


def checkpoint(model, optimizer, scheduler, epoch, steps, best,
               history, fine_sha, code_sha):
    return {"rounds": model.rounds, "use_dns": model.use_dns,
            "refiner_state_dict": model.refiner.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch, "optimizer_steps": steps,
            "selection_val21_final_epe_512px": best,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state()
                if torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "history": history, "fine_sha256": fine_sha,
            "code_sha256": code_sha, "test_pairs_accessed": False}


def train_arm(model, trainset, val21, device, output, epochs,
              fine_sha, code_sha, code_check):
    label = f"round{model.rounds}" + ("" if model.use_dns else "_no_dns")
    fixed = snapshot_base(model)
    optimizer = torch.optim.AdamW(model.refiner.parameters(), lr=LR)
    steps_target = 1 if code_check else epochs * len(trainset)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps_target, eta_min=MIN_LR)
    best = validation_epe(model, val21, device)
    best_epoch = 0
    history = [{"epoch": 0, "optimizer_steps": 0,
                "val21_final_epe_512px": best}]
    save_atomic(checkpoint(model, optimizer, scheduler, 0, 0, best,
                           history, fine_sha, code_sha),
                output / f"best_{label}.pth")
    steps = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, (1 if code_check else epochs) + 1):
        model.train()
        loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                            generator=torch.Generator().manual_seed(SEED + epoch),
                            num_workers=0, pin_memory=device.type == "cuda")
        order = hashlib.sha256()
        loss_sum = 0.
        processed = 0
        start = time.perf_counter()
        for batch in loader:
            if code_check and steps == 1:
                break
            order.update((batch["name"][0] + "\n").encode("utf-8"))
            optimizer.zero_grad(set_to_none=True)
            loss = iteration_loss(model, batch, device)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite {label} loss at step {steps}")
            loss.backward()
            grad = model.refiner.update_head.weight.grad
            if steps == 0 and (grad is None or not torch.isfinite(grad).all() or
                               grad.abs().sum() == 0):
                raise RuntimeError("New residual head has no finite gradient")
            if any(parameter.grad is not None for parameter in model.base.parameters()):
                raise RuntimeError("Frozen original network received a gradient")
            torch.nn.utils.clip_grad_norm_(model.refiner.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            steps += 1
            processed += 1
            loss_sum += float(loss.detach())
        value = validation_epe(model, val21, device)
        improved = value < best
        if improved:
            best, best_epoch = value, epoch
        record = {"epoch": epoch, "optimizer_steps": steps,
                  "image_order_sha256": order.hexdigest(),
                  "train_loss": loss_sum / processed,
                  "val21_final_epe_512px": value,
                  "best_val21_epe_512px": best,
                  "learning_rate": optimizer.param_groups[0]["lr"],
                  "train_seconds": time.perf_counter() - start,
                  "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20
                    if device.type == "cuda" else None}
        history.append(record)
        (output / f"history_{label}.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8")
        if improved:
            save_atomic(checkpoint(model, optimizer, scheduler, epoch, steps,
                                   best, history, fine_sha, code_sha),
                        output / f"best_{label}.pth")
        print(f"{label} epoch {epoch}/{epochs}: val21 final={value:.4f}, "
              f"best={best:.4f}, steps={steps}", flush=True)
    if steps != steps_target:
        raise RuntimeError(f"Unexpected training steps {steps}, expected {steps_target}")
    change = base_change(model, fixed)
    if change != 0:
        raise RuntimeError(f"Frozen parameter or BN changed: {change}")
    selected = torch.load(output / f"best_{label}.pth", map_location="cpu",
                          weights_only=False)
    model.refiner.load_state_dict(selected["refiner_state_dict"], strict=True)
    model.eval()
    return {"label": label, "best_epoch": best_epoch,
            "selected_steps": selected["optimizer_steps"],
            "executed_steps": steps, "val21_best_epe_512px": best,
            "frozen_max_abs_change": change,
            "peak_allocated_mib": max(
                row.get("peak_allocated_mib") or 0 for row in history),
            "history": history}


@torch.no_grad()
def round_outputs(model, valset, device, directory):
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for batch in DataLoader(valset, batch_size=1, shuffle=False):
        name = batch["name"][0]
        source, target, source256, target256 = inputs(model, batch, device)
        result = model.run(target, source, target256, source256)
        truth = batch["flow_map"].to(device).float()
        valid = batch["correspondence_mask"].to(device).bool()
        valid_image = valid[0]
        edge = edge_quartile(batch["target_image"].to(device), valid_image)
        large = valid_image & (torch.linalg.vector_norm(truth[0], dim=0) >= 64)
        pre32 = model.base.deconv4(result["coarse"][0]) * 2
        _, inside32, selected32 = search_stage(
            pre32, result["coarse"][1] * 2, truth, valid, 32)
        outside = F.interpolate((selected32 & ~inside32)[:, None].float(),
                                (512, 512), mode="nearest")[0, 0].bool()
        masks = {"edge": edge, "large_motion_64plus": large,
                 "first32_outside": valid_image & outside}
        initial128 = result["local"][-1]
        initial256 = F.interpolate(initial128, (256, 256),
                                   mode="bilinear", align_corners=False)
        for size, prediction in ((128, initial128), (256, initial256)):
            pooled, selected = pooled_truth(truth, valid, size)
            outside = selected & ((pooled - prediction).abs().amax(dim=1) >
                                  RADIUS[size] * 512 / size)
            assigned = F.interpolate(outside[:, None].float(),
                                     (512, 512), mode="nearest")[0, 0].bool()
            masks[f"initial{size}_window_outside"] = valid_image & assigned
        outputs = [result["initial512"], *result["rounds512"]]
        row = {"name": name, "valid_pixels": int(valid.sum())}
        arrays = {}
        for index, prediction in enumerate(outputs):
            label = "initial" if index == 0 else f"round{index}"
            errors = torch.linalg.vector_norm(prediction - truth, dim=1)
            row[f"{label}_epe_512px"] = float(errors[valid].mean())
            for threshold in (1, 3, 5):
                row[f"{label}_fraction_below_{threshold}px"] = float(
                    ((errors[0] < threshold) & valid_image).sum() /
                    valid_image.sum())
            for group, mask in masks.items():
                pixels = int(mask.sum())
                row[f"{group}_pixels"] = pixels
                row[f"{label}_{group}_epe_512px"] = (
                    float(errors[0][mask].mean()) if pixels else None)
            arrays[label] = prediction[0].detach().cpu().numpy().astype(np.float32)
        for index, info in enumerate(result["round_diagnostics"], start=1):
            row[f"round{index}_valid_update_fraction"] = float(
                info["update_valid"].float().mean())
            row[f"round{index}_mean_delta_512px"] = float(
                torch.linalg.vector_norm(info["delta512"], dim=1).mean())
        np.savez_compressed(directory / f"{name}.npz", **arrays)
        rows.append(row)
    return rows


def paired_rows(one, two):
    first = {row["name"]: row for row in one}
    second = {row["name"]: row for row in two}
    if len(first) != 23 or set(first) != set(second):
        raise RuntimeError("One/two-round validation rows do not pair")
    rows = []
    for name in sorted(first):
        a, b = first[name], second[name]
        if a["valid_pixels"] != b["valid_pixels"]:
            raise RuntimeError("Paired GT masks differ")
        row = {"name": name, "in_diagnostic21": name not in EXCLUDED,
               "valid_pixels": a["valid_pixels"],
               "baseline_epe_512px": a["baseline_epe_512px"],
               "one_round_epe_512px": a["new_epe_512px"],
               "two_round_epe_512px": b["new_epe_512px"],
               "two_minus_one_epe_512px": b["new_epe_512px"] -
                                              a["new_epe_512px"]}
        for threshold in (1, 3, 5):
            key = f"new_fraction_epe_below_{threshold}px"
            row[f"one_fraction_below_{threshold}px"] = a[key]
            row[f"two_fraction_below_{threshold}px"] = b[key]
        for group in ("edge_top_quartile", "large_motion_64plus",
                      "first32_outside"):
            row[f"{group}_pixels"] = a[f"{group}_pixels"]
            row[f"one_{group}_epe_512px"] = a[
                f"{group}_new_epe_512px"]
            row[f"two_{group}_epe_512px"] = b[
                f"{group}_new_epe_512px"]
        rows.append(row)
    return rows


def summarize_round_rows(rows):
    count = sum(row["valid_pixels"] for row in rows)
    labels = ["initial", "round1"]
    if "round2_epe_512px" in rows[0]:
        labels.append("round2")
    summary = {"pairs": len(rows), "valid_pixels": count, "outputs": {}}
    for label in labels:
        item = {"epe_512px": sum(row[f"{label}_epe_512px"] *
                                  row["valid_pixels"] for row in rows) / count,
                "pixel_fraction_below": {
                    str(t): sum(row[f"{label}_fraction_below_{t}px"] *
                                row["valid_pixels"] for row in rows) / count
                    for t in (1, 3, 5)}}
        for group in ("edge", "large_motion_64plus", "first32_outside",
                      "initial128_window_outside",
                      "initial256_window_outside"):
            pixels = sum(row[f"{group}_pixels"] for row in rows)
            item[f"{group}_epe_512px"] = (
                sum((row[f"{label}_{group}_epe_512px"] or 0) *
                    row[f"{group}_pixels"] for row in rows) / pixels
                if pixels else None)
        summary["outputs"][label] = item
    if "round2" in labels:
        summary["round2_improved_pairs_over_round1"] = sum(
            row["round2_epe_512px"] < row["round1_epe_512px"] for row in rows)
    return summary


def paired_round_rows(one, two):
    first = {row["name"]: row for row in one}
    second = {row["name"]: row for row in two}
    if len(first) != 23 or set(first) != set(second):
        raise RuntimeError("One/two-round output rows do not pair")
    rows = []
    for name in sorted(first):
        a, b = first[name], second[name]
        row = {"name": name, "in_diagnostic21": name not in EXCLUDED,
               "valid_pixels": a["valid_pixels"],
               "initial_epe_512px": a["initial_epe_512px"],
               "one_final_epe_512px": a["round1_epe_512px"],
               "two_final_epe_512px": b["round2_epe_512px"],
               "two_internal_first_epe_512px": b["round1_epe_512px"]}
        for group in ("edge", "large_motion_64plus", "first32_outside",
                      "initial128_window_outside",
                      "initial256_window_outside"):
            row[f"{group}_pixels"] = a[f"{group}_pixels"]
            row[f"one_{group}_epe_512px"] = a[
                f"round1_{group}_epe_512px"]
            row[f"two_{group}_epe_512px"] = b[
                f"round2_{group}_epe_512px"]
        rows.append(row)
    return rows


def paired_gain(one_rows, two_rows):
    a = {row["name"]: row for row in one_rows if row["name"] not in EXCLUDED}
    b = {row["name"]: row for row in two_rows if row["name"] not in EXCLUDED}
    if len(a) != 21 or set(a) != set(b):
        raise RuntimeError("One/two round results are not on the same 21 pairs")
    first = group_summary(list(a.values()))
    second = group_summary(list(b.values()))
    gain = first["new_final_epe_512px"] - second["new_final_epe_512px"]
    improved = sum(b[name]["new_epe_512px"] < a[name]["new_epe_512px"]
                   for name in a)
    return {"val21_gain_two_vs_one_512px": gain,
            "val21_improved_pairs_two_vs_one": improved,
            "adoption_rule": ADOPTION,
            "adopt_two_rounds": gain >= .1 and improved >= 15}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--coarse-checkpoint", type=Path, required=True)
    parser.add_argument("--fine-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--code-check", action="store_true")
    parser.add_argument("--coverage-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be a new, empty directory")
    if args.epochs < 1 or args.code_check and args.coverage_only:
        parser.error("Invalid epoch or mode combination")
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
    if len(trainset) != 176 or len(valset) != 23:
        raise ValueError("Expected original train176/val23 splits")
    val21 = Subset(valset, selected_indices(valset))
    if args.code_check:
        trainset = Subset(trainset, range(1))
        val21 = Subset(valset, [selected_indices(valset)[0]])
    glunet = importlib.import_module("models.our_models.GLUNet")
    original = glunet.MutualMatching
    glunet.MutualMatching = lambda corr: original(corr.clamp_min(0))
    try:
        parent, _ = make_model(args.coarse_checkpoint, args.fine_checkpoint,
                               args.pretrained, device)
        coverage_rows, coverage_summary = initial_coverage(
            parent, val21 if args.code_check else valset, device)
        coverage = {"geometry": geometry, "summary": coverage_summary,
                    "per_image": coverage_rows,
                    "special_cases": {r["name"]: r for r in coverage_rows
                                      if r["name"] in EXCLUDED}}
        (args.output / "coverage_initial.json").write_text(
            json.dumps(coverage, indent=2), encoding="utf-8")
        if args.coverage_only:
            print(json.dumps(coverage_summary, indent=2), flush=True)
            return
        del parent
        report = {"code_check_only": args.code_check,
                  "test_pairs_accessed": False,
                  "experiment": "frozen_fine_cross_scale_dns_local_attention_recurrent",
                  "base_has_preexisting_dns_and_32grid_dcn": True,
                  "new_branch_has_dcn": False,
                  "new_branch_global_attention": False,
                  "direction": "VI target -> IR source",
                  "flow_unit": "512 image pixels; 128 grid flow/4; 256 grid flow/2",
                  "local_windows": {"128": "5x5", "256": "3x3"},
                  "budget": {"epochs_per_arm": 1 if args.code_check else args.epochs,
                             "steps_per_arm": 1 if args.code_check else
                                 args.epochs * 176,
                             "batch_size": BATCH_SIZE,
                             "optimizer": "AdamW, 1e-4",
                             "schedule": "per-step cosine to 1e-5",
                             "loss": "valid-mask Charbonnier; two rounds normalized weights 0.9,1.0",
                             "selection": "strict minimum weighted final 512px EPE on fixed val21"},
                  "pretrained_sha256": sha256_file(args.pretrained),
                  "coarse_sha256": sha256_file(args.coarse_checkpoint),
                  "fine_sha256": sha256_file(args.fine_checkpoint),
                  "code_sha256": sha256_file(Path(__file__)),
                  "matching_fix": FIX,
                  "initial_coverage": coverage_summary,
                  "arms": {}}
        rows_by_arm = {}
        round_rows_by_arm = {}
        initial_refiner_sha = None
        initial_round2_state = None
        for rounds, use_dns in ((1, True), (2, True)):
            torch.manual_seed(SEED)
            base, _ = make_model(args.coarse_checkpoint, args.fine_checkpoint,
                                 args.pretrained, device)
            model = FrozenFineWithRecurrent(base.base, rounds,
                                            use_dns=use_dns).to(device).eval()
            label = f"round{rounds}"
            initialization_sha = state_sha256(model.refiner)
            if initial_refiner_sha is None:
                initial_refiner_sha = initialization_sha
            elif initialization_sha != initial_refiner_sha:
                raise RuntimeError("One/two round refiner starts differ")
            if rounds == 2:
                initial_round2_state = {
                    key: tensor.detach().cpu().clone()
                    for key, tensor in model.refiner.state_dict().items()}
            zero = zero_check(model, val21 if args.code_check else valset,
                              device)
            training = train_arm(model, trainset, val21, device,
                                 args.output, args.epochs,
                                 report["fine_sha256"], report["code_sha256"],
                                 args.code_check)
            evaluated = val21 if args.code_check else valset
            rows, _ = evaluate(model, evaluated, device,
                               args.output / label / "edge_ghosts",
                               args.code_check)
            round_rows = round_outputs(
                model, evaluated, device, args.output / label / "flows_val")
            write_rows(args.output / f"per_image_{label}.csv", rows)
            write_rows(args.output / f"per_image_rounds_{label}.csv", round_rows)
            rows_by_arm[label] = rows
            round_rows_by_arm[label] = round_rows
            entry = {"initial_zero_max_abs_diff_512px": zero,
                     "initial_refiner_sha256": initialization_sha,
                     "training": training,
                     "new_parameters": sum(p.numel()
                                           for p in model.refiner.parameters())}
            if args.code_check:
                entry["smoke_val1"] = group_summary(rows)
            else:
                entry["val21"] = group_summary(
                    [r for r in rows if r["name"] not in EXCLUDED])
                entry["val23"] = group_summary(rows)
                entry["excluded2"] = group_summary(
                    [r for r in rows if r["name"] in EXCLUDED])
                entry["round_outputs"] = {
                    "val21": summarize_round_rows(
                        [r for r in round_rows if r["name"] not in EXCLUDED]),
                    "val23": summarize_round_rows(round_rows),
                    "excluded2": summarize_round_rows(
                        [r for r in round_rows if r["name"] in EXCLUDED])}
            report["arms"][label] = entry
            (args.output / "report_val.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps({"arm": label, "selected_epoch":
                              training["best_epoch"],
                              "selection_epe_512px": training["val21_best_epe_512px"]},
                             indent=2), flush=True)
            del model, base
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if not args.code_check:
            write_rows(args.output / "per_image_paired_one_vs_two.csv",
                       paired_rows(rows_by_arm["round1"],
                                   rows_by_arm["round2"]))
            write_rows(args.output / "per_image_paired_round_outputs.csv",
                       paired_round_rows(round_rows_by_arm["round1"],
                                         round_rows_by_arm["round2"]))
            report["decision"] = paired_gain(rows_by_arm["round1"],
                                              rows_by_arm["round2"])
            a = {r["name"]: r for r in round_rows_by_arm["round2"]}
            selected = [r for r in a.values() if r["name"] not in EXCLUDED]
            report["within_two_round_model"] = {
                "val21": summarize_round_rows(selected),
                "val23": summarize_round_rows(list(a.values())),
                "excluded2": summarize_round_rows(
                    [r for r in a.values() if r["name"] in EXCLUDED])}
            (args.output / "report_val.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8")
            if report["decision"]["adopt_two_rounds"]:
                torch.manual_seed(SEED)
                base, _ = make_model(args.coarse_checkpoint,
                                     args.fine_checkpoint,
                                     args.pretrained, device)
                model = FrozenFineWithRecurrent(base.base, 2,
                                                use_dns=False).to(device).eval()
                # Remove only DNS; every shared feature/fusion/update weight
                # starts from the same initialization as the two-round arm.
                common_initial = {key: value for key, value in
                                  initial_round2_state.items()
                                  if not key.startswith("dns")}
                model.refiner.load_state_dict(common_initial, strict=True)
                zero = zero_check(model, valset, device)
                training = train_arm(model, trainset, val21, device,
                                     args.output, args.epochs,
                                     report["fine_sha256"],
                                     report["code_sha256"], False)
                rows, _ = evaluate(model, valset, device,
                                   args.output / "round2_no_dns" / "edge_ghosts",
                                   False)
                per_round = round_outputs(
                    model, valset, device,
                    args.output / "round2_no_dns" / "flows_val")
                write_rows(args.output / "per_image_round2_no_dns.csv", rows)
                write_rows(args.output / "per_image_rounds_round2_no_dns.csv",
                           per_round)
                report["arms"]["round2_no_dns"] = {
                    "initial_zero_max_abs_diff_512px": zero,
                    "training": training,
                    "new_parameters": sum(p.numel()
                                          for p in model.refiner.parameters()),
                    "val21": group_summary([r for r in rows
                                            if r["name"] not in EXCLUDED]),
                    "val23": group_summary(rows),
                    "excluded2": group_summary([r for r in rows
                                                if r["name"] in EXCLUDED])}
                dns_val21 = report["arms"]["round2"]["val21"]
                no_dns_val21 = report["arms"]["round2_no_dns"]["val21"]
                dns_rows = {r["name"]: r for r in rows_by_arm["round2"]
                            if r["name"] not in EXCLUDED}
                no_dns_rows = {r["name"]: r for r in rows
                               if r["name"] not in EXCLUDED}
                report["dns_contribution_val21"] = {
                    "epe_gain_with_dns_512px":
                        no_dns_val21["new_final_epe_512px"] -
                        dns_val21["new_final_epe_512px"],
                    "pairs_improved_with_dns": sum(
                        dns_rows[name]["new_epe_512px"] <
                        no_dns_rows[name]["new_epe_512px"]
                        for name in dns_rows)}
                (args.output / "report_val.json").write_text(
                    json.dumps(report, indent=2), encoding="utf-8")
    finally:
        glunet.MutualMatching = original


if __name__ == "__main__":
    main()
