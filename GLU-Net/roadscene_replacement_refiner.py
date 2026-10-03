"""One-round 64/128-grid feature-alignment refiner replacing the 32-grid DCN.

The selected fresh_mutual_fix coarse checkpoint is fixed. Original local
decoders plus a lightweight projected-feature refiner train for the same six
epochs / 528 updates as the matched local ablation. DCN is bypassed. The
two-round option is gated by a successful one-round validation report.
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
from models.our_models.multiscale_local_iter import MultiScaleLocalIter
from roadscene_coarse import sha256_file
from roadscene_local_ablation import (
    checked_payload, evaluate_arm, frozen_change, frozen_snapshot,
    new_model, save_atomic, validation_epe)
from roadscene_multiscale_iter import synthetic_geometry_check
from roadscene_staged_mutual_fix import FIX
import roadscene_staged as staged


SEED = staged.SEED


class ReplacementRefiner(nn.Module):
    """Expose the unchanged GLU-Net evaluator interface with DCN bypassed."""

    def __init__(self, base, rounds):
        super().__init__()
        self.base = base
        self.base.local_dcn_steps = 0
        self.base.train_coarse_encoder = False
        self.refiner = MultiScaleLocalIter(rounds=rounds)
        self.capture_rounds = False
        self.last_rounds = None

    @property
    def pyramid(self):
        return self.base.pyramid

    @property
    def deconv4(self):
        return self.base.deconv4

    @property
    def deconv2(self):
        return self.base.deconv2

    def pre_process_data(self, *args, **kwargs):
        return self.base.pre_process_data(*args, **kwargs)

    def enrich_coarse_features(self, *args, **kwargs):
        return self.base.enrich_coarse_features(*args, **kwargs)

    def coarsest_resolution_flow(self, *args, **kwargs):
        return self.base.coarsest_resolution_flow(*args, **kwargs)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()  # every original BatchNorm statistic is fixed
        return self

    def forward(self, target, source, target256, source256):
        features64, features128 = [], []

        def remember64(_module, _inputs, output):
            if len(features64) < 2:
                features64.append(output.detach())

        def remember128(_module, _inputs, output):
            if len(features128) < 2:
                features128.append(output.detach())

        # Full-resolution VGG is called target first, then source. The 256px
        # pyramid is called later; capture only the first pair of each level.
        handles = [
            self.base.pyramid._modules["level_3"].register_forward_hook(
                remember64),
            self.base.pyramid._modules["level_2"].register_forward_hook(
                remember128)]
        try:
            flow256, flow512 = self.base(target, source, target256, source256)
        finally:
            for handle in handles:
                handle.remove()
        if (len(features64) != 2 or len(features128) != 2 or
                features64[0].shape[1:] != (256, 64, 64) or
                features128[0].shape[1:] != (128, 128, 128)):
            raise RuntimeError("Unexpected feature grid or modality order")
        updates = self.refiner(
            features64[0], features64[1],
            features128[0], features128[1],
            flow512[0], flow512[1])
        if self.capture_rounds:
            self.last_rounds = (updates["flow64_rounds"],
                                updates["flow128_rounds"])
        else:
            self.last_rounds = None
        return flow256, [updates["flow64_rounds"][-1],
                         updates["flow128_rounds"][-1]]


def prepare_model(coarse, device, rounds):
    torch.manual_seed(SEED)
    base = new_model(coarse, device, bypass=True)
    local_params = staged.groups(base)["local"]
    local_ids = {id(parameter) for parameter in local_params}
    for parameter in local_params:
        parameter.requires_grad_(True)
    for parameter in base.parameters():
        if parameter.requires_grad and id(parameter) not in local_ids:
            raise RuntimeError("Unexpected trainable base parameter")
    model = ReplacementRefiner(base, rounds).to(device)
    model.eval()
    return model


@torch.no_grad()
def zero_head_check(model, valset, device):
    maxima = {name: 0.0 for name in
              ("coarse16", "local32", "local64", "final128")}
    for batch in DataLoader(valset, batch_size=1, shuffle=False):
        source, target, s256, t256, *_ = model.pre_process_data(
            batch["source_image"].to(device),
            batch["target_image"].to(device), device=device)
        base256, base512 = model.base(target, source, t256, s256)
        new256, new512 = model(target, source, t256, s256)
        for label, before, after in zip(
                maxima, (*base256, *base512), (*new256, *new512)):
            maxima[label] = max(maxima[label],
                                float((before - after).abs().max()))
    if any(value != 0 for value in maxima.values()):
        raise RuntimeError(f"Zero-head output differs from baseline: {maxima}")
    return maxima


@torch.no_grad()
def round_metrics(model, valset, device):
    model.eval()
    model.capture_rounds = True
    sums, count, per_image = {}, 0, {}
    for batch in DataLoader(valset, batch_size=1, shuffle=False):
        source, target, s256, t256, *_ = model.pre_process_data(
            batch["source_image"].to(device),
            batch["target_image"].to(device), device=device)
        model(target, source, t256, s256)
        truth = batch["flow_map"].to(device).float()
        valid = batch["correspondence_mask"].to(device).bool()
        count += int(valid.sum())
        row = {}
        for size, predictions in zip((64, 128), model.last_rounds):
            for index, prediction in enumerate(predictions, 1):
                full = F.interpolate(prediction, size=(512, 512),
                                     mode="bilinear", align_corners=False)
                error = torch.linalg.vector_norm(full - truth, dim=1)
                label = f"grid{size}_round{index}_epe_512px"
                sums[label] = sums.get(label, 0.0) + float(error[valid].sum())
                row[label] = float(error[valid].mean())
        per_image[batch["name"][0]] = row
    model.capture_rounds = False
    model.last_rounds = None
    return {name: total / count for name, total in sums.items()}, per_image


def load_controls(path, coarse_sha, code_check):
    if code_check:
        return None, {}
    if not path or not path.is_file():
        raise ValueError("Complete local-only/DCN ablation report is required")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("code_check_only") or
            report.get("coarse_checkpoint_sha256") != coarse_sha or
            report.get("optimizer_steps_per_trained_arm") != 528 or
            set(("local_only", "dcn_only", "local_dcn",
                 "prior_5plus1_local_dcn")) - set(report.get("arms", {}))):
        raise ValueError("Local ablation report is incomplete or from another coarse weight")
    csv_path = path.parent / "per_image_val.csv"
    with csv_path.open(newline="", encoding="utf-8") as file:
        rows = {row["name"]: row for row in csv.DictReader(file)}
    if len(rows) != 23:
        raise ValueError("Local ablation lacks 23 validation image rows")
    return report, rows


def check_two_round_gate(path, coarse_sha, code_check):
    if code_check:
        return
    if not path or not path.is_file():
        raise ValueError("Two rounds require a successful one-round report")
    prior = json.loads(path.read_text(encoding="utf-8"))
    if (prior.get("rounds") != 1 or prior.get("code_check_only") or
            prior.get("coarse_checkpoint_sha256") != coarse_sha or
            not prior.get("decision", {}).get("allow_two_round")):
        raise ValueError("One-round validation did not permit two rounds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--coarse-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-fine-checkpoint", type=Path, required=True)
    parser.add_argument("--local-ablation-report", type=Path)
    parser.add_argument("--one-round-report", type=Path)
    parser.add_argument("--rounds", type=int, choices=(1, 2), default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-check", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be a new, empty directory")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coarse = checked_payload(args.coarse_checkpoint, args.pretrained, False)
    coarse_sha = sha256_file(args.coarse_checkpoint)
    reference = torch.load(args.reference_fine_checkpoint,
                           map_location="cpu", weights_only=False)
    if (reference.get("stage") != "fine" or
            reference.get("parent_sha256") != coarse_sha or
            reference.get("matching_fix") != FIX):
        raise ValueError("Prior fine checkpoint is not from this fresh coarse")
    controls, control_rows = load_controls(
        args.local_ablation_report, coarse_sha, args.code_check)
    if args.rounds == 2:
        check_two_round_gate(args.one_round_report, coarse_sha, args.code_check)
    trainset = RoadScenePairs(args.data_root, "train")
    valset = RoadScenePairs(args.data_root, "val")
    if len(trainset) != 176 or len(valset) != 23:
        raise ValueError("Expected 176 training and 23 validation pairs")
    if args.code_check:
        trainset, valset = Subset(trainset, range(2)), Subset(valset, range(1))
    reference_history = json.loads(
        (args.reference_fine_checkpoint.parent / "report_fine.json")
        .read_text(encoding="utf-8"))["history"]
    reference_orders = {row["epoch"]: row["image_order_sha256"]
                        for row in reference_history
                        if 1 <= row["epoch"] <= 6}
    if not args.code_check and len(reference_orders) != 6:
        raise ValueError("Prior fine report lacks six data-order hashes")

    glunet_module = importlib.import_module("models.our_models.GLUNet")
    original_matching = glunet_module.MutualMatching
    glunet_module.MutualMatching = (
        lambda corr: original_matching(corr.clamp_min(0)))
    try:
        geometry = synthetic_geometry_check()
        model = prepare_model(coarse, device, args.rounds)
        zero = zero_head_check(model, valset, device)
        frozen = frozen_snapshot(model.base, ("local",))
        local_params = staged.groups(model.base)["local"]
        optimizer = torch.optim.AdamW([
            {"params": local_params, "lr": 1e-5, "group_name": "local"},
            {"params": model.refiner.parameters(), "lr": 1e-4,
             "group_name": "replacement_refiner"}])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=.5, patience=4,
            threshold=.01, threshold_mode="abs", min_lr=1e-7)
        best = validation_epe(model, valset, device)
        history = [{"epoch": 0, "optimizer_steps": 0,
                    "val_final_epe_512px": best, "train_loss": None}]
        best_epoch, steps = 0, 0
        initial_state = {name: tensor.detach().cpu().clone()
                         for name, tensor in model.state_dict().items()}
        epochs = 1 if args.code_check else 6
        for epoch in range(1, epochs + 1):
            model.train()
            loader = DataLoader(
                trainset, batch_size=2, shuffle=True,
                generator=torch.Generator().manual_seed(SEED + epoch),
                num_workers=0, pin_memory=device.type == "cuda")
            order, losses = hashlib.sha256(), []
            started = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            for batch in loader:
                for name in batch["name"]:
                    order.update(name.encode("utf-8") + b"\n")
                optimizer.zero_grad(set_to_none=True)
                loss, _ = staged.dense_training_loss(
                    model, batch, device, coarse_auxiliary=False)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite replacement loss")
                loss.backward()
                if steps == 0 and (model.refiner.update_head.weight.grad is None or
                                   not torch.isfinite(
                                       model.refiner.update_head.weight.grad).all() or
                                   model.refiner.update_head.weight.grad.abs().sum() == 0):
                    raise RuntimeError("Replacement head receives no finite gradient")
                if any(p.grad is not None for p in staged.groups(model.base)["coarse"]):
                    raise RuntimeError("Frozen coarse stage received gradient")
                torch.nn.utils.clip_grad_norm_(
                    list(local_params) + list(model.refiner.parameters()), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
                steps += 1
            digest = order.hexdigest()
            if not args.code_check and digest != reference_orders[epoch]:
                raise RuntimeError("Image order differs from prior fine training")
            val = validation_epe(model, valset, device)
            scheduler.step(val)
            record = {"epoch": epoch, "optimizer_steps": steps,
                      "train_loss": float(np.mean(losses)),
                      "val_final_epe_512px": val,
                      "image_order_sha256": digest,
                      "learning_rates": {g["group_name"]: g["lr"]
                                         for g in optimizer.param_groups},
                      "train_seconds": time.perf_counter() - started,
                      "peak_train_allocated_mib": (
                          torch.cuda.max_memory_allocated() / 2**20
                          if device.type == "cuda" else None)}
            history.append(record)
            if val < best - .01:
                best, best_epoch = val, epoch
                save_atomic({
                    "epoch": epoch, "rounds": args.rounds,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                    "numpy_rng_state": np.random.get_state(),
                    "python_rng_state": random.getstate(),
                    "cuda_rng_state": torch.cuda.get_rng_state()
                    if device.type == "cuda" else None,
                    "coarse_sha256": coarse_sha,
                    "optimizer_steps": steps, "selection_metric": best,
                    "matching_fix": FIX, "dcn_bypassed": True,
                    "code_sha256": sha256_file(Path(__file__)),
                    "code_check_only": args.code_check,
                }, args.output / "best_replacement.pth")
            (args.output / "history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8")
            print(f"replacement rounds={args.rounds} epoch {epoch}/{epochs}: "
                  f"val final={val:.4f}, best={best:.4f}, steps={steps}",
                  flush=True)
        if steps != (1 if args.code_check else 528):
            raise RuntimeError(f"Unexpected update count: {steps}")
        change = frozen_change(model.base, frozen)
        if change != 0:
            raise RuntimeError(f"Frozen coarse weights or BN changed: {change}")
        if best_epoch:
            state = torch.load(args.output / "best_replacement.pth",
                               map_location="cpu", weights_only=False)
            model.load_state_dict(state["model_state_dict"], strict=True)
        else:
            model.load_state_dict(initial_state, strict=True)
            save_atomic({"epoch": 0, "rounds": args.rounds,
                         "model_state_dict": initial_state,
                         "optimizer_state_dict": optimizer.state_dict(),
                         "scheduler_state_dict": scheduler.state_dict(),
                         "torch_rng_state": torch.get_rng_state(),
                         "numpy_rng_state": np.random.get_state(),
                         "python_rng_state": random.getstate(),
                         "cuda_rng_state": torch.cuda.get_rng_state()
                         if device.type == "cuda" else None,
                         "coarse_sha256": coarse_sha,
                         "optimizer_steps": 0, "selection_metric": best,
                         "matching_fix": FIX, "dcn_bypassed": True,
                         "code_sha256": sha256_file(Path(__file__)),
                         "code_check_only": args.code_check},
                        args.output / "best_replacement.pth")
        model.eval()
        summary, image_rows = evaluate_arm(model, valset, device)
        rounds, per_round = round_metrics(model, valset, device)
        paired = []
        for name, record in image_rows.items():
            row = {"name": name, "valid_full": record["valid_full"],
                   "new_final_epe_512px": record["final_aepe_512px"],
                   **{key: value for key, value in record.items()
                      if key != "valid_full"},
                   **per_round[name]}
            if controls:
                prior = control_rows[name]
                if int(prior["valid_full"]) != record["valid_full"]:
                    raise RuntimeError("Valid mask differs from local controls")
                for arm in ("coarse_frozen", "dcn_only", "local_only",
                            "local_dcn", "prior_5plus1_local_dcn"):
                    row[f"{arm}_final_epe_512px"] = float(
                        prior[f"{arm}_final_aepe_512px"])
                    row[f"new_minus_{arm}_epe_512px"] = (
                        record["final_aepe_512px"] -
                        row[f"{arm}_final_epe_512px"])
            paired.append(row)
        with (args.output / "per_image_val.csv").open(
                "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(paired[0]))
            writer.writeheader()
            writer.writerows(paired)
        decision = {"allow_two_round": False,
                    "reason": "requires complete one-round validation"}
        if controls and args.rounds == 1:
            reference_arm = min(
                ("local_only", "local_dcn", "prior_5plus1_local_dcn"),
                key=lambda arm: controls["arms"][arm]["final_epe_512px"])
            reference_summary = controls["arms"][reference_arm]
            improved = sum(
                row[f"new_minus_{reference_arm}_epe_512px"] < 0
                for row in paired)
            gain = (reference_summary["final_epe_512px"] -
                    summary["final_epe_512px"])
            safe_large = (summary["large_motion_final_epe_512px"] <=
                          reference_summary["large_motion_final_epe_512px"])
            safe_outside = (
                summary["first_window_outside_final_epe_512px"] <=
                reference_summary["first_window_outside_final_epe_512px"])
            decision = {
                "reference_arm": reference_arm, "final_gain_512px": gain,
                "improved_pairs": improved,
                "large_motion_not_worse": safe_large,
                "window_outside_not_worse": safe_outside,
                "allow_two_round": (gain >= .1 and improved >= 15 and
                                    safe_large and safe_outside),
                "rule": "gain>=0.1px, >=15/23 improved, no large/outside regression"}
        report = {
            "split": "val", "test_pairs_accessed": False,
            "code_check_only": args.code_check, "rounds": args.rounds,
            "coarse_checkpoint_sha256": coarse_sha,
            "reference_fine_sha256":
                sha256_file(args.reference_fine_checkpoint),
            "local_ablation_report_sha256":
                sha256_file(args.local_ablation_report) if controls else None,
            "matching_fix": FIX, "dcn_bypassed": True,
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad),
            "added_refiner_parameters": sum(
                p.numel() for p in model.refiner.parameters()),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "optimizer_steps": steps, "selected_epoch": best_epoch,
            "selection": "validation final 512px EPE, min_delta=0.01",
            "loss": "same masked final-flow Charbonnier as matched local controls",
            "flow_units": "64/128 grid flows use 512-image-pixel XY units",
            "geometry": geometry, "zero_init_max_abs_difference": zero,
            "frozen_base_max_abs_change": change,
            "history": history, "summary": summary,
            "round_epe_512px": rounds,
            "decision": decision}
        (args.output / "report_val.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"code_check_only": args.code_check,
                          "rounds": args.rounds, "summary": summary,
                          "round_epe_512px": rounds,
                          "decision": decision}, indent=2), flush=True)
    finally:
        glunet_module.MutualMatching = original_matching


if __name__ == "__main__":
    main()
