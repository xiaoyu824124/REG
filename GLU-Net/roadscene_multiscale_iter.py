"""Frozen GLU-Net + isolated two-scale recurrent local-flow experiment.

Uses the original selected epoch-37 best_coarse checkpoint. Only the new
MultiScaleLocalIter branch trains; no original model file or checkpoint is
changed. Train/validation splits only, five epochs = 440 optimizer updates,
matching the original DCN-only warmup budget and validation selection rule.
"""

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.roadscene import RoadScenePairs
from models.our_models.GLUNet import GLUNet_model
from models.our_models.multiscale_local_iter import (
    MultiScaleLocalIter, warp_source_to_target)
from roadscene_coarse import sha256_file
from roadscene_refinement_audit import pooled_truth


SEED = 2026
LABELS = ("64_round1", "64_round2", "128_round1", "128_round2")


class FrozenGLUNetWithIter(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base.eval()
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.branch = MultiScaleLocalIter()

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, raw_source, raw_target):
        device = raw_source.device
        source, target, source256, target256, *_ = self.base.pre_process_data(
            raw_source, raw_target, device=device)
        features128, features64 = [], []

        def remember128(_module, _inputs, output):
            if len(features128) < 2:
                features128.append(output.detach())

        def remember64(_module, _inputs, output):
            if len(features64) < 2:
                features64.append(output.detach())

        hook128 = self.base.pyramid._modules["level_2"].register_forward_hook(
            remember128)
        hook64 = self.base.pyramid._modules["level_3"].register_forward_hook(
            remember64)
        try:
            with torch.no_grad():
                flows256, flows512 = self.base(target, source, target256,
                                                source256)
        finally:
            hook128.remove()
            hook64.remove()
        if (len(features128) != 2 or len(features64) != 2 or
                features128[0].shape[1:] != (128, 128, 128) or
                features64[0].shape[1:] != (256, 64, 64)):
            raise RuntimeError(f"Unexpected VGG tensor shape or order: "
                               f"level_2={[tuple(x.shape) for x in features128]}, "
                               f"level_3={[tuple(x.shape) for x in features64]}")
        updates = self.branch(features64[0], features64[1],
                              features128[0], features128[1],
                              flows512[0], flows512[1])
        return {"base_flows256": flows256, "base_flows512": flows512,
                **updates}


def synthetic_geometry_check():
    for size in (64, 128):
        impulse = torch.zeros(1, 1, size, size)
        impulse[0, 0, 4, 5] = 1
        zero = torch.zeros(1, 2, size, size)
        same, valid = warp_source_to_target(impulse, zero)
        flow = zero.clone()
        flow[:, 0] = 512 / size
        shifted, _ = warp_source_to_target(impulse, flow)
        if (not torch.allclose(same, impulse, atol=1e-6) or
                not torch.allclose(shifted[0, 0, 4, 4],
                                   torch.tensor(1.), atol=1e-6) or
                not bool(valid[0, 0, 4, 4])):
            raise RuntimeError(f"Target-to-source warp failed at grid {size}")
    return {"flow_direction": "visible target -> infrared source",
            "grid64_one_cell_in_512px": 8,
            "grid128_one_cell_in_512px": 4,
            "align_corners": True,
            "impulse_direction_check": "passed"}


def make_base(path, pretrained, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("stage") != "coarse" or payload.get("epoch") != 37 or
            payload.get("arm") != "dns_attention" or
            payload.get("base_sha256") != sha256_file(pretrained)):
        raise ValueError("Expected original epoch-37 DNS+SA/CA coarse best")
    model = GLUNet_model(evaluation=False, pyramid_type="VGG",
                        cyclic_consistency=True, backbone_pretrained=False,
                        coarse_attention=True, coarse_dns=True,
                        local_dcn_steps=1)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval(), payload


def epe_loss(prediction, truth, valid):
    if not valid.any():
        raise ValueError("No valid ground truth for supervised flow")
    norm = torch.sqrt((prediction - truth).square().sum(dim=1) + .01)
    return norm[valid].mean()


def training_loss(result, truth, valid):
    scale_terms = []
    for size, predictions in ((64, result["flow64_rounds"]),
                              (128, result["flow128_rounds"])):
        gt, selected = pooled_truth(truth, valid, size)
        for prediction in predictions:
            scale_terms.append(epe_loss(prediction, gt, selected))
    final = epe_loss(result["final_512px"], truth, valid)
    # Keep the same main 512px loss as DCN-only. The four 64/128 round losses
    # provide the requested local supervision at small auxiliary weight.
    return final + .1 * torch.stack(scale_terms).mean(), {
        "final_loss": float(final.detach()),
        **{f"{name}_loss": float(loss.detach()) for name, loss in
           zip(LABELS, scale_terms)}}


def timing_on_device(forward, source, target, device):
    with torch.no_grad():
        for _ in range(3):
            forward(source, target)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(5):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            forward(source, target)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def group_stats(error, mask):
    count = int(mask.sum())
    return {"pixels": count,
            "sum": float(error[mask].sum()) if count else 0.0,
            "epe": float(error[mask].mean()) if count else None}


@torch.no_grad()
def evaluate_branch(model, dataset, device, timed=False):
    model.eval()
    rows = []
    if timed and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for batch in DataLoader(dataset, batch_size=1, shuffle=False):
        source = batch["source_image"].to(device)
        target = batch["target_image"].to(device)
        truth = batch["flow_map"].to(device).float()
        valid = batch["correspondence_mask"].to(device).bool()
        result = model(source, target)
        iterations = (*result["flow64_rounds"],
                      *result["flow128_rounds"])
        errors = {}
        for label, flow in zip(LABELS, iterations):
            full = F.interpolate(flow, size=(512, 512), mode="bilinear",
                                 align_corners=False)
            errors[label] = torch.linalg.vector_norm(full - truth, dim=1)
        final_error = torch.linalg.vector_norm(
            result["final_512px"] - truth, dim=1)
        base_final = F.interpolate(result["base_flows512"][-1],
                                   size=(512, 512), mode="bilinear",
                                   align_corners=False)
        base_error = torch.linalg.vector_norm(base_final - truth, dim=1)
        displacement = torch.linalg.vector_norm(truth, dim=1)
        large = valid & (displacement >= 64)
        gt32, selected = pooled_truth(truth, valid, 32)
        centre32 = model.base.deconv4(result["base_flows256"][0]) * 2
        inside32 = ((gt32 - centre32).abs().amax(dim=1) <= 64) & selected
        assigned_inside = F.interpolate(inside32[:, None].float(),
                                        size=(512, 512), mode="nearest")[:, 0].bool()
        assigned_outside = F.interpolate((selected & ~inside32)[:, None].float(),
                                         size=(512, 512), mode="nearest")[:, 0].bool()
        row = {"name": batch["name"][0], "valid_full": int(valid.sum()),
               "base_final": group_stats(base_error, valid),
               "final": group_stats(final_error, valid),
               "large": group_stats(final_error, large),
               "base_large": group_stats(base_error, large),
               "first_window_inside": group_stats(final_error,
                                                   valid & assigned_inside),
               "first_window_outside": group_stats(final_error,
                                                    valid & assigned_outside),
               "base_first_window_inside": group_stats(base_error,
                                                        valid & assigned_inside),
               "base_first_window_outside": group_stats(base_error,
                                                         valid & assigned_outside),
               "first_window_queries": int(selected.sum()),
               "first_window_outside_queries": int((selected & ~inside32).sum())}
        for label in LABELS:
            row[label] = group_stats(errors[label], valid)
        row["warp_valid_fraction"] = [
            float(info["valid_fraction"]) for info in result["diagnostics"]]
        if timed:
            row["inference_ms"] = timing_on_device(model, source, target,
                                                    device)
        rows.append(row)
    count = sum(row["valid_full"] for row in rows)
    summary = {"samples": len(rows), "valid_full": count,
               "final_epe_512px": sum(row["final"]["sum"] for row in rows) / count,
               "base_final_epe_512px": sum(row["base_final"]["sum"]
                                       for row in rows) / count,
               "improved_pairs_vs_base": sum(
                   row["final"]["epe"] < row["base_final"]["epe"]
                   for row in rows),
               "first_window_outside_fraction": sum(
                   row["first_window_outside_queries"] for row in rows) /
                   sum(row["first_window_queries"] for row in rows)}
    for label in LABELS:
        summary[f"{label}_epe_512px"] = sum(
            row[label]["sum"] for row in rows) / count
    for label in ("large", "first_window_inside", "first_window_outside"):
        pixels = sum(row[label]["pixels"] for row in rows)
        summary[f"{label}_pixels"] = pixels
        summary[f"{label}_epe_512px"] = (
            sum(row[label]["sum"] for row in rows) / pixels if pixels else None)
        summary[f"base_{label}_epe_512px"] = (
            sum(row[f"base_{label}"]["sum"] for row in rows) / pixels
            if pixels else None)
    if timed:
        summary["inference_ms_mean_pair_median"] = float(np.mean([
            row["inference_ms"] for row in rows]))
        summary["peak_allocated_mib"] = (
            torch.cuda.max_memory_allocated() / 2**20
            if device.type == "cuda" else None)
    return summary, rows


@torch.no_grad()
def evaluate_plain(base, dataset, device, timed=False):
    base.eval()
    rows = []
    if timed and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for batch in DataLoader(dataset, batch_size=1, shuffle=False):
        source = batch["source_image"].to(device)
        target = batch["target_image"].to(device)
        truth = batch["flow_map"].to(device).float()
        valid = batch["correspondence_mask"].to(device).bool()

        def forward(s, t):
            s, t, s256, t256, *_ = base.pre_process_data(s, t, device=device)
            flows256, flows512 = base(t, s, t256, s256)
            return flows256, flows512, F.interpolate(
                flows512[-1], size=(512, 512),
                mode="bilinear", align_corners=False)

        flows256, _, prediction = forward(source, target)
        error = torch.linalg.vector_norm(prediction - truth, dim=1)
        large = valid & (torch.linalg.vector_norm(truth, dim=1) >= 64)
        gt32, selected = pooled_truth(truth, valid, 32)
        centre32 = base.deconv4(flows256[0]) * 2
        inside32 = ((gt32 - centre32).abs().amax(dim=1) <= 64) & selected
        assigned_inside = F.interpolate(inside32[:, None].float(),
                                        size=(512, 512), mode="nearest")[:, 0].bool()
        assigned_outside = F.interpolate((selected & ~inside32)[:, None].float(),
                                         size=(512, 512), mode="nearest")[:, 0].bool()
        row = {"name": batch["name"][0], "valid_full": int(valid.sum()),
               "epe": float(error[valid].mean()),
               "large": group_stats(error, large),
               "first_window_inside": group_stats(error, valid & assigned_inside),
               "first_window_outside": group_stats(error, valid & assigned_outside)}
        if timed:
            row["inference_ms"] = timing_on_device(
                lambda s, t: forward(s, t)[-1], source, target, device)
        rows.append(row)
    summary = {"samples": len(rows), "valid_full": sum(
        row["valid_full"] for row in rows)}
    summary["final_epe_512px"] = sum(row["epe"] * row["valid_full"]
                                   for row in rows) / summary["valid_full"]
    for label in ("large", "first_window_inside", "first_window_outside"):
        pixels = sum(row[label]["pixels"] for row in rows)
        summary[f"{label}_pixels"] = pixels
        summary[f"{label}_epe_512px"] = (
            sum(row[label]["sum"] for row in rows) / pixels if pixels else None)
    if timed:
        summary["inference_ms_mean_pair_median"] = float(np.mean([
            row["inference_ms"] for row in rows]))
        summary["peak_allocated_mib"] = (
            torch.cuda.max_memory_allocated() / 2**20
            if device.type == "cuda" else None)
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--coarse-checkpoint", type=Path, required=True)
    parser.add_argument("--dcn-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--code-check", action="store_true",
                        help="One training update plus geometry/freeze checks; no full training")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geometry = synthetic_geometry_check()
    base, coarse_payload = make_base(args.coarse_checkpoint,
                                     args.pretrained, device)
    model = FrozenGLUNetWithIter(base).to(device).eval()
    trainset = RoadScenePairs(args.data_root, "train")
    valset = RoadScenePairs(args.data_root, "val")
    if len(trainset) != 176 or len(valset) != 23:
        raise ValueError("Expected 176 train and 23 validation pairs")
    first = next(iter(DataLoader(valset, batch_size=1, shuffle=False)))
    with torch.no_grad():
        initial = model(first["source_image"].to(device),
                        first["target_image"].to(device))
    frozen_final = F.interpolate(initial["base_flows512"][-1],
                                  size=(512, 512), mode="bilinear",
                                  align_corners=False)
    checks = {
        "tensor_shapes": {
            "coarse16": list(initial["base_flows256"][0].shape),
            "local32": list(initial["base_flows256"][1].shape),
            "local64": list(initial["base_flows512"][0].shape),
            "final128": list(initial["base_flows512"][1].shape)},
        "zero_head_final_max_abs_difference": float((
            initial["final_512px"] - frozen_final).abs().max()),
        "zero_head_64_max_abs_difference": max(float((
            flow - initial["base_flows512"][0]).abs().max())
            for flow in initial["flow64_rounds"]),
        "zero_head_128_max_abs_difference": max(float((
            flow - initial["base_flows512"][1]).abs().max())
            for flow in initial["flow128_rounds"]),
        "geometry": geometry,
        "frozen_base_parameters": all(
            not p.requires_grad for p in base.parameters())}
    if (any(checks[key] != 0 for key in (
            "zero_head_final_max_abs_difference",
            "zero_head_64_max_abs_difference",
            "zero_head_128_max_abs_difference")) or
            not checks["frozen_base_parameters"]):
        raise RuntimeError(f"Initialization/freeze check failed: {checks}")
    (args.output / "code_checks.json").write_text(json.dumps(
        checks, indent=2), encoding="utf-8")
    if args.geometry_only:
        print(json.dumps(checks, indent=2), flush=True)
        return
    base_before = {name: value.detach().cpu().clone()
                   for name, value in base.state_dict().items()}
    branch_initial = {name: value.detach().cpu().clone()
                      for name, value in model.branch.state_dict().items()}
    optimizer = torch.optim.AdamW(model.branch.parameters(), lr=1e-4)
    if args.code_check:
        batch = next(iter(DataLoader(trainset, batch_size=2, shuffle=False)))
        source = batch["source_image"].to(device)
        target = batch["target_image"].to(device)
        truth = batch["flow_map"].to(device).float()
        valid = batch["correspondence_mask"].to(device).bool()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(source, target)
        loss, terms = training_loss(output, truth, valid)
        loss.backward()
        head_grad = model.branch.update_head.weight.grad
        if (not torch.isfinite(loss) or head_grad is None or
                not torch.isfinite(head_grad).all() or head_grad.abs().sum() == 0):
            raise RuntimeError("Code-check loss or branch gradient failed")
        torch.nn.utils.clip_grad_norm_(model.branch.parameters(), 1.0)
        optimizer.step()
        frozen_change = max(float((value.detach().cpu() - base_before[name]).abs().max())
                            for name, value in base.state_dict().items())
        if frozen_change != 0:
            raise RuntimeError("Frozen base changed in code-check")
        check_result = {"code_check_only": True, "train_pairs": list(batch["name"]),
                        "loss": float(loss.detach()), "terms": terms,
                        "head_grad_abs_sum": float(head_grad.abs().sum()),
                        "frozen_base_max_abs_change": frozen_change,
                        "branch_head_max_abs_after_step": float(
                            model.branch.update_head.weight.detach().abs().max()),
                        "zero_init_checks": checks,
                        "test_pairs_accessed": False}
        (args.output / "code_check_train.json").write_text(
            json.dumps(check_result, indent=2), encoding="utf-8")
        print(json.dumps(check_result, indent=2), flush=True)
        return
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=4,
        threshold=.01, threshold_mode="abs", min_lr=1e-7)
    baseline, _ = evaluate_branch(model, valset, device)
    best = baseline["final_epe_512px"]
    history = [{"epoch": 0, "optimizer_steps": 0,
                "val": baseline, "train_loss": None}]
    best_epoch = 0
    step = 0
    old_fine_report = json.loads((args.dcn_checkpoint.parent /
                                  "report_fine.json").read_text(encoding="utf-8"))
    reference_order = {row["epoch"]: row.get("image_order_sha256")
                       for row in old_fine_report["history"]}
    for epoch in range(1, 6):
        model.train()
        loader = DataLoader(trainset, batch_size=2, shuffle=True,
                            generator=torch.Generator().manual_seed(SEED + epoch),
                            num_workers=0, pin_memory=device.type == "cuda")
        order = hashlib.sha256()
        losses = []
        components = []
        for batch in loader:
            for name in batch["name"]:
                order.update(name.encode("utf-8") + b"\n")
            source = batch["source_image"].to(device)
            target = batch["target_image"].to(device)
            truth = batch["flow_map"].to(device).float()
            valid = batch["correspondence_mask"].to(device).bool()
            optimizer.zero_grad(set_to_none=True)
            result = model(source, target)
            loss, terms = training_loss(result, truth, valid)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at epoch {epoch}, step {step}")
            loss.backward()
            if step == 0 and not (model.branch.update_head.weight.grad is not None
                                  and torch.isfinite(
                                      model.branch.update_head.weight.grad).all()
                                  and model.branch.update_head.weight.grad.abs().sum() > 0):
                raise RuntimeError("Residual head receives no finite gradient")
            torch.nn.utils.clip_grad_norm_(model.branch.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            components.append(terms)
            step += 1
        if order.hexdigest() != reference_order[epoch]:
            raise RuntimeError(f"DCN-only data order mismatch at epoch {epoch}")
        val, _ = evaluate_branch(model, valset, device)
        scheduler.step(val["final_epe_512px"])
        record = {"epoch": epoch, "optimizer_steps": step,
                  "train_loss": float(np.mean(losses)),
                  "train_terms": {name: float(np.mean([row[name]
                               for row in components])) for name in components[0]},
                  "val": val, "image_order_sha256": order.hexdigest(),
                  "image_order_matches_dcn_only": True,
                  "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(record)
        if val["final_epe_512px"] < best - .01:
            best = val["final_epe_512px"]
            best_epoch = epoch
            torch.save({"epoch": epoch, "model_state_dict": model.branch.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "parent_sha256": sha256_file(args.coarse_checkpoint),
                        "optimizer_steps": step,
                        "selection_metric": best,
                        "test_pairs_accessed": False},
                       args.output / "best_iter_branch.pth")
        (args.output / "history.json").write_text(json.dumps(
            history, indent=2), encoding="utf-8")
        print(f"iter epoch {epoch}/5: final={val['final_epe_512px']:.4f}, "
              f"r64_1={val['64_round1_epe_512px']:.4f}, "
              f"r64_2={val['64_round2_epe_512px']:.4f}, "
              f"r128_1={val['128_round1_epe_512px']:.4f}, "
              f"r128_2={val['128_round2_epe_512px']:.4f}, "
              f"best={best:.4f}", flush=True)
    if best_epoch:
        state = torch.load(args.output / "best_iter_branch.pth",
                           map_location="cpu", weights_only=False)
        model.branch.load_state_dict(state["model_state_dict"], strict=True)
    else:
        model.branch.load_state_dict(branch_initial, strict=True)
        torch.save({"epoch": 0, "model_state_dict": branch_initial,
                    "parent_sha256": sha256_file(args.coarse_checkpoint),
                    "selection_metric": best, "test_pairs_accessed": False},
                   args.output / "best_iter_branch.pth")
    base_change = max(float((value.detach().cpu() - base_before[name]).abs().max())
                      for name, value in base.state_dict().items())
    if base_change != 0:
        raise RuntimeError("Frozen GLU-Net parameters/buffers changed")
    selected_summary, selected_rows = evaluate_branch(model, valset,
                                                      device, timed=True)
    base_summary, base_rows = evaluate_plain(base, valset, device, timed=True)
    dcn = GLUNet_model(evaluation=False, pyramid_type="VGG",
                       cyclic_consistency=True, backbone_pretrained=False,
                       coarse_attention=True, coarse_dns=True,
                       local_dcn_steps=1)
    fine_payload = torch.load(args.dcn_checkpoint, map_location="cpu",
                              weights_only=False)
    if fine_payload.get("parent_sha256") != sha256_file(args.coarse_checkpoint):
        raise ValueError("DCN-only best does not descend from this coarse best")
    dcn.load_state_dict(fine_payload["model_state_dict"], strict=True)
    dcn = dcn.to(device).eval()
    dcn_summary, dcn_rows = evaluate_plain(dcn, valset, device, timed=True)
    indexed_base = {row["name"]: row for row in base_rows}
    indexed_dcn = {row["name"]: row for row in dcn_rows}
    per_image = []
    for row in selected_rows:
        name = row["name"]
        item = {"name": name, "valid_full": row["valid_full"],
                "coarse_best_final_epe_512px": indexed_base[name]["epe"],
                "dcn_best_final_epe_512px": indexed_dcn[name]["epe"],
                "iter_final_epe_512px": row["final"]["epe"],
                "iter_minus_dcn_final_epe_512px": row["final"]["epe"] -
                                                    indexed_dcn[name]["epe"],
                "large_epe_512px": row["large"]["epe"],
                "large_valid_pixels": row["large"]["pixels"],
                "dcn_large_epe_512px": indexed_dcn[name]["large"]["epe"],
                "first_window_inside_epe_512px":
                    row["first_window_inside"]["epe"],
                "dcn_first_window_inside_epe_512px":
                    indexed_dcn[name]["first_window_inside"]["epe"],
                "first_window_outside_epe_512px":
                    row["first_window_outside"]["epe"],
                "dcn_first_window_outside_epe_512px":
                    indexed_dcn[name]["first_window_outside"]["epe"],
                "first_window_outside_queries": row["first_window_outside_queries"],
                "first_window_queries": row["first_window_queries"],
                "iter_inference_ms": row["inference_ms"]}
        item.update({f"{label}_epe_512px": row[label]["epe"]
                     for label in LABELS})
        per_image.append(item)
    with (args.output / "per_image_val.csv").open(
            "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    report = {"split": "val", "test_pairs_accessed": False,
              "coarse_checkpoint_sha256": sha256_file(args.coarse_checkpoint),
              "dcn_checkpoint_sha256": sha256_file(args.dcn_checkpoint),
              "selected_epoch": best_epoch, "optimizer_steps_per_arm": 440,
              "frozen_base_max_abs_change": base_change,
              "zero_initialization_and_geometry": checks,
              "selection": "val final 512px pixel-weighted EPE, min_delta=0.01",
              "history": history,
              "coarse_best": base_summary, "dcn_only_best": dcn_summary,
              "iterative_best": selected_summary,
              "improved_pairs_vs_dcn": sum(row[
                  "iter_minus_dcn_final_epe_512px"] < 0 for row in per_image),
              "worsened_pairs_vs_dcn": sum(row[
                  "iter_minus_dcn_final_epe_512px"] > 0 for row in per_image)}
    (args.output / "report_val.json").write_text(json.dumps(
        report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("history", "zero_initialization_and_geometry")},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
