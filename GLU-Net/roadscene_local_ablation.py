"""Matched six-epoch local refinement ablation from one fixed coarse weight.

Arms: frozen baseline, DCN only, original local layers only with DCN bypassed,
and original local layers plus DCN. The three trained arms share data order,
528-update maximum budget, loss, validation selection and GT masks. The
previous five-epoch-DCN plus one-epoch-joint checkpoint is evaluated separately
as a schedule reference. Train/validation only; no joint coarse training.
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
from torch.utils.data import DataLoader, Subset

from datasets.roadscene import RoadScenePairs
from models.our_models.GLUNet import GLUNet_model
from roadscene_coarse import sha256_file
from roadscene_compare import PROTOCOL, evaluate_common
from roadscene_refinement_audit import (
    STAGES, combine_groups, combine_search, evaluate as evaluate_stages)
import roadscene_staged as staged
from roadscene_staged_mutual_fix import FIX


SEED = staged.SEED
ARMS = ("dcn_only", "local_only", "local_dcn")
GRID_GROUPS = ("gt_displacement_64+",
               "outside_first_window_assigned_pixels",
               "inside_first_window_assigned_pixels")


def checked_payload(path, pretrained, smoke):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("stage") != "coarse" or
            payload.get("arm") != "dns_attention" or
            payload.get("base_sha256") != sha256_file(pretrained)):
        raise ValueError("Coarse checkpoint or base weight mismatch")
    if smoke:
        if payload.get("code_check_only"):
            raise ValueError("Smoke check still needs a full coarse checkpoint")
    elif (payload.get("matching_fix") != FIX or
          payload.get("code_check_only") or
          payload.get("training_origin") !=
          "original pretrained GLU-Net, not epoch-37 continuation"):
        raise ValueError("Expected fresh_mutual_fix/best_coarse.pth")
    return payload


def new_model(coarse, device, bypass=False):
    model = GLUNet_model(evaluation=False, pyramid_type="VGG",
                        cyclic_consistency=True, backbone_pretrained=False,
                        coarse_attention=True, coarse_dns=True,
                        local_dcn_steps=1)
    model.load_state_dict(coarse["model_state_dict"], strict=True)
    model.local_dcn_steps = 0 if bypass else 1
    model.train_coarse_encoder = False
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def flow_tensors(model, batch, device):
    source, target, s256, t256, *_ = model.pre_process_data(
        batch["source_image"].to(device), batch["target_image"].to(device),
        device=device)
    return model(target, source, t256, s256)


@torch.no_grad()
def bypass_check(coarse, valset, device):
    model = new_model(coarse, device)
    differences = {label: 0.0 for label in STAGES}
    for batch in DataLoader(valset, batch_size=1, shuffle=False):
        model.local_dcn_steps = 0
        off256, off512 = flow_tensors(model, batch, device)
        model.local_dcn_steps = 1
        on256, on512 = flow_tensors(model, batch, device)
        for label, a, b in zip(STAGES, (*off256, *off512),
                               (*on256, *on512)):
            differences[label] = max(differences[label],
                                     float((a - b).abs().max()))
    if differences["coarse16"] != 0:
        raise RuntimeError("DCN bypass altered coarse flow")
    # The original coarse-stage DCN head is zero initialized. A tiny
    # floating-point deviation at later grids is measured, not concealed.
    if max(differences.values()) > .003:
        raise RuntimeError(f"Unexpected nonzero coarse-stage DCN: {differences}")
    return {"max_abs_difference_activating_untrained_dcn": differences,
            "bypass": "local_dcn_steps=0 skips only the 32-grid DCN block",
            "flow_units": {"coarse16": "256px image pixels",
                           "local32": "256px image pixels",
                           "local64": "512px image pixels",
                           "final128": "512px image pixels"},
            "direction": "VI target -> IR source"}


def configure(model, arm):
    model.eval()
    model.train_coarse_encoder = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = staged.groups(model)
    active = {"dcn_only": ("dcn",), "local_only": ("local",),
              "local_dcn": ("local", "dcn")}[arm]
    model.local_dcn_steps = 0 if arm == "local_only" else 1
    for name in active:
        for parameter in selected[name]:
            parameter.requires_grad_(True)
    if any(p.requires_grad for name in staged.FROZEN_FLOW_UPSAMPLERS
           for p in getattr(model, name).parameters()):
        raise RuntimeError("Pretrained flow upsampler became trainable")
    return selected, active


def frozen_snapshot(model, active):
    names = {id(p) for name in active for p in staged.groups(model)[name]}
    parameters = {name: p.detach().cpu().clone()
                  for name, p in model.named_parameters() if id(p) not in names}
    buffers = {name: b.detach().cpu().clone()
               for name, b in model.named_buffers()}
    return parameters, buffers


def frozen_change(model, snapshot):
    parameters, buffers = snapshot
    differences = [float((p.detach().cpu() - parameters[name]).abs().max())
                   for name, p in model.named_parameters() if name in parameters]
    differences += [float((b.detach().cpu() - buffers[name]).abs().max())
                    for name, b in model.named_buffers() if name in buffers]
    return max(differences, default=0.0)


@torch.no_grad()
def validation_epe(model, dataset, device):
    model.eval()
    total, count = 0., 0
    for batch in DataLoader(dataset, batch_size=1, shuffle=False):
        _, flow512 = flow_tensors(model, batch, device)
        predicted = torch.nn.functional.interpolate(
            flow512[-1], size=(512, 512), mode="bilinear",
            align_corners=False)
        truth = batch["flow_map"].to(device).float()
        valid = batch["correspondence_mask"].to(device).bool()
        error = torch.linalg.vector_norm(predicted - truth, dim=1)
        total += float(error[valid].sum())
        count += int(valid.sum())
    return total / count


def save_atomic(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_arm(arm, coarse, coarse_sha, pretrained_sha, trainset, valset,
              reference_orders, output, device, code_check):
    model = new_model(coarse, device, bypass=arm == "local_only")
    selected, active = configure(model, arm)
    configured_trainable = sum(
        p.numel() for name in active for p in selected[name])
    snapshot = frozen_snapshot(model, active)
    optimizer, scheduler = staged.make_optimizer(model, "fine")
    best = validation_epe(model, valset, device)
    history = [{"epoch": 0, "steps": 0, "val_final_epe_512px": best,
                "train_loss": None, "image_order_sha256": None}]
    best_epoch, steps = 0, 0
    arm_dir = output / arm
    arm_dir.mkdir(parents=True, exist_ok=False)
    max_epochs = 1 if code_check else 6
    for epoch in range(1, max_epochs + 1):
        model.eval()  # freeze every BatchNorm running statistic
        loader = DataLoader(
            trainset, batch_size=2, shuffle=True,
            generator=torch.Generator().manual_seed(SEED + epoch),
            num_workers=0, pin_memory=device.type == "cuda")
        order = hashlib.sha256()
        losses = []
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
                raise RuntimeError(f"{arm} nonfinite loss at epoch {epoch}")
            loss.backward()
            if steps == 0 and not any(
                    p.grad is not None and torch.isfinite(p.grad).all()
                    and p.grad.abs().sum() > 0
                    for name in active for p in selected[name]):
                raise RuntimeError(f"{arm} has no finite trainable gradient")
            if any(p.grad is not None for p in selected["coarse"]):
                raise RuntimeError("Frozen coarse network received gradients")
            torch.nn.utils.clip_grad_norm_(
                [p for name in active for p in selected[name]], 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            steps += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        digest = order.hexdigest()
        if reference_orders and digest != reference_orders[epoch]:
            raise RuntimeError(f"{arm} data order differs from prior fine run")
        val = validation_epe(model, valset, device)
        scheduler.step(val)
        record = {"epoch": epoch, "steps": steps,
                  "val_final_epe_512px": val,
                  "train_loss": float(np.mean(losses)),
                  "image_order_sha256": digest,
                  "same_order_as_prior_fine": bool(reference_orders),
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
                "arm": arm, "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "numpy_rng_state": np.random.get_state(),
                "python_rng_state": random.getstate(),
                "cuda_rng_state": torch.cuda.get_rng_state()
                if device.type == "cuda" else None,
                "steps": steps, "selection_metric": best,
                "coarse_sha256": coarse_sha,
                "pretrained_sha256": pretrained_sha,
                "dcn_bypassed": arm == "local_only",
                "matching_fix": FIX,
                "code_sha256": sha256_file(Path(__file__)),
                "code_check_only": code_check,
            }, arm_dir / "best.pth")
        (arm_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8")
        print(f"{arm} epoch {epoch}/{max_epochs}: "
              f"val final={val:.4f}, best={best:.4f}, steps={steps}",
              flush=True)
    if steps != (1 if code_check else 528):
        raise RuntimeError(f"Unexpected {arm} update count: {steps}")
    change = frozen_change(model, snapshot)
    if change != 0:
        raise RuntimeError(f"{arm} changed frozen parameters or BN: {change}")
    if best_epoch:
        selected_state = torch.load(
            arm_dir / "best.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(selected_state["model_state_dict"], strict=True)
    else:
        # Epoch 0 is a valid selected candidate.
        model = new_model(coarse, device, bypass=arm == "local_only")
        save_atomic({"arm": arm, "epoch": 0,
                     "model_state_dict": model.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict(),
                     "scheduler_state_dict": scheduler.state_dict(),
                     "torch_rng_state": torch.get_rng_state(),
                     "numpy_rng_state": np.random.get_state(),
                     "python_rng_state": random.getstate(),
                     "cuda_rng_state": torch.cuda.get_rng_state()
                     if device.type == "cuda" else None,
                     "selection_metric": best, "steps": 0,
                     "coarse_sha256": coarse_sha,
                     "pretrained_sha256": pretrained_sha,
                     "dcn_bypassed": arm == "local_only",
                     "matching_fix": FIX,
                     "code_sha256": sha256_file(Path(__file__)),
                     "code_check_only": code_check}, arm_dir / "best.pth")
    return model.eval(), {"selected_epoch": best_epoch,
                          "selected_final_epe_512px": best,
                          "max_steps": steps, "frozen_max_abs_change": change,
                          "configured_trainable_parameters":
                              configured_trainable,
                          "history": history}


@torch.no_grad()
def evaluate_arm(model, valset, device):
    loader = DataLoader(valset, batch_size=1, shuffle=False)
    common, common_rows = evaluate_common(model, loader, device)
    detail_rows = evaluate_stages(model, loader, device, preview_dir=None)
    indexed = {row["name"]: row for row in common_rows}
    if indexed.keys() != {row["name"] for row in detail_rows}:
        raise RuntimeError("Evaluator image pair mismatch")
    by_image = {}
    for row in detail_rows:
        external = indexed[row["name"]]
        if external["valid_full"] != row["valid_full"]:
            raise RuntimeError("Evaluator valid-mask mismatch")
        image = {"valid_full": row["valid_full"],
                 "final_aepe_512px": external["aepe_512px"],
                 "inference_ms": external["inference_ms"],
                 "gt_displacement_p90_512px":
                     row["gt_displacement_p90_512px"],
                 "first_window_outside_fraction": (
                     row["search"]["local32"]["outside"]["queries"] /
                     row["search"]["local32"]["valid_queries"])}
        image.update({f"{name}_epe_512px": row["stage_epe_512px"][name]
                      for name in STAGES})
        for group in GRID_GROUPS:
            part = row["groups"][group]
            image[f"{group}_pixels"] = part["pixels"]
            image[f"{group}_final_epe_512px"] = (
                part["final128"] / part["pixels"] if part["pixels"] else None)
        by_image[row["name"]] = image
    summary = {"samples": len(detail_rows),
               "final_epe_512px": common["final_flow_epe_512px"],
               "pair_mean_aepe_512px": common["aepe_pair_mean_512px"],
               "valid_pixels_full": common["valid_pixels_full"],
               "four_level_epe_512px": combine_groups(
                   detail_rows, "all")["epe_512px"],
               "large_motion_final_epe_512px": combine_groups(
                   detail_rows, "gt_displacement_64+")["epe_512px"]["final128"],
               "first_window_outside_final_epe_512px": combine_groups(
                   detail_rows, "outside_first_window_assigned_pixels"
                   )["epe_512px"]["final128"],
               "first_window_outside_query_fraction": combine_search(
                   detail_rows, "local32")["outside"]["fraction_of_valid_queries"],
               "inference_ms_mean_pair_median":
                   common["inference_ms_mean_pair_median"],
               "peak_inference_allocated_mib": common["peak_allocated_mib"]}
    base = model.base if hasattr(model, "base") else model
    dcn_parameters = sum(p.numel() for p in base.local_dcn32.parameters())
    summary["total_parameters"] = sum(p.numel() for p in model.parameters())
    summary["effective_inference_parameters"] = (
        summary["total_parameters"] - dcn_parameters
        if base.local_dcn_steps == 0 else summary["total_parameters"])
    summary["trainable_parameters"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    return summary, by_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--coarse-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-fine-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-check", action="store_true",
                        help="One optimizer update per arm, one val pair")
    parser.add_argument("--smoke-legacy-coarse", action="store_true",
                        help="Code-check only with old epoch-37 local weight")
    args = parser.parse_args()
    if args.smoke_legacy_coarse and not args.code_check:
        parser.error("Legacy coarse is allowed only for a code check")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty and independent")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coarse = checked_payload(args.coarse_checkpoint, args.pretrained,
                             args.smoke_legacy_coarse)
    coarse_sha = sha256_file(args.coarse_checkpoint)
    pretrained_sha = sha256_file(args.pretrained)
    reference = torch.load(args.reference_fine_checkpoint, map_location="cpu",
                           weights_only=False)
    if (reference.get("stage") != "fine" or
            reference.get("parent_sha256") != coarse_sha or
            reference.get("base_sha256") != pretrained_sha or
            reference.get("code_check_only")):
        raise ValueError("Reference fine best must descend from this coarse")
    if not args.smoke_legacy_coarse and reference.get("matching_fix") != FIX:
        raise ValueError("Reference fine checkpoint lacks the matching fix")
    report_path = args.reference_fine_checkpoint.parent / "report_fine.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Prior fine report missing: {report_path}")
    prior = json.loads(report_path.read_text(encoding="utf-8"))
    if not args.code_check and prior["result"]["selected_epoch"] != 6:
        raise ValueError("Expected selected fine epoch 6 for matched budget")
    reference_orders = {}
    if not args.code_check:
        reference_orders = {row["epoch"]: row["image_order_sha256"]
                            for row in prior["history"]
                            if 1 <= row["epoch"] <= 6}
        if len(reference_orders) != 6:
            raise ValueError("Prior fine history lacks six data-order hashes")
    trainset = RoadScenePairs(args.data_root, "train")
    valset = RoadScenePairs(args.data_root, "val")
    if len(trainset) != 176 or len(valset) != 23:
        raise ValueError("Expected 176 train and 23 validation pairs")
    if args.code_check:
        trainset, valset = Subset(trainset, range(2)), Subset(valset, range(1))
    glunet_module = importlib.import_module("models.our_models.GLUNet")
    original_matching = glunet_module.MutualMatching
    glunet_module.MutualMatching = (
        lambda corr: original_matching(corr.clamp_min(0)))
    try:
        geometry = bypass_check(coarse, valset, device)
        baseline = new_model(coarse, device, bypass=True)
        models = {"coarse_frozen": baseline}
        training = {}
        for arm in ARMS:
            models[arm], training[arm] = train_arm(
                arm, coarse, coarse_sha, pretrained_sha,
                trainset, valset, reference_orders, args.output, device,
                args.code_check)
        reference_model = new_model(coarse, device)
        reference_model.load_state_dict(reference["model_state_dict"], strict=True)
        models["prior_5plus1_local_dcn"] = reference_model.eval()
        summaries, per_arm = {}, {}
        for label, model in models.items():
            summaries[label], per_arm[label] = evaluate_arm(
                model, valset, device)
        names = per_arm["coarse_frozen"].keys()
        if any(rows.keys() != names for rows in per_arm.values()):
            raise RuntimeError("Arm image pair sets differ")
        per_image = []
        for name in names:
            record = {"name": name, "valid_full":
                      per_arm["coarse_frozen"][name]["valid_full"]}
            for label, rows in per_arm.items():
                if rows[name]["valid_full"] != record["valid_full"]:
                    raise RuntimeError("Arm valid pixel masks differ")
                for key, value in rows[name].items():
                    if key != "valid_full":
                        record[f"{label}_{key}"] = value
            per_image.append(record)
        with (args.output / "per_image_val.csv").open(
                "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(per_image[0]))
            writer.writeheader()
            writer.writerows(per_image)
        reference_rows = per_arm["prior_5plus1_local_dcn"]
        contrasts = {label: {
            "improved_pairs_vs_prior": sum(
                rows[name]["final_aepe_512px"] <
                reference_rows[name]["final_aepe_512px"]
                for name in names),
            "worsened_pairs_vs_prior": sum(
                rows[name]["final_aepe_512px"] >
                reference_rows[name]["final_aepe_512px"]
                for name in names)}
            for label, rows in per_arm.items()
            if label != "prior_5plus1_local_dcn"}
        report = {
            "split": "val", "test_pairs_accessed": False,
            "code_check_only": args.code_check,
            "smoke_legacy_coarse": args.smoke_legacy_coarse,
            "pretrained_sha256": pretrained_sha,
            "coarse_checkpoint_sha256": coarse_sha,
            "prior_fine_checkpoint_sha256":
                sha256_file(args.reference_fine_checkpoint),
            "matching_fix": FIX,
            "code_sha256": sha256_file(Path(__file__)),
            "selection": "min val final EPE 512px, min_delta=0.01",
            "max_epochs": 1 if args.code_check else 6,
            "optimizer_steps_per_trained_arm":
                1 if args.code_check else 528,
            "prior_schedule":
                "DCN-only epochs 1-5, local+DCN epoch 6; separate reference",
            "matched_schedule":
                "active module group from epoch 1 through epoch 6",
            "loss": "same final-flow Charbonnier EPE and valid mask as staged fine",
            "protocol": PROTOCOL, "geometry_and_bypass": geometry,
            "training": training, "arms": summaries,
            "paired_vs_prior": contrasts}
        (args.output / "report_val.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"code_check_only": args.code_check,
                          "arms": summaries,
                          "paired_vs_prior": contrasts},
                         indent=2), flush=True)
    finally:
        glunet_module.MutualMatching = original_matching


if __name__ == "__main__":
    main()
