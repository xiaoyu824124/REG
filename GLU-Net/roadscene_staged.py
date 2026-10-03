"""RoadScene DNS+SA/CA registration: coarse -> fine DCN -> joint training.

Run --stage coarse, then fine, then joint. Each stage starts from the prior
stage's best checkpoint. Training uses train/validation only; the separate
test evaluator is locked until validation has selected a final stage.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from datasets.roadscene import RoadScenePairs
from models.our_models.GLUNet import GLUNet_model
from roadscene_coarse import (coarse_statistics, evaluate as evaluate_coarse,
                              evaluate_full, load_base_weights, predict_coarse,
                              prepare, sha256_file)
from roadscene_compare import PROTOCOL, evaluate_common
from roadscene_dcn import geometry_check
from roadscene_refinement_audit import (STAGES, combine_groups, combine_search,
                                        evaluate as evaluate_stages)


SEED = 2026
ARM = "dns_attention"
STAGE_CONFIG = {
    "coarse": {"max_epochs": 200, "patience": 20, "min_delta": 0.01,
               "selection": "coarse_epe_256px"},
    "fine": {"max_epochs": 80, "patience": 12, "min_delta": 0.01,
             "selection": "final_epe_512px"},
    "joint": {"max_epochs": 30, "patience": 6, "min_delta": 0.01,
              "selection": "final_epe_512px"},
}
FINE_WARMUP_EPOCHS = 5
COARSE_MODULES = ("coarse_dns", "coarse_attention", "decoder4")
LOCAL_MODULES = ("decoder3", "dc_conv1", "dc_conv2", "dc_conv3",
                 "dc_conv4", "dc_conv5", "dc_conv6", "dc_conv7",
                 "decoder2", "decoder1", "upfeat2", "l_dc_conv1",
                 "l_dc_conv2", "l_dc_conv3", "l_dc_conv4", "l_dc_conv5",
                 "l_dc_conv6", "l_dc_conv7")
FROZEN_FLOW_UPSAMPLERS = ("deconv4", "deconv2")


def stage_budget(args):
    config = STAGE_CONFIG[args.stage]
    return {"stage": args.stage, "arm": ARM,
            "max_epochs": (2 if args.stage == "fine" else 1)
            if args.code_check else config["max_epochs"],
            "patience": config["patience"], "min_delta": config["min_delta"],
            "batch_size": 1 if args.code_check else 2, "seed": SEED,
            "selection": config["selection"],
            "fine_dcn_only_warmup_epochs": (1 if args.code_check else FINE_WARMUP_EPOCHS)
            if args.stage == "fine" else 0,
            "train_pairs": 2 if args.code_check else 176,
            "val_pairs": 1 if args.code_check else 23,
            "coarse_encoder": "VGG pyramid.level_4 only",
            "flow_upsamplers_trainable": False,
            "code_check": args.code_check}


def source_stage(stage):
    return {"coarse": None, "fine": "coarse", "joint": "fine"}[stage]


def best_path(args, stage=None):
    return args.output / f"best_{stage or args.stage}.pth"


def latest_path(args):
    return args.output / f"latest_{args.stage}.pth"


def provenance(args):
    root = Path(__file__).resolve().parent
    names = ("roadscene_staged.py", "roadscene_coarse.py",
             "roadscene_compare.py", "roadscene_metrics.py",
             "roadscene_refinement_audit.py", "datasets/roadscene.py",
             "models/our_models/GLUNet.py",
             "models/our_models/coarse_attention.py",
             "models/our_models/coarse_dns.py",
             "models/our_models/local_dcn.py")
    return {"base_sha256": sha256_file(args.pretrained),
            "code_sha256": {name: sha256_file(root / name) for name in names}}


def new_model(args, device, from_previous=True):
    model = GLUNet_model(evaluation=False, pyramid_type="VGG",
                        cyclic_consistency=True, backbone_pretrained=False,
                        coarse_attention=True, coarse_dns=True,
                        local_dcn_steps=1)
    load_base_weights(model, args.pretrained)
    parent = source_stage(args.stage)
    parent_sha = None
    cumulative_steps = 0
    if from_previous and parent:
        path = best_path(args, parent)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (payload.get("stage") != parent or payload.get("arm") != ARM or
                payload.get("base_sha256") != sha256_file(args.pretrained) or
                payload.get("code_check_only") != args.code_check):
            raise ValueError(f"Incompatible {parent} checkpoint: {path}")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        parent_sha = sha256_file(path)
        cumulative_steps = payload["cumulative_optimizer_steps"]
    model = model.to(device).eval()
    return model, parent_sha, cumulative_steps


def groups(model):
    level4 = model.pyramid._modules["level_4"]
    coarse = list(level4.parameters()) + [
        p for name in COARSE_MODULES for p in getattr(model, name).parameters()]
    local = [p for name in LOCAL_MODULES for p in getattr(model, name).parameters()]
    dcn = list(model.local_dcn32.parameters())
    if len({id(p) for p in coarse + local + dcn}) != len(coarse + local + dcn):
        raise RuntimeError("Optimization parameter groups overlap")
    return {"coarse": coarse, "local": local, "dcn": dcn}


def configure_trainability(model, stage, epoch, fine_warmup=FINE_WARMUP_EPOCHS):
    model.train_coarse_encoder = stage == "joint"
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = groups(model)
    if stage == "coarse":
        active = ("coarse",)
    elif stage == "fine":
        active = ("dcn",) if epoch <= fine_warmup else ("local", "dcn")
    else:
        active = ("coarse", "local", "dcn")
    for name in active:
        for parameter in selected[name]:
            parameter.requires_grad_(True)
    for name in FROZEN_FLOW_UPSAMPLERS:
        if any(p.requires_grad for p in getattr(model, name).parameters()):
            raise RuntimeError(f"Flow upsampler {name} must remain frozen")
    return selected, active


def make_optimizer(model, stage):
    selected = groups(model)
    if stage == "coarse":
        rates = {"coarse": 1e-4}
    elif stage == "fine":
        rates = {"local": 1e-5, "dcn": 1e-4}
    else:
        rates = {"coarse": 1e-6, "local": 1e-5, "dcn": 1e-4}
    optimizer = torch.optim.AdamW([
        {"params": selected[name], "lr": rate, "group_name": name}
        for name, rate in rates.items()])
    config = STAGE_CONFIG[stage]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=max(2, config["patience"] // 3),
        threshold=config["min_delta"], threshold_mode="abs",
        min_lr=1e-7)
    return optimizer, scheduler


@torch.no_grad()
def zero_gate_check(args, valset, device):
    batch = next(iter(DataLoader(valset, batch_size=1, shuffle=False)))
    original, _, _ = new_model(args, device, from_previous=False)
    source, target, source256, target256, *_ = original.pre_process_data(
        batch["source_image"].to(device), batch["target_image"].to(device),
        device=device)
    baseline = GLUNet_model(evaluation=False, pyramid_type="VGG",
                           cyclic_consistency=True, backbone_pretrained=False,
                           coarse_attention=False, coarse_dns=False,
                           local_dcn_steps=0)
    load_base_weights(baseline, args.pretrained)
    baseline = baseline.to(device).eval()
    baseline256, baseline512 = baseline(target, source, target256, source256)
    original.local_dcn_steps = 0
    before256, before512 = original(target, source, target256, source256)
    original.local_dcn_steps = 1
    after256, after512 = original(target, source, target256, source256)
    differences = [float((a - b).abs().max()) for a, b in
                   zip((*before256, *before512), (*after256, *after512))]
    base_differences = [float((a - b).abs().max()) for a, b in
                        zip((*baseline256, *baseline512),
                            (*before256, *before512))]
    if max(base_differences) > 1e-5:
        raise RuntimeError(f"Zero-gate DNS/SA/CA changed base flow: {base_differences}")
    if max(differences[:3]) > 1e-5 or differences[3] > 0.003:
        raise RuntimeError(f"Zero-initialized DCN changed starting flow: {differences}")
    if (float(original.coarse_dns.gate) != 0 or
            float(original.coarse_attention.gate) != 0):
        raise RuntimeError("New coarse interaction gates must start at zero")
    del original, baseline
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"dcn_stage_flow_max_abs": dict(zip(STAGES, differences)),
            "dns_attention_vs_base_max_abs": dict(zip(STAGES, base_differences))}


def coarse_training_loss(model, batch, device):
    source, target, _, _, truth, valid, _ = prepare(batch, device)
    # Unlike roadscene_coarse.extract_coarse_features, this forward preserves
    # gradients through VGG level_4; levels_0..3 remain frozen.
    target_features = model.pyramid(target)[-3]
    source_features = model.pyramid(source)[-3]
    if target_features.shape[-2:] != (16, 16):
        raise RuntimeError("Expected 16x16 global matching features")
    target_features, source_features = model.enrich_coarse_features(
        target_features, source_features, target, source)
    predicted, corr = model.coarsest_resolution_flow(
        target_features, source_features, 256, 256, return_corr=True)
    epe, ce, _ = coarse_statistics(predicted, corr, truth, valid)
    return epe + 2 * ce, {"coarse_epe_256px": float(epe.detach()),
                           "corr_ce": float(ce.detach())}


def dense_training_loss(model, batch, device, coarse_auxiliary):
    source, target, source256, target256, *_ = model.pre_process_data(
        batch["source_image"].to(device), batch["target_image"].to(device),
        device=device)
    flow256, flow512 = model(target, source, target256, source256)
    truth = batch["flow_map"].to(device).float()
    valid = batch["correspondence_mask"].to(device).bool()
    if not valid.any():
        raise ValueError("Training batch has no valid GT pixels")
    final = F.interpolate(flow512[-1], truth.shape[-2:], mode="bilinear",
                          align_corners=False)
    def term(predicted):
        error = torch.sqrt((predicted - truth).square().sum(dim=1) + 0.01)
        return error[valid].mean()
    final_loss = term(final)
    if not coarse_auxiliary:
        return final_loss, {"fine_512px": float(final_loss.detach())}
    coarse = F.interpolate(flow256[0], truth.shape[-2:], mode="bilinear",
                           align_corners=False) * 2
    coarse_loss = term(coarse)
    return final_loss + 0.25 * coarse_loss, {
        "fine_512px": float(final_loss.detach()),
        "coarse_512px": float(coarse_loss.detach())}


@torch.no_grad()
def selection_metric(model, stage, val_loader, device):
    coarse = evaluate_coarse(model, val_loader, device)
    final = evaluate_full(model, val_loader, device)
    metric = (coarse["epe_256px"] if stage == "coarse"
              else final["final_flow_epe_512px"])
    return float(metric), coarse, final


@torch.no_grad()
def joint_epoch_diagnostics(model, val_loader, device):
    rows = evaluate_stages(model, val_loader, device, preview_dir=None)
    grouped = combine_groups(rows, "all")
    search = combine_search(rows, "local32")
    return {"coarse_epe_512px": grouped["epe_512px"]["coarse16"],
            "final_epe_512px": grouped["epe_512px"]["final128"],
            "first_window_outside_fraction":
                search["outside"]["fraction_of_valid_queries"],
            "per_image": {row["name"]: {
                "coarse_epe_512px": row["stage_epe_512px"]["coarse16"],
                "final_epe_512px": row["stage_epe_512px"]["final128"],
                "first_window_outside_fraction":
                    row["search"]["local32"]["outside"]["queries"] /
                    row["search"]["local32"]["valid_queries"]}
                for row in rows}}


def checkpoint_payload(model, optimizer, scheduler, args, epoch, metric,
                       stage_steps, cumulative_steps, parent_sha, bad_epochs,
                       history, finished):
    return {"stage": args.stage, "arm": ARM, "epoch": epoch,
            "selection_metric": metric, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "cuda_rng_state": (torch.cuda.get_rng_state()
                               if next(model.parameters()).is_cuda else None),
            "stage_optimizer_steps": stage_steps,
            "cumulative_optimizer_steps": cumulative_steps,
            "bad_epochs": bad_epochs, "history": history,
            "finished": finished, "base_sha256": sha256_file(args.pretrained),
            "parent_sha256": parent_sha, "budget": stage_budget(args),
            "code_check_only": args.code_check,
            "code_sha256": provenance(args)["code_sha256"]}


def atomic_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_stage(args, trainset, valset, device):
    torch.manual_seed(SEED)
    model, parent_sha, parent_steps = new_model(args, device)
    optimizer, scheduler = make_optimizer(model, args.stage)
    val_loader = DataLoader(valset, batch_size=1, shuffle=False)
    best_file = best_path(args)
    latest_file = latest_path(args)
    config = STAGE_CONFIG[args.stage]
    max_epochs = stage_budget(args)["max_epochs"]
    patience = config["patience"]
    min_delta = config["min_delta"]
    history, stage_steps, bad_epochs, first_epoch = [], 0, 0, 1
    cumulative_steps = parent_steps
    best = float("inf")
    if args.resume and best_file.exists() and not latest_file.exists():
        raise ValueError("Best checkpoint exists without resumable latest state")
    if args.resume and latest_file.exists():
        saved = torch.load(latest_file, map_location="cpu", weights_only=False)
        if (saved.get("stage") != args.stage or saved.get("arm") != ARM or
                saved.get("budget") != stage_budget(args) or
                saved.get("base_sha256") != sha256_file(args.pretrained) or
                saved.get("parent_sha256") != parent_sha or
                saved.get("code_sha256") != provenance(args)["code_sha256"]):
            raise ValueError("Resume state does not match stage/source/code")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        torch.set_rng_state(saved["torch_rng_state"])
        np.random.set_state(saved["numpy_rng_state"])
        random.setstate(saved["python_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng_state"])
        stage_steps = saved["stage_optimizer_steps"]
        cumulative_steps = saved["cumulative_optimizer_steps"]
        bad_epochs = saved["bad_epochs"]
        history = saved["history"]
        best = min(row["selection_metric"] for row in history)
        first_epoch = saved["epoch"] + 1
        if saved["finished"]:
            return history, saved
    else:
        configure_trainability(model, args.stage, 0,
                               stage_budget(args)["fine_dcn_only_warmup_epochs"])
        best, coarse, final = selection_metric(model, args.stage, val_loader, device)
        initial_record = {"epoch": 0, "selection_metric": best,
                        "val_coarse_epe_256px": coarse["epe_256px"],
                        "val_corr_top1": coarse["corr_top1"],
                        "val_final_epe_512px": final["final_flow_epe_512px"],
                        "train_loss": None, "stage_optimizer_steps": 0,
                        "cumulative_optimizer_steps": cumulative_steps,
                        "image_order_sha256": None,
                        "learning_rates": {group["group_name"]: group["lr"]
                                           for group in optimizer.param_groups}}
        if args.stage == "joint":
            initial_record["val_joint_diagnostics"] = joint_epoch_diagnostics(
                model, val_loader, device)
        history.append(initial_record)
        atomic_save(checkpoint_payload(model, optimizer, scheduler, args, 0,
                    best, 0, cumulative_steps, parent_sha, 0, history, False), best_file)
        print(f"{args.stage} epoch 0: coarse={coarse['epe_256px']:.4f}, "
              f"final={final['final_flow_epe_512px']:.4f}", flush=True)
    for epoch in range(first_epoch, max_epochs + 1):
        model.eval()
        selected, active = configure_trainability(
            model, args.stage, epoch,
            stage_budget(args)["fine_dcn_only_warmup_epochs"])
        generator = torch.Generator().manual_seed(SEED + epoch)
        loader = DataLoader(trainset, batch_size=1 if args.code_check else 2,
                            shuffle=True, generator=generator,
                            num_workers=args.workers,
                            pin_memory=(device.type == "cuda"))
        total, seen, order = 0., 0, hashlib.sha256()
        component_sums = {}
        epoch_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for batch in loader:
            for name in batch["name"]:
                order.update(name.encode("utf-8") + b"\n")
            optimizer.zero_grad(set_to_none=True)
            if args.stage == "coarse":
                loss, components = coarse_training_loss(model, batch, device)
            else:
                loss, components = dense_training_loss(
                    model, batch, device, coarse_auxiliary=args.stage == "joint")
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite {args.stage} training loss")
            loss.backward()
            if args.code_check and seen == 0:
                for name in active:
                    if not any(p.grad is not None and torch.isfinite(p.grad).all()
                               and p.grad.abs().sum() for p in selected[name]):
                        raise RuntimeError(f"{args.stage}: {name} receives no gradient")
                if args.stage in ("coarse", "joint"):
                    level4 = model.pyramid._modules["level_4"]
                    if not any(p.grad is not None and p.grad.abs().sum()
                               for p in level4.parameters()):
                        raise RuntimeError("Selected VGG level_4 receives no gradient")
                if any(p.grad is not None for name in FROZEN_FLOW_UPSAMPLERS
                       for p in getattr(model, name).parameters()):
                    raise RuntimeError("Pretrained flow upsampler received a gradient")
            torch.nn.utils.clip_grad_norm_(
                [p for name in active for p in selected[name]], 1.0)
            optimizer.step()
            count = len(batch["name"])
            total += float(loss.detach()) * count
            for key, value in components.items():
                component_sums[key] = component_sums.get(key, 0.) + value * count
            seen += count
            stage_steps += 1
            cumulative_steps += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - epoch_started
        metric, coarse, final = selection_metric(model, args.stage, val_loader, device)
        scheduler.step(metric)
        improved = metric < best - min_delta
        bad_epochs = 0 if improved else bad_epochs + 1
        record = {"epoch": epoch, "selection_metric": metric,
                  "val_coarse_epe_256px": coarse["epe_256px"],
                  "val_corr_top1": coarse["corr_top1"],
                  "val_final_epe_512px": final["final_flow_epe_512px"],
                  "train_loss": total / seen,
                  "train_components": {key: value / seen
                                       for key, value in component_sums.items()},
                  "active_parameter_groups": active,
                  "stage_optimizer_steps": stage_steps,
                  "cumulative_optimizer_steps": cumulative_steps,
                  "image_order_sha256": order.hexdigest(),
                  "learning_rates": {group["group_name"]: group["lr"]
                                     for group in optimizer.param_groups},
                  "train_seconds": seconds,
                  "train_pairs_per_second": seen / seconds,
                  "peak_allocated_mib": (torch.cuda.max_memory_allocated() / 2**20
                                         if device.type == "cuda" else None),
                  "peak_reserved_mib": (torch.cuda.max_memory_reserved() / 2**20
                                        if device.type == "cuda" else None)}
        if args.stage == "joint":
            record["val_joint_diagnostics"] = joint_epoch_diagnostics(
                model, val_loader, device)
        history.append(record)
        if improved:
            best = metric
            atomic_save(checkpoint_payload(model, optimizer, scheduler, args,
                        epoch, metric, stage_steps, cumulative_steps,
                        parent_sha, 0, history, False), best_file)
        print(f"{args.stage} epoch {epoch}/{max_epochs}: "
              f"coarse={coarse['epe_256px']:.4f}, "
              f"final={final['final_flow_epe_512px']:.4f}, "
              f"best={best:.4f}, patience={bad_epochs}/{patience}", flush=True)
        (args.output / f"history_{args.stage}.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8")
        finished = bad_epochs >= patience or epoch == max_epochs
        atomic_save(checkpoint_payload(model, optimizer, scheduler, args,
                    epoch, metric, stage_steps, cumulative_steps,
                    parent_sha, bad_epochs, history, finished), latest_file)
        if finished:
            break
    return history, torch.load(latest_file, map_location="cpu", weights_only=False)


@torch.no_grad()
def evaluate_selected(args, valset, device):
    payload = torch.load(best_path(args), map_location="cpu", weights_only=False)
    if (payload.get("stage") != args.stage or
            payload.get("budget") != stage_budget(args) or
            payload.get("base_sha256") != sha256_file(args.pretrained)):
        raise ValueError("Best checkpoint is incompatible with requested stage")
    model, parent_sha, _ = new_model(args, device)
    if payload.get("parent_sha256") != parent_sha:
        raise ValueError("Stage checkpoint was not trained from this parent")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    configure_trainability(model, args.stage, payload["epoch"],
                           stage_budget(args)["fine_dcn_only_warmup_epochs"])
    loader = DataLoader(valset, batch_size=1, shuffle=False)
    common, common_rows = evaluate_common(model, loader, device)
    stage_rows = evaluate_stages(model, loader, device, preview_dir=None)
    indexed = {row["name"]: row for row in common_rows}
    if indexed.keys() != {row["name"] for row in stage_rows}:
        raise RuntimeError("Common/stage evaluator pair mismatch")
    per_image = []
    for row in stage_rows:
        name = row["name"]
        if indexed[name]["valid_full"] != row["valid_full"]:
            raise RuntimeError("Common/stage evaluator mask mismatch")
        item = {"name": name, "valid_full": row["valid_full"],
                "inference_ms": indexed[name]["inference_ms"],
                "aepe_512px": indexed[name]["aepe_512px"],
                "cmr5_success": indexed[name]["aepe_512px"] < 5,
                "gt_displacement_p90_512px": row["gt_displacement_p90_512px"]}
        item.update({f"{stage}_epe_512px": row["stage_epe_512px"][stage]
                     for stage in STAGES})
        for group in ("gt_displacement_64+",
                      "outside_first_window_assigned_pixels",
                      "inside_first_window_assigned_pixels"):
            part = row["groups"][group]
            item[f"{group}_pixels"] = part["pixels"]
            item[f"{group}_final_epe_512px"] = (
                part["final128"] / part["pixels"] if part["pixels"] else None)
        item["first_window_outside_fraction"] = (
            row["search"]["local32"]["outside"]["queries"] /
            row["search"]["local32"]["valid_queries"])
        per_image.append(item)
    detail = {"groups": {group: combine_groups(stage_rows, group)
                         for group in ("all", "gt_displacement_64+",
                                       "outside_first_window_assigned_pixels",
                                       "inside_first_window_assigned_pixels")},
              "search": {level: combine_search(stage_rows, level)
                         for level in ("local32", "local64", "final128")}}
    result = {"stage": args.stage, "selected_epoch": payload["epoch"],
              "selected_metric": payload["selection_metric"],
              "checkpoint": {"path": str(best_path(args)),
                             "sha256": sha256_file(best_path(args))},
              "common": common, "detail": detail, "per_image": per_image,
              "total_parameters": sum(p.numel() for p in model.parameters()),
              "flow_upsamplers_frozen": all(
                  not p.requires_grad for name in FROZEN_FLOW_UPSAMPLERS
                  for p in getattr(model, name).parameters())}
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def save_stage_report(args, result, history, latest):
    report = {"stage": args.stage, "arm": ARM,
              "code_check_only": args.code_check,
              "test_pairs_accessed": False,
              "dataset": str(args.data_root.resolve()),
              "budget": stage_budget(args), "protocol": PROTOCOL,
              "base_sha256": sha256_file(args.pretrained),
              "history": history, "stopped_at_epoch": latest["epoch"],
              "stop_reason": ("early_stopping" if latest["bad_epochs"] >=
                              STAGE_CONFIG[args.stage]["patience"]
                              else "max_epochs"),
              "cumulative_optimizer_steps": latest["cumulative_optimizer_steps"],
              "result": result}
    (args.output / f"report_{args.stage}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    rows = result["per_image"]
    with (args.output / f"per_image_{args.stage}.csv").open(
            "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.stage == "joint":
        coarse_report = json.loads((args.output / "report_coarse.json").read_text(
            encoding="utf-8"))
        fine = json.loads((args.output / "report_fine.json").read_text(
            encoding="utf-8"))
        before = fine["result"]["common"]["final_flow_epe_512px"]
        after = result["common"]["final_flow_epe_512px"]
        selected = "joint" if after < before - STAGE_CONFIG["joint"]["min_delta"] else "fine"
        decision = {"selected_stage": selected,
                    "fine_final_epe_512px": before,
                    "joint_final_epe_512px": after,
                    "selected_checkpoint": str(best_path(args, selected)),
                    "reason": "joint improves validation final EPE beyond fixed min_delta"
                    if selected == "joint" else "retain stage-two best; joint did not improve validation final EPE"}
        (args.output / "selection.json").write_text(
            json.dumps(decision, indent=2), encoding="utf-8")
        reports = {"coarse": coarse_report, "fine": fine, "joint": report}
        stage_summary = {}
        for stage, item in reports.items():
            outcome = item["result"]
            rows = outcome["per_image"]
            stage_summary[stage] = {
                "selected_epoch": outcome["selected_epoch"],
                "coarse_epe_256px": outcome["common"]["coarse_epe_256px"],
                "four_level_epe_512px": outcome["detail"]["groups"]["all"]["epe_512px"],
                "final_epe_512px": outcome["common"]["final_flow_epe_512px"],
                "first_window_outside_fraction": outcome["detail"]["search"]["local32"]["outside"]["fraction_of_valid_queries"],
                "large_motion_final_epe_512px": outcome["detail"]["groups"]["gt_displacement_64+"]["epe_512px"]["final128"],
                "inference_ms_per_pair": outcome["common"]["inference_ms_mean_pair_median"],
                "peak_inference_allocated_mib": outcome["common"]["peak_allocated_mib"],
                "worst_five_pairs": sorted(
                    ({"name": row["name"],
                      "final_epe_512px": row["final128_epe_512px"]}
                     for row in rows),
                    key=lambda row: row["final_epe_512px"], reverse=True)[:5]}
        def paired_change(before_stage, after_stage):
            previous = {row["name"]: row for row in
                        reports[before_stage]["result"]["per_image"]}
            current = reports[after_stage]["result"]["per_image"]
            rows = [{"name": row["name"],
                     "final_epe_change_512px": row["final128_epe_512px"] -
                     previous[row["name"]]["final128_epe_512px"],
                     "coarse_epe_change_512px": row["coarse16_epe_512px"] -
                     previous[row["name"]]["coarse16_epe_512px"],
                     "window_outside_change": row["first_window_outside_fraction"] -
                     previous[row["name"]]["first_window_outside_fraction"]}
                    for row in current]
            return {"improved_pairs": sum(row["final_epe_change_512px"] < 0 for row in rows),
                    "worsened_pairs": sum(row["final_epe_change_512px"] > 0 for row in rows),
                    "per_image": rows}
        comparison = {"selected_stage": selected,
                      "stages": stage_summary,
                      "fine_vs_coarse": paired_change("coarse", "fine"),
                      "joint_vs_fine": paired_change("fine", "joint")}
        fine_rows = {row["name"]: row for row in fine["result"]["per_image"]}
        fine_coarse = stage_summary["fine"]["four_level_epe_512px"]["coarse16"]
        fine_window = stage_summary["fine"]["first_window_outside_fraction"]
        onset = {"final_epe": None, "coarse_epe": None,
                 "first_window_outside_fraction": None,
                 "per_image_final_epe": {name: None for name in fine_rows}}
        for entry in history:
            if entry["epoch"] == 0:
                continue
            diag = entry["val_joint_diagnostics"]
            if onset["final_epe"] is None and diag["final_epe_512px"] > before + 0.01:
                onset["final_epe"] = entry["epoch"]
            if onset["coarse_epe"] is None and diag["coarse_epe_512px"] > fine_coarse + 0.01:
                onset["coarse_epe"] = entry["epoch"]
            if (onset["first_window_outside_fraction"] is None and
                    diag["first_window_outside_fraction"] > fine_window):
                onset["first_window_outside_fraction"] = entry["epoch"]
            for name, metrics in diag["per_image"].items():
                if (onset["per_image_final_epe"][name] is None and
                        metrics["final_epe_512px"] >
                        fine_rows[name]["final128_epe_512px"] + 0.01):
                    onset["per_image_final_epe"][name] = entry["epoch"]
        comparison["joint_regression_onset_vs_fine_epoch"] = onset
        (args.output / "stage_comparison.json").write_text(
            json.dumps(comparison, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGE_CONFIG), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-split", choices=("val", "test"), default="val")
    parser.add_argument("--code-check", action="store_true")
    args = parser.parse_args()
    if args.workers < 0 or (args.code_check and (args.resume or args.eval_only)):
        parser.error("Invalid worker count or code-check mode")
    if args.eval_split == "test" and (not args.eval_only or args.code_check):
        parser.error("Test is only available with --eval-only after validation selection")
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.resume and not args.eval_only and best_path(args).exists():
        parser.error("This stage already has a best checkpoint; use --resume")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.eval_split == "test":
        decision_path = args.output / "selection.json"
        if not decision_path.exists():
            parser.error("Validation selection.json must exist before test evaluation")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if args.stage != decision["selected_stage"]:
            parser.error("Test evaluation must use the validation-selected stage")
    valset = RoadScenePairs(args.data_root, args.eval_split)
    if args.eval_split == "test" and len(valset) != 22:
        parser.error("Expected 22 locked RoadScene test pairs")
    if args.eval_split == "val" and not args.code_check and len(valset) != 23:
        parser.error("Expected 23 RoadScene validation pairs")
    if args.code_check:
        valset = Subset(valset, range(1))
    geometry = geometry_check(device)
    if args.stage == "coarse":
        identity = zero_gate_check(args, valset, device)
    else:
        identity = {"parent_stage": source_stage(args.stage),
                    "parent_sha256": sha256_file(best_path(args, source_stage(args.stage)))}
    if args.eval_only:
        result = evaluate_selected(args, valset, device)
        if args.eval_split == "test":
            output = {"split": "test", "protocol": PROTOCOL,
                      "validation_selection": decision,
                      "test_selection_used": False,
                      "result": result}
            (args.output / "report_test_selected.json").write_text(
                json.dumps(output, indent=2), encoding="utf-8")
            with (args.output / "per_image_test_selected.csv").open(
                    "w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(result["per_image"][0]))
                writer.writeheader()
                writer.writerows(result["per_image"])
        print(json.dumps({"stage": args.stage, "selected_epoch": result["selected_epoch"],
                          "final_epe_512px": result["common"]["final_flow_epe_512px"],
                          "coarse_epe_256px": result["common"]["coarse_epe_256px"]},
                         indent=2), flush=True)
        return
    trainset = RoadScenePairs(args.data_root, "train")
    if not args.code_check and len(trainset) != 176:
        parser.error("Expected 176 RoadScene training pairs")
    if args.code_check:
        trainset = Subset(trainset, range(2))
    history, latest = train_stage(args, trainset, valset, device)
    result = evaluate_selected(args, valset, device)
    report = save_stage_report(args, result, history, latest)
    report["geometry_check"] = geometry
    report["initialization_check"] = identity
    (args.output / f"report_{args.stage}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"stage": args.stage, "code_check_only": args.code_check,
                      "selected_epoch": result["selected_epoch"],
                      "stopped_at_epoch": latest["epoch"],
                      "coarse_epe_256px": result["common"]["coarse_epe_256px"],
                      "final_epe_512px": result["common"]["final_flow_epe_512px"],
                      "cumulative_steps": latest["cumulative_optimizer_steps"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
