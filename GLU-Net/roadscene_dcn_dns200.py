"""Matched 200-epoch joint RoadScene SA/CA+DCN versus DNS+SA/CA+DCN study.

Both arms use a single 32x32 deformable residual update and train/validate
only. This script cannot open the locked RoadScene test split.
"""

import argparse
import csv
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from datasets.roadscene import RoadScenePairs
from models.our_models.GLUNet import GLUNet_model
from roadscene_coarse import load_base_weights, sha256_file
from roadscene_compare import PROTOCOL, evaluate_common
from roadscene_dcn import LOCAL_NAMES, geometry_check, select_metric
from roadscene_refinement_audit import (STAGES, combine_groups,
                                        combine_search, evaluate as evaluate_stages)


SEED = 2026
COARSE_LOSS_WEIGHT = 0.25
ARMS = {"saca_dcn": "attention", "dns_saca_dcn": "dns_attention"}
GROUPS = ("all", "gt_displacement_64+",
          "outside_first_window_assigned_pixels",
          "inside_first_window_assigned_pixels")
SEARCH_LEVELS = ("local32", "local64", "final128")


def source_path(args, arm):
    return args.attention_checkpoint if arm == "saca_dcn" else args.dns_attention_checkpoint


def provenance(args):
    root = Path(__file__).resolve().parent
    names = ("roadscene_dcn_dns200.py", "roadscene_dcn.py",
             "roadscene_compare.py", "roadscene_refinement_audit.py",
             "roadscene_coarse.py", "roadscene_metrics.py",
             "datasets/roadscene.py", "models/our_models/GLUNet.py",
             "models/our_models/local_dcn.py", "models/our_models/coarse_dns.py",
             "models/our_models/coarse_attention.py",
             "models/correlation/correlation.py")
    return {"base_sha256": sha256_file(args.pretrained),
            "source_sha256": {arm: sha256_file(source_path(args, arm)) for arm in ARMS},
            "code_sha256": {name: sha256_file(root / name) for name in names}}


def model_from_weights(args, arm, device, selected=None, allow_smoke=False):
    dns = arm == "dns_saca_dcn"
    model = GLUNet_model(evaluation=False, pyramid_type="VGG",
                        cyclic_consistency=True, backbone_pretrained=False,
                        coarse_attention=True, coarse_dns=dns, local_dcn_steps=1)
    load_base_weights(model, args.pretrained)
    source = torch.load(source_path(args, arm), map_location="cpu", weights_only=False)
    if source.get("arm") != ARMS[arm] or source.get("base_sha256") != sha256_file(args.pretrained):
        raise ValueError(f"{arm}: coarse checkpoint arm/base weight mismatch")
    if source.get("budget", {}).get("code_check"):
        raise ValueError(f"{arm}: smoke-check coarse checkpoint is not a training source")
    model.decoder4.load_state_dict(source["decoder4_state_dict"])
    model.coarse_attention.load_state_dict(source["attention_state_dict"])
    if dns:
        model.coarse_dns.load_state_dict(source["dns_state_dict"])
    if selected is not None:
        payload = torch.load(selected, map_location="cpu", weights_only=False)
        if (payload.get("arm") != arm or
                payload.get("base_sha256") != sha256_file(args.pretrained) or
                payload.get("source_sha256") != sha256_file(source_path(args, arm)) or
                (payload.get("code_check_only") and not allow_smoke)):
            raise ValueError(f"{arm}: DCN checkpoint or source mismatch")
        for name in LOCAL_NAMES:
            getattr(model, name).load_state_dict(payload["local_state_dicts"][name])
        model.local_dcn32.load_state_dict(payload["dcn_state_dict"])
        model.decoder4.load_state_dict(payload["decoder4_state_dict"])
        model.coarse_attention.load_state_dict(payload["attention_state_dict"])
        if dns:
            model.coarse_dns.load_state_dict(payload["dns_state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.decoder4.parameters():
        parameter.requires_grad_(True)
    for parameter in model.coarse_attention.parameters():
        parameter.requires_grad_(True)
    if dns:
        for parameter in model.coarse_dns.parameters():
            parameter.requires_grad_(True)
    for name in LOCAL_NAMES:
        for parameter in getattr(model, name).parameters():
            parameter.requires_grad_(True)
    for parameter in model.local_dcn32.parameters():
        parameter.requires_grad_(True)
    # Keep the pretrained BN running statistics fixed in both arms.
    return model.to(device).eval()


@torch.no_grad()
def initialization_check(args, valset, device):
    batch = next(iter(DataLoader(valset, batch_size=1, shuffle=False)))
    outcomes = {}
    for arm in ARMS:
        model = model_from_weights(args, arm, device)
        source, target, source256, target256, *_ = model.pre_process_data(
            batch["source_image"].to(device), batch["target_image"].to(device), device=device)
        # A disabled DCN is the corresponding trained coarse network, with
        # identical decoder and later local weights.
        model.local_dcn_steps = 0
        before256, before512 = model(target, source, target256, source256)
        model.local_dcn_steps = 1
        after256, after512, trace = model(target, source, target256, source256,
                                         return_dcn_trace=True)
        differences = [float((a - b).abs().max()) for a, b in
                       zip((*before256, *before512), (*after256, *after512))]
        offset = float(trace[0]["offset_grid"].abs().max())
        # The legacy final upsampling may exhibit small nondeterministic CUDA
        # differences; require each preceding stage to agree exactly.
        if max(differences[:3]) > 1e-5 or differences[3] > 0.003 or offset != 0:
            raise RuntimeError(f"{arm}: zero-initialized DCN changed starting flow: {differences}")
        outcomes[arm] = {"per_stage_max_abs_256_or_512px": dict(zip(STAGES, differences)),
                         "initial_offset_grid_max_abs": offset}
        del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return outcomes


def budget(args):
    return {"epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "seed": SEED, "dcn_steps": 1,
            "loss": "masked sqrt(dx^2+dy^2+0.01) at 512px",
            "coarse_loss_weight": COARSE_LOSS_WEIGHT,
            "train_pairs": 2 if args.code_check else 176,
            "val_pairs": 1 if args.code_check else 23}


def local_state(model):
    return {name: getattr(model, name).state_dict() for name in LOCAL_NAMES}


def selected_payload(model, arm, epoch, metric, steps, args):
    return {"arm": arm, "epoch": epoch, "validation_final_epe_512px": metric,
            "optimizer_steps": steps, "base_sha256": sha256_file(args.pretrained),
            "source_sha256": sha256_file(source_path(args, arm)),
            "source_arm": ARMS[arm], "budget": budget(args),
            "code_check_only": args.code_check,
            "local_state_dicts": local_state(model),
            "dcn_state_dict": model.local_dcn32.state_dict(),
            "decoder4_state_dict": model.decoder4.state_dict(),
            "attention_state_dict": model.coarse_attention.state_dict(),
            "dns_state_dict": (model.coarse_dns.state_dict()
                               if arm == "dns_saca_dcn" else None)}


def atomic_save(payload, destination):
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def joint_loss(model, batch, device):
    source, target, source256, target256, *_ = model.pre_process_data(
        batch["source_image"].to(device), batch["target_image"].to(device), device=device)
    flows256, flows512 = model(target, source, target256, source256)
    truth = batch["flow_map"].to(device).float()
    valid = batch["correspondence_mask"].to(device).bool()
    if not valid.any():
        raise ValueError("Training batch has no valid GT flow pixels")
    final = F.interpolate(flows512[-1], size=truth.shape[-2:],
                          mode="bilinear", align_corners=False)
    # flow4 is in 256-image-pixel units even though its feature grid is 16x16.
    # Resizing only changes sampling; multiplying by 2 changes vector units.
    coarse = F.interpolate(flows256[0], size=truth.shape[-2:],
                           mode="bilinear", align_corners=False) * 2
    def masked_error(prediction):
        error = torch.sqrt((prediction - truth).square().sum(dim=1) + 0.01)
        return error[valid].mean()
    fine_loss = masked_error(final)
    coarse_loss = masked_error(coarse)
    return fine_loss + COARSE_LOSS_WEIGHT * coarse_loss, fine_loss, coarse_loss


def train_arm(args, arm, trainset, val_loader, device):
    torch.manual_seed(SEED)
    model = model_from_weights(args, arm, device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    latest = args.output / f"latest_{arm}.pth"
    best_path = args.output / f"best_{arm}.pth"
    if args.resume and best_path.exists() and not latest.exists():
        raise ValueError(f"{arm}: selected checkpoint exists but resumable state is missing")
    history, best, steps, first_epoch = [], float("inf"), 0, 1
    elapsed_prior, peak_allocated, peak_reserved = 0., 0., 0.
    if args.resume and latest.exists():
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        if (payload.get("arm") != arm or payload.get("budget") != budget(args) or
                payload.get("base_sha256") != sha256_file(args.pretrained) or
                payload.get("source_sha256") != sha256_file(source_path(args, arm)) or
                payload.get("code_sha256") != provenance(args)["code_sha256"]):
            raise ValueError(f"{arm}: incompatible resume state")
        for name in LOCAL_NAMES:
            getattr(model, name).load_state_dict(payload["local_state_dicts"][name])
        model.local_dcn32.load_state_dict(payload["dcn_state_dict"])
        model.decoder4.load_state_dict(payload["decoder4_state_dict"])
        model.coarse_attention.load_state_dict(payload["attention_state_dict"])
        if arm == "dns_saca_dcn":
            model.coarse_dns.load_state_dict(payload["dns_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        history, best, steps = payload["history"], payload["best_metric"], payload["optimizer_steps"]
        first_epoch = payload["completed_epoch"] + 1
        elapsed_prior = payload["training_seconds"]
        peak_allocated = payload["training_peak_allocated_mib"]
        peak_reserved = payload["training_peak_reserved_mib"]
        torch.set_rng_state(payload["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(payload["cuda_rng_state"])
        if first_epoch <= args.epochs and not best_path.exists():
            raise ValueError(f"{arm}: resume state lacks selected best checkpoint")
    started = time.perf_counter()
    generator = torch.Generator()
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                              generator=generator, num_workers=args.workers,
                              pin_memory=(device.type == "cuda"),
                              persistent_workers=args.workers > 0)
    for epoch in range(first_epoch, args.epochs + 1):
        model.eval()
        generator.manual_seed(SEED + epoch)
        order_digest = hashlib.sha256()
        total, fine_total, coarse_total, count = 0., 0., 0., 0
        epoch_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for batch in train_loader:
            for name in batch["name"]:
                order_digest.update(name.encode("utf-8") + b"\n")
            optimizer.zero_grad(set_to_none=True)
            loss, fine_part, coarse_part = joint_loss(model, batch, device)
            loss.backward()
            if args.code_check and steps == 0:
                grad = model.local_dcn32.delta.weight.grad
                if grad is None or not torch.isfinite(grad).all() or not grad.abs().sum():
                    raise RuntimeError(f"{arm}: final-flow loss does not train DCN residual")
                coarse_grad = model.decoder4.final.weight.grad
                attention_grads = [p.grad for p in model.coarse_attention.parameters()]
                if (coarse_grad is None or not torch.isfinite(coarse_grad).all() or
                        not coarse_grad.abs().sum() or not any(
                            g is not None and torch.isfinite(g).all() and g.abs().sum()
                            for g in attention_grads)):
                    raise RuntimeError(f"{arm}: final-flow loss does not train coarse flow/SA-CA")
                if arm == "dns_saca_dcn" and not any(
                    p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()
                    for p in model.coarse_dns.parameters()):
                    raise RuntimeError("DNS+SA/CA coarse branch does not receive gradients")
            if args.code_check and steps == 1:
                grad = model.local_dcn32.offset.weight.grad
                if grad is None or not torch.isfinite(grad).all() or not grad.abs().sum():
                    raise RuntimeError(f"{arm}: DCN offsets do not receive gradients")
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(batch["name"])
            fine_total += float(fine_part.detach()) * len(batch["name"])
            coarse_total += float(coarse_part.detach()) * len(batch["name"])
            count += len(batch["name"])
            steps += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated() / 2**20)
            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved() / 2**20)
        seconds = time.perf_counter() - epoch_started
        metric = select_metric(model, val_loader, device)
        record = {"epoch": epoch, "train_loss": total / count,
                  "train_fine_loss_512px": fine_total / count,
                  "train_coarse_loss_512px": coarse_total / count,
                  "val_final_epe_512px": metric, "optimizer_steps": steps,
                  "image_order_sha256": order_digest.hexdigest(),
                  "train_seconds": seconds, "train_pairs_per_second": count / seconds,
                  "peak_allocated_mib": peak_allocated, "peak_reserved_mib": peak_reserved}
        history.append(record)
        if metric < best:
            best = metric
            atomic_save(selected_payload(model, arm, epoch, metric, steps, args), best_path)
        print(f"{arm} epoch {epoch}/{args.epochs}: final val EPE={metric:.4f}, "
              f"train={count / seconds:.2f} pairs/s, peak allocated={peak_allocated:.0f} MiB",
              flush=True)
        (args.output / f"history_{arm}.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8")
        if epoch % args.save_every == 0 or epoch == args.epochs:
            atomic_save({**selected_payload(model, arm, epoch, metric, steps, args),
                         "optimizer_state_dict": optimizer.state_dict(),
                         "completed_epoch": epoch, "best_metric": best,
                         "history": history,
                         "training_seconds": elapsed_prior + time.perf_counter() - started,
                         "training_peak_allocated_mib": peak_allocated,
                         "training_peak_reserved_mib": peak_reserved,
                         "torch_rng_state": torch.get_rng_state(),
                         "cuda_rng_state": (torch.cuda.get_rng_state() if device.type == "cuda"
                                            else None),
                         "code_sha256": provenance(args)["code_sha256"]}, latest)
    return {"history": history, "optimizer_steps": steps,
            "training_seconds": elapsed_prior + time.perf_counter() - started,
            "training_peak_allocated_mib": peak_allocated,
            "training_peak_reserved_mib": peak_reserved}


def validate_arm(args, arm, valset, device):
    path = args.output / f"best_{arm}.pth"
    model = model_from_weights(args, arm, device, selected=path,
                               allow_smoke=args.code_check)
    loader = DataLoader(valset, batch_size=1, shuffle=False)
    common, common_rows = evaluate_common(model, loader, device)
    stages = evaluate_stages(model, loader, device, preview_dir=None)
    grouped = {group: combine_groups(stages, group) for group in GROUPS}
    searched = {level: combine_search(stages, level) for level in SEARCH_LEVELS}
    indexed = {row["name"]: row for row in common_rows}
    rows = []
    for row in stages:
        name = row["name"]
        if indexed[name]["valid_full"] != row["valid_full"]:
            raise RuntimeError("Common and stage metrics disagree on valid GT mask")
        item = {"name": name, "valid_full": row["valid_full"],
                "inference_ms": indexed[name]["inference_ms"],
                "peak_allocated_mib": indexed[name]["peak_allocated_mib"],
                "cmr5_success": indexed[name]["aepe_512px"] < 5,
                "first_window_outside_fraction":
                    row["search"]["local32"]["outside"]["queries"] /
                    row["search"]["local32"]["valid_queries"]}
        item.update({f"{stage}_epe_512px": row["stage_epe_512px"][stage]
                     for stage in STAGES})
        for group in GROUPS[1:]:
            part = row["groups"][group]
            item[f"{group}_pixels"] = part["pixels"]
            item[f"{group}_final_epe_512px"] = (
                part["final128"] / part["pixels"] if part["pixels"] else None)
        rows.append(item)
    return {"common": common, "stage_summary": {"groups": grouped, "search": searched},
            "per_image": rows, "checkpoint": {"path": str(path), "sha256": sha256_file(path)},
            "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_parameter_count": sum(p.numel() for p in model.parameters())}


def paired_analysis(results):
    left = {row["name"]: row for row in results["saca_dcn"]["per_image"]}
    right = {row["name"]: row for row in results["dns_saca_dcn"]["per_image"]}
    if left.keys() != right.keys() or any(left[name]["valid_full"] != right[name]["valid_full"]
                                           for name in left):
        raise RuntimeError("Paired arms used different validation pairs or valid masks")
    pairs = []
    for name in left:
        a, b = left[name], right[name]
        pairs.append({"name": name,
                      "saca_dcn_final_epe_512px": a["final128_epe_512px"],
                      "dns_saca_dcn_final_epe_512px": b["final128_epe_512px"],
                      "dns_minus_saca_final_epe_512px":
                          b["final128_epe_512px"] - a["final128_epe_512px"],
                      "saca_dcn_first_window_outside_fraction":
                          a["first_window_outside_fraction"],
                      "dns_saca_dcn_first_window_outside_fraction":
                          b["first_window_outside_fraction"],
                      "saca_dcn_cmr5_success": a["cmr5_success"],
                      "dns_saca_dcn_cmr5_success": b["cmr5_success"]})
    delta = [row["dns_minus_saca_final_epe_512px"] for row in pairs]
    return {"improved_pairs": sum(value < 0 for value in delta),
            "worsened_pairs": sum(value > 0 for value in delta),
            "median_pair_delta_512px": float(np.median(delta)),
            "pairs": pairs}


def batch_preflight(args, device):
    """One train-only update per arm to measure batch fit; no checkpoint saved."""
    trainset = RoadScenePairs(args.data_root, "train")
    if len(trainset) != 176:
        raise ValueError("Batch preflight expects all 176 RoadScene training pairs")
    batch = next(iter(DataLoader(trainset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.workers,
                                 pin_memory=(device.type == "cuda"))))
    results = {}
    for arm in ARMS:
        gc.collect()
        torch.manual_seed(SEED)
        model = model_from_weights(args, arm, device)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                       lr=args.lr)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        def update():
            optimizer.zero_grad(set_to_none=True)
            loss, fine, coarse = joint_loss(model, batch, device)
            loss.backward()
            optimizer.step()
            return loss, fine, coarse
        # The first local-correlation CUDA/CuPy invocation may compile a kernel.
        # Exclude it from the throughput measurement for both arms.
        loss, fine, coarse = update()
        del loss, fine, coarse
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        allocated_before = (torch.cuda.memory_allocated() / 2**20
                            if device.type == "cuda" else None)
        timings = []
        for _ in range(3):
            started = time.perf_counter()
            loss, fine, coarse = update()
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - started)
        seconds = float(np.median(timings))
        results[arm] = {"batch_size": len(batch["name"]),
                        "warmup_updates": 1, "timed_updates": 3,
                        "median_step_seconds_excluding_data_load": seconds,
                        "training_pairs_per_second_excluding_data_load":
                            len(batch["name"]) / seconds,
                        "total_loss": float(loss.detach()),
                        "fine_loss": float(fine.detach()),
                        "coarse_loss": float(coarse.detach()),
                        "allocated_before_timed_mib": allocated_before,
                        "peak_allocated_mib": (torch.cuda.max_memory_allocated() / 2**20
                                               if device.type == "cuda" else None),
                        "peak_reserved_mib": (torch.cuda.max_memory_reserved() / 2**20
                                              if device.type == "cuda" else None)}
        del update, model, optimizer, loss, fine, coarse
    return {"code_check_only": True, "training_split_only": True,
            "batch_size": args.batch_size,
            "gpu_total_mib": (torch.cuda.get_device_properties(device).total_memory / 2**20
                              if device.type == "cuda" else None),
            "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--attention-checkpoint", type=Path, required=True)
    parser.add_argument("--dns-attention-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--code-check", action="store_true")
    parser.add_argument("--batch-preflight", action="store_true",
                        help="one train-only forward/backward/update per arm at batch 4")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.workers < 0 or args.save_every < 1:
        parser.error("Epochs, batch size, LR and save interval must be positive")
    if args.code_check and (args.resume or args.eval_only or args.batch_preflight):
        parser.error("Code check cannot resume, preflight or evaluate")
    if args.batch_preflight and (args.resume or args.eval_only):
        parser.error("Batch preflight cannot resume or evaluate")
    if args.code_check:
        args.epochs, args.batch_size = 1, 1
    elif (args.epochs, args.batch_size, args.lr) != (200, 4, 1e-4):
        parser.error("Formal comparison is fixed at 200 epochs, batch 4, lr 1e-4")
    if not args.resume and not args.eval_only and any(args.output.glob("best_*.pth")):
        parser.error("Output already has checkpoints; use --resume or a new directory")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.batch_preflight:
        result = batch_preflight(args, device)
        (args.output / "batch_preflight.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        return
    geometry = geometry_check(device)
    valset = RoadScenePairs(args.data_root, "val")
    if not args.code_check and len(valset) != 23:
        parser.error("Expected all 23 validation pairs")
    if args.code_check:
        valset = Subset(valset, range(1))
    identity = initialization_check(args, valset, device)
    trainset = None
    if not args.eval_only:
        trainset = RoadScenePairs(args.data_root, "train")
        if not args.code_check and len(trainset) != 176:
            parser.error("Expected all 176 training pairs")
        if args.code_check:
            trainset = Subset(trainset, range(2))
    hashes = provenance(args)
    source_meta = {}
    for arm in ARMS:
        payload = torch.load(source_path(args, arm), map_location="cpu", weights_only=False)
        source_meta[arm] = {"path": str(source_path(args, arm)),
                            "sha256": hashes["source_sha256"][arm],
                            "selected_epoch": payload.get("epoch"),
                            "source_budget": payload.get("budget")}
    report = {"code_check_only": args.code_check, "split": "train/val only",
              "test_pairs_accessed": False, "dataset": str(args.data_root.resolve()),
              "train_pairs": len(trainset) if trainset is not None else 176,
              "val_pairs": len(valset), "device": str(device),
              "gpu_name": torch.cuda.get_device_name() if device.type == "cuda" else None,
              "gpu_total_mib": (torch.cuda.get_device_properties(device).total_memory / 2**20
                                if device.type == "cuda" else None),
              "budget": budget(args), "source_checkpoints": source_meta,
              "provenance": hashes, "protocol": PROTOCOL,
              "geometry_check": geometry, "initialization_check": identity,
              "trainable": "joint decoder4 + SA/CA (+ DNS in combination arm) + decoder3 + existing 32-grid dilated refiners + one 32-grid DCN; VGG, later local stages and pretrained flow upsamplers frozen in both arms",
              "training_loss": "L=mean_valid sqrt(final512-GT squared norm+0.01) + 0.25*mean_valid sqrt(upsampled_coarse256*2-GT squared norm+0.01); both terms on same 512px GT/mask",
              "checkpoint_rule": "lowest valid-pixel-weighted final 512px validation EPE within each arm",
              "training": {}, "results": {}}
    if not args.eval_only:
        loader = DataLoader(valset, batch_size=1, shuffle=False)
        for arm in ARMS:
            report["training"][arm] = train_arm(args, arm, trainset, loader, device)
            if arm == "dns_saca_dcn":
                first = report["training"]["saca_dcn"]
                second = report["training"][arm]
                if first["optimizer_steps"] != second["optimizer_steps"] or [
                    row["image_order_sha256"] for row in first["history"]] != [
                    row["image_order_sha256"] for row in second["history"]]:
                    raise RuntimeError("Arms received different optimization steps or image order")
            (args.output / "progress.json").write_text(json.dumps(report, indent=2),
                                                        encoding="utf-8")
            if device.type == "cuda":
                torch.cuda.empty_cache()
    for arm in ARMS:
        report["results"][arm] = validate_arm(args, arm, valset, device)
        (args.output / "progress.json").write_text(json.dumps(report, indent=2),
                                                    encoding="utf-8")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report["paired"] = paired_analysis(report["results"])
    rows = [{"arm": arm, **item} for arm, result in report["results"].items()
            for item in result["per_image"]]
    with (args.output / "per_image.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(dict.fromkeys(
            key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2),
                                                  encoding="utf-8")
    print(json.dumps({"code_check_only": args.code_check,
                      "val_final_epe_512px": {
                          arm: result["common"]["final_flow_epe_512px"]
                          for arm, result in report["results"].items()},
                      "improved_pairs_dns_vs_saca": report["paired"]["improved_pairs"],
                      "optimizer_steps_each": {arm: record["optimizer_steps"]
                                               for arm, record in report["training"].items()}},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
