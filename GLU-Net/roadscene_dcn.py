"""Matched, staged RoadScene fine-refinement study; train/val only.

Phase 1 compares the selected SA/CA network after the same local fine-tuning
budget with one zero-initialized 32-grid deformable residual update. Phase 2
is gated by phase-1 validation and adds two shared-weight update iterations.
The 22 test pairs are never opened by this script.
"""

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from datasets.roadscene import RoadScenePairs
from models.our_models.GLUNet import GLUNet_model
from models.our_models.mod import warp
from roadscene_coarse import load_base_weights, sha256_file
from roadscene_compare import PROTOCOL, evaluate_common
from roadscene_refinement_audit import (BIN_NAMES, STAGES, combine_groups,
                                        combine_search, evaluate as evaluate_stages)


SEED = 2026
STEPS = {"attention": 0, "dcn_single": 1, "dcn_iter2": 2}
LOCAL_NAMES = ("decoder3", "dc_conv1", "dc_conv2", "dc_conv3", "dc_conv4",
               "dc_conv5", "dc_conv6", "dc_conv7")


def model_from_weights(base_path, attention_path, arm, device, checkpoint=None):
    model = GLUNet_model(evaluation=False, pyramid_type="VGG",
                        cyclic_consistency=True, backbone_pretrained=False,
                        coarse_attention=True, local_dcn_steps=STEPS[arm])
    load_base_weights(model, base_path)
    selected = torch.load(attention_path, map_location="cpu", weights_only=False)
    if selected.get("arm") != "attention":
        raise ValueError("Expected selected SA/CA attention checkpoint")
    if selected.get("base_sha256") != sha256_file(base_path):
        raise ValueError("Selected SA/CA was trained from another base weight")
    model.decoder4.load_state_dict(selected["decoder4_state_dict"])
    model.coarse_attention.load_state_dict(selected["attention_state_dict"])
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (payload["arm"] != arm or
                payload["base_sha256"] != sha256_file(base_path) or
                payload["attention_sha256"] != sha256_file(attention_path) or
                payload.get("code_check_only", True)):
            raise ValueError("Incompatible or smoke-check DCN checkpoint")
        for name, state in payload["local_state_dicts"].items():
            getattr(model, name).load_state_dict(state)
        if model.local_dcn32 is not None:
            model.local_dcn32.load_state_dict(payload["dcn_state_dict"])
    for param in model.parameters():
        param.requires_grad_(False)
    for name in LOCAL_NAMES:
        for param in getattr(model, name).parameters():
            param.requires_grad_(True)
    if model.local_dcn32 is not None:
        for param in model.local_dcn32.parameters():
            param.requires_grad_(True)
    return model.to(device).eval()


def geometry_check(device):
    """A +2 feature-pixel target-to-source translation is +16 at 256px."""
    from models.our_models.local_dcn import LocalDeformableFlowUpdate
    torch.manual_seed(SEED)
    target = torch.randn(1, 8, 32, 32, device=device)
    source = torch.zeros_like(target)
    source[..., 2:] = target[..., :-2]
    shifted = warp(source, torch.cat((torch.full((1, 1, 32, 32), 2., device=device),
                                      torch.zeros((1, 1, 32, 32), device=device)), 1))
    translation_error = float((shifted[..., :-2] - target[..., :-2]).abs().max())
    module = LocalDeformableFlowUpdate().to(device)
    flow = torch.zeros(1, 2, 32, 32, device=device)
    flow[:, 0] = 16.0
    output, info = module(torch.zeros(1, 81, 32, 32, device=device),
                          target, shifted, flow, 256, 256)
    identity_error = float((output - flow).abs().max())
    offset_error = float(info["offset_grid"].abs().max())
    inside_count = int(info["source_inside"].sum())
    with torch.no_grad():
        module.offset.bias.fill_(math.atanh(0.5))  # 1 grid pixel in every offset
        module.delta.bias[0] = math.atanh(0.25)   # +1 x grid pixel per update
    once, nonzero = module(torch.zeros(1, 81, 32, 32, device=device),
                           target, shifted, flow, 256, 256)
    twice, _ = module(torch.zeros(1, 81, 32, 32, device=device),
                      target, shifted, once, 256, 256)
    update_error = max(float((once[:, 0] - 24).abs().max()),
                       float((twice[:, 0] - 32).abs().max()),
                       float(once[:, 1].abs().max()),
                       float((nonzero["offset_grid"] - 1).abs().max()))
    if (translation_error > 2e-5 or identity_error != 0 or
            offset_error != 0 or inside_count != 32 * 30 or update_error > 1e-5):
        raise RuntimeError("DCN XY/scale/offset identity check failed: "
                           f"warp={translation_error}, flow={identity_error}, "
                           f"offset={offset_error}, inside={inside_count}, "
                           f"nonzero_update={update_error}")
    return {"direction": "visible target -> IR source",
            "translation_grid_px": [2, 0], "translation_image256_px": [16, 0],
            "warp_max_abs": translation_error, "identity_flow_max_abs": identity_error,
            "initial_offset_grid_max_abs": offset_error,
            "nonzero_two_updates_max_abs": update_error,
            "predicted_source_inside_queries": inside_count}


@torch.no_grad()
def initialization_check(args, valset, device):
    """Every initial DCN round must leave all four SA/CA flows unchanged."""
    batch = next(iter(DataLoader(valset, batch_size=1, shuffle=False)))
    baseline = model_from_weights(args.pretrained, args.attention_checkpoint,
                                  "attention", device)
    source, target, source256, target256, *_ = baseline.pre_process_data(
        batch["source_image"].to(device), batch["target_image"].to(device), device=device)
    base256, base512 = baseline(target, source, target256, source256)
    repeat256, repeat512 = baseline(target, source, target256, source256)
    repeat_diffs = [float((a - b).abs().max()) for a, b in
                    zip((*base256, *base512), (*repeat256, *repeat512))]
    values = {}
    for arm in ("dcn_single", "dcn_iter2"):
        variant = model_from_weights(args.pretrained, args.attention_checkpoint,
                                     arm, device)
        out256, out512, trace = variant(target, source, target256, source256,
                                        return_dcn_trace=True)
        stage_diffs = [float((before - after).abs().max())
                       for before, after in zip((*base256, *base512),
                                                (*out256, *out512))]
        stage_diff = max(stage_diffs)
        trace_diff = max(float((step["flow_256px"] - out256[1]).abs().max())
                         for step in trace)
        offset_diff = max(float(step["offset_grid"].abs().max())
                          for step in trace)
        if any(diff > (max(0.003, repeat_diffs[i] * 2)
                       if i == 3 else 1e-5) for i, diff in
               enumerate(stage_diffs)) or trace_diff != 0 or offset_diff != 0:
            raise RuntimeError(f"{arm} initial flow/offset differs from SA/CA: "
                               f"stages={stage_diffs}, repeat={repeat_diffs}, "
                               f"trace={trace_diff}, "
                               f"offset={offset_diff}")
        values[arm] = {"all_four_stage_max_abs": stage_diff,
                       "per_stage_max_abs": dict(zip(STAGES, stage_diffs)),
                       "baseline_repeat_per_stage_max_abs": dict(zip(STAGES, repeat_diffs)),
                       "each_round_flow_max_abs": trace_diff,
                       "each_round_offset_grid_max_abs": offset_diff}
        del variant
    del baseline
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return values


def final_loss(model, batch, device):
    source, target, source256, target256, *_ = model.pre_process_data(
        batch["source_image"].to(device), batch["target_image"].to(device), device=device)
    flows256, flows512 = model(target, source, target256, source256)
    truth = batch["flow_map"].to(device).float()
    valid = batch["correspondence_mask"].to(device).bool()
    prediction = F.interpolate(flows512[-1], size=truth.shape[-2:],
                               mode="bilinear", align_corners=False)
    error = torch.sqrt((prediction - truth).square().sum(dim=1) + 0.01)
    if not valid.any():
        raise ValueError("Training pair contains no valid GT flow pixels")
    return error[valid].mean(), prediction


@torch.no_grad()
def select_metric(model, loader, device):
    model.eval()
    total, count = 0., 0
    for batch in loader:
        source, target, source256, target256, *_ = model.pre_process_data(
            batch["source_image"].to(device), batch["target_image"].to(device), device=device)
        _, flows512 = model(target, source, target256, source256)
        truth = batch["flow_map"].to(device).float()
        valid = batch["correspondence_mask"].to(device).bool()
        pred = F.interpolate(flows512[-1], size=truth.shape[-2:],
                             mode="bilinear", align_corners=False)
        error = torch.linalg.vector_norm(pred - truth, dim=1)
        total += float(error[valid].sum())
        count += int(valid.sum())
    return total / count


def save_checkpoint(path, model, arm, epoch, metric, args, steps_done):
    torch.save({"arm": arm, "epoch": epoch, "validation_final_epe_512px": metric,
                "base_sha256": sha256_file(args.pretrained),
                "attention_sha256": sha256_file(args.attention_checkpoint),
                "local_state_dicts": {name: getattr(model, name).state_dict()
                                      for name in LOCAL_NAMES},
                "dcn_state_dict": (model.local_dcn32.state_dict()
                                   if model.local_dcn32 is not None else None),
                "optimizer_steps": steps_done, "code_check_only": args.code_check,
                "budget": {"epochs": args.epochs, "batch_size": args.batch_size,
                           "lr": args.lr, "seed": SEED}}, path)


def train_arm(arm, args, trainset, val_loader, device):
    torch.manual_seed(SEED)
    model = model_from_weights(args.pretrained, args.attention_checkpoint,
                               arm, device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    best = float("inf")
    history = []
    steps_done = 0
    started = time.perf_counter()
    peak = 0
    for epoch in range(1, args.epochs + 1):
        # eval() keeps frozen pretrained BatchNorm statistics fixed. Trainable
        # affine parameters still receive gradients.
        model.eval()
        order = torch.Generator().manual_seed(SEED + epoch)
        loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                            generator=order, num_workers=0)
        total, seen = 0., 0
        order_digest = hashlib.sha256()
        for batch in loader:
            for name in batch["name"]:
                order_digest.update(name.encode("utf-8") + b"\n")
            optimizer.zero_grad(set_to_none=True)
            loss, _ = final_loss(model, batch, device)
            loss.backward()
            if args.code_check and steps_done == 0:
                delta = (model.local_dcn32.delta.weight.grad if model.local_dcn32 else
                         model.decoder3.predict_flow.weight.grad)
                if delta is None or not torch.isfinite(delta).all() or not delta.abs().sum():
                    raise RuntimeError(f"{arm}: final-flow loss did not backpropagate")
            if args.code_check and steps_done == 1 and model.local_dcn32:
                grad = model.local_dcn32.offset.weight.grad
                if grad is None or not torch.isfinite(grad).all() or not grad.abs().sum():
                    raise RuntimeError(f"{arm}: DCN offsets did not receive gradients")
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(batch["name"])
            seen += len(batch["name"])
            steps_done += 1
            if device.type == "cuda":
                peak = max(peak, torch.cuda.max_memory_allocated() / 2**20)
        metric = select_metric(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": total / seen,
                        "val_final_epe_512px": metric, "optimizer_steps": steps_done,
                        "image_order_sha256": order_digest.hexdigest()})
        if metric < best:
            best = metric
            save_checkpoint(args.output / f"best_{arm}.pth", model, arm,
                            epoch, metric, args, steps_done)
        print(f"{arm} epoch {epoch}: final val EPE={metric:.4f}", flush=True)
        (args.output / f"history_{arm}.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8")
    return {"history": history, "optimizer_steps": steps_done,
            "training_seconds": time.perf_counter() - started,
            "training_peak_allocated_mib": peak}


@torch.no_grad()
def evaluate_rounds(model, loader, device):
    if model.local_dcn_steps == 0:
        return {"summary_epe_512px": {}, "per_image": {}}
    totals = [0.0] * model.local_dcn_steps
    count = 0
    per_image = {}
    for batch in loader:
        source, target, source256, target256, *_ = model.pre_process_data(
            batch["source_image"].to(device), batch["target_image"].to(device), device=device)
        _, _, rounds = model(target, source, target256, source256,
                              return_dcn_trace=True)
        truth = batch["flow_map"].to(device).float()
        valid = batch["correspondence_mask"].to(device).bool()
        valid_count = int(valid.sum())
        item = {}
        for index, round_data in enumerate(rounds):
            flow512 = F.interpolate(round_data["flow_256px"], (512, 512),
                                    mode="bilinear", align_corners=False) * 2
            error = torch.linalg.vector_norm(flow512 - truth, dim=1)
            subtotal = float(error[valid].sum())
            totals[index] += subtotal
            item[f"dcn_round_{index + 1}_epe_512px"] = subtotal / valid_count
        per_image[batch["name"][0]] = item
        count += valid_count
    return {"summary_epe_512px": {f"round_{i+1}": value / count
                                      for i, value in enumerate(totals)},
            "per_image": per_image}


def validate_arm(arm, args, valset, device, selected_only=False):
    if selected_only and arm != "attention":
        raise ValueError("Only the original selected SA/CA is a fixed reference")
    model = model_from_weights(args.pretrained, args.attention_checkpoint, arm,
                               device, (args.output / f"best_{arm}.pth"
                                        if not args.code_check and not selected_only else None))
    if args.code_check and not selected_only:
        # Smoke-check checkpoints cannot be used for a formal comparison.
        payload = torch.load(args.output / f"best_{arm}.pth", map_location="cpu",
                             weights_only=False)
        for name, state in payload["local_state_dicts"].items():
            getattr(model, name).load_state_dict(state)
        if model.local_dcn32 is not None:
            model.local_dcn32.load_state_dict(payload["dcn_state_dict"])
    loader = DataLoader(valset, batch_size=1, shuffle=False)
    common, common_rows = evaluate_common(model, loader, device, previews=None)
    stage_rows = evaluate_stages(model, loader, device, preview_dir=None)
    rounds = evaluate_rounds(model, loader, device)
    summary = {"groups": {group: combine_groups(stage_rows, group) for group in
                           ("all", "gt_displacement_64+",
                            "outside_first_window_assigned_pixels",
                            "inside_first_window_assigned_pixels")},
               "search": {level: combine_search(stage_rows, level)
                          for level in ("local32", "local64", "final128")}}
    indexed = {row["name"]: row for row in common_rows}
    per_image = []
    for row in stage_rows:
        name = row["name"]
        if indexed[name]["valid_full"] != row["valid_full"]:
            raise RuntimeError("Common and stage evaluators used different masks")
        record = {"name": name, "valid_full": row["valid_full"],
                  "inference_ms": indexed[name]["inference_ms"],
                  "peak_allocated_mib": indexed[name]["peak_allocated_mib"],
                  "first_window_outside_fraction":
                    row["search"]["local32"]["outside"]["queries"] /
                    row["search"]["local32"]["valid_queries"]}
        record.update({f"{stage}_epe_512px": row["stage_epe_512px"][stage]
                       for stage in STAGES})
        record.update(rounds["per_image"].get(name, {}))
        for group in ("gt_displacement_64+", "outside_first_window_assigned_pixels",
                      "inside_first_window_assigned_pixels"):
            part = row["groups"][group]
            record[f"{group}_pixels"] = part["pixels"]
            record[f"{group}_final_epe_512px"] = (
                part["final128"] / part["pixels"] if part["pixels"] else None)
        per_image.append(record)
    return {"common": common, "stage_summary": summary,
            "iteration_summary": rounds["summary_epe_512px"],
            "per_image": per_image,
            "checkpoint": {"path": str(args.attention_checkpoint if selected_only else
                                        args.output / f"best_{arm}.pth"),
                           "sha256": sha256_file(args.attention_checkpoint if selected_only else
                                                  args.output / f"best_{arm}.pth")}}


def phase2_gate(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("code_check_only") or report.get("phase") != 1:
        raise ValueError("Phase 2 needs a complete phase-1 validation report")
    if not report["phase2_gate"]["passed"]:
        raise ValueError("Phase-1 DCN did not meet the predeclared validation gate")
    return report


FAILURE_IDS = ("000014", "000009", "000015", "000002", "000007")


def comparison_gate(baseline, candidate):
    """Validation-only gate, fixed before any formal DCN training."""
    def metric(result, group):
        return result["stage_summary"]["groups"][group]["epe_512px"]["final128"]
    before = {row["name"]: row for row in baseline["per_image"]}
    after = {row["name"]: row for row in candidate["per_image"]}
    if before.keys() != after.keys():
        raise ValueError("Validation arms saw different image pairs")
    if any(before[name]["valid_full"] != after[name]["valid_full"] for name in before):
        raise ValueError("Validation arms used different GT masks")
    improved = sum(after[name]["final128_epe_512px"] <
                   before[name]["final128_epe_512px"] for name in before)
    failed = [name for name in FAILURE_IDS if name in before]
    failure_before = np.mean([before[name]["final128_epe_512px"] for name in failed])
    failure_after = np.mean([after[name]["final128_epe_512px"] for name in failed])
    speed_ratio = (candidate["common"]["inference_ms_mean_pair_median"] /
                   baseline["common"]["inference_ms_mean_pair_median"])
    checks = {
        "final_epe_improves": metric(candidate, "all") < metric(baseline, "all"),
        "large_motion_epe_not_worse": metric(candidate, "gt_displacement_64+") <=
                                       metric(baseline, "gt_displacement_64+"),
        "first_window_outside_epe_not_worse":
            metric(candidate, "outside_first_window_assigned_pixels") <=
            metric(baseline, "outside_first_window_assigned_pixels"),
        "at_least_12_of_23_pairs_improve": improved >= 12,
        "prior_validation_failures_not_worse_on_average": failure_after <= failure_before,
        "inference_under_1_5x": speed_ratio <= 1.5,
    }
    return {"passed": all(checks.values()), "checks": checks,
            "improved_pairs": improved, "evaluated_pairs": len(before),
            "prior_failure_ids": failed, "prior_failures_mean_epe_before": float(failure_before),
            "prior_failures_mean_epe_after": float(failure_after),
            "speed_ratio": float(speed_ratio),
            "final_epe_before": metric(baseline, "all"),
            "final_epe_after": metric(candidate, "all"),
            "outside_final_epe_before": metric(baseline, "outside_first_window_assigned_pixels"),
            "outside_final_epe_after": metric(candidate, "outside_first_window_assigned_pixels")}


def code_hashes():
    root = Path(__file__).parent
    names = ("roadscene_dcn.py", "roadscene_compare.py",
             "roadscene_refinement_audit.py", "roadscene_coarse.py",
             "datasets/roadscene.py", "models/our_models/GLUNet.py",
             "models/our_models/local_dcn.py", "models/our_models/mod.py",
             "models/correlation/correlation.py")
    return {name: sha256_file(root / name) for name in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, choices=(1, 2), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--attention-checkpoint", type=Path, required=True)
    parser.add_argument("--phase1-report", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-check", action="store_true")
    parser.add_argument("--eval-only", action="store_true",
                        help="reevaluate a saved arm on validation without training")
    parser.add_argument("--arm", choices=tuple(STEPS),
                        help="arm to reevaluate with --eval-only")
    args = parser.parse_args()
    if args.eval_only:
        if args.code_check or args.arm is None:
            parser.error("--eval-only requires --arm and excludes --code-check")
        if args.arm not in (("attention", "dcn_single") if args.phase == 1
                            else ("dcn_iter2",)):
            parser.error("Arm does not belong to the requested phase")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        valset = RoadScenePairs(args.data_root, "val")
        if len(valset) != 23:
            parser.error("Expected all 23 RoadScene validation pairs")
        result = validate_arm(args.arm, args, valset, device)
        (args.output / f"eval_{args.arm}.json").write_text(
            json.dumps({"arm": args.arm, "split": "val", "result": result,
                        "code_hashes": code_hashes()}, indent=2), encoding="utf-8")
        with (args.output / f"eval_{args.arm}_per_image.csv").open(
                "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(result["per_image"][0]))
            writer.writeheader()
            writer.writerows(result["per_image"])
        print(json.dumps({"arm": args.arm,
                          "final_epe_512px": result["common"]["final_flow_epe_512px"]},
                         indent=2), flush=True)
        return
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch size and lr must be positive")
    if args.output.exists() and any(args.output.glob("best_*.pth")):
        parser.error("Output directory already has checkpoints")
    if not args.code_check and (args.epochs, args.batch_size, args.lr) != (20, 2, 1e-4):
        parser.error("Formal study uses the fixed 20-epoch, batch-2, lr-1e-4 budget")
    if args.phase == 2 and not args.code_check and not args.phase1_report:
        parser.error("Phase 2 requires --phase1-report")
    previous = (phase2_gate(args.phase1_report) if args.phase == 2 and
                not args.code_check else None)
    if previous is not None and (previous["base_sha256"] != sha256_file(args.pretrained) or
            previous["attention_sha256"] != sha256_file(args.attention_checkpoint) or
            previous["dataset"] != str(args.data_root.resolve()) or
            previous["budget"] != {"epochs": args.epochs, "batch_size": args.batch_size,
                                    "lr": args.lr, "seed": SEED} or
            previous["code_hashes"] != code_hashes()):
        parser.error("Phase-1 weights, dataset, budget or code differ")
    if args.code_check:
        args.epochs = 1
        args.batch_size = 1
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geometry = geometry_check(device)
    trainset = RoadScenePairs(args.data_root, "train")
    valset = RoadScenePairs(args.data_root, "val")
    if not args.code_check and (len(trainset), len(valset)) != (176, 23):
        parser.error("Formal study requires 176 train and 23 validation pairs")
    if args.code_check:
        trainset = Subset(trainset, range(2))
        valset = Subset(valset, range(1))
    val_loader = DataLoader(valset, batch_size=1, shuffle=False)
    identity = initialization_check(args, valset, device)
    arms = ("attention", "dcn_single") if args.phase == 1 else ("dcn_iter2",)
    report = {"phase": args.phase, "code_check_only": args.code_check,
              "dataset": str(args.data_root.resolve()), "split": "train/val only",
              "train_pairs": len(trainset), "val_pairs": len(valset),
              "base_sha256": sha256_file(args.pretrained),
              "attention_sha256": sha256_file(args.attention_checkpoint),
              "protocol": PROTOCOL, "code_hashes": code_hashes(),
              "geometry_check": geometry,
              "initialization_check": identity,
              "budget": {"epochs": args.epochs, "batch_size": args.batch_size,
                         "lr": args.lr, "seed": SEED},
              "selection": "lowest valid-pixel-weighted final 512px val EPE per arm",
              "trainable": "decoder3 and existing 32-grid dilated refiners in all arms; DCN only in DCN arms; coarse stage, later local levels and pretrained upsamplers frozen",
              "arms": arms, "training": {}, "results": {}}
    if args.phase == 1:
        report["results"]["selected_attention_reference"] = validate_arm(
            "attention", args, valset, device, selected_only=True)
    for arm in arms:
        report["training"][arm] = train_arm(arm, args, trainset, val_loader, device)
        if arm != arms[0]:
            first = report["training"][arms[0]]
            current = report["training"][arm]
            if (first["optimizer_steps"] != current["optimizer_steps"] or
                [row["image_order_sha256"] for row in first["history"]] !=
                [row["image_order_sha256"] for row in current["history"]]):
                raise RuntimeError("Matched arms saw different steps or image order")
        if previous is not None:
            reference = previous["training"]["attention"]
            current = report["training"][arm]
            if (reference["optimizer_steps"] != current["optimizer_steps"] or
                [row["image_order_sha256"] for row in reference["history"]] !=
                [row["image_order_sha256"] for row in current["history"]]):
                raise RuntimeError("Phase-2 arm differs from phase-1 training budget/order")
        (args.output / "progress.json").write_text(json.dumps(report, indent=2),
                                                      encoding="utf-8")
        report["results"][arm] = validate_arm(arm, args, valset, device)
        (args.output / "progress.json").write_text(json.dumps(report, indent=2),
                                                      encoding="utf-8")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if previous is not None:
        report["phase1_report_sha256"] = sha256_file(args.phase1_report)
        report["phase1_results"] = previous["results"]
    if args.phase == 1 and not args.code_check:
        report["phase2_gate"] = comparison_gate(report["results"]["attention"],
                                                 report["results"]["dcn_single"])
        fixed_epe = report["results"]["selected_attention_reference"]["common"][
            "final_flow_epe_512px"]
        single_epe = report["results"]["dcn_single"]["common"][
            "final_flow_epe_512px"]
        report["phase2_gate"]["checks"]["beats_fixed_selected_sa_ca"] = (
            single_epe < fixed_epe)
        report["phase2_gate"]["fixed_selected_final_epe_512px"] = fixed_epe
        report["phase2_gate"]["passed"] = all(
            report["phase2_gate"]["checks"].values())
    else:
        if previous is not None:
            report["phase3_gate"] = comparison_gate(
                previous["results"]["dcn_single"], report["results"]["dcn_iter2"])
    rows = []
    for arm, data in report["results"].items():
        for item in data["per_image"]:
            rows.append({"arm": arm, **item})
    with (args.output / "per_image.csv").open("w", newline="", encoding="utf-8") as file:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2),
                                                   encoding="utf-8")
    print(json.dumps({"code_check_only": args.code_check,
                      "phase": args.phase,
                      "val_final_epe_512px": {arm: result["common"]["final_flow_epe_512px"]
                                              for arm, result in report["results"].items()}},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
