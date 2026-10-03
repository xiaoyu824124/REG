"""Single 256-grid residual after the selected fresh_mutual_fix fine model.

Only train/val are accepted. The 22-pair test split is deliberately unavailable.
The predeclared selection rule is applied after validation, never during training.
"""

import argparse
import csv
import importlib
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from datasets.roadscene import RoadScenePairs
from models.our_models.local_256_refiner import Local256Residual
from roadscene_coarse import sha256_file
from roadscene_compare import PROTOCOL, synchronize
from roadscene_local_ablation import checked_payload, new_model, save_atomic
from roadscene_refinement_audit import edge_quartile, search_stage
from roadscene_staged_mutual_fix import FIX
import roadscene_staged as staged


SEED = staged.SEED
MAX_EPOCHS = 20
PATIENCE = 5
MIN_DELTA = .01
# Fixed before training. All EPE margins are in 512-image-pixel units.
DECISION = {"minimum_final_epe_gain_512px": .5,
            "minimum_improved_pairs": 15,
            "edge_epe_must_improve": True,
            "maximum_large_motion_epe_increase_512px": .1,
            "maximum_window_outside_epe_increase_512px": .1}
GROUPS = ("all", "edge_top_quartile", "large_motion_64plus",
          "first32_inside", "first32_outside")
THRESHOLDS = (1, 3, 5)


class FrozenFineWith256(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base.eval()
        self.base.train_coarse_encoder = False
        self.refiner = Local256Residual()
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()  # includes all BatchNorm running statistics
        return self

    def pre_process_data(self, *args, **kwargs):
        return self.base.pre_process_data(*args, **kwargs)

    def run(self, target, source, target256, source256):
        features = []

        def remember(_module, _inputs, output):
            if len(features) < 2:
                features.append(output.detach())

        # The full-image pyramid runs first, target VI before source IR.
        handle = self.base.pyramid._modules["level_1"].register_forward_hook(
            remember)
        try:
            with torch.no_grad():
                coarse, local = self.base(target, source, target256, source256)
        finally:
            handle.remove()
        if (len(features) != 2 or
                any(item.shape[1:] != (64, 256, 256) for item in features)):
            raise RuntimeError("Expected VI then IR VGG level_1 features")
        refined512, refined256, residual256 = self.refiner(
            features[0], features[1], local[-1])
        return {"coarse": coarse, "local": local,
                "final512": refined512, "grid256": refined256,
                "residual256": residual256}

    def forward(self, target, source, target256, source256):
        result = self.run(target, source, target256, source256)
        return result["coarse"], [*result["local"], result["final512"]]


def geometry_check(device):
    module = Local256Residual().to(device).eval()
    target = torch.zeros(1, 64, 256, 256, device=device)
    source = torch.zeros_like(target)
    target[0, 0, 100, 100] = 1
    source[0, 0, 100, 102] = 1
    flow128 = torch.zeros(1, 2, 128, 128, device=device)
    flow128[:, 0] = 4  # +4 at 512 corresponds to +2 on the 256 grid.
    with torch.no_grad():
        correct, _ = module.local_correlation(
            target, source, F.interpolate(flow128, (256, 256),
                                           mode="bilinear", align_corners=False))
        wrong = flow128.clone()
        wrong[:, 0] = -4
        opposite, _ = module.local_correlation(
            target, source, F.interpolate(wrong, (256, 256),
                                           mode="bilinear", align_corners=False))
        final, grid, delta = module(target, source, flow128)
        baseline = F.interpolate(flow128, (512, 512), mode="bilinear",
                                 align_corners=False)
        zero_difference = float((final - baseline).abs().max())
        module.head.bias[0] = float(np.arctanh(.5))
        moved, _, _ = module(target, source, flow128)
        conversion = float((moved - baseline)[0, 0, 256, 256])
    if (float(correct[0, 4, 100, 100]) < .999 or
            float(opposite[0, 4, 100, 100]) != 0 or
            zero_difference != 0 or abs(conversion - 1) > 1e-5 or
            grid.shape[-2:] != (256, 256) or delta.abs().max() != 0):
        raise RuntimeError("256-grid direction, unit or zero-head check failed")
    return {"synthetic_source_offset_256grid": [2, 0],
            "given_target_to_source_flow_512px": [4, 0],
            "correct_center_correlation": float(correct[0, 4, 100, 100]),
            "opposite_center_correlation": float(opposite[0, 4, 100, 100]),
            "zero_head_final_max_abs_diff": zero_difference,
            "half_grid_pixel_to_512px_delta": conversion}


def make_model(coarse_path, fine_path, pretrained, device):
    coarse = checked_payload(coarse_path, pretrained, smoke=False)
    fine = torch.load(fine_path, map_location="cpu", weights_only=False)
    if (fine.get("stage") != "fine" or fine.get("arm") != "dns_attention" or
            fine.get("parent_sha256") != sha256_file(coarse_path) or
            fine.get("base_sha256") != sha256_file(pretrained) or
            fine.get("matching_fix") != FIX or fine.get("code_check_only") or
            fine.get("epoch") != 6):
        raise ValueError("Expected selected fresh_mutual_fix epoch-6 fine weight")
    base = new_model(coarse, device)
    base.load_state_dict(fine["model_state_dict"], strict=True)
    return FrozenFineWith256(base).to(device).eval(), fine


def inputs(model, batch, device):
    return model.pre_process_data(batch["source_image"].to(device),
                                  batch["target_image"].to(device),
                                  device=device)[:4]


@torch.no_grad()
def zero_check(model, dataset, device):
    maximum = 0.0
    for batch in DataLoader(dataset, batch_size=1, shuffle=False):
        source, target, source256, target256 = inputs(model, batch, device)
        result = model.run(target, source, target256, source256)
        baseline = F.interpolate(result["local"][-1], (512, 512),
                                 mode="bilinear", align_corners=False)
        maximum = max(maximum, float((result["final512"] - baseline).abs().max()))
    if maximum != 0:
        raise RuntimeError(f"Zero-head final output changed: {maximum}")
    return maximum


@torch.no_grad()
def validation_epe(model, dataset, device):
    model.eval()
    total, count = 0.0, 0
    for batch in DataLoader(dataset, batch_size=1, shuffle=False):
        source, target, source256, target256 = inputs(model, batch, device)
        prediction = model.run(target, source, target256, source256)["final512"]
        truth = batch["flow_map"].to(device).float()
        valid = batch["correspondence_mask"].to(device).bool()
        error = torch.linalg.vector_norm(prediction - truth, dim=1)
        total += float(error[valid].sum())
        count += int(valid.sum())
    return total / count


def snapshot_base(model):
    return ({name: value.detach().cpu().clone()
             for name, value in model.base.named_parameters()},
            {name: value.detach().cpu().clone()
             for name, value in model.base.named_buffers()})


def base_change(model, snapshot):
    params, buffers = snapshot
    differences = [float((value.detach().cpu() - params[name]).abs().max())
                   for name, value in model.base.named_parameters()]
    differences += [float((value.detach().cpu() - buffers[name]).abs().max())
                    for name, value in model.base.named_buffers()]
    return max(differences, default=0.0)


def checkpoint(model, optimizer, scheduler, epoch, best, steps, coarse_sha,
               fine_sha, history):
    return {"epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state()
            if torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "optimizer_steps": steps, "selection_metric": best,
            "coarse_sha256": coarse_sha, "fine_sha256": fine_sha,
            "code_sha256": sha256_file(Path(__file__)),
            "history": history, "split": "val", "test_pairs_accessed": False}


def train(model, trainset, valset, device, output, coarse_sha, fine_sha,
          code_check):
    optimizer = torch.optim.AdamW(model.refiner.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=2,
        threshold=MIN_DELTA, threshold_mode="abs", min_lr=1e-7)
    fixed = snapshot_base(model)
    best = validation_epe(model, valset, device)
    history = [{"epoch": 0, "val_final_epe_512px": best,
                "train_loss": None, "optimizer_steps": 0}]
    best_epoch, bad, steps = 0, 0, 0
    save_atomic(checkpoint(model, optimizer, scheduler, 0, best, steps,
                           coarse_sha, fine_sha, history),
                output / "best_local256.pth")
    limit = 1 if code_check else MAX_EPOCHS
    for epoch in range(1, limit + 1):
        model.train()
        loader = DataLoader(trainset, batch_size=2, shuffle=True,
                            generator=torch.Generator().manual_seed(SEED + epoch),
                            num_workers=0, pin_memory=device.type == "cuda")
        losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = staged.dense_training_loss(
                model, batch, device, coarse_auxiliary=False)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite local256 training loss")
            loss.backward()
            if steps == 0 and (model.refiner.head.weight.grad is None or
                               not torch.isfinite(model.refiner.head.weight.grad).all() or
                               model.refiner.head.weight.grad.abs().sum() == 0):
                raise RuntimeError("Local256 residual head has no finite gradient")
            if any(p.grad is not None for p in model.base.parameters()):
                raise RuntimeError("Frozen GLU-Net received a gradient")
            torch.nn.utils.clip_grad_norm_(model.refiner.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            steps += 1
        value = validation_epe(model, valset, device)
        scheduler.step(value)
        improved = value < best - MIN_DELTA
        if improved:
            best, best_epoch, bad = value, epoch, 0
        else:
            bad += 1
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "val_final_epe_512px": value, "best_epe_512px": best,
                        "optimizer_steps": steps, "lr": optimizer.param_groups[0]["lr"],
                        "patience": f"{bad}/{PATIENCE}"})
        (output / "history.json").write_text(json.dumps(history, indent=2),
                                               encoding="utf-8")
        if improved:
            save_atomic(checkpoint(model, optimizer, scheduler, epoch, best,
                                   steps, coarse_sha, fine_sha, history),
                        output / "best_local256.pth")
        print(f"local256 epoch {epoch}/{limit}: final val EPE={value:.4f}, "
              f"best={best:.4f}, patience={bad}/{PATIENCE}", flush=True)
        if bad >= PATIENCE:
            break
    frozen_difference = base_change(model, fixed)
    if frozen_difference != 0:
        raise RuntimeError(f"Frozen weights or BN changed: {frozen_difference}")
    selected = torch.load(output / "best_local256.pth", map_location="cpu",
                          weights_only=False)
    model.load_state_dict(selected["model_state_dict"], strict=True)
    model.eval()
    return {"selected_epoch": best_epoch, "stopped_at_epoch": epoch,
            "optimizer_steps_executed": steps, "best_val_final_epe_512px": best,
            "frozen_base_max_abs_change": frozen_difference,
            "history": history}


def per_group(errors, masks):
    result = {}
    for label, mask in masks.items():
        count = int(mask.sum())
        result[label] = {"pixels": count,
                         "baseline_epe_512px": float(errors["baseline"][mask].mean())
                         if count else None,
                         "new_epe_512px": float(errors["new"][mask].mean())
                         if count else None}
    return result


def edge_map(image):
    gray = (image.float() / 255 *
            image.new_tensor((.299, .587, .114), dtype=torch.float32)
            [:, None, None]).sum(dim=0)[None, None]
    kx = gray.new_tensor(((-1, 0, 1), (-2, 0, 2), (-1, 0, 1)))[None, None] / 8
    ky = kx.transpose(-1, -2)
    strength = torch.sqrt(F.conv2d(gray, kx, padding=1).square() +
                          F.conv2d(gray, ky, padding=1).square())[0, 0]
    return strength


def warp_image(source, flow):
    y, x = torch.meshgrid(torch.arange(512, device=source.device),
                          torch.arange(512, device=source.device), indexing="ij")
    grid = torch.stack((2 * (x + flow[0]) / 511 - 1,
                        2 * (y + flow[1]) / 511 - 1), dim=-1)[None]
    return F.grid_sample(source[None].float(), grid, align_corners=True)[0]


def ghost_panel(path, source, target, baseline, refined, valid):
    target_edge = edge_map(target)
    target_edge = target_edge / target_edge[valid].quantile(.95).clamp_min(1e-6)
    panels = []
    for label, flow in (("baseline 128 grid", baseline),
                        ("new 256 grid", refined)):
        warped = warp_image(source, flow)
        ir_edge = edge_map(warped) / edge_map(warped)[valid].quantile(.95).clamp_min(1e-6)
        overlay = torch.zeros(3, 512, 512, device=source.device)
        overlay[0] = target_edge.clamp(0, 1)  # visible edges: red
        overlay[1] = ir_edge.clamp(0, 1)      # warped IR edges: cyan
        overlay[2] = ir_edge.clamp(0, 1)
        overlay[:, ~valid] = 0
        panels.append((label, overlay))
    canvas = Image.new("RGB", (1024, 540))
    draw = ImageDraw.Draw(canvas)
    for index, (label, panel) in enumerate(panels):
        rgb = (panel.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        canvas.paste(Image.fromarray(rgb), (512 * index, 28))
        draw.text((512 * index + 8, 8), label, fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


@torch.no_grad()
def evaluate(model, dataset, device, preview_dir, code_check):
    model.eval()
    rows, accum = [], {group: {"pixels": 0, "baseline_sum": 0.0,
                               "new_sum": 0.0} for group in GROUPS}
    threshold_count = {arm: {limit: 0 for limit in THRESHOLDS}
                       for arm in ("baseline", "new")}
    total_valid = 0
    for batch in DataLoader(dataset, batch_size=1, shuffle=False):
        name = batch["name"][0]
        raw_source = batch["source_image"].to(device)
        raw_target = batch["target_image"].to(device)
        truth = batch["flow_map"].to(device).float()
        valid = batch["correspondence_mask"].to(device).bool()[0]

        def base_forward():
            s, t, s256, t256 = inputs(model, batch, device)
            _, local = model.base(t, s, t256, s256)
            return F.interpolate(local[-1], (512, 512), mode="bilinear",
                                 align_corners=False)

        def new_forward():
            s, t, s256, t256 = inputs(model, batch, device)
            return model.run(t, s, t256, s256)["final512"]

        timings = {}
        repeats = 1 if code_check else PROTOCOL["timed_repeats_per_pair"]
        warmups = 0 if code_check else PROTOCOL["warmup_forwards"]
        for label, function in (("baseline", base_forward),
                                ("new", new_forward)):
            for _ in range(warmups):
                function()
            durations = []
            for _ in range(repeats):
                synchronize(device)
                started = time.perf_counter()
                function()
                synchronize(device)
                durations.append((time.perf_counter() - started) * 1000)
            timings[label] = float(np.median(durations))

        s, t, s256, t256 = inputs(model, batch, device)
        result = model.run(t, s, t256, s256)
        baseline = F.interpolate(result["local"][-1], (512, 512),
                                 mode="bilinear", align_corners=False)
        refined = result["final512"]
        errors = {"baseline": torch.linalg.vector_norm(
                      baseline[0] - truth[0], dim=0),
                  "new": torch.linalg.vector_norm(
                      refined[0] - truth[0], dim=0)}
        edge = edge_quartile(raw_target, valid)
        coarse16 = result["coarse"][0]
        pre32 = model.base.deconv4(coarse16) * 2
        _, inside32, selected32 = search_stage(
            pre32, result["coarse"][1] * 2, truth, valid[None], 32)
        outside = F.interpolate((selected32 & ~inside32)[:, None].float(),
                                (512, 512), mode="nearest")[0, 0].bool()
        inside = F.interpolate(inside32[:, None].float(), (512, 512),
                               mode="nearest")[0, 0].bool()
        masks = {"all": valid, "edge_top_quartile": edge,
                 "large_motion_64plus": valid &
                    (torch.linalg.vector_norm(truth[0], dim=0) >= 64),
                 "first32_inside": valid & inside,
                 "first32_outside": valid & outside}
        groups = per_group(errors, masks)
        total_valid += int(valid.sum())
        for group in GROUPS:
            count = groups[group]["pixels"]
            accum[group]["pixels"] += count
            if count:
                accum[group]["baseline_sum"] += (
                    groups[group]["baseline_epe_512px"] * count)
                accum[group]["new_sum"] += groups[group]["new_epe_512px"] * count
        for arm in threshold_count:
            for limit in THRESHOLDS:
                threshold_count[arm][limit] += int(((errors[arm] < limit) &
                                                     valid).sum())
        row = {"name": name, "valid_pixels": int(valid.sum()),
               "baseline_epe_512px": groups["all"]["baseline_epe_512px"],
               "new_epe_512px": groups["all"]["new_epe_512px"],
               "new_minus_baseline_epe_512px": (
                   groups["all"]["new_epe_512px"] -
                   groups["all"]["baseline_epe_512px"]),
               "baseline_ms": timings["baseline"], "new_ms": timings["new"]}
        for arm in threshold_count:
            for limit in THRESHOLDS:
                row[f"{arm}_fraction_epe_below_{limit}px"] = float(
                    ((errors[arm] < limit) & valid).sum() / valid.sum())
        for group in GROUPS[1:]:
            row[f"{group}_pixels"] = groups[group]["pixels"]
            row[f"{group}_baseline_epe_512px"] = groups[group]["baseline_epe_512px"]
            row[f"{group}_new_epe_512px"] = groups[group]["new_epe_512px"]
        rows.append(row)
        if preview_dir and name in {"000014", "000002"}:
            ghost_panel(preview_dir / f"{name}_edge_ghost.png",
                        raw_source[0], raw_target[0], baseline[0],
                        refined[0], valid)
    summary_groups = {group: {"pixels": values["pixels"],
                              "baseline_epe_512px": values["baseline_sum"] /
                                  values["pixels"] if values["pixels"] else None,
                              "new_epe_512px": values["new_sum"] /
                                  values["pixels"] if values["pixels"] else None}
                      for group, values in accum.items()}
    return rows, {"samples": len(rows), "valid_pixels": total_valid,
                  "groups": summary_groups,
                  "pixel_fraction_epe_below": {
                      arm: {str(limit): threshold_count[arm][limit] / total_valid
                            for limit in THRESHOLDS}
                      for arm in threshold_count},
                  "improved_pairs": sum(row["new_minus_baseline_epe_512px"] < 0
                                        for row in rows),
                  "timing_ms_mean_pair_median": {
                      arm: float(np.mean([row[f"{arm}_ms"] for row in rows]))
                      for arm in ("baseline", "new")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--coarse-checkpoint", type=Path)
    parser.add_argument("--fine-checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--code-check", action="store_true")
    parser.add_argument("--geometry-check-only", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geometry = geometry_check(device)
    if args.geometry_check_only:
        print(json.dumps({"geometry": geometry}, indent=2), flush=True)
        return
    if any(value is None for value in (args.data_root, args.pretrained,
                                       args.coarse_checkpoint,
                                       args.fine_checkpoint, args.output)):
        parser.error("Training needs dataset, pretrained and both selected weights")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be a new, empty directory")
    args.output.mkdir(parents=True, exist_ok=True)
    trainset = RoadScenePairs(args.data_root, "train")
    valset = RoadScenePairs(args.data_root, "val")
    if len(trainset) != 176 or len(valset) != 23:
        raise ValueError("Expected 176 training and 23 validation pairs")
    if args.code_check:
        trainset = Subset(trainset, range(2))
        valset = Subset(valset, range(1))
    glunet_module = importlib.import_module("models.our_models.GLUNet")
    original_matching = glunet_module.MutualMatching
    glunet_module.MutualMatching = (
        lambda correlation: original_matching(correlation.clamp_min(0)))
    try:
        model, fine = make_model(args.coarse_checkpoint,
                                 args.fine_checkpoint, args.pretrained, device)
        initial_difference = zero_check(model, valset, device)
        fine_report = json.loads((args.fine_checkpoint.parent /
                                  "report_fine.json").read_text(encoding="utf-8"))
        if (not args.code_check and
                abs(fine_report["result"]["common"]["final_flow_epe_512px"] -
                    5.74643290998514) > .01):
            raise ValueError("Selected baseline validation result is not 5.7464px")
        training = train(model, trainset, valset, device, args.output,
                         sha256_file(args.coarse_checkpoint),
                         sha256_file(args.fine_checkpoint), args.code_check)
        rows, metrics = evaluate(model, valset, device,
                                 args.output / "edge_ghosts", args.code_check)
        with (args.output / "per_image_val.csv").open(
                "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        groups = metrics["groups"]
        baseline = groups["all"]["baseline_epe_512px"]
        new = groups["all"]["new_epe_512px"]
        if (not args.code_check and
                abs(baseline - fine_report["result"]["common"]
                    ["final_flow_epe_512px"]) > .02):
            raise RuntimeError("Paired baseline differs from selected fine report")
        def group_increase(group):
            before = groups[group]["baseline_epe_512px"]
            after = groups[group]["new_epe_512px"]
            return after - before if before is not None and after is not None else None

        decision = {
            "predeclared_rule": DECISION,
            "final_gain_512px": baseline - new,
            "improved_pairs": metrics["improved_pairs"],
            "edge_gain_512px": -group_increase("edge_top_quartile"),
            "large_motion_increase_512px": group_increase("large_motion_64plus"),
            "window_outside_increase_512px": group_increase("first32_outside")}
        decision["retain_new_layer"] = (
            not args.code_check
            and decision["large_motion_increase_512px"] is not None
            and decision["window_outside_increase_512px"] is not None
            and decision["final_gain_512px"] >= DECISION["minimum_final_epe_gain_512px"]
            and decision["improved_pairs"] >= DECISION["minimum_improved_pairs"]
            and decision["edge_gain_512px"] > 0
            and decision["large_motion_increase_512px"] <=
                DECISION["maximum_large_motion_epe_increase_512px"]
            and decision["window_outside_increase_512px"] <=
                DECISION["maximum_window_outside_epe_increase_512px"])
        report = {"experiment": "single_256_grid_local_residual",
                  "split": "val", "test_pairs_accessed": False,
                  "code_check_only": args.code_check,
                  "pretrained_sha256": sha256_file(args.pretrained),
                  "coarse_checkpoint_sha256": sha256_file(args.coarse_checkpoint),
                  "fine_checkpoint_sha256": sha256_file(args.fine_checkpoint),
                  "matching_fix": FIX, "geometry": geometry,
                  "zero_head_max_abs_difference": initial_difference,
                  "feature": "frozen full-image VGG level_1, 64 channels, 256x256",
                  "flow_units": "all new/final XY flows use 512-image-pixel units; 256-grid sampling uses flow/2",
                  "direction": "visible target -> infrared source; warp infrared features",
                  "correlation": "3x3 source-offset cosine correlation with validity channels",
                  "maximum_residual_per_axis_512px": 2.0,
                  "upsampling": "frozen pretrained deconv4/deconv2; new 128-to-256 residual uses bilinear, baseline 128-to-512 output is preserved at zero initialization",
                  "trainable_parameters": sum(p.numel() for p in model.refiner.parameters()),
                  "loss": "same masked final-flow Charbonnier EPE as staged fine",
                  "edge_definition": "top quartile of visible Sobel magnitude among valid pixels, per pair",
                  "window_definition": "first 32-grid pre-correlation centre, 9x9 radius 4; >80% valid pooled GT; nearest assignment to 512-grid pixels",
                  "timing_protocol": PROTOCOL,
                  "training": training, "metrics": metrics,
                  "special_cases": {row["name"]: row for row in rows
                                    if row["name"] in {"000014", "000002"}},
                  "decision": decision}
        (args.output / "report_val.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"selected_epoch": training["selected_epoch"],
                          "metrics": metrics, "decision": decision},
                         indent=2), flush=True)
    finally:
        glunet_module.MutualMatching = original_matching


if __name__ == "__main__":
    main()
