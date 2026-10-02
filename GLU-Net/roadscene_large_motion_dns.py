"""Train/val-only large-motion coarse-match and independent DNS ablations.

Arms: fixed selected SA/CA; SA/CA with train-only displacement-stratified
sampling; SA/CA with train-only coarse loss weighting; GLU-Net+DNS without
SA/CA or contrastive learning. Full training uses 20 epochs by default.
The independent test split is never opened by this script.
"""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from datasets.roadscene import RoadScenePairs, read_flo
from roadscene_coarse import (coarse_statistics, correspondence_targets,
                              evaluate, make_model, predict_coarse, prepare,
                              sha256_file)
from roadscene_compare import evaluate_common, load_glu
from roadscene_refinement_audit import pooled_truth


SEED = 2026
ARMS = ("attention", "attention_stratified", "attention_weighted", "dns")
NEW_ARMS = ARMS[1:]
LARGE_GT_512PX = 64.0
STRATUM_MIN_LARGE_FRACTION = 0.75
HIGH_STRATUM_DRAW_FRACTION = 0.75
HIGH_MOTION_LOSS_WEIGHT = 2.0


def stratify_train_pairs(dataset):
    high, low = [], []
    fractions = {}
    for index, paths in enumerate(dataset.samples):
        flow = read_flo(paths[2])
        height, width = flow.shape[:2]
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        valid = (np.isfinite(flow).all(axis=2) &
                 (x + flow[..., 0] >= 0) & (x + flow[..., 0] <= width - 1) &
                 (y + flow[..., 1] >= 0) & (y + flow[..., 1] <= height - 1))
        if not valid.any():
            raise ValueError(f"No finite GT in {paths[2]}")
        fraction = float((np.linalg.norm(flow[valid], axis=1) >= LARGE_GT_512PX).mean())
        fractions[paths[2].stem] = fraction
        (high if fraction >= STRATUM_MIN_LARGE_FRACTION else low).append(index)
    if not high or not low:
        raise ValueError("Stratified arm needs both high and low train-only GT strata")
    return high, low, fractions


def epoch_loader(dataset, arm, batch_size, epoch, high, low):
    generator = torch.Generator().manual_seed(SEED)
    if arm != "attention_stratified":
        # Matches the existing SA/CA experiment's DataLoader shuffle stream.
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            generator=generator, num_workers=0)
        # The same generator must advance across epochs, so this DataLoader
        # is constructed outside the epoch loop by train_arm instead.
        return loader
    high_draws = round(len(dataset) * HIGH_STRATUM_DRAW_FRACTION)
    low_draws = len(dataset) - high_draws
    epoch_generator = torch.Generator().manual_seed(SEED + epoch)
    high_indices = torch.tensor(high)[torch.randint(len(high), (high_draws,),
                                                    generator=epoch_generator)]
    low_indices = torch.tensor(low)[torch.randint(len(low), (low_draws,),
                                                 generator=epoch_generator)]
    draws = torch.cat((high_indices, low_indices))
    draws = draws[torch.randperm(len(draws), generator=epoch_generator)].tolist()
    return DataLoader(Subset(dataset, draws), batch_size=batch_size,
                      shuffle=False, num_workers=0)


def weighted_coarse_loss(flow, corr, truth, valid):
    """Weight both flow and global-correlation CE at large GT displacements."""
    indices, corr_valid = correspondence_targets(truth, valid)
    # Truth is in 256px units; 64px at full 512 corresponds to 32 here.
    large = torch.linalg.vector_norm(truth, dim=1) >= LARGE_GT_512PX / 2
    weight = torch.where(large, HIGH_MOTION_LOSS_WEIGHT, 1.0)
    error = torch.linalg.vector_norm(flow - truth, dim=1)
    scores = corr.permute(0, 2, 3, 1)
    ce = F.cross_entropy((scores / 0.1).reshape(-1, 256),
                         indices.reshape(-1), reduction="none").reshape_as(indices)
    epe_loss = (error * weight)[valid].sum() / weight[valid].sum()
    ce_loss = (ce * weight)[corr_valid].sum() / weight[corr_valid].sum()
    return epe_loss + 2 * ce_loss


def train_arm(arm, args, train_dataset, val_loader, device, high, low):
    torch.manual_seed(SEED)
    has_dns = arm in {"dns", "dns_attention"}
    has_attention = arm != "dns"
    model = make_model(args.pretrained, attention=has_attention, device=device,
                       train_decoder=True, dns=has_dns)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    normal_loader = None if arm == "attention_stratified" else epoch_loader(
        train_dataset, arm, args.batch_size, 1, high, low)
    best = float("inf")
    history = []
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        loader = (epoch_loader(train_dataset, arm, args.batch_size, epoch, high, low)
                  if arm == "attention_stratified" else normal_loader)
        model.eval()  # retain pretrained batch-norm statistics
        order, losses = [], []
        for batch in loader:
            order.extend(batch["name"])
            source, target, _, _, truth, valid, _ = prepare(batch, device)
            predicted, corr = predict_coarse(model, target, source)
            if arm == "attention_weighted":
                loss = weighted_coarse_loss(predicted, corr, truth, valid)
            else:
                epe, ce, _ = coarse_statistics(predicted, corr, truth, valid)
                loss = epe + 2 * ce
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite train loss in {arm}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = evaluate(model, val_loader, device)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                  "train_steps": len(losses),
                  "image_order_sha256": hashlib.sha256(json.dumps(order).encode()).hexdigest(),
                  "validation": validation}
        history.append(record)
        print(f"{arm} epoch {epoch}: val coarse EPE={validation['epe_256px']:.4f}",
              flush=True)
        # Match the existing selected SA/CA checkpoint rule. Arm selection
        # below uses final EPE, actual window misses and pair stability.
        if validation["epe_256px"] < best:
            best = validation["epe_256px"]
            payload = {"arm": arm if arm in {"dns", "dns_attention"} else "attention",
                       "variant": arm, "epoch": epoch,
                       "decoder4_state_dict": model.decoder4.state_dict(),
                       "validation": validation,
                       "base_sha256": sha256_file(args.pretrained),
                       "budget": {"epochs": args.epochs, "lr": args.lr,
                                  "batch_size": args.batch_size, "seed": SEED,
                                  "code_check": args.code_check}}
            if has_dns:
                payload["dns_state_dict"] = model.coarse_dns.state_dict()
            if has_attention:
                payload["attention_state_dict"] = model.coarse_attention.state_dict()
            torch.save(payload, args.output / f"best_{arm}.pth")
    if device.type == "cuda":
        torch.cuda.synchronize()
    cost = {"training_seconds": time.perf_counter() - started,
            "trainable_parameters": sum(p.numel() for p in trainable),
            "peak_training_allocated_mib": (torch.cuda.max_memory_allocated() / 2**20
                                            if device.type == "cuda" else None)}
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return history, cost


@torch.no_grad()
def coarse_and_window(model, loader, device):
    rows = {}
    for batch in loader:
        source, target, _, _, _, _, _ = prepare(batch, device)
        coarse, _ = predict_coarse(model, target, source)
        full = F.interpolate(coarse, size=(512, 512), mode="bilinear",
                             align_corners=False) * 2
        gt = batch["flow_map"].to(device).float()
        valid = batch["correspondence_mask"].to(device).bool()
        error = torch.linalg.vector_norm(full - gt, dim=1)
        displacement = torch.linalg.vector_norm(gt, dim=1)
        large = valid & (displacement >= LARGE_GT_512PX)
        grid_gt, selected = pooled_truth(gt, valid, 32)
        actual_warp_centre = model.deconv4(coarse) * 2
        residual = grid_gt - actual_warp_centre
        outside = (residual.abs().amax(dim=1) > 64) & selected
        grid_large = selected & (torch.linalg.vector_norm(grid_gt, dim=1) >= LARGE_GT_512PX)
        rows[batch["name"][0]] = {
            "coarse_full_epe_sum_512px": float(error[valid].double().sum()),
            "valid_full": int(valid.sum()),
            "coarse_full_large_epe_sum_512px": float(error[large].double().sum()),
            "valid_large_full": int(large.sum()),
            "first_window_valid_queries": int(selected.sum()),
            "first_window_outside_queries": int(outside.sum()),
            "first_window_large_valid_queries": int(grid_large.sum()),
            "first_window_large_outside_queries": int((outside & grid_large).sum())}
    return rows


def combine_diagnostics(rows):
    count = sum(r["valid_full"] for r in rows.values())
    large = sum(r["valid_large_full"] for r in rows.values())
    queries = sum(r["first_window_valid_queries"] for r in rows.values())
    large_queries = sum(r["first_window_large_valid_queries"] for r in rows.values())
    return {"valid_full": count, "coarse_full_epe_512px":
            sum(r["coarse_full_epe_sum_512px"] for r in rows.values()) / count,
            "coarse_full_large_epe_512px":
            (sum(r["coarse_full_large_epe_sum_512px"] for r in rows.values()) / large
             if large else None),
            "first_window_valid_queries": queries,
            "first_window_outside_queries": sum(r["first_window_outside_queries"] for r in rows.values()),
            "first_window_outside_fraction":
            sum(r["first_window_outside_queries"] for r in rows.values()) / queries,
            "first_window_large_valid_queries": large_queries,
            "first_window_large_outside_fraction":
            (sum(r["first_window_large_outside_queries"] for r in rows.values()) / large_queries
             if large_queries else None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--attention-checkpoint", type=Path, required=True,
                        help="fixed selected SA/CA reference, unchanged")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--code-check", action="store_true",
                        help="two train/one val pair, one epoch; never research evidence")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch-size and lr must be positive")
    if not args.code_check and (args.epochs, args.batch_size, args.lr) != (20, 2, 1e-4):
        parser.error("Full comparison must match selected SA/CA budget: 20 epochs, batch 2, lr 1e-4")
    if args.output.exists() and any(args.output.glob("best_*.pth")):
        parser.error("Output already contains weights; use a fresh directory")
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
    high, low, fractions = stratify_train_pairs(full_train)
    if args.code_check:
        train_dataset = Subset(full_train, (high[0], low[0]))
        # Within this two-element Subset, the stratum positions are 0/1.
        high, low = [0], [1]
        val_dataset = Subset(full_val, range(1))
    else:
        train_dataset, val_dataset = full_train, full_val
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    report = {"code_check_only": args.code_check, "split": "val",
              "arms": ARMS, "train_pairs": len(train_dataset),
              "val_pairs": len(val_dataset),
              "base_sha256": sha256_file(args.pretrained),
              "selected_attention_sha256": sha256_file(args.attention_checkpoint),
              "seed": SEED, "epochs": args.epochs, "batch_size": args.batch_size,
              "lr": args.lr, "train_steps_per_epoch":
                  (len(train_dataset) + args.batch_size - 1) // args.batch_size,
              "training_design": {
                  "stratum_threshold": f">= {STRATUM_MIN_LARGE_FRACTION:.0%} valid train pixels with GT magnitude >=64px",
                  "stratified_draws": f"{HIGH_STRATUM_DRAW_FRACTION:.0%} high-stratum, with replacement; equal epoch length",
                  "loss_weight": "2x at 16-grid GT magnitude >=32px (64px at512) for EPE and correlation CE; normalized weighted means",
                  "dns": "existing shared 2D two-ring centre-free neighbour similarity, learned projection and zero residual gate; no SA/CA or contrastive loss",
                  "selection_within_arm": "minimum validation coarse EPE, matching the fixed SA/CA checkpoint; final flow/window/pairs decide across arms",
                  "test": "never accessed"},
              "train_stratum_counts": {"high": len(high), "low": len(low)},
              "train_large_fraction_by_pair": fractions,
              "history": {}, "costs": {}, "results": {}}
    # A zero DNS gate must reproduce the pretrained global match exactly.
    with torch.no_grad():
        source, target, *_ = prepare(next(iter(val_loader)), device)
        baseline = make_model(args.pretrained, False, device)
        dns_zero = make_model(args.pretrained, False, device, dns=True)
        base_flow, base_corr = predict_coarse(baseline, target, source)
        dns_flow, dns_corr = predict_coarse(dns_zero, target, source)
        report["dns_zero_gate_check"] = {
            "flow_max_abs_diff": float((base_flow - dns_flow).abs().max()),
            "corr_max_abs_diff": float((base_corr - dns_corr).abs().max())}
        if max(report["dns_zero_gate_check"].values()) > 1e-6:
            raise RuntimeError("DNS zero-gate changed pretrained output")
        del baseline, dns_zero
    for arm in NEW_ARMS:
        history, cost = train_arm(arm, args, train_dataset, val_loader,
                                  device, high, low)
        report["history"][arm], report["costs"][arm] = history, cost
        (args.output / "progress.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if [r["image_order_sha256"] for r in report["history"]["attention_weighted"]] != \
       [r["image_order_sha256"] for r in report["history"]["dns"]]:
        raise RuntimeError("Weighted and DNS arms saw different train image order")
    common_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    for arm in ARMS:
        path = args.attention_checkpoint if arm == "attention" else args.output / f"best_{arm}.pth"
        model, meta = load_glu(args.pretrained, path,
                               "dns" if arm == "dns" else "attention", device)
        if meta.get("code_check_only") and not args.code_check:
            raise ValueError("Smoke-check weights cannot enter full study")
        if not args.code_check:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if arm != "attention" and payload["budget"] != {
                "epochs": 20, "lr": 1e-4, "batch_size": 2,
                "seed": SEED, "code_check": False}:
                raise ValueError("New arm budget differs from fixed SA/CA control")
        summary, rows = evaluate_common(model, common_loader, device,
                                         previews=None)
        details = coarse_and_window(model, common_loader, device)
        by_name = {row["name"]: row for row in rows}
        if by_name.keys() != details.keys():
            raise RuntimeError("Pair IDs mismatch between final and window metrics")
        for name, item in details.items():
            by_name[name].update(item)
        report["results"][arm] = {"summary": summary,
                                  "diagnostics": combine_diagnostics(details),
                                  "per_image": rows, "checkpoint": meta}
        print(f"{arm}: final={summary['final_flow_epe_512px']:.4f}, "
              f"window_outside={report['results'][arm]['diagnostics']['first_window_outside_fraction']:.4f}",
              flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    pairs = []
    indexed = {arm: {row["name"]: row for row in report["results"][arm]["per_image"]}
               for arm in ARMS}
    for name in indexed["attention"]:
        row = {"name": name, "valid_full": indexed["attention"][name]["valid_full"]}
        for arm in ARMS:
            item = indexed[arm][name]
            if item["valid_full"] != row["valid_full"]:
                raise RuntimeError("Different valid masks across arms")
            row[f"{arm}_final_epe_512px"] = item["aepe_512px"]
            row[f"{arm}_coarse_epe_256px"] = item["coarse_epe_256px"]
            row[f"{arm}_coarse_full_epe_512px"] = (
                item["coarse_full_epe_sum_512px"] / item["valid_full"])
            large = item["full_bins_512px"]["64+"]
            row[f"{arm}_large_final_epe_512px"] = (
                large["epe_sum"] / large["pixels"] if large["pixels"] else "")
            row[f"{arm}_large_coarse_epe_512px"] = (
                item["coarse_full_large_epe_sum_512px"] / item["valid_large_full"]
                if item["valid_large_full"] else "")
            row[f"{arm}_first_window_outside_fraction"] = (
                item["first_window_outside_queries"] / item["first_window_valid_queries"])
            row[f"{arm}_inference_ms"] = item["inference_ms"]
        pairs.append(row)
    report["paired_stability"] = {arm: {
        "improved_pairs_vs_attention": sum(row[f"{arm}_final_epe_512px"] < row["attention_final_epe_512px"]
                                           for row in pairs),
        "worsened_pairs_vs_attention": sum(row[f"{arm}_final_epe_512px"] > row["attention_final_epe_512px"]
                                           for row in pairs)} for arm in NEW_ARMS}
    with (args.output / "per_image.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"code_check_only": args.code_check,
                      "arms": {arm: {"coarse_256px": report["results"][arm]["summary"]["coarse_epe_256px"],
                                     "final_512px": report["results"][arm]["summary"]["final_flow_epe_512px"],
                                     "window_outside": report["results"][arm]["diagnostics"]["first_window_outside_fraction"]}
                               for arm in ARMS}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
