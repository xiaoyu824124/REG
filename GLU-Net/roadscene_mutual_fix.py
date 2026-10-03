"""Isolated RoadScene coarse continuation with nonnegative MutualMatching input.

Restores the full selected epoch-37 state, trains epochs 38..57, and writes
only to a new output directory. The original GLU-Net implementation is never
edited or overwritten. Train and validation splits only.
"""

import argparse
import csv
import hashlib
import importlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import roadscene_staged as staged
from datasets.roadscene import RoadScenePairs
from models.our_models.GLUNet import GLUNet_model
from roadscene_coarse import (coarse_statistics, evaluate as evaluate_coarse,
                              evaluate_full, predict_coarse, prepare,
                              sha256_file)


def distribution(values):
    return {"min": float(values.min()), "max": float(values.max()),
            "finite": bool(torch.isfinite(values).all())}


def paired_epoch37(model, valset, device, glunet_module, original_matching):
    """Check whether clamping changes already-normal validation pairs."""
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(valset, batch_size=1, shuffle=False):
            source_in, target_in, _, _, truth, valid, _ = prepare(batch, device)
            row = {"name": batch["name"][0], "valid_coarse": int(valid.sum())}
            for label, matching in (
                    ("original", original_matching),
                    ("fixed", lambda corr: original_matching(corr.clamp_min(0)))):
                glunet_module.MutualMatching = matching
                flow, corr = predict_coarse(model, target_in, source_in)
                epe, ce, hit = coarse_statistics(flow, corr, truth, valid)
                row[f"{label}_coarse_epe_256px"] = float(epe)
                row[f"{label}_corr_ce"] = float(ce)
                row[f"{label}_corr_top1"] = float(hit)
            row["delta_coarse_epe_256px"] = (
                row["fixed_coarse_epe_256px"] - row["original_coarse_epe_256px"])
            row["delta_corr_top1"] = (
                row["fixed_corr_top1"] - row["original_corr_top1"])
            rows.append(row)
    return rows


def save_checkpoint(payload, output, filename):
    staged.atomic_save(payload, output / filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--coarse-checkpoint", type=Path, required=True)
    parser.add_argument("--original-history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.stage = "coarse"
    args.code_check = False
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty; this run never overwrites results")
    if args.output.resolve() == args.coarse_checkpoint.resolve().parent:
        parser.error("Output must be separate from original checkpoints")
    original = torch.load(args.coarse_checkpoint, map_location="cpu",
                          weights_only=False)
    if (original.get("stage") != "coarse" or original.get("epoch") != 37 or
            original.get("arm") != "dns_attention" or
            original.get("base_sha256") != sha256_file(args.pretrained) or
            original.get("code_check_only")):
        raise ValueError("Expected original full epoch-37 coarse checkpoint")
    root = Path(__file__).resolve().parent
    mismatch = [name for name, digest in original["code_sha256"].items()
                if sha256_file(root / name) != digest]
    if mismatch:
        raise ValueError(f"Original training code differs: {mismatch}")
    needed = ("model_state_dict", "optimizer_state_dict",
              "scheduler_state_dict", "torch_rng_state", "numpy_rng_state",
              "python_rng_state", "cuda_rng_state")
    if any(key not in original for key in needed):
        raise ValueError("Epoch-37 checkpoint lacks full training state")
    original_history = json.loads(args.original_history.read_text(encoding="utf-8"))
    reference = {row["epoch"]: row for row in original_history}
    trainset = RoadScenePairs(args.data_root, "train")
    valset = RoadScenePairs(args.data_root, "val")
    if len(trainset) != 176 or len(valset) != 23:
        raise ValueError("Expected 176 training and 23 validation pairs")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GLUNet_model(evaluation=False, pyramid_type="VGG",
                         cyclic_consistency=True, backbone_pretrained=False,
                         coarse_attention=True, coarse_dns=True,
                         local_dcn_steps=1)
    model.load_state_dict(original["model_state_dict"], strict=True)
    model = model.to(device).eval()
    selected, active = staged.configure_trainability(model, "coarse", 37)
    parameters = [p for name in active for p in selected[name]]
    optimizer, scheduler = staged.make_optimizer(model, "coarse")
    optimizer.load_state_dict(original["optimizer_state_dict"])
    scheduler.load_state_dict(original["scheduler_state_dict"])
    # Construct everything first; restore RNG immediately before paired
    # validation and the deterministic per-epoch training order.
    torch.set_rng_state(original["torch_rng_state"])
    np.random.set_state(original["numpy_rng_state"])
    random.setstate(original["python_rng_state"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(original["cuda_rng_state"])
    glunet_module = importlib.import_module("models.our_models.GLUNet")
    original_matching = glunet_module.MutualMatching
    zero = torch.zeros(1, 1, 16, 16, 16, 16, device=device)
    zero_output = original_matching(zero.clamp_min(0))
    if not (torch.isfinite(zero_output).all() and
            torch.count_nonzero(zero_output) == 0):
        raise RuntimeError("All-zero correlation normalization is unsafe")
    pair_rows = paired_epoch37(model, valset, device, glunet_module,
                               original_matching)
    args.output.mkdir(parents=True)
    with (args.output / "epoch37_pair_check.csv").open(
            "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    capture = {"enabled": False, "row": None}

    def fixed_matching(corr):
        positive = corr.clamp_min(0)
        result = original_matching(positive)
        if capture["enabled"]:
            b, _, hs, ws, ht, wt = positive.shape
            source_max = positive.reshape(b, hs * ws, ht, wt).amax(dim=1)
            target_max = positive.reshape(b, hs, ws, ht * wt).amax(dim=-1)
            capture["row"] = {"raw_corr": distribution(corr.detach()),
                              "matched_corr": distribution(result.detach()),
                              "all_zero_queries": int((source_max == 0).sum() +
                                                      (target_max == 0).sum()),
                              "total_queries": source_max.numel() + target_max.numel()}
        return result

    glunet_module.MutualMatching = fixed_matching
    val_loader = DataLoader(valset, batch_size=1, shuffle=False)
    history = []
    batch_rows = []
    best = None
    try:
        coarse37 = evaluate_coarse(model, val_loader, device)
        final37 = evaluate_full(model, val_loader, device)
        baseline = {"epoch": 37,
                    "val_coarse_epe_256px": coarse37["epe_256px"],
                    "val_corr_top1": coarse37["corr_top1"],
                    "val_final_epe_512px": final37["final_flow_epe_512px"],
                    "train_corr_ce": None, "preclip_grad_norm_p50": None,
                    "preclip_grad_norm_p95": None,
                    "preclip_grad_norm_max": None,
                    "raw_corr_min": None, "raw_corr_max": None,
                    "matched_corr_min": None, "matched_corr_max": None,
                    "all_zero_query_fraction": None,
                    "image_order_matches_original": None,
                    "optimizer_steps": original["stage_optimizer_steps"]}
        history.append(baseline)
        best = coarse37["epe_256px"]
        seed_checkpoint = dict(original)
        seed_checkpoint.update({"selection_metric": best,
                                "mutual_fix": "relu_correlation_before_matching",
                                "parent_original_sha256": sha256_file(args.coarse_checkpoint),
                                "experiment_script_sha256": sha256_file(Path(__file__)),
                                "history": history})
        save_checkpoint(seed_checkpoint, args.output, "best_coarse.pth")
        (args.output / "epoch37_baseline.json").write_text(
            json.dumps({"fixed": baseline,
                        "original_history": reference[37],
                        "zero_query_synthetic_safe": True,
                        "pair_check": {"pairs": len(pair_rows),
                                       "worsened_coarse_pairs": sum(
                                           row["delta_coarse_epe_256px"] > 1e-4
                                           for row in pair_rows)}}, indent=2),
            encoding="utf-8")
        print(f"epoch 37 fixed: coarse={best:.4f}, "
              f"top1={coarse37['corr_top1']:.4f}, "
              f"final={final37['final_flow_epe_512px']:.4f}", flush=True)
        steps = original["stage_optimizer_steps"]
        for epoch in range(38, 58):
            model.eval()
            staged.configure_trainability(model, "coarse", epoch)
            generator = torch.Generator().manual_seed(staged.SEED + epoch)
            loader = DataLoader(trainset, batch_size=2, shuffle=True,
                                generator=generator, num_workers=0,
                                pin_memory=device.type == "cuda")
            order = hashlib.sha256()
            epoch_rows = []
            for batch_index, batch in enumerate(loader, 1):
                names = list(batch["name"])
                for name in names:
                    order.update(name.encode("utf-8") + b"\n")
                optimizer.zero_grad(set_to_none=True)
                capture["enabled"] = True
                capture["row"] = None
                loss, components = staged.coarse_training_loss(model, batch,
                                                                device)
                corr_info = capture["row"]
                capture["enabled"] = False
                if corr_info is None:
                    raise RuntimeError("No global correlation was captured")
                loss.backward()
                preclip = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
                postclip = sum(float(p.grad.detach().square().sum())
                               for p in parameters if p.grad is not None) ** .5
                if not (np.isfinite(preclip) and torch.isfinite(loss) and
                        corr_info["raw_corr"]["finite"] and
                        corr_info["matched_corr"]["finite"]):
                    raise RuntimeError(f"Nonfinite value at epoch {epoch}, batch {batch_index}")
                optimizer.step()
                if not all(torch.isfinite(p).all() for p in parameters):
                    raise RuntimeError(f"Nonfinite weights at epoch {epoch}, batch {batch_index}")
                steps += 1
                epoch_rows.append({"epoch": epoch, "batch": batch_index,
                                   "pairs": names, "loss": float(loss.detach()),
                                   **components, "grad_norm_pre_clip": preclip,
                                   "grad_norm_post_clip": postclip,
                                   **corr_info})
            batch_rows.extend(epoch_rows)
            image_order = order.hexdigest()
            if image_order != reference[epoch]["image_order_sha256"]:
                raise RuntimeError(f"Image order mismatch in epoch {epoch}")
            coarse = evaluate_coarse(model, val_loader, device)
            final = evaluate_full(model, val_loader, device)
            scheduler.step(coarse["epe_256px"])
            grad = np.array([row["grad_norm_pre_clip"] for row in epoch_rows])
            count = sum(row["total_queries"] for row in epoch_rows)
            record = {"epoch": epoch,
                      "train_loss": float(np.mean([row["loss"] for row in epoch_rows])),
                      "train_coarse_epe_256px": float(np.mean([
                          row["coarse_epe_256px"] for row in epoch_rows])),
                      "train_corr_ce": float(np.mean([
                          row["corr_ce"] for row in epoch_rows])),
                      "preclip_grad_norm_p50": float(np.quantile(grad, .5)),
                      "preclip_grad_norm_p95": float(np.quantile(grad, .95)),
                      "preclip_grad_norm_max": float(grad.max()),
                      "raw_corr_min": min(row["raw_corr"]["min"]
                                          for row in epoch_rows),
                      "raw_corr_max": max(row["raw_corr"]["max"]
                                          for row in epoch_rows),
                      "matched_corr_min": min(row["matched_corr"]["min"]
                                              for row in epoch_rows),
                      "matched_corr_max": max(row["matched_corr"]["max"]
                                              for row in epoch_rows),
                      "all_zero_query_fraction": sum(
                          row["all_zero_queries"] for row in epoch_rows) / count,
                      "val_coarse_epe_256px": coarse["epe_256px"],
                      "val_corr_top1": coarse["corr_top1"],
                      "val_final_epe_512px": final["final_flow_epe_512px"],
                      "image_order_sha256": image_order,
                      "image_order_matches_original": True,
                      "optimizer_steps": steps,
                      "learning_rate_after_scheduler": optimizer.param_groups[0]["lr"]}
            history.append(record)
            if coarse["epe_256px"] < best - .01:
                best = coarse["epe_256px"]
                selected_epoch = epoch
                improved = True
            else:
                improved = False
            state = staged.checkpoint_payload(
                model, optimizer, scheduler, args, epoch,
                coarse["epe_256px"], steps, steps,
                sha256_file(args.coarse_checkpoint), 0, history, epoch == 57)
            state["mutual_fix"] = "relu_correlation_before_matching"
            state["parent_original_sha256"] = sha256_file(args.coarse_checkpoint)
            state["experiment_script_sha256"] = sha256_file(Path(__file__))
            save_checkpoint(state, args.output, "latest_coarse.pth")
            if improved:
                save_checkpoint(state, args.output, "best_coarse.pth")
            (args.output / "history_fixed.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8")
            with (args.output / "batch_trace.jsonl").open(
                    "a", encoding="utf-8") as file:
                for row in epoch_rows:
                    file.write(json.dumps(row) + "\n")
            print(f"epoch {epoch}: CE={record['train_corr_ce']:.4f}, "
                  f"grad95={record['preclip_grad_norm_p95']:.1f}, "
                  f"zero={record['all_zero_query_fraction']:.4%}, "
                  f"coarse={coarse['epe_256px']:.4f}, "
                  f"top1={coarse['corr_top1']:.4f}, "
                  f"final={final['final_flow_epe_512px']:.4f}, "
                  f"best={best:.4f}", flush=True)
        selected = torch.load(args.output / "best_coarse.pth",
                              map_location="cpu", weights_only=False)
        outcome = {"split": "train+val", "test_pairs_accessed": False,
                   "original_checkpoint_sha256": sha256_file(args.coarse_checkpoint),
                   "original_best_epoch": 37,
                   "selected_epoch": selected["epoch"],
                   "selected_coarse_epe_256px": best,
                   "improved_beyond_0.01": selected["epoch"] > 37,
                   "all_image_orders_match": all(
                       row["image_order_matches_original"] is True
                       for row in history[1:]),
                   "zero_query_synthetic_safe": True,
                   "history": history}
        (args.output / "report_fixed_coarse.json").write_text(
            json.dumps(outcome, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in outcome.items()
                          if key != "history"}, indent=2), flush=True)
    finally:
        glunet_module.MutualMatching = original_matching


if __name__ == "__main__":
    main()
