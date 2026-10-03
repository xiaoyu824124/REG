"""Read-only forensic replay of coarse epoch 38 from the selected epoch-37 state.

The original epoch 30 checkpoint does not exist, so epoch 31 cannot be replayed
exactly. This script never writes a model checkpoint or reads the test split.
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
from roadscene_coarse import correspondence_targets, evaluate, sha256_file


def stats(tensor):
    values = tensor.detach().float()
    return {"min": float(values.min()), "max": float(values.max()),
            "abs_max": float(values.abs().max()),
            "finite": bool(torch.isfinite(values).all())}


def norm_of_grads(parameters):
    return sum(float(p.grad.detach().float().square().sum())
               for p in parameters if p.grad is not None) ** .5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nonnegative-matching", action="store_true",
                        help="In-memory counterfactual: ReLU correlation before MutualMatching")
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("stage") != "coarse" or payload.get("epoch") != 37:
        raise ValueError("This audit requires the selected epoch-37 coarse checkpoint")
    if payload.get("base_sha256") != sha256_file(args.pretrained):
        raise ValueError("Pretrained weight mismatch")
    root = Path(__file__).resolve().parent
    mismatch = [name for name, digest in payload["code_sha256"].items()
                if sha256_file(root / name) != digest]
    if mismatch:
        raise ValueError(f"Training source changed: {mismatch}")
    history = json.loads(args.reference_history.read_text(encoding="utf-8"))
    reference = next(row for row in history if row["epoch"] == 38)
    torch.set_rng_state(payload["torch_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    random.setstate(payload["python_rng_state"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_rng_state(payload["cuda_rng_state"])
    model = staged.GLUNet_model(evaluation=False, pyramid_type="VGG",
                                cyclic_consistency=True,
                                backbone_pretrained=False,
                                coarse_attention=True, coarse_dns=True,
                                local_dcn_steps=1)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(device).eval()
    selected, active = staged.configure_trainability(model, "coarse", 38)
    parameters = [p for name in active for p in selected[name]]
    optimizer, scheduler = staged.make_optimizer(model, "coarse")
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    trainset = RoadScenePairs(args.data_root, "train")
    valset = RoadScenePairs(args.data_root, "val")
    if len(trainset) != 176 or len(valset) != 23:
        raise ValueError("Expected 176 train and 23 val pairs")
    generator = torch.Generator().manual_seed(staged.SEED + 38)
    loader = DataLoader(trainset, batch_size=2, shuffle=True,
                        generator=generator, num_workers=0,
                        pin_memory=device.type == "cuda")
    glunet_module = importlib.import_module("models.our_models.GLUNet")
    original_matching = glunet_module.MutualMatching
    original_stats = staged.coarse_statistics
    original_flow = model.coarsest_resolution_flow
    capture = {}

    def traced_matching(corr4d):
        shape = corr4d.shape
        b, _, hs, ws, ht, wt = shape
        capture["raw_corr"] = stats(corr4d)
        if args.nonnegative_matching:
            corr4d = corr4d.clamp_min(0)
        max_source = corr4d.reshape(b, hs * ws, ht, wt).max(dim=1).values
        max_target = corr4d.reshape(b, hs, ws, ht * wt).max(dim=-1).values
        denom = torch.cat(((max_source + 1e-5).abs().flatten(),
                           (max_target + 1e-5).abs().flatten()))
        capture["matching_min_abs_denominator"] = float(denom.min())
        capture["matching_denominator_below_1e-3"] = int((denom < 1e-3).sum())
        capture["matching_nonpositive_maxima"] = int(
            (max_source <= 0).sum() + (max_target <= 0).sum())
        result = original_matching(corr4d)
        capture["matched_corr"] = stats(result)
        return result

    def traced_flow(target_features, source_features, height, width,
                    return_corr=False):
        capture["target_feature"] = stats(target_features)
        capture["source_feature"] = stats(source_features)
        result = original_flow(target_features, source_features, height,
                               width, return_corr=return_corr)
        capture["corr_logits"] = stats(result[1])
        capture["predicted_flow"] = stats(result[0])
        return result

    def traced_statistics(predicted, corr, truth, valid, return_counts=False):
        indices, corr_valid = correspondence_targets(truth, valid)
        capture["valid_coarse"] = int(valid.sum())
        capture["valid_corr"] = int(corr_valid.sum())
        capture["truth_flow"] = stats(truth[valid[:, None].expand_as(truth)])
        capture["corr_target_min"] = int(indices[corr_valid].min())
        capture["corr_target_max"] = int(indices[corr_valid].max())
        return original_stats(predicted, corr, truth, valid,
                              return_counts=return_counts)

    glunet_module.MutualMatching = traced_matching
    model.coarsest_resolution_flow = traced_flow
    staged.coarse_statistics = traced_statistics
    rows, image_order = [], hashlib.sha256()
    try:
        for number, batch in enumerate(loader, 1):
            capture = {}
            names = list(batch["name"])
            for name in names:
                image_order.update(name.encode("utf-8") + b"\n")
            optimizer.zero_grad(set_to_none=True)
            loss, components = staged.coarse_training_loss(model, batch, device)
            loss.backward()
            pre_clip = norm_of_grads(parameters)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            post_clip = norm_of_grads(parameters)
            before = [p.detach().clone() for p in parameters]
            optimizer.step()
            update_max = max(float((p.detach() - old).abs().max())
                             for p, old in zip(parameters, before))
            row = {"batch": number, "pairs": names, "loss": float(loss.detach()),
                   **components, **capture,
                   "grad_norm_pre_clip": pre_clip,
                   "grad_norm_post_clip": post_clip,
                   "parameter_update_abs_max": update_max,
                   "parameters_finite_after_update": all(
                       bool(torch.isfinite(p).all()) for p in parameters)}
            rows.append(row)
            if not (np.isfinite(row["loss"]) and np.isfinite(pre_clip)
                    and row["parameters_finite_after_update"]):
                break
        observed_order = image_order.hexdigest()
        val = evaluate(model, DataLoader(valset, batch_size=1, shuffle=False),
                       device) if len(rows) == 88 else None
    finally:
        glunet_module.MutualMatching = original_matching
        staged.coarse_statistics = original_stats
        model.coarsest_resolution_flow = original_flow
    summary = {"replayed_from_epoch": 37, "replayed_epoch": 38,
               "nonnegative_matching": args.nonnegative_matching,
               "checkpoint_sha256": sha256_file(args.checkpoint),
               "batches": len(rows), "image_order_sha256": observed_order,
               "image_order_matches_original": observed_order == reference["image_order_sha256"],
               "validation_replay": val,
               "validation_original": {"epe_256px": reference["val_coarse_epe_256px"],
                                       "corr_top1": reference["val_corr_top1"]},
               "first_corr_ce_above_5": next((row["batch"] for row in rows
                                              if row["corr_ce"] > 5), None),
               "first_nonfinite": next((row["batch"] for row in rows
                                        if not np.isfinite(row["loss"]) or
                                        not row["parameters_finite_after_update"]), None),
               "first_matching_denominator_below_1e-3": next(
                   (row["batch"] for row in rows
                    if row["matching_denominator_below_1e-3"]), None),
               "rows": rows}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "coarse_epoch38_replay.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output / "coarse_epoch38_batches.csv").open(
            "w", newline="", encoding="utf-8") as file:
        fields = ("batch", "pairs", "loss", "coarse_epe_256px", "corr_ce",
                  "valid_coarse", "valid_corr", "grad_norm_pre_clip",
                  "grad_norm_post_clip", "parameter_update_abs_max",
                  "matching_min_abs_denominator",
                  "matching_denominator_below_1e-3",
                  "matching_nonpositive_maxima")
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
