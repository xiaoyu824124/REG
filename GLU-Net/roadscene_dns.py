"""Fair train/validation ablation: GLU-Net, SA/CA, DNS, DNS+point contrast.

All four arms start from one GLU-Net checkpoint, use identical train/val pairs,
image order, coarse decoder optimization, learning rate, epoch budget, and
evaluation code. The held-out test split is never opened by this script.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.roadscene import RoadScenePairs
from roadscene_coarse import (coarse_statistics, correspondence_targets,
                              evaluate, evaluate_full, extract_coarse_features,
                              make_model, predict_coarse, prepare, sha256_file)


ARMS = ("baseline", "attention", "dns", "dns_contrastive")
SEED = 2026


def make_arm(name, pretrained, device, train_decoder=False):
    return make_model(pretrained, attention=name == "attention", device=device,
                      train_decoder=train_decoder,
                      dns=name in {"dns", "dns_contrastive"})


def point_contrastive_loss(model, features, truth, valid, temperature):
    """GT-warped 16x16 IR positives; spatially separated same-pair negatives.

    This is a supervised 2D adaptation for RoadScene, not the DSIR paper's
    same-image intensity-augmentation objective.
    """
    target, source = model.coarse_dns(*features)
    _, _, height, width = target.shape
    y, x = torch.meshgrid(torch.arange(height, device=truth.device),
                          torch.arange(width, device=truth.device), indexing="ij")
    source_x = x[None].float() + truth[:, 0] / 16.0
    source_y = y[None].float() + truth[:, 1] / 16.0
    grid = torch.stack((source_x / (width - 1) * 2 - 1,
                        source_y / (height - 1) * 2 - 1), dim=-1)
    sampled_source = F.grid_sample(source, grid, mode="bilinear",
                                   padding_mode="zeros", align_corners=True)
    _, corr_valid = correspondence_targets(truth, valid)
    losses = []
    for batch_index in range(target.shape[0]):
        mask = corr_valid[batch_index].flatten()
        if int(mask.sum()) < 2:
            continue
        target_vectors = F.normalize(target[batch_index].flatten(1).T[mask], dim=1)
        source_vectors = F.normalize(sampled_source[batch_index].flatten(1).T[mask], dim=1)
        logits = target_vectors @ source_vectors.T / temperature
        query_xy = torch.stack((x.flatten()[mask], y.flatten()[mask]), dim=1).float()
        match_xy = torch.stack((source_x[batch_index].flatten()[mask],
                                source_y[batch_index].flatten()[mask]), dim=1)
        separated = ((torch.cdist(query_xy, query_xy) >= 2.0) &
                     (torch.cdist(match_xy, match_xy) >= 2.0))
        diagonal = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(~(separated | diagonal), -1e4)
        labels = torch.arange(logits.shape[0], device=logits.device)
        losses.append((F.cross_entropy(logits, labels) +
                       F.cross_entropy(logits.T, labels)) / 2)
    return torch.stack(losses).mean() if losses else target.sum() * 0


def train_arm(name, args, device, train_dataset, val_loader):
    torch.manual_seed(SEED)
    model = make_arm(name, args.pretrained, device, train_decoder=True)
    trainable = list(model.decoder4.parameters())
    if name == "attention":
        trainable += list(model.coarse_attention.parameters())
    elif name in {"dns", "dns_contrastive"}:
        trainable += list(model.coarse_dns.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    order = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, generator=order, num_workers=0)
    best_epe = float("inf")
    history = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, args.epochs + 1):
        losses = []
        contrast_losses = []
        for batch in train_loader:
            source_in, target_in, _, _, truth, valid, _ = prepare(batch, device)
            features = extract_coarse_features(model, target_in, source_in)
            predicted, corr = predict_coarse(model, target_in, source_in, features)
            epe, ce, _ = coarse_statistics(predicted, corr, truth, valid)
            loss = epe + 2.0 * ce
            if name == "dns_contrastive":
                contrast = point_contrastive_loss(model, features, truth, valid,
                                                  args.temperature)
                loss = loss + args.contrastive_weight * contrast
                contrast_losses.append(contrast.item())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        validation = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "train_contrastive_loss": (float(np.mean(contrast_losses))
                                                    if contrast_losses else None),
                        "validation": validation})
        print(f"{name} epoch {epoch}: loss={np.mean(losses):.4f}, "
              f"val_epe={validation['epe_256px']:.4f}, "
              f"corr_top1={validation['corr_top1']:.4f}", flush=True)
        if validation["epe_256px"] < best_epe:
            best_epe = validation["epe_256px"]
            payload = {"arm": name, "epoch": epoch,
                       "decoder4_state_dict": model.decoder4.state_dict(),
                       "validation": validation,
                       "base_sha256": sha256_file(args.pretrained),
                       "budget": {"epochs": args.epochs, "lr": args.lr,
                                  "batch_size": args.batch_size, "seed": SEED,
                                  "code_check": False}}
            if name == "attention":
                payload["attention_state_dict"] = model.coarse_attention.state_dict()
            elif name in {"dns", "dns_contrastive"}:
                payload["dns_state_dict"] = model.coarse_dns.state_dict()
            torch.save(payload, args.output / f"best_{name}.pth")
    if device.type == "cuda":
        torch.cuda.synchronize()
    cost = {"training_seconds": time.perf_counter() - started,
            "training_peak_allocated_mib": (torch.cuda.max_memory_allocated() / 2**20
                                            if device.type == "cuda" else None)}
    del optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return history, cost


def load_selected(name, args, device):
    model = make_arm(name, args.pretrained, device)
    path = args.output / f"best_{name}.pth"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("arm") != name:
        raise ValueError(f"Checkpoint {path} belongs to a different arm")
    model.decoder4.load_state_dict(payload["decoder4_state_dict"])
    if name == "attention":
        model.coarse_attention.load_state_dict(payload["attention_state_dict"])
    elif name in {"dns", "dns_contrastive"}:
        model.coarse_dns.load_state_dict(payload["dns_state_dict"])
    return model.eval(), path, payload


def initial_equivalence(args, device, batch):
    predictions = {}
    for name in ARMS:
        torch.manual_seed(SEED)
        model = make_arm(name, args.pretrained, device)
        source, target, *_ = prepare(batch, device)
        with torch.no_grad():
            predictions[name] = predict_coarse(model, target, source)
        del model
    base_flow, base_corr = predictions["baseline"]
    differences = {}
    for name in ARMS[1:]:
        flow, corr = predictions[name]
        differences[name] = {
            "flow_max_abs_diff": float((flow - base_flow).abs().max().item()),
            "corr_max_abs_diff": float((corr - base_corr).abs().max().item()),
        }
    if any(max(values.values()) > 1e-6 for values in differences.values()):
        raise RuntimeError(f"Initial gated branches differ from baseline: {differences}")
    return differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--skip-full", action="store_true",
                        help="run only coarse validation if CuPy/CUDA is unavailable")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch size, and learning rate must be positive")
    if args.contrastive_weight < 0 or args.temperature <= 0:
        parser.error("contrastive weight must be nonnegative; temperature positive")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    train_dataset = RoadScenePairs(args.data_root, "train")
    val_loader = DataLoader(RoadScenePairs(args.data_root, "val"),
                            batch_size=args.batch_size, shuffle=False, num_workers=0)
    checks = initial_equivalence(args, device, next(iter(val_loader)))
    print(f"Initial gate checks: {checks}", flush=True)
    histories = {}
    costs = {}
    for name in ARMS:
        histories[name], costs[name] = train_arm(name, args, device, train_dataset,
                                                val_loader)
        (args.output / "training_progress.json").write_text(
            json.dumps({"history": histories, "costs": costs}, indent=2),
            encoding="utf-8")
    results = {}
    for name in ARMS:
        model, path, payload = load_selected(name, args, device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        coarse = evaluate(model, val_loader, device)
        coarse_peak = (torch.cuda.max_memory_allocated() / 2**20
                       if device.type == "cuda" else None)
        result = {"coarse": coarse,
                  "selected_epoch": payload["epoch"],
                  "checkpoint_sha256": sha256_file(path),
                  "cost": {**costs[name],
                           "inference_coarse_peak_allocated_mib": coarse_peak},
                  "total_parameters": sum(parameter.numel()
                                          for parameter in model.parameters()),
                  "trained_parameters": (sum(parameter.numel()
                                             for parameter in model.decoder4.parameters()) +
                                         (sum(parameter.numel() for parameter in
                                              (model.coarse_attention if name == "attention"
                                               else model.coarse_dns).parameters())
                                          if name != "baseline" else 0))}
        if not args.skip_full:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            result["final_flow"] = evaluate_full(model, val_loader, device)
            result["cost"]["inference_full_peak_allocated_mib"] = (
                torch.cuda.max_memory_allocated() / 2**20
                if device.type == "cuda" else None)
        results[name] = result
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {"dataset": str(args.data_root), "split": "val",
              "train_pairs": len(train_dataset), "val_pairs": len(val_loader.dataset),
              "pretrained_sha256": sha256_file(args.pretrained),
              "seed": SEED, "epochs": args.epochs, "batch_size": args.batch_size,
              "lr": args.lr, "contrastive_weight": args.contrastive_weight,
              "temperature": args.temperature, "torch": torch.__version__,
              "cuda": torch.version.cuda, "device": str(device),
              "initial_equivalence": checks, "history": histories,
              "results": results,
              "selection": "minimum validation coarse EPE for each arm",
              "note": "No test images or test metrics were read by this script."}
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2),
                                                   encoding="utf-8")
    print(json.dumps({"results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
