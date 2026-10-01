"""Matched six-arm MIND train/val ablation. No test split is accessible.

baseline, SA/CA, A, A+SA/CA, B, B+SA/CA start from the SAME original weight,
train decoder4 + only their added modules for 20 epochs, and select minimum
validation coarse EPE. Existing historical checkpoints remain untouched.
Do not compare newly trained MIND to a control with a different epoch budget.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets.roadscene import RoadScenePairs
from roadscene_coarse import coarse_statistics, evaluate, prepare, predict_coarse, sha256_file
from roadscene_compare import PROTOCOL, evaluator_hashes, evaluate_common, load_glu, save_comparison

ARMS = ("baseline", "attention", "mind_a", "mind_a_attention", "mind_b", "mind_b_attention")
SEED = 2026


def train_arm(arm, args, train_dataset, val_loader, device):
    torch.manual_seed(SEED)
    # load_glu intentionally refuses untrained MIND accuracy comparisons;
    # training constructs a fresh model directly.
    from roadscene_coarse import make_model
    attention = arm in {"attention", "mind_a_attention", "mind_b_attention",
                        "phase_attention"}
    route = "a" if arm.startswith("mind_a") else "b" if arm.startswith("mind_b") else None
    phase = arm == "phase_attention"
    model = make_model(args.pretrained, attention, device, train_decoder=True,
                       mind=route, phase=phase)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED), num_workers=0)
    best = float("inf")
    history = []
    synchronize = lambda: torch.cuda.synchronize() if device.type == "cuda" else None
    synchronize()
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, args.epochs + 1):
        order = []
        losses = []
        model.eval()  # matches existing experiment: fixed BN statistics
        for batch in loader:
            order.extend(batch["name"])
            source, target, _, _, truth, valid, _ = prepare(batch, device)
            flow, corr = predict_coarse(model, target, source)
            epe, ce, _ = coarse_statistics(flow, corr, truth, valid)
            loss = epe + 2 * ce
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss in {arm}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = evaluate(model, val_loader, device)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                  "image_order_sha256": hashlib.sha256(json.dumps(order).encode()).hexdigest(),
                  "validation": validation}
        history.append(record)
        print(f"{arm} epoch {epoch}: val coarse EPE={validation['epe_256px']:.4f}", flush=True)
        if validation["epe_256px"] < best:
            best = validation["epe_256px"]
            payload = {"arm": arm, "epoch": epoch, "validation": validation,
                       "decoder4_state_dict": model.decoder4.state_dict(),
                       "base_sha256": sha256_file(args.pretrained),
                       "budget": {"epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
                                  "seed": SEED, "code_check": args.code_check}}
            if attention:
                payload["attention_state_dict"] = model.coarse_attention.state_dict()
            if route:
                payload["mind_state_dict"] = model.coarse_mind.state_dict()
            if phase:
                payload["phase_state_dict"] = model.coarse_phase.state_dict()
            torch.save(payload, args.output / f"best_{arm}.pth")
    synchronize()
    costs = {"training_seconds": time.perf_counter() - started,
             "trainable_parameters": sum(parameter.numel() for parameter in trainable),
             "peak_training_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None}
    del trainable, optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return history, costs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--code-check", action="store_true", help="2 train + 1 val pair, one epoch; never experimental evidence")
    parser.add_argument("--skip-full", action="store_true", help="only if CUDA/CuPy local stages are unavailable")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch size, lr must be positive")
    if args.output.exists() and any(args.output.glob("best_*.pth")):
        parser.error("Output already contains selected checkpoints; choose a new output directory")
    if args.code_check:
        args.epochs = 1
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(SEED)
    train_dataset = RoadScenePairs(args.data_root, "train")
    val_dataset = RoadScenePairs(args.data_root, "val")
    if args.code_check:
        train_dataset = Subset(train_dataset, range(2))
        val_dataset = Subset(val_dataset, range(1))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    report = {"code_check_only": args.code_check, "train_pairs": len(train_dataset),
              "protocol": PROTOCOL, "evaluator_hashes": evaluator_hashes(),
              "val_pairs": len(val_dataset), "epochs": args.epochs, "batch_size": args.batch_size,
              "lr": args.lr, "seed": SEED, "base_sha256": sha256_file(args.pretrained),
              "selection": "minimum val coarse EPE per arm; no test access",
              "history": {}, "costs": {}, "results": {}}
    for arm in ARMS:
        history, costs = train_arm(arm, args, train_dataset, val_loader, device)
        report["history"][arm] = history
        report["costs"][arm] = costs
        if arm != "baseline":
            if [row["image_order_sha256"] for row in history] != [row["image_order_sha256"] for row in report["history"]["baseline"]]:
                raise RuntimeError("Arms saw different training image order")
        (args.output / "progress.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    common_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    for arm in ARMS:
        model, meta = load_glu(args.pretrained, args.output / f"best_{arm}.pth", arm, device)
        if args.skip_full:
            result = {"coarse": evaluate(model, val_loader, device), "checkpoint": meta}
        else:
            summary, rows = evaluate_common(model, common_loader, device)
            result = {"summary": summary, "per_image": rows, "checkpoint": meta}
        report["results"][arm] = result
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not args.skip_full:
        save_comparison(args.output, report["results"])
    print(json.dumps({"code_check_only": args.code_check, "arms": list(report["results"])}, indent=2))


if __name__ == "__main__":
    main()
