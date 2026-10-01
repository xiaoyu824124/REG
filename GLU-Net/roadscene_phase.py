"""Matched train/validation study of SA/CA, MIND B guide, and phase guide.

Only RoadScene train and val are opened. All three arms start from the same
original GLU-Net weight and have the same optimizer, image order, updates,
selection metric, and full-flow evaluator. No test-based decision is possible.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset

from datasets.roadscene import RoadScenePairs
from models.our_models.phase_congruency import FixedPhaseCongruency2D
from roadscene_coarse import prepare, predict_coarse, sha256_file
from roadscene_compare import (PROTOCOL, evaluator_hashes, evaluate_common,
                               load_glu, save_comparison)
from roadscene_mind import SEED, train_arm


ARMS = ("attention", "mind_b_attention", "phase_attention")
PREVIEW_IDS = ("000002", "000004", "000013")  # selected from earlier validation only


def _save_gray(path, array, scale=1):
    image = Image.fromarray((array.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    if scale != 1:
        image = image.resize((image.width * scale, image.height * scale),
                             resample=Image.Resampling.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


@torch.no_grad()
def save_structure_and_correlation(output, models, dataset, device,
                                   names=PREVIEW_IDS):
    descriptor = FixedPhaseCongruency2D().to(device).eval()
    wanted = set(names)
    for batch in DataLoader(dataset, batch_size=1, shuffle=False):
        name = batch["name"][0]
        if name not in wanted:
            continue
        source, target, source_rgb, target_rgb, *_ = prepare(batch, device)
        for label, rgb in (("ir_source", source_rgb), ("visible_target", target_rgb)):
            maps = descriptor(rgb)
            _save_gray(output / "structure_maps" / f"{name}_{label}.png",
                       maps[0].amax(dim=0))
            _save_gray(output / "structure_maps" / f"{name}_{label}_16.png",
                       torch.nn.functional.interpolate(maps, size=(16, 16),
                                                       mode="area")[0].amax(dim=0), scale=16)
        for arm, model in models.items():
            _, corr = predict_coarse(model, target, source)
            # At the same target-grid query, channels enumerate source bins.
            heat = corr[0, :, 8, 8].reshape(16, 16)
            heat = (heat - heat.amin()) / (heat.amax() - heat.amin()).clamp_min(1e-8)
            _save_gray(output / "correlation_maps" / arm / f"{name}_target_8_8.png",
                       heat, scale=16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--code-check", action="store_true",
                        help="two train/one val pair, one epoch; never report as research result")
    parser.add_argument("--skip-full", action="store_true",
                        help="local code check if CuPy full stages are unavailable")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch size, and learning rate must be positive")
    if args.output.exists() and any(args.output.glob("best_*.pth")):
        parser.error("Output already contains checkpoints; choose a fresh output directory")
    if args.code_check:
        args.epochs = 1
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train_dataset = RoadScenePairs(args.data_root, "train")
    val_dataset = RoadScenePairs(args.data_root, "val")
    if args.code_check:
        train_dataset = Subset(train_dataset, range(2))
        val_dataset = Subset(val_dataset, range(1))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    report = {"code_check_only": args.code_check, "arms": ARMS,
              "train_pairs": len(train_dataset), "val_pairs": len(val_dataset),
              "base_sha256": sha256_file(args.pretrained), "protocol": PROTOCOL,
              "evaluator_hashes": evaluator_hashes(), "epochs": args.epochs,
              "batch_size": args.batch_size, "lr": args.lr, "seed": SEED,
              "selection": "minimum val coarse EPE within each arm; no test access",
              "history": {}, "costs": {}, "results": {}}
    for arm in ARMS:
        history, costs = train_arm(arm, args, train_dataset, val_loader, device)
        report["history"][arm] = history
        report["costs"][arm] = costs
        if arm != ARMS[0]:
            reference = [row["image_order_sha256"] for row in report["history"][ARMS[0]]]
            if [row["image_order_sha256"] for row in history] != reference:
                raise RuntimeError("Arms saw different training image order")
        (args.output / "progress.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    common_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    for arm in ARMS:
        model, meta = load_glu(args.pretrained, args.output / f"best_{arm}.pth", arm, device)
        if args.skip_full:
            from roadscene_coarse import evaluate
            result = {"coarse": evaluate(model, val_loader, device), "checkpoint": meta}
        else:
            summary, rows = evaluate_common(model, common_loader, device,
                previews=args.output / "previews" / arm,
                preview_names=PREVIEW_IDS if not args.code_check else ())
            result = {"summary": summary, "per_image": rows, "checkpoint": meta}
        report["results"][arm] = result
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not args.skip_full:
        models = {arm: load_glu(args.pretrained, args.output / f"best_{arm}.pth",
                                arm, device)[0] for arm in ARMS}
        save_structure_and_correlation(args.output, models, val_dataset, device,
                                       names=PREVIEW_IDS if not args.code_check else
                                       (val_dataset[0]["name"],))
        save_comparison(args.output, report["results"])
        del models
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"code_check_only": args.code_check,
                      "arms": list(report["results"])}, indent=2))


if __name__ == "__main__":
    main()
