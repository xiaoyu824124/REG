"""Retrain staged fine refinement from the isolated, fixed coarse best.

The only coarse-path change remains clamping signed correlation before the
original MutualMatching. Existing staged checkpoints are never overwritten.
"""

import argparse
import csv
import importlib
import json
import random
from pathlib import Path

import numpy as np
import torch

import roadscene_staged as staged
from datasets.roadscene import RoadScenePairs
from roadscene_coarse import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.stage = "fine"
    args.code_check = False
    args.workers = 0
    args.resume = False
    args.eval_only = False
    args.eval_split = "val"
    coarse_path = args.output / "best_coarse.pth"
    if not coarse_path.is_file() or (args.output / "best_fine.pth").exists():
        parser.error("Requires isolated best_coarse and no existing fine checkpoint")
    coarse = torch.load(coarse_path, map_location="cpu", weights_only=False)
    if (coarse.get("mutual_fix") != "relu_correlation_before_matching" or
            coarse.get("epoch") <= 37 or coarse.get("stage") != "coarse" or
            coarse.get("base_sha256") != sha256_file(args.pretrained)):
        raise ValueError("Selected fixed coarse checkpoint is incompatible")
    if not (args.output / "report_fixed_coarse.json").is_file():
        raise ValueError("The completed 38..57 fixed coarse report is missing")
    torch.manual_seed(staged.SEED)
    np.random.seed(staged.SEED)
    random.seed(staged.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset = RoadScenePairs(args.data_root, "train")
    valset = RoadScenePairs(args.data_root, "val")
    if len(trainset) != 176 or len(valset) != 23:
        raise ValueError("Expected 176 train and 23 validation pairs")
    glunet_module = importlib.import_module("models.our_models.GLUNet")
    original_matching = glunet_module.MutualMatching
    original_provenance = staged.provenance

    def fixed_matching(corr):
        return original_matching(corr.clamp_min(0))

    def fixed_provenance(recipe_args):
        record = original_provenance(recipe_args)
        record["code_sha256"]["roadscene_mutual_fix_fine.py"] = sha256_file(
            Path(__file__))
        return record

    glunet_module.MutualMatching = fixed_matching
    staged.provenance = fixed_provenance
    try:
        history, latest = staged.train_stage(args, trainset, valset, device)
        result = staged.evaluate_selected(args, valset, device)
        report = staged.save_stage_report(args, result, history, latest)
        report["coarse_matching_fix"] = "relu_correlation_before_matching"
        report["selected_coarse_epoch"] = coarse["epoch"]
        report["selected_coarse_sha256"] = sha256_file(coarse_path)
        report["test_pairs_accessed"] = False
        (args.output / "report_fine.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"stage": "fine", "coarse_epoch": coarse["epoch"],
                          "selected_fine_epoch": result["selected_epoch"],
                          "final_epe_512px": result["common"]["final_flow_epe_512px"],
                          "coarse_epe_256px": result["common"]["coarse_epe_256px"],
                          "test_pairs_accessed": False}, indent=2), flush=True)
    finally:
        glunet_module.MutualMatching = original_matching
        staged.provenance = original_provenance


if __name__ == "__main__":
    main()
