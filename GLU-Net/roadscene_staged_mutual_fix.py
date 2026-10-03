"""Fresh RoadScene coarse -> fine training with nonnegative MutualMatching.

This is an isolated recipe. It starts coarse training from the original
pretrained GLU-Net weights, never from the epoch-37 continuation. The only
matching change is clamp_min(0) immediately before the original
MutualMatching. The existing staged trainer still controls parameter groups,
losses, validation selection, full-state checkpoints and early stopping.
Joint training and the locked test split are intentionally unavailable here.
"""

import argparse
import importlib
import json
from pathlib import Path

import torch

import roadscene_staged as staged
from roadscene_coarse import sha256_file


FIX = "nonnegative_correlation_before_mutual_matching"
SCRIPT = Path(__file__).resolve()


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("coarse", "fine"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--code-check", action="store_true")
    args = parser.parse_args()
    if args.resume and args.code_check:
        parser.error("A one-batch code check cannot resume")
    if not args.pretrained.is_file():
        parser.error("Pretrained GLU-Net weight is missing")
    if not args.data_root.is_dir():
        parser.error("RoadScene root is missing")

    manifest = args.output / "fresh_mutual_fix_recipe.json"
    expected = {
        "matching_fix": FIX,
        "pretrained_sha256": sha256_file(args.pretrained),
        "script_sha256": sha256_file(SCRIPT),
        "train_split": "train",
        "selection_split": "val",
        "test_pairs_accessed": False,
        "source": "original pretrained GLU-Net, not epoch-37 continuation",
    }
    if manifest.exists():
        recorded = json.loads(manifest.read_text(encoding="utf-8"))
        if recorded != expected:
            parser.error("Existing recipe manifest differs from this run")
    elif args.output.exists() and any(args.output.iterdir()):
        parser.error("Output is not empty and has no matching recipe manifest")
    elif args.stage == "fine" or args.resume:
        parser.error("Start with a fresh coarse stage in this output directory")
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(expected, indent=2), encoding="utf-8")

    glunet_module = importlib.import_module("models.our_models.GLUNet")
    original_matching = glunet_module.MutualMatching
    original_provenance = staged.provenance
    original_checkpoint_payload = staged.checkpoint_payload
    original_new_model = staged.new_model
    original_save_report = staged.save_stage_report

    def fixed_matching(correlation):
        return original_matching(correlation.clamp_min(0))

    all_zero = fixed_matching(torch.zeros(1, 1, 16, 16, 16, 16))
    if not (torch.isfinite(all_zero).all() and
            torch.count_nonzero(all_zero) == 0):
        raise RuntimeError("All-zero correlation query is unsafe")

    def fixed_provenance(recipe_args):
        record = original_provenance(recipe_args)
        record["code_sha256"][SCRIPT.name] = sha256_file(SCRIPT)
        return record

    def fixed_checkpoint_payload(*payload_args, **payload_kwargs):
        payload = original_checkpoint_payload(*payload_args, **payload_kwargs)
        payload["matching_fix"] = FIX
        payload["training_origin"] = expected["source"]
        return payload

    def checked_new_model(recipe_args, device, from_previous=True):
        if recipe_args.stage == "fine" and from_previous:
            parent = staged.best_path(recipe_args, "coarse")
            if not parent.is_file():
                raise FileNotFoundError(f"Fresh coarse checkpoint missing: {parent}")
            payload = torch.load(parent, map_location="cpu", weights_only=False)
            if (payload.get("matching_fix") != FIX or
                    payload.get("training_origin") != expected["source"] or
                    payload.get("code_check_only") != recipe_args.code_check):
                raise ValueError("Fine stage requires this recipe's fresh coarse best")
        return original_new_model(recipe_args, device, from_previous)

    def fixed_save_report(*report_args, **report_kwargs):
        report = original_save_report(*report_args, **report_kwargs)
        report["matching_fix"] = FIX
        report["training_origin"] = expected["source"]
        report["test_pairs_accessed"] = False
        return report

    glunet_module.MutualMatching = fixed_matching
    staged.provenance = fixed_provenance
    staged.checkpoint_payload = fixed_checkpoint_payload
    staged.new_model = checked_new_model
    staged.save_stage_report = fixed_save_report
    try:
        staged.main()
    finally:
        glunet_module.MutualMatching = original_matching
        staged.provenance = original_provenance
        staged.checkpoint_payload = original_checkpoint_payload
        staged.new_model = original_new_model
        staged.save_stage_report = original_save_report


if __name__ == "__main__":
    run()
