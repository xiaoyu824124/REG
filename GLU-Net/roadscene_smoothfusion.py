"""Paired RoadScene evaluation of SmoothFusion keyframe registration and SA/CA.

Run each method in a separate Python process: SmoothFusion and this project
both define a top-level ``models`` package. Only validation may be used to
establish the protocol; test requires a complete paired validation report.
No SmoothFusion source, weights, registration logic, or fusion code is edited.
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.roadscene import RoadScenePairs
from roadscene_metrics import aggregate_flow, flow_metrics


PROTOCOL = {
    "version": 1,
    "name": "SmoothFusion keyframe registration versus selected GLU-Net+SA/CA",
    "direction": "visible target -> warped infrared source",
    "input": "same RoadScene native 512x512 uint8 RGB source and target; each model applies its original preprocessing",
    "output": "dense target-to-source displacement at 512x512 in image pixels",
    "gt": "the same RoadScene .flo target-to-source flow",
    "valid_mask": "finite GT and GT-mapped source coordinate inside the image, independent of prediction",
    "weighted_epe": "sum valid pixel EPE divided by count valid pixels",
    "pair_aepe": "per-pair mean valid pixel EPE; pair mean is unweighted arithmetic mean",
    "batch_size": 1,
    "warmup_forwards": 3,
    "timed_repeats_per_pair": 5,
    "timing": "median of five synchronized on-device forwards per pair, including model preprocessing and 512px output, excluding disk IO, GT transfer and metrics",
    "precision": "float32, no autocast",
    "selection": "no model or hyperparameter selection on independent test",
}
BIN_EDGES = (0, 8, 16, 32, 64, float("inf"))
BIN_NAMES = ("0-8", "8-16", "16-32", "32-64", "64+")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluator_hashes():
    root = Path(__file__).resolve().parent
    return {name: sha256(root / name) for name in (
        "roadscene_smoothfusion.py", "roadscene_metrics.py",
        "datasets/roadscene.py")}


def smooth_source_hashes(root):
    akrf = root / "AKRF"
    return {str(path.relative_to(root)).replace("\\", "/"): sha256(path)
            for path in sorted((akrf / "model" / "reg").rglob("*.py"))}


def model_metadata(args):
    if args.method == "smoothfusion_keyframe":
        akrf = args.smooth_root / "AKRF"
        weight = akrf / "model/reg/GLUNet/pre_trained_models/GLUNet_DPED_CityScape_ADE.pth"
        if not weight.is_file():
            raise FileNotFoundError(weight)
        return {"name": "SmoothFusion keyframe registration module",
                "source_root": str(args.smooth_root.resolve()),
                "source_hashes": smooth_source_hashes(args.smooth_root),
                "pretrained_sha256": sha256(weight),
                "weight_path": str(weight),
                "training_data": "GLU-Net DPED/CityScape/ADE pretraining; no RoadScene registration fine-tune"}
    if not args.pretrained.is_file() or not args.attention_checkpoint.is_file():
        raise FileNotFoundError("SA/CA base or selected checkpoint is missing")
    root = Path(__file__).resolve().parent
    return {"name": "GLU-Net+SA/CA",
            "source_hashes": {name: sha256(root / name) for name in (
                "roadscene_compare.py", "roadscene_coarse.py",
                "models/our_models/GLUNet.py", "models/our_models/coarse_attention.py")},
            "pretrained_sha256": sha256(args.pretrained),
            "weight_path": str(args.pretrained),
            "attention_checkpoint_path": str(args.attention_checkpoint),
            "attention_checkpoint_sha256": sha256(args.attention_checkpoint),
            "training_data": "same GLU-Net base plus 20-epoch RoadScene train-split decoder4 and SA/CA fine-tuning"}


def load_predictor(args, device):
    if args.method == "smoothfusion_keyframe":
        # Match AKRF/model/reg/__init__.py LazyGLUNet.load_GLU_Reg. Its
        # forward calls estimate_flow(ir_image, vi_image), returning dense
        # target->source flow; it never calls the fusion module.
        package = args.smooth_root / "AKRF/model/reg/GLUNet"
        # Import under the GLUNet namespace so its relative ``..utils``
        # references resolve. Keep it separate from this project's models.
        sys.path.insert(0, str(package.parent.resolve()))
        from GLUNet.models.models_compared import GLU_Net
        # The legacy constructor separately requests ImageNet VGG weights.
        # Its complete GLU checkpoint is loaded strictly just afterwards,
        # including the VGG pyramid. Suppress only that redundant download.
        from torchvision import models as torchvision_models
        original_vgg16 = torchvision_models.vgg16
        def vgg_without_download(*vgg_args, **vgg_kwargs):
            vgg_kwargs.pop("pretrained", None)
            vgg_kwargs["weights"] = None
            return original_vgg16(*vgg_args, **vgg_kwargs)
        torchvision_models.vgg16 = vgg_without_download
        try:
            model = GLU_Net(
                path_pre_trained_models=str(package / "pre_trained_models"),
                model_type="DPED_CityScape_ADE", consensus_network=False,
                cyclic_consistency=True, iterative_refinement=True,
                apply_flipping_condition=False)
        finally:
            torchvision_models.vgg16 = original_vgg16
        if model.net.training:
            raise RuntimeError("SmoothFusion registration model is not in eval mode")
        def predict(source, target):
            return model.estimate_flow(source, target, device, mode="channel_first")
        count = sum(p.numel() for p in model.net.parameters())
        return predict, count
    from roadscene_compare import _predict_full, load_glu
    model, checkpoint = load_glu(args.pretrained, args.attention_checkpoint,
                                 "attention", device)
    if checkpoint.get("code_check_only"):
        raise ValueError("SA/CA smoke-check checkpoint cannot be compared")
    def predict(source, target):
        return _predict_full(model, source, target, crft=False)[0]
    count = sum(p.numel() for p in model.parameters())
    return predict, count


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def displacement_bins(predicted, gt, valid):
    displacement = torch.linalg.vector_norm(gt, dim=0)
    error = torch.linalg.vector_norm(predicted - gt, dim=0)
    groups = {}
    for index, label in enumerate(BIN_NAMES):
        selected = valid & (displacement >= BIN_EDGES[index]) & (
            displacement < BIN_EDGES[index + 1])
        groups[label] = {"valid_pixels": int(selected.sum()),
                         "epe_sum": float(error[selected].double().sum())}
    return groups


@torch.no_grad()
def evaluate(predict, dataset, device, max_samples):
    rows = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for index, batch in enumerate(loader):
        if max_samples is not None and index >= max_samples:
            break
        source = batch["source_image"].to(device)
        target = batch["target_image"].to(device)
        if source.shape[-2:] != (512, 512) or target.shape[-2:] != (512, 512):
            raise ValueError("Protocol requires 512x512 inputs")
        if index == 0:
            for _ in range(PROTOCOL["warmup_forwards"]):
                predict(source, target)
            sync(device)
        times = []
        pred = None
        for _ in range(PROTOCOL["timed_repeats_per_pair"]):
            del pred
            sync(device)
            started = time.perf_counter()
            pred = predict(source, target)
            sync(device)
            times.append((time.perf_counter() - started) * 1000)
        gt = batch["flow_map"].to(device).float()
        valid = batch["correspondence_mask"].to(device).bool()
        if pred.shape != gt.shape:
            raise ValueError(f"Expected dense 512px flow {gt.shape}, got {pred.shape}")
        row = {"name": batch["name"][0],
               **flow_metrics(pred[0], gt[0], valid[0]),
               "inference_ms": float(np.median(times)),
               "by_gt_displacement_512px": displacement_bins(pred[0], gt[0], valid[0])}
        rows.append(row)
    summary = aggregate_flow(rows)
    summary["inference_ms_mean_pair_median"] = float(np.mean([r["inference_ms"] for r in rows]))
    summary["by_gt_displacement_512px"] = {}
    for label in BIN_NAMES:
        count = sum(r["by_gt_displacement_512px"][label]["valid_pixels"] for r in rows)
        total = sum(r["by_gt_displacement_512px"][label]["epe_sum"] for r in rows)
        summary["by_gt_displacement_512px"][label] = {
            "valid_pixels": count, "epe_512px": total / count if count else None}
    return summary, rows


def manifest(dataset):
    entries = [{"name": paths[2].stem,
                "files": {str(p.relative_to(dataset.root)).replace("\\", "/"): sha256(p)
                          for p in paths}} for paths in dataset.samples]
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def evaluation(args):
    if args.split == "test" and (args.max_samples or not args.validated_report):
        raise ValueError("Complete paired validation report required before locked test")
    torch.manual_seed(2026)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = model_metadata(args)
    hashes = evaluator_hashes()
    if args.split == "test":
        previous = json.loads(args.validated_report.read_text(encoding="utf-8"))
        if previous.get("split") != "val" or previous.get("protocol") != PROTOCOL or \
           previous.get("evaluator_hashes") != hashes or \
           previous.get("summary", {}).get("samples") != 23 or \
           previous.get("methods", {}).get(args.method, {}).get("model") != metadata:
            raise ValueError("Test protocol/model differs from complete paired validation")
    predictor, params = load_predictor(args, device)
    dataset = RoadScenePairs(args.data_root, args.split)
    summary, rows = evaluate(predictor, dataset, device, args.max_samples)
    report = {"method": args.method, "split": args.split,
              "dataset": str(args.data_root), "pair_manifest_sha256": manifest(dataset),
              "protocol": PROTOCOL, "evaluator_hashes": hashes,
              "model": metadata, "parameters": params,
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
              "max_samples": args.max_samples, "summary": summary, "per_image": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"method": args.method, "split": args.split,
                      "summary": summary}, indent=2), flush=True)


def compare(args):
    a = json.loads(args.smooth_report.read_text(encoding="utf-8"))
    b = json.loads(args.attention_report.read_text(encoding="utf-8"))
    if a["method"] != "smoothfusion_keyframe" or b["method"] != "attention":
        raise ValueError("Expected SmoothFusion keyframe and SA/CA reports")
    if a["split"] != b["split"] or a["protocol"] != b["protocol"] or \
       a["evaluator_hashes"] != b["evaluator_hashes"] or \
       a["pair_manifest_sha256"] != b["pair_manifest_sha256"]:
        raise ValueError("Image pair, evaluator or protocol mismatch")
    expected = 23 if a["split"] == "val" else 22
    if a["max_samples"] or b["max_samples"] or \
       a["summary"]["samples"] != expected or b["summary"]["samples"] != expected:
        raise ValueError("Full split required for paired comparison")
    left = {row["name"]: row for row in a["per_image"]}
    right = {row["name"]: row for row in b["per_image"]}
    if left.keys() != right.keys():
        raise ValueError("Methods evaluated different pairs")
    pairs = []
    for name in left:
        x, y = left[name], right[name]
        if x["valid_full"] != y["valid_full"]:
            raise ValueError(f"GT mask mismatch for {name}")
        pairs.append({"name": name, "valid_pixels": x["valid_full"],
                      "smoothfusion_keyframe_epe_512px": x["aepe_512px"],
                      "attention_epe_512px": y["aepe_512px"],
                      "smooth_minus_attention_epe_512px": x["aepe_512px"] - y["aepe_512px"],
                      "smoothfusion_keyframe_ms": x["inference_ms"],
                      "attention_ms": y["inference_ms"]})
    combined = {"split": a["split"], "protocol": PROTOCOL,
                "evaluator_hashes": a["evaluator_hashes"],
                "pair_manifest_sha256": a["pair_manifest_sha256"],
                "summary": {"smoothfusion_keyframe": a["summary"],
                            "attention": b["summary"]},
                "methods": {"smoothfusion_keyframe": {"model": a["model"],
                                                       "parameters": a["parameters"]},
                            "attention": {"model": b["model"],
                                          "parameters": b["parameters"]}},
                "per_image": pairs}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(combined, indent=2),
                                                   encoding="utf-8")
    with (args.output / "per_image.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    print(json.dumps({name: item["final_flow_epe_512px"]
                      for name, item in combined["summary"].items()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = commands.add_parser("eval")
    evaluate_parser.add_argument("--method", choices=("smoothfusion_keyframe", "attention"), required=True)
    evaluate_parser.add_argument("--data-root", type=Path, required=True)
    evaluate_parser.add_argument("--smooth-root", type=Path)
    evaluate_parser.add_argument("--pretrained", type=Path)
    evaluate_parser.add_argument("--attention-checkpoint", type=Path)
    evaluate_parser.add_argument("--split", choices=("val", "test"), default="val")
    evaluate_parser.add_argument("--validated-report", type=Path)
    evaluate_parser.add_argument("--max-samples", type=int)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--smooth-report", type=Path, required=True)
    compare_parser.add_argument("--attention-report", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "eval":
        if args.method == "smoothfusion_keyframe" and not args.smooth_root:
            parser.error("SmoothFusion requires --smooth-root")
        if args.method == "attention" and (not args.pretrained or not args.attention_checkpoint):
            parser.error("SA/CA requires --pretrained and --attention-checkpoint")
        if args.max_samples is not None and args.max_samples < 1:
            parser.error("max-samples must be positive")
        evaluation(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
