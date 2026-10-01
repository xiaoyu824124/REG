"""Validation-only inference probe for GLU-Net's pretrained flow upsamplers.

This deliberately changes the model at inference time without retraining. It
tests whether a geometric improvement in local search coverage is sufficient
to improve the *full* decoder. It is not a candidate model comparison.
"""

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.roadscene import RoadScenePairs
from roadscene_coarse import sha256_file
from roadscene_compare import load_glu


class BilinearFlowUpsample(nn.Module):
    def forward(self, flow):
        return F.interpolate(flow, scale_factor=2, mode="bilinear",
                             align_corners=False)


@torch.no_grad()
def evaluate(model, loader, device):
    rows = []
    for batch in loader:
        source = batch["source_image"].to(device)
        target = batch["target_image"].to(device)
        gt = batch["flow_map"].to(device).float()
        valid = batch["correspondence_mask"].to(device).bool()
        source, target, source_256, target_256, *_ = model.pre_process_data(
            source, target, device=device)
        _, flows = model(target, source, target_256, source_256)
        pred = F.interpolate(flows[-1], size=(512, 512), mode="bilinear",
                             align_corners=False)
        error = torch.linalg.vector_norm(pred - gt, dim=1)
        rows.append({"name": batch["name"][0], "valid_pixels": int(valid.sum()),
                     "epe_sum": float(error[valid].sum())})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--attention-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_glu(args.pretrained, args.attention_checkpoint,
                           "attention", device)
    original4, original2 = model.deconv4, model.deconv2
    bilinear = BilinearFlowUpsample()
    loader = DataLoader(RoadScenePairs(args.data_root, "val"), batch_size=1,
                        shuffle=False)
    variants = (("original", original4, original2),
                ("bilinear_deconv4_only", bilinear, original2),
                ("bilinear_deconv2_only", original4, bilinear),
                ("bilinear_both", bilinear, bilinear))
    results = {}
    try:
        for name, layer4, layer2 in variants:
            model.deconv4, model.deconv2 = layer4, layer2
            rows = evaluate(model, loader, device)
            pixels = sum(row["valid_pixels"] for row in rows)
            results[name] = {"weighted_epe_512px": sum(row["epe_sum"] for row in rows) / pixels,
                             "valid_pixels": pixels,
                             "pairs": len(rows),
                             "per_image": [{"name": row["name"],
                                            "epe_512px": row["epe_sum"] / row["valid_pixels"]}
                                           for row in rows]}
    finally:
        model.deconv4, model.deconv2 = original4, original2
    report = {"split": "val", "protocol": "same 512px GT/mask; inference-only layer substitution without retraining; no test access",
              "pretrained_sha256": sha256_file(args.pretrained),
              "attention_checkpoint_sha256": sha256_file(args.attention_checkpoint),
              "selected_checkpoint_epoch": meta.get("epoch"),
              "results": results}
    (args.output / "upsampler_probe.json").write_text(json.dumps(report, indent=2),
                                                        encoding="utf-8")
    with (args.output / "upsampler_probe.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("name", *results))
        for index, row in enumerate(results["original"]["per_image"]):
            writer.writerow((row["name"], *(results[name]["per_image"][index]["epe_512px"]
                                          for name in results)))
    print(json.dumps({name: result["weighted_epe_512px"]
                      for name, result in results.items()}, indent=2))


if __name__ == "__main__":
    main()
