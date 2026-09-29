"""Check RoadScene paths and target-to-source flow with a known translation."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from datasets.roadscene import RoadScenePairs
from models.our_models.consensus_network_modules import FeatureCorrelation, MutualMatching
from roadscene_coarse import prepare, correspondence_targets, warp_edge_error


def synthetic_translation():
    rng = np.random.default_rng(7)
    target = rng.integers(0, 256, (512, 512), dtype=np.uint8)
    source = np.zeros_like(target)
    # source(y-32, x+32) = target(y, x)
    source[:480, 32:] = target[32:, :480]
    flow = torch.empty(1, 2, 512, 512)
    flow[:, 0] = 32
    flow[:, 1] = -32
    valid = torch.zeros(1, 512, 512, dtype=torch.bool)
    valid[:, 32:, :480] = True
    batch = {
        "target_image": torch.from_numpy(np.stack([target] * 3)[None].copy()),
        "source_image": torch.from_numpy(np.stack([source] * 3)[None].copy()),
        "flow_map": flow,
        "correspondence_mask": valid,
    }
    _, _, source_256, target_256, coarse_flow, coarse_valid, mask = prepare(
        batch, torch.device("cpu"))
    indices, corr_valid = correspondence_targets(coarse_flow, coarse_valid)
    source_features = torch.eye(256).reshape(1, 256, 16, 16)
    target_features = torch.zeros_like(source_features)
    target_features[:, :, 1:, :15] = source_features[:, :, :15, 1:]
    corr = FeatureCorrelation(shape="4D", normalization=False)(
        source_features, target_features)
    corr = MutualMatching(corr).reshape(1, 256, 16, 16)
    observed_corr_index = int(corr[0, :, 8, 8].argmax())
    gt_edge_error, grid = warp_edge_error(source_256, target_256, coarse_flow, mask)
    opposite_edge_error, _ = warp_edge_error(source_256, target_256, -coarse_flow, mask)
    warped = F.grid_sample(source_256, grid, align_corners=True)
    mask_256 = F.interpolate(mask, size=(256, 256), mode="nearest").bool()
    photometric_mae = (warped - target_256).abs()[mask_256.expand_as(warped)].mean()
    result = {
        "known_flow_512px": [32, -32],
        "observed_coarse_flow_256px_center": coarse_flow[0, :, 8, 8].tolist(),
        "expected_source_bin_at_target_8_8": [9, 7],
        "observed_source_flat_index": int(indices[0, 8, 8]),
        "observed_correlation_peak_index": observed_corr_index,
        "valid_full_pixels": int(mask.sum()),
        "valid_coarse_pixels": int(coarse_valid.sum()),
        "valid_corr_queries": int(corr_valid.sum()),
        "photometric_mae_correct_warp": float(photometric_mae),
        "edge_error_correct": float(gt_edge_error),
        "edge_error_opposite": float(opposite_edge_error),
    }
    assert result["observed_coarse_flow_256px_center"] == [16.0, -16.0]
    assert result["observed_source_flat_index"] == 7 * 16 + 9
    assert result["observed_correlation_peak_index"] == 7 * 16 + 9
    assert result["valid_full_pixels"] == 480 * 480
    assert result["valid_coarse_pixels"] == 15 * 15
    assert result["edge_error_correct"] < result["edge_error_opposite"]
    return result


def real_direction_check(root, samples=5):
    loader = DataLoader(Subset(RoadScenePairs(root, "val"), range(samples)), batch_size=1)
    results = []
    for batch in loader:
        _, _, source, target, truth, _, mask = prepare(batch, torch.device("cpu"))
        correct, _ = warp_edge_error(source, target, truth, mask)
        opposite, _ = warp_edge_error(source, target, -truth, mask)
        zero, _ = warp_edge_error(source, target, torch.zeros_like(truth), mask)
        results.append({"pair": batch["name"][0], "valid_full_pixels": int(mask.sum()),
                        "gt_direction": float(correct), "opposite_direction": float(opposite),
                        "zero_flow": float(zero)})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    counts = {split: len(RoadScenePairs(args.data_root, split))
              for split in ("train", "val", "test")}
    report = {"dataset": str(args.data_root), "split_counts": counts,
              "synthetic": synthetic_translation(),
              "real_val_direction_check": real_direction_check(args.data_root)}
    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
