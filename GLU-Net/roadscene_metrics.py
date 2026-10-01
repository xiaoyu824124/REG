"""Uniform full-flow metrics. CMR's statistical unit is an IMAGE PAIR.

CRFT paper §4.1 and test_epoch_end use mean per-pair AEPE < threshold, not
per-pixel EPE < threshold. Our AEPE includes every pair and uses the common
GT-valid mask. CRFT's source instead uses all pixels and filters infinite
AEPEs; that legacy failure exclusion is intentionally not used here.
"""

import numpy as np
import torch

CMR_THRESHOLDS = (5, 4, 3, 2, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)


def flow_metrics(predicted, truth, valid):
    if predicted.shape != truth.shape or valid.shape != truth.shape[-2:]:
        raise ValueError("Metrics require matching 2xHxW flow and HxW GT mask")
    valid = valid.bool()
    if not valid.any():
        raise ValueError("Image pair has no valid GT pixels")
    error = torch.linalg.vector_norm(predicted - truth, dim=0)[valid]
    # Fail on nonfinite predictions; never discard failed pixels/pairs.
    if not torch.isfinite(error).all():
        raise ValueError("Nonfinite predicted flow on valid GT pixels")
    return {"valid_full": int(valid.sum().item()),
            "epe_sum": float(error.double().sum().item()),
            "aepe_512px": float(error.double().mean().item()),
            "pixel_hits": {str(t): int((error < t).sum().item()) for t in CMR_THRESHOLDS}}


def aggregate_flow(rows):
    if not rows:
        raise ValueError("No evaluated pairs")
    rows = list(rows)
    count = sum(row["valid_full"] for row in rows)
    aepe = np.array([row["aepe_512px"] for row in rows])
    return {"samples": len(rows), "valid_pixels_full": count,
            "final_flow_epe_512px": sum(row["epe_sum"] for row in rows) / count,
            "aepe_pair_mean_512px": float(aepe.mean()),
            "cmr_pair_percent": {str(t): float((aepe < t).mean() * 100)
                                 for t in CMR_THRESHOLDS},
            "pixel_success_percent_auxiliary": {
                str(t): sum(row["pixel_hits"][str(t)] for row in rows) * 100 / count
                for t in CMR_THRESHOLDS}}
