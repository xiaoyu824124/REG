"""Train and compare GLU-Net's 16x16 global stage on RoadScene.

The local CUDA/CuPy correlation stages are never called by this experiment.
Both models use the same pretrained GLU-Net weights and training split.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from datasets.roadscene import RoadScenePairs
from models.our_models.GLUNet import GLUNet_model


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_base_weights(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not key.startswith("coarse_attention.") for key in missing):
        raise RuntimeError(f"Incompatible GLU-Net checkpoint: missing={missing}, unexpected={unexpected}")


def make_model(checkpoint, attention, device, train_decoder=False):
    model = GLUNet_model(evaluation=False, pyramid_type="VGG",
                        cyclic_consistency=True, coarse_attention=attention,
                        backbone_pretrained=False)
    load_base_weights(model, checkpoint)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if train_decoder:
        for parameter in model.decoder4.parameters():
            parameter.requires_grad_(True)
    if attention:
        for parameter in model.coarse_attention.parameters():
            parameter.requires_grad_(True)
    return model.to(device).eval()


def prepare(batch, device):
    source = batch["source_image"].to(device).float()
    target = batch["target_image"].to(device).float()
    height, width = target.shape[-2:]
    if (height, width) != source.shape[-2:]:
        raise ValueError("source and target sizes must match")
    # Match GLUNet_model.pre_process_data, including its uint8 quantization.
    source_256 = F.interpolate(source, size=(256, 256), mode="area").byte().float() / 255.0
    target_256 = F.interpolate(target, size=(256, 256), mode="area").byte().float() / 255.0
    mean = target_256.new_tensor(IMAGENET_MEAN)[None, :, None, None]
    std = target_256.new_tensor(IMAGENET_STD)[None, :, None, None]
    source_input = (source_256 - mean) / std
    target_input = (target_256 - mean) / std

    mask = batch["correspondence_mask"].to(device).float()[:, None]
    flow = batch["flow_map"].to(device).float()
    mask_coarse = F.interpolate(mask, size=(16, 16), mode="area")
    flow_coarse = F.interpolate(flow * mask, size=(16, 16), mode="area")
    flow_coarse = flow_coarse / mask_coarse.clamp_min(1e-6)
    flow_coarse[:, 0] *= 256.0 / width
    flow_coarse[:, 1] *= 256.0 / height
    valid_coarse = mask_coarse[:, 0] > 0.8
    return source_input, target_input, source_256, target_256, flow_coarse, valid_coarse, mask


def extract_coarse_features(model, target_input, source_input):
    with torch.no_grad():
        target_features = model.pyramid(target_input)[-3]
        source_features = model.pyramid(source_input)[-3]
    if target_features.shape[-2:] != (16, 16):
        raise ValueError(f"Expected 16x16 coarse features, got {target_features.shape[-2:]}")
    return target_features, source_features


def predict_coarse(model, target_input, source_input, features=None):
    if features is None:
        features = extract_coarse_features(model, target_input, source_input)
    target_features, source_features = features
    return model.coarsest_resolution_flow(target_features, source_features,
                                          256, 256, return_corr=True)


def correspondence_targets(flow, valid):
    y, x = torch.meshgrid(torch.arange(16, device=flow.device),
                          torch.arange(16, device=flow.device), indexing="ij")
    mapped_x = x[None].float() + flow[:, 0] / 16.0
    mapped_y = y[None].float() + flow[:, 1] / 16.0
    nearest_x = mapped_x.round().long()
    nearest_y = mapped_y.round().long()
    valid = valid & (nearest_x >= 0) & (nearest_x < 16) & (nearest_y >= 0) & (nearest_y < 16)
    indices = nearest_y.clamp(0, 15) * 16 + nearest_x.clamp(0, 15)
    return indices, valid


def coarse_statistics(predicted, corr, truth, valid, return_counts=False):
    indices, corr_valid = correspondence_targets(truth, valid)
    epe_map = torch.linalg.vector_norm(predicted - truth, dim=1)
    corr_scores = corr.permute(0, 2, 3, 1)
    nll = F.cross_entropy((corr_scores / 0.1).reshape(-1, 256),
                          indices.reshape(-1), reduction="none").reshape_as(indices)
    valid_count = int(valid.sum().item())
    corr_count = int(corr_valid.sum().item())
    if valid_count == 0 or corr_count == 0:
        raise ValueError("A batch has no valid coarse correspondences")
    epe_sum = epe_map[valid].sum()
    ce_sum = nll[corr_valid].sum()
    hit_sum = (corr_scores.argmax(dim=-1) == indices)[corr_valid].sum()
    values = (epe_sum / valid_count, ce_sum / corr_count, hit_sum.float() / corr_count)
    if return_counts:
        return values, (epe_sum.item(), valid_count, ce_sum.item(),
                        corr_count, int(hit_sum.item()))
    return values


def edge_magnitude(rgb):
    gray = rgb[:, :1] * 0.299 + rgb[:, 1:2] * 0.587 + rgb[:, 2:3] * 0.114
    kernel_x = gray.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])[None, None] / 8
    kernel_y = kernel_x.transpose(-1, -2)
    edge = torch.sqrt(F.conv2d(gray, kernel_x, padding=1).square() +
                      F.conv2d(gray, kernel_y, padding=1).square() + 1e-8)
    return edge / edge.mean(dim=(-1, -2), keepdim=True).clamp_min(1e-6)


def warp_edge_error(source, target, flow, mask):
    flow_256 = F.interpolate(flow, size=(256, 256), mode="bilinear", align_corners=False)
    y, x = torch.meshgrid(torch.arange(256, device=flow.device),
                          torch.arange(256, device=flow.device), indexing="ij")
    mapped_x = x[None].float() + flow_256[:, 0]
    mapped_y = y[None].float() + flow_256[:, 1]
    grid = torch.stack((mapped_x / 255 * 2 - 1, mapped_y / 255 * 2 - 1), dim=-1)
    warped_edge = F.grid_sample(edge_magnitude(source), grid, align_corners=True)
    target_edge = edge_magnitude(target)
    valid = F.interpolate(mask, size=(256, 256), mode="nearest")[:, 0].bool()
    valid = valid & (mapped_x >= 0) & (mapped_x <= 255) & (mapped_y >= 0) & (mapped_y <= 255)
    return (warped_edge[:, 0] - target_edge[:, 0]).abs()[valid].mean(), grid


def save_diagnostics(folder, name, predicted, truth, corr, valid, source, target, grid):
    folder.mkdir(parents=True, exist_ok=True)
    predicted_np = predicted[0].detach().cpu().numpy()
    truth_np = truth[0].detach().cpu().numpy()
    corr_np = corr[0].detach().cpu().numpy()
    np.savez_compressed(folder / f"{name}.npz", correlation=corr_np,
                        coarse_flow=predicted_np, gt_coarse_flow=truth_np,
                        valid=valid[0].detach().cpu().numpy())
    confidence = corr_np.max(axis=0)
    confidence = (255 * (confidence - confidence.min()) /
                  max(float(np.ptp(confidence)), 1e-6)).astype(np.uint8)
    Image.fromarray(confidence).resize((256, 256)).save(folder / f"{name}_corr_confidence.png")
    center_query = corr_np[:, 8, 8].reshape(16, 16)
    center_query = (255 * (center_query - center_query.min()) /
                    max(float(np.ptp(center_query)), 1e-6)).astype(np.uint8)
    Image.fromarray(center_query).resize((256, 256)).save(folder / f"{name}_corr_center.png")
    clipped = np.clip(predicted_np / 64.0, -1, 1)
    flow_rgb = np.stack(((clipped[0] + 1) * 127.5,
                         (clipped[1] + 1) * 127.5,
                         np.linalg.norm(clipped, axis=0) * 127.5), axis=-1).astype(np.uint8)
    Image.fromarray(flow_rgb).resize((256, 256)).save(folder / f"{name}_coarse_flow.png")
    warped = F.grid_sample(source[:1], grid[:1], align_corners=True)
    preview = torch.cat((target[:1], warped), dim=-1)[0].permute(1, 2, 0)
    Image.fromarray((preview.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(
        folder / f"{name}_visible_warped_ir.png")


@torch.no_grad()
def evaluate(model, loader, device, diagnostics=None, save_limit=4):
    model.eval()
    totals = {"epe_sum": 0.0, "ce_sum": 0.0, "corr_hits": 0,
              "valid_coarse": 0, "valid_corr": 0, "valid_full": 0,
              "warp_error_sum": 0.0, "inference_seconds": 0.0}
    count = 0
    for batch in loader:
        source_in, target_in, source, target, truth, valid, mask = prepare(batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        flow, corr = predict_coarse(model, target_in, source_in)
        if device.type == "cuda":
            torch.cuda.synchronize()
        totals["inference_seconds"] += time.perf_counter() - start
        _, counts = coarse_statistics(flow, corr, truth, valid, return_counts=True)
        epe_sum, valid_count, ce_sum, corr_count, hit_count = counts
        totals["epe_sum"] += epe_sum
        totals["ce_sum"] += ce_sum
        totals["corr_hits"] += hit_count
        totals["valid_coarse"] += valid_count
        totals["valid_corr"] += corr_count
        totals["valid_full"] += int(mask.sum().item())
        warp_error, grid = warp_edge_error(source, target, flow, mask)
        totals["warp_error_sum"] += warp_error.item() * flow.shape[0]
        if diagnostics is not None and count < save_limit:
            save_diagnostics(diagnostics, batch["name"][0], flow, truth, corr,
                             valid, source, target, grid)
        count += flow.shape[0]
    return {
        "samples": count,
        "valid_pixels_full": totals["valid_full"],
        "valid_pixels_coarse": totals["valid_coarse"],
        "valid_corr_queries": totals["valid_corr"],
        "epe_256px": totals["epe_sum"] / totals["valid_coarse"],
        "corr_ce": totals["ce_sum"] / totals["valid_corr"],
        "corr_top1": totals["corr_hits"] / totals["valid_corr"],
        "warp_edge_error": totals["warp_error_sum"] / count,
        "inference_ms_per_pair": totals["inference_seconds"] * 1000 / count,
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def evaluate_full(model, loader, device):
    """Evaluate the unchanged local stages when CuPy/CUDA is available."""
    model.eval()
    epe_sum = 0.0
    valid_count = 0
    inference_seconds = 0.0
    samples = 0
    warmed_up = False
    for batch in loader:
        source, target, source_256, target_256, *_ = model.pre_process_data(
            batch["source_image"], batch["target_image"], device=device)
        if not warmed_up:
            model(target, source, target_256, source_256)
            warmed_up = True
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _, flows_original = model(target, source, target_256, source_256)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_seconds += time.perf_counter() - start
        truth = batch["flow_map"].to(device).float()
        valid = batch["correspondence_mask"].to(device)
        # The original GLU-Net flow1 is in full-image pixels at 1/4 resolution.
        # Its training code interpolates it without multiplying the flow vectors.
        predicted = F.interpolate(flows_original[-1], size=truth.shape[-2:],
                                  mode="bilinear", align_corners=False)
        epe = torch.linalg.vector_norm(predicted - truth, dim=1)
        epe_sum += epe[valid].sum().item()
        valid_count += int(valid.sum().item())
        samples += truth.shape[0]
    return {"samples": samples, "valid_pixels_full": valid_count,
            "final_flow_epe_512px": epe_sum / valid_count,
            "inference_ms_per_pair": inference_seconds * 1000 / samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "eval"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("roadscene_runs/coarse_sa_ca"))
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--attention-checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--eval-split", choices=("val", "test"), default="val")
    parser.add_argument("--max-eval-samples", type=int,
                        help="limit evaluation pairs for a quick code check")
    parser.add_argument("--full-model", action="store_true",
                        help="also evaluate final flow; requires CuPy and CUDA Toolkit")
    args = parser.parse_args()
    if args.full_model and args.mode != "eval":
        parser.error("--full-model is only available in eval mode")
    if args.max_eval_samples is not None and args.mode != "eval":
        parser.error("--max-eval-samples is only available in eval mode")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2026)
    np.random.seed(2026)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    print(f"Device: {device}", flush=True)
    eval_dataset = RoadScenePairs(args.data_root, args.eval_split)
    if args.max_eval_samples is not None:
        if args.max_eval_samples < 1:
            parser.error("--max-eval-samples must be positive")
        eval_dataset = Subset(eval_dataset, range(min(args.max_eval_samples,
                                                      len(eval_dataset))))
    loader = DataLoader(eval_dataset,
                        batch_size=args.batch_size, num_workers=args.workers)
    baseline = make_model(args.pretrained, False, device,
                          train_decoder=args.mode == "train")
    attention = make_model(args.pretrained, True, device,
                           train_decoder=args.mode == "train")
    first_batch = next(iter(loader))
    source_in, target_in, *_ = prepare(first_batch, device)
    with torch.no_grad():
        initial_base_flow, initial_base_corr = predict_coarse(baseline, target_in, source_in)
        initial_attn_flow, initial_attn_corr = predict_coarse(attention, target_in, source_in)
    initialization_check = {
        "flow_max_abs_diff": (initial_base_flow - initial_attn_flow).abs().max().item(),
        "corr_max_abs_diff": (initial_base_corr - initial_attn_corr).abs().max().item(),
    }
    if max(initialization_check.values()) > 1e-6:
        raise RuntimeError(f"Zero-gate attention changed baseline outputs: {initialization_check}")
    print(f"Zero-gate check: {initialization_check}", flush=True)
    if args.baseline_checkpoint:
        payload = torch.load(args.baseline_checkpoint, map_location="cpu", weights_only=False)
        baseline.decoder4.load_state_dict(payload["decoder4_state_dict"])
    if args.attention_checkpoint:
        payload = torch.load(args.attention_checkpoint, map_location="cpu", weights_only=False)
        attention.coarse_attention.load_state_dict(payload["attention_state_dict"])
        if "decoder4_state_dict" in payload:
            attention.decoder4.load_state_dict(payload["decoder4_state_dict"])

    if args.mode == "train":
        train_loader = DataLoader(RoadScenePairs(args.data_root, "train"),
                                  batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers)
        optimizers = {
            "baseline": torch.optim.AdamW(baseline.decoder4.parameters(), lr=args.lr),
            "attention": torch.optim.AdamW(
                list(attention.decoder4.parameters()) +
                list(attention.coarse_attention.parameters()), lr=args.lr),
        }
        best_epe = {"baseline": float("inf"), "attention": float("inf")}
        for epoch in range(args.epochs):
            losses = {"baseline": [], "attention": []}
            for batch in train_loader:
                source_in, target_in, _, _, truth, valid, _ = prepare(batch, device)
                features = extract_coarse_features(baseline, target_in, source_in)
                for name, model in (("baseline", baseline), ("attention", attention)):
                    flow, corr = predict_coarse(model, target_in, source_in, features)
                    epe, ce, _ = coarse_statistics(flow, corr, truth, valid)
                    loss = epe + 2.0 * ce
                    optimizers[name].zero_grad(set_to_none=True)
                    loss.backward()
                    optimizers[name].step()
                    losses[name].append(loss.item())
            for name, model in (("baseline", baseline), ("attention", attention)):
                metrics = evaluate(model, loader, device)
                print(f"Epoch {epoch + 1} {name}: loss={np.mean(losses[name]):.4f}, "
                      f"val_epe={metrics['epe_256px']:.4f}, "
                      f"corr_top1={metrics['corr_top1']:.4f}", flush=True)
                if metrics["epe_256px"] < best_epe[name]:
                    best_epe[name] = metrics["epe_256px"]
                    payload = {"epoch": epoch + 1,
                               "decoder4_state_dict": model.decoder4.state_dict(),
                               "validation": metrics}
                    if name == "attention":
                        payload["attention_state_dict"] = model.coarse_attention.state_dict()
                    torch.save(payload, args.output / f"best_{name}.pth")
        for name, model in (("baseline", baseline), ("attention", attention)):
            payload = torch.load(args.output / f"best_{name}.pth", map_location="cpu",
                                 weights_only=False)
            model.decoder4.load_state_dict(payload["decoder4_state_dict"])
            if name == "attention":
                model.coarse_attention.load_state_dict(payload["attention_state_dict"])

    results = {
        "baseline": evaluate(baseline, loader, device, args.output / "baseline"),
        "attention": evaluate(attention, loader, device, args.output / "attention"),
        "dataset": str(args.data_root), "split": args.eval_split,
        "flow_unit": "pixels at 256x256 image scale",
        "warp_error": "normalized edge magnitude L1 at 256x256",
        "initialization_check": initialization_check,
        "pretrained": str(args.pretrained),
        "pretrained_sha256": sha256_file(args.pretrained),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "train_seed": 2026,
        "epochs": args.epochs if args.mode == "train" else None,
        "max_eval_samples": args.max_eval_samples,
        "training": "matched decoder4 fine-tuning; SA/CA branch also trains its attention",
    }
    selected_paths = {
        "baseline": (args.output / "best_baseline.pth") if args.mode == "train"
                    else args.baseline_checkpoint,
        "attention": (args.output / "best_attention.pth") if args.mode == "train"
                     else args.attention_checkpoint,
    }
    results["selected_checkpoints"] = {}
    for name, path in selected_paths.items():
        if path is not None:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            results["selected_checkpoints"][name] = {
                "path": str(path), "sha256": sha256_file(path),
                "epoch": payload.get("epoch"),
                "validation": payload.get("validation"),
            }
    full_error = None
    if args.full_model:
        results["final_flow"] = {}
        for name, model in (("baseline", baseline), ("attention", attention)):
            try:
                results["final_flow"][name] = evaluate_full(model, loader, device)
            except Exception as error:
                results["final_flow"][name] = {
                    "status": "blocked",
                    "reason": f"{type(error).__name__}: {error}",
                }
                full_error = error
    (args.output / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)
    if full_error is not None:
        raise RuntimeError("Full GLU-Net evaluation was blocked; see comparison.json") from full_error


if __name__ == "__main__":
    main()
