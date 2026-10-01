"""Use an external, unmodified official CRFT checkout without Lightning.

Core inference only needs torch and einops. Read CRFT config assignments with
AST instead of importing its yacs/Lightning training dependencies. Unknown
configuration expressions fail rather than silently changing the model.
"""

import ast
import importlib
import json
import operator
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from roadscene_coarse import sha256_file


def _chunk_attention(query, key, value, scale, chunk=64):
    """Exact softmax over ALL keys, chunk only independent query rows.

    Inference only, float32 retained. This bounds temporary score matrices;
    it does not reduce the quadratic FLOPs or turn global attention into windows.
    """
    output = torch.empty_like(query)
    key_t = key.transpose(-2, -1).contiguous()
    for start in range(0, query.shape[-2], chunk):
        end = min(start + chunk, query.shape[-2])
        scores = (query[..., start:end, :] @ key_t) * scale
        output[..., start:end, :] = scores.softmax(dim=-1) @ value
    return output


def enable_chunked_attention(model, chunk=64):
    """Runtime adapter for the official fine_process attention, same weights.

    Unmodified 512px fine_process flattens ALL overlapping windows into long
    sequences and attempts 87.5GiB in its first attention. No window partition,
    pruning, parameter, scale, or normalization changes are introduced here.
    """
    if model.training:
        raise ValueError("Chunked CRFT adapter is for eval only")
    changed = []
    for module in model.modules():
        kind = type(module).__name__
        if kind not in {"VanillaAttention", "CrossBidirectionalAttention"}:
            continue
        if hasattr(module, "_official_forward"):
            continue
        module._official_forward = module.forward
        if kind == "VanillaAttention":
            def forward(self, x_q, x_kv=None):
                if self.training or torch.is_grad_enabled():
                    raise RuntimeError("Chunked CRFT forward requires eval and no_grad")
                x_kv = x_q if x_kv is None else x_kv
                bs, _, dim = x_q.shape
                kv = self.kv_proj(x_kv).reshape(bs, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
                q = self.q_proj(x_q).reshape(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                output = _chunk_attention(q, kv[0], kv[1], self.softmax_temp, chunk)
                return self.merge(output.transpose(1, 2).reshape(bs, -1, dim))
        else:
            def forward(self, x0, x1):
                if self.training or torch.is_grad_enabled():
                    raise RuntimeError("Chunked CRFT forward requires eval and no_grad")
                bs = x0.shape[0]
                qk0, qk1 = self.qk_proj(x0), self.qk_proj(x1)
                v0, v1 = self.v_proj(x0), self.v_proj(x1)
                qk0, qk1, v0, v1 = [tensor.reshape(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
                                    for tensor in (qk0, qk1, v0, v1)]
                qk0, qk1 = qk0 * self.softmax_temp ** 0.5, qk1 * self.softmax_temp ** 0.5
                out0 = _chunk_attention(qk0, qk1, v1, 1, chunk)
                out1 = _chunk_attention(qk1, qk0, v0, 1, chunk)
                return (self.merge(out0.transpose(1, 2).flatten(start_dim=-2)),
                        self.merge(out1.transpose(1, 2).flatten(start_dim=-2)))
        module.forward = types.MethodType(forward, module)
        changed.append(kind)
    return {"query_chunk": chunk, "modified_runtime_modules": changed,
            "algorithm": "exact global softmax, independent query row chunks, float32"}


def _attribute_path(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    return [node.id] + list(reversed(parts)) if isinstance(node, ast.Name) else []


def _config_literal(node, config):
    if isinstance(node, ast.Name) and node.id == "INFERENCE":
        return False
    if isinstance(node, ast.BinOp):
        operations = {ast.Add: operator.add, ast.Mult: operator.mul,
                      ast.Pow: operator.pow, ast.Div: operator.truediv}
        return operations[type(node.op)](_config_literal(node.left, config), _config_literal(node.right, config))
    if isinstance(node, ast.Attribute):
        keys = _attribute_path(node)
        if not keys or keys[0] != "_CN":
            raise ValueError(f"Unsupported config reference: {keys}")
        value = config
        for key in keys[1:]:
            value = value[key.lower()]
        return value
    return ast.literal_eval(node)


def official_config(root):
    root = Path(root).resolve()
    paths = [root / "src/config/default.py",
             root / "configs/crft/outdoor/visible_thermal.py"]
    config = {}
    for path in paths:
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                keys = _attribute_path(target)
                if len(keys) < 2 or keys[0] not in ("_CN", "cfg"):
                    continue
                value = node.value
                if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "CN":
                    parsed = {}
                elif isinstance(value, ast.Name) and value.id == "INFERENCE":
                    parsed = False  # official default.py INFERENCE=False, even in test.py
                else:
                    parsed = _config_literal(value, config)
                cursor = config
                lower_keys = [key.lower() for key in keys[1:]]
                if not lower_keys:
                    config = parsed
                    continue
                for key in lower_keys[:-1]:
                    cursor = cursor.setdefault(key, {})
                cursor[lower_keys[-1]] = parsed
    files = sorted((root / "src/crft").rglob("*.py")) + paths + [
        root / "configs/data/roadscence_512.py", root / "src/lightning/lightning_crft.py"]
    hashes = {str(path.relative_to(root)).replace("\\", "/"): sha256_file(path) for path in files}
    return json.loads(json.dumps(config["crft"])), hashes


def import_official(root):
    root = Path(root).resolve()
    sys.path.insert(0, str(root))
    module = importlib.import_module("src.crft")
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise RuntimeError("A different src.crft module is already imported")
    return module.CRFT


def load_crft(root, checkpoint, device):
    config, source_hashes = official_config(root)
    model = import_official(root)(config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    state = {key.removeprefix("module.").removeprefix("matcher."): value
             for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    # The PRIMARY control uses the official native64 config and unmodified
    # attention. Chunking is only an opt-in runtime/code check for native512.
    adapter = {"algorithm": "official unchunked attention", "native_input_size": 64}
    return model, {"path": str(checkpoint),
        "sha256": sha256_file(checkpoint), "config": config, "source_hashes": source_hashes,
        "epoch": payload.get("epoch"), "runtime_attention": adapter,
        "native_input_size": 64,
        "training_provenance": "unknown unless supplied by checkpoint owner"}


def resize_crft_input(rgb, size):
    if rgb.shape[-2:] == (size, size):
        return rgb.float()
    # Native512 ->64 is an integer factor8. Bilinear half-pixel averaging with
    # round-half-up matches official OpenCV uint8 INTER_LINEAR in this protocol.
    resized = F.interpolate(rgb.float(), (size, size), mode="bilinear", align_corners=False)
    return (resized + 0.5).floor().clamp(0, 255)


def resize_image_flow(flow, size):
    height, width = flow.shape[-2:]
    resized = F.interpolate(flow, size=size, mode="bilinear", align_corners=False)
    scale = resized.new_tensor((size[1] / width, size[0] / height))[None, :, None, None]
    return resized * scale


@torch.no_grad()
def predict_crft(model, target_rgb, source_rgb, input_size=64):
    # Official loader supplies float RGB 0..255; CRFT does its own per-channel
    # mean/std normalization. Never apply GLU ImageNet normalization here.
    data = {"image0": resize_crft_input(target_rgb, input_size),
            "image1": resize_crft_input(source_rgb, input_size)}
    model(data)
    flow = data["flow_f_full"]  # image0(target) -> image1(source); already image pixels
    if flow.shape[-2:] != (input_size, input_size):
        raise RuntimeError("CRFT did not return the requested native resolution")
    if flow.shape[-2:] != target_rgb.shape[-2:]:
        flow = resize_image_flow(flow, target_rgb.shape[-2:])
    return flow, data["flow_c"]


@torch.no_grad()
def audit_crft_coordinates(root):
    config, hashes = official_config(root)
    import_official(root)
    matching = importlib.import_module("src.crft.crft_module.coarse_matching")
    fine = importlib.import_module("src.crft.crft_module.fine_matching")
    features = torch.eye(256).reshape(1, 256, 16, 16) * 100
    source = torch.roll(features, shifts=(-1, 1), dims=(-2, -1))
    flow, probability = matching.global_correlation_softmax(features, source)
    observed = flow[0, :, 8, 8]
    torch.testing.assert_close(observed, torch.tensor([1., -1.]), atol=1e-6, rtol=0)
    final = fine.FineMatching(config)._upsample_to_image_resolution(flow,
        {"hw0_i": (128, 128), "hw0_f": (64, 64)})
    torch.testing.assert_close(final[0, :, 64, 64], torch.tensor([8., -8.]), atol=1e-5, rtol=0)
    resized = resize_image_flow(torch.tensor([4., -4.])[None, :, None, None].expand(1, 2, 64, 64), (512, 512))
    torch.testing.assert_close(resized, torch.tensor([32., -32.])[None, :, None, None].expand_as(resized))
    return {"coarse_target_8_8_to_source_xy": [9, 7],
            "coarse_flow_xy": observed.tolist(), "image_flow_xy": final[0, :, 64, 64].tolist(),
            "source_flat_index": int(probability[0, 8 * 16 + 8].argmax()),
            "native64_flow_xy": [4, -4], "rescaled512_flow_xy": resized[0, :, 256, 256].tolist(),
            "source_hashes": hashes, "config": config}
