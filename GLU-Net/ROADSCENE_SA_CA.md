# RoadScene coarse SA/CA experiment

This experiment follows the first, single-frame phase. `GLUNet_model` applies
shared self attention and bidirectional cross attention to the 16x16 VGG
features, immediately before global correlation. The local matching and flow
decoders are unchanged. The attention gate starts at zero, so an untrained
attention model gives the same result as the baseline.

RoadScene uses visible as the target image and warped infrared as the source.
Its `.flo` file maps a pixel in visible to the corresponding position in IR.
The loader reads `train`, `val`, and `test` independently; no temporal frames
from VTMOT are used. Invalid or out-of-bounds flow pixels are masked.
The CRFT checkout's `RoadScence.py` currently points at a different
`wrapped/` directory layout, so this loader follows the files actually present
under `train/512_...`, `val/512_...`, and `test/512_...`.

## Environment

Use Python 3.10, PyTorch 2.4.0 with CUDA 12.1, and the pinned dependencies in
`requirements.txt`. The coarse-only script does not call the CuPy local
correlation kernel. A later full GLU-Net forward pass still needs the CuPy
package and a matching CUDA Toolkit.

## Train and compare

Run from the GLU-Net repository root. Supply your own RoadScene and pretrained
weight paths on the server.

First check the image/flow direction with a known 512x512 translation and five
real validation pairs:

```bash
python audit_roadscene.py --data-root /path/to/RoadScence \
  --output roadscene_runs/direction_audit.json
```

```bash
python roadscene_coarse.py train \
  --data-root /path/to/RoadScence \
  --pretrained /path/to/GLUNet_DPED_CityScape_ADE.pth \
  --output roadscene_runs/sa_ca \
  --epochs 20 --batch-size 2
```

Both branches start from the same checkpoint and fine tune the 16x16 `decoder4`
with the same images, optimizer, learning rate, loss, and epochs. The SA/CA
branch also trains its attention module. The VGG encoder and local stages stay
frozen. Each branch uses validation EPE to select its own best epoch. The
outputs are `best_baseline.pth` and `best_attention.pth`; the original
pretrained checkpoint is still required to reconstruct either full model.
Before training, the script verifies that zero-gate SA/CA gives exactly the
same coarse flow and correlation as the baseline.

To evaluate a trained module on held-out test images:

```bash
python roadscene_coarse.py eval \
  --data-root /path/to/RoadScence \
  --pretrained /path/to/GLUNet_DPED_CityScape_ADE.pth \
  --baseline-checkpoint roadscene_runs/sa_ca/best_baseline.pth \
  --attention-checkpoint roadscene_runs/sa_ca/best_attention.pth \
  --eval-split test --output roadscene_runs/test
```

The output contains `comparison.json` for baseline versus SA/CA, plus
`baseline/` and `attention/` folders with correlation volumes, confidence
images, center-query correlation images, coarse flows, and warped IR previews.
EPE is in pixels at the 256x256 model input scale; multiply by two to compare
with RoadScene's 512x512 pixel scale. `warp_edge_error` is the L1 difference
of normalized edge magnitudes at 256x256 and serves as a cross-modal proxy,
not a photometric error. The JSON also records valid pixel/query counts,
correlation top-1 bin accuracy, inference milliseconds per pair, PyTorch/CUDA
versions, and the pretrained weight SHA-256. Inference timing includes the
encoder, global correlation, attention when enabled, and coarse decoder; it
excludes image loading, preprocessing, and metric computation.

By default the script evaluates only the global stage. After establishing that
global matching improves, run the same `eval` command with `--full-model` to
also evaluate the final 512x512 flow EPE through GLU-Net's unchanged local
stages. This requires the pinned CuPy package and a working CUDA Toolkit.
The script records a blocked status in `comparison.json` if full inference
fails; coarse metrics remain available and must not be reported as final flow
registration quality. Keep `--eval-split val` for development. Run test once
after selecting both checkpoints on validation, then use `--eval-split test`
for the final held-out report.
