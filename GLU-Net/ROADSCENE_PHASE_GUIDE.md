# RoadScene: ACFR-Net-inspired phase structure guide at GLU coarse level

## Paper mechanism and scope

ACFR-Net, *Multi-modal image registration network via attention to common
features*, Infrared Physics & Technology 158 (2026), 106791,
https://doi.org/10.1016/j.infrared.2026.106791, describes two distinct stages:

| ACFR-Net stage | Paper mechanism | This controlled GLU-Net experiment |
|---|---|---|
| Global affine registration | A phase-congruency edge mask (PCEM) guides feature extraction toward shared structural edges; an affine estimator predicts the global transform | Use only the structural-guidance idea. Independently compute a fixed phase-congruency map for visible target and IR source, area-pool each to the 16×16 global feature grid, and add a zero-gated learned residual to the corresponding original encoder feature. GLU global correlation and coarse flow remain. |
| Local registration | Hierarchical **channel attention** on shallow features and **spatial attention** on deep features | Not imported. GLU local correlation and refinements remain unchanged. GLU's existing SA/CA means **self-attention/cross-attention** at 16×16, not the paper's spatial/channel attention. |

Only the publisher abstract/introduction/method overview were available for
verification; the full publisher text was inaccessible. This code is an
independent phase-congruency-inspired implementation, **not** a reproduction
of ACFR-Net's exact PCEM equation or reported results. No affine estimator,
new local module, phase loss, or other paper loss is added.

## Fixed prior and adapter

The fixed descriptor uses 3 log-Gabor scales (wavelengths 3, 6, 12 pixels),
8 orientations, 32-pixel reflection padding, a quadrature even/odd response,
and a predeclared per-orientation noise floor of 1.5 times the spatial median
finest-scale response. Each map is the positive local energy after that floor
divided by total response amplitude, clipped to [0,1]. These choices are
fixed before formal validation; they must not be tuned on the held-out test.
Eight direction maps are pooled from 256×256 to 16×16 without flow scaling.
Their maximum is saved as a single visible structure image for review.

The PC adapter has the same 8→64→512 pointwise projection and scalar tanh
gate as MIND route B. It applies independently and symmetrically to each
original encoded feature **before** existing SA/CA and global correlation.
The gate starts at zero, so the new arm reproduces SA/CA outputs at init.
Only decoder4, existing SA/CA, and the chosen guide are trained. The VGG
encoder and GLU local decoders stay at the same pretrained weight.

## Existing matched MIND evidence (server, 20 epochs, val only)

The report is in `roadscene_runs/mind_matched_20/comparison.json` (server commit
`917c6ae`). All arms use the same original GLU weight, 176 train/23 val pairs,
batch 2, lr 1e-4, seed 2026, and minimum validation coarse EPE checkpoint
selection. Numbers below are actual full-model validation results, not test.

| Arm | Coarse EPE 256px | Corr Top-1 | Final flow EPE 512px | Mean pair AEPE | CMR@5px | Full inference ms/pair | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| SA/CA | 10.2983 | 16.05% | 11.7238 | 12.1963 | 39.13% | 47.17 | 746.46 |
| MIND A+SA/CA | 17.4613 | 4.12% | 22.92 | 24.03 | 17.39% | 55.31 | 747.46 |
| MIND B+SA/CA | 10.3964 | 16.28% | 12.4067 | 12.9203 | 39.13% | 51.36 | 747.59 |

MIND A replaces the original input to the frozen pretrained VGG at the
global feature branch, changing the input distribution and losing the original
feature information; that is a plausible mechanism, not an experimentally
isolated cause. MIND B preserves original features but still worsens final
EPE versus matched SA/CA. It improves 11/23 pairs and degrades 12/23.
Examples of final pair AEPE deterioration are `000009` +6.51, `000014`
+4.78 and `000021` +3.73 pixels. The `64+` GT-displacement bucket worsens
from 16.05 to 17.02 final EPE; only the `[0,8)` bucket has a small decrease
from 4.079 to 4.041. Seven pairs have better coarse but worse final error.
Thus a small Top-1 gain does not establish better dense final flow.

## Precommitted PC decision rule

Train and evaluate three **fresh matched arms**: `attention`,
`mind_b_attention`, `phase_attention`. Each starts from the original base
weight; no arm resumes from another. Use the same train image order (checked
by SHA per epoch), optimizer, 20 epochs, batch 2, lr 1e-4, seed 2026, loss
`coarse EPE + 2*corr CE`, checkpoint choice `minimum val coarse EPE`, and
common full evaluator. Use only train and val. The locked 22-pair test must
not be opened to choose a phase filter or checkpoint.

Continue this PC route only if, against its **new matched SA/CA control** on
23 val pairs, full pixel-weighted EPE falls at least 5%, mean pair AEPE falls,
at least 16/23 pairs improve, no predeclared GT-displacement bucket worsens
over 5%, CMR@5px loses at most one pair, and inference latency and peak
allocated GPU memory are each <=1.25× control. A coarse-only improvement
with worse final flow fails. These are project decisions, not significance
tests. If it fails, keep SA/CA and stop this PC branch. A later local-MIND
study would be a separate single-factor experiment; it is not implemented
here.

## Local code verification, not experimental accuracy

`audit_phase.py` checked IR and visible maps on val pair `000000`: both
shapes `[1,8,256,256]`, finite and within [0,1], flat input output 0,
contrast-inversion max difference `7.82e-7`.
An 8-pixel synthetic shift in each axis produced a mean absolute difference
of about `0.00185` between correspondingly shifted maps in the central
160×160 region; the FFT/reflection boundary was excluded.
With a trained SA/CA checkpoint copied to the zero-gate PC arm, coarse
flow/correlation and full flow were
identical (`max abs diff 0`) using deterministic CuDNN settings. One-epoch
2-train/1-val smoke ran all three arms and generated all checkpoint, metric,
structure-map, and correlation-map outputs. Its numbers are **code-check only**.
The local CuPy cache had to be redirected to a writable directory; no new
Python package or CUDA version was installed.

## Server commands (Windows CMD)

Run after pulling the new commit. The existing server `glunet` environment
has Torch 2.4.0+cu121; no extra package is needed for torch FFT.

```bat
cd /d G:\cxj\REG
git pull --ff-only origin main
cd GLU-Net
conda activate glunet
python audit_phase.py --data-root "G:\cxj\RoadScence" --pretrained "pre_trained_models\GLUNet_DPED_CityScape_ADE.pth" --attention-checkpoint "roadscene_runs\sa_ca\best_attention.pth" --full-model --output "roadscene_runs\phase_audit\audit.json"
python roadscene_phase.py --data-root "G:\cxj\RoadScence" --pretrained "pre_trained_models\GLUNet_DPED_CityScape_ADE.pth" --epochs 20 --batch-size 2 --lr 0.0001 --output "roadscene_runs\phase_matched_20"
```

The formal run writes `comparison.json`, `per_image.csv`,
`paired_differences.json`, `structure_maps/*.png`,
`correlation_maps/<arm>/*.png`, and full-flow preview images. It prints only
per-epoch coarse EPE and final arm names; full metrics are in the JSON/CSV.
If an output directory already has `best_*.pth`, choose a new output path;
the script refuses to overwrite selected checkpoints.

**No formal PC accuracy or PC-versus-MIND decision has been measured yet.**
The A4000 run above is needed before reporting that result.
