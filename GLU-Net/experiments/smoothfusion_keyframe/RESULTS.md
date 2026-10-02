# SmoothFusion keyframe registration module on RoadScene validation

## What was evaluated

Local source: `E:\res and fus\SmoothFusion`. The video keyframe path in
`AKRF/utils/predict_video_GLU_keyframe2.py` calls `reg(ir_tensor, vi_tensor)`.
`AKRF/model/reg/__init__.py` implements that call with the bundled
`GLU_Net.estimate_flow(ir_image, vi_image)`. It returns **dense target-to-source
flow**, not a homography or affine matrix. Video frame selection, history
averaging and the ASPFusion image fusion call happen outside this registration
forward. We therefore evaluate each pair as its own first keyframe and name
the arm **SmoothFusion keyframe registration module**. RoadScene single frames
cannot test adaptive frame selection or cross-frame smoothing.

This registration core is the project's packaged **original GLU-Net**, not a
separately trained SmoothFusion registration architecture. Its checkpoint
`GLUNet_DPED_CityScape_ADE.pth` has SHA-256
`2eac424ee4c9998cafed7ca60e7181d9f54adcee7e8e71250c5f2c7c0c9e50b1`,
identical to the base weight used by the SA/CA experiment. The filename
identifies the DPED/CityScape/ADE pretraining source; no RoadScene fine-tune
is loaded for this SmoothFusion arm. SA/CA additionally uses its RoadScene
train-split epoch-20 decoder4 and attention checkpoint SHA-256
`3d5550b78f8492e85c9f3c69a5767ba311b46c7b76732a362b130fd1bf56edbd`.
This is an existing-system comparison, not equal RoadScene training budgets.

## Fixed single-pair protocol

All 23 validation pairs have the same raw 512×512 uint8 RGB visible target
and warped IR source. Flow maps target/visible coordinates to source/IR
coordinates, as confirmed by the SmoothFusion call order and its remap
convention `source(x + flow_x, y + flow_y)`. Both arms output 512×512 dense
flow in 512px units, compared with the same `.flo` GT and the same 5,413,286
valid pixels (finite GT and mapped source position inside the image). No
prediction-dependent masking or image-pair exclusion is used. Each arm keeps
its original internal image preprocessing. Timings are per-pair medians of
five synchronized GPU forwards after three warmups, including preprocessing
and output, excluding disk IO/GT/metrics; the table averages those medians.

The SmoothFusion constructor separately requests ImageNet VGG weights even
though the complete GLU checkpoint overwrites the pyramid. The adapter
suppresses **only that redundant download** during construction, then lets
SmoothFusion load its unchanged complete checkpoint strictly. No registration
layer, parameter, warp or fusion code was edited. Both arms ran separately on
the same local RTX 3060 Laptop GPU with PyTorch 2.4.0+cu121.

## Actual validation result

| Arm | Final EPE, valid-pixel weighted, 512px | Mean pair EPE, 512px | Pair CMR@5px | Mean inference ms/pair | Parameters |
|---|---:|---:|---:|---:|---:|
| SmoothFusion keyframe registration module | 32.9388 | 34.4256 | 5/23 = 21.74% | 73.47 | 13,590,488 |
| Selected GLU-Net+SA/CA | **11.7249** | **12.1976** | 9/23 = 39.13% | 73.82 | 16,744,921 |

The separate earlier formal SA/CA validation result was 11.7238; this
common-protocol local replay gives 11.7249. The small difference does not
change the comparison. Runtime differences of this size are not evidence of
a consistent speed advantage on a single laptop GPU.

| GT displacement in 512px | SmoothFusion final EPE | SA/CA final EPE |
|---|---:|---:|
| 0–8px | 4.12 | 4.08 |
| 8–16px | 5.20 | 4.77 |
| 16–32px | 10.54 | 7.05 |
| 32–64px | 14.59 | 8.09 |
| ≥64px | **51.96** | **16.05** |

Largest SmoothFusion minus SA/CA pair-EPE gaps were `000002` +98.23px,
`000007` +95.14px and `000015` +63.57px. SmoothFusion was better on
`000003` by 1.78px and `000013` by 1.48px. See `per_image.csv` for all
23 paired values and `comparison.json` for complete protocol hashes,
denominators and weight/source provenance.

## Test status

The validation adapter and paired protocol ran successfully. The later user
instruction keeps the 22-pair independent test locked until the DNS and
large-displacement validation study chooses its plan. **No SmoothFusion test
EPE is reported here.** Test can later be run without retuning using the
`--validated-report` gate in `roadscene_smoothfusion.py` once that selection
is complete.

## Reproduce the validation comparison

From `GLU-Net`, in separate Python processes (same CUDA/CuPy environment):

```powershell
python roadscene_smoothfusion.py eval --method smoothfusion_keyframe --data-root E:\datasets\RoadScence --smooth-root 'E:\res and fus\SmoothFusion' --output roadscene_runs\smoothfusion_val\smooth.json
python roadscene_smoothfusion.py eval --method attention --data-root E:\datasets\RoadScence --pretrained pre_trained_models\GLUNet_DPED_CityScape_ADE.pth --attention-checkpoint roadscene_runs\phase_matched_20\best_attention.pth --output roadscene_runs\smoothfusion_val\attention.json
python roadscene_smoothfusion.py compare --smooth-report roadscene_runs\smoothfusion_val\smooth.json --attention-report roadscene_runs\smoothfusion_val\attention.json --output roadscene_runs\smoothfusion_val\paired
```
