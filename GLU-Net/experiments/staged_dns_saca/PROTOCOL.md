# RoadScene three-stage DNS+SA/CA with one local DCN update

## Why this candidate

The matched 50-epoch validation comparison in
`experiments/dns_saca/formal_val_50/RESULTS.md` used the same 176 training
pairs, 23 validation pairs, original GLU-Net base checkpoint, 4,400 update
steps per arm, source/target order, GT and valid mask. The same evaluator
reported SA/CA coarse EPE 8.5717 (256px) and final EPE 9.7658 (512px), versus
DNS+SA/CA 6.5352 and 7.2885. DNS+SA/CA is therefore the coarse candidate
for this new schedule. Eleven of 23 pair means were worse, so this is not a
claim of uniformly better matching. These are previous validation results,
not results of the three-stage schedule. The 22-pair test split is not used
for training or selection.

## Fixed model and data protocol

- Original GLU-Net base weight `pre_trained_models/GLUNet_DPED_CityScape_ADE.pth`.
- 16×16 VGG coarse features → shared 2D DNS residual → existing self attention
  and cross attention → Global Correlation → coarse flow. Both DNS and
  attention use their existing zero-initialized gates.
- Existing local correlation/decoders at 32, 64 and 128 grids. Exactly one
  `torchvision.ops.DeformConv2d` residual update at 32×32; no further
  iterations or new convolution variants.
- Target-grid flow points from visible target pixels to infrared source
  sampling locations. Original 512×512 RoadScene image pairs and GT; the
  existing `correspondence_mask` determines valid pixels. Full outputs are
  compared at 512×512. Coarse flow is evaluated at 256px image units and
  after ×2 rescaling on the 512px mask. All four level EPEs, actual first
  local window misses, inside/outside errors, and GT displacement ≥64px use
  the same 512px unit via `roadscene_refinement_audit.py`.
- Runtime and inference allocation use `roadscene_compare.py`'s warmup,
  repeated, synchronized full forward protocol. This reports model forward
  time, not data loading or preprocessing of labels. Per-image results are
  saved separately.
- Training only reads `train`; checkpoint choice only reads `val`. The
  `test` evaluator refuses to run until `selection.json` exists and its
  chosen stage is requested.

## Optimization schedule

| Stage | Starting weights | Trainable parameters | Loss and selection | Cap / early stop |
|---|---|---|---|---|
| Coarse | Original base; new DNS, SA/CA, DCN zero-gated | VGG `level_4` only, DNS, SA/CA, `decoder4`; LR 1e-4 | Train: coarse EPE256 + 2× correspondence CE. Choose minimum validation coarse EPE256. Record final EPE512. | 200 epochs maximum; patience 20, minimum improvement 0.01px |
| Fine | Best coarse checkpoint | Epochs 1–5: single DCN only, LR 1e-4. Then existing local decoders/refinement at 1e-5 plus DCN at 1e-4. | Train: masked final-flow Charbonnier EPE512. Choose validation final EPE512. | 80 epochs maximum; patience 12, minimum improvement 0.01px |
| Joint | Best fine checkpoint | Coarse group LR 1e-6; existing local group 1e-5; DCN 1e-4 | Train: final Charbonnier EPE512 + 0.25× coarse Charbonnier EPE512. Choose validation final EPE512. | 30 epochs maximum; patience 6, minimum improvement 0.01px |

All stages keep model modules in evaluation mode while explicitly enabling
gradients for listed parameters. This freezes BatchNorm running statistics.
The original `deconv4` and `deconv2` flow upsamplers and their pretrained
weights remain **frozen in every stage**. The earlier VGG blocks are frozen.
During joint training only, the selected VGG level receives gradients through
the full forward; the default GLU-Net inference path remains unchanged.

Every stage evaluates its starting point as epoch 0 and saves the best
checkpoint separately from `latest_*` resume state. Both contain model,
optimizer, scheduler, Python/NumPy/Torch/CUDA RNG state, epoch, cumulative
optimizer steps, base/parent hashes, and the fixed budget. Resume with
`--resume`; the script rejects a changed source checkpoint, base weight,
schedule or code hash. The model is kept at the best fine checkpoint unless
the best joint checkpoint lowers validation final EPE by more than 0.01px.
`stage_comparison.json` records the four-level EPE, window misses, large
motion, worst five pairs, speed, memory and paired changes. The chosen stage
is written to `selection.json`. During joint training, every validation epoch
also records full-scale coarse/final EPE, actual first-window misses and
per-image values, so `stage_comparison.json` identifies the first epoch that
regressed against the best fine checkpoint.

## Server commands

Run in `G:\cxj\REG\GLU-Net` with the `glunet` environment and RoadScene data
at `G:\cxj\RoadScence`. The three commands share one output directory.

```bat
python roadscene_staged.py --stage coarse --data-root "G:\cxj\RoadScence" --pretrained "pre_trained_models\GLUNet_DPED_CityScape_ADE.pth" --output "roadscene_runs\staged_dns_saca"
python roadscene_staged.py --stage fine --data-root "G:\cxj\RoadScence" --pretrained "pre_trained_models\GLUNet_DPED_CityScape_ADE.pth" --output "roadscene_runs\staged_dns_saca"
python roadscene_staged.py --stage joint --data-root "G:\cxj\RoadScence" --pretrained "pre_trained_models\GLUNet_DPED_CityScape_ADE.pth" --output "roadscene_runs\staged_dns_saca"
```

Add `--resume` to the interrupted stage command. The first two stages need
their predecessors' `best_*.pth` in that output directory. Inspect
`selection.json` before testing. If its `selected_stage` is `fine`, use
`--stage fine` in the following command; otherwise use `--stage joint`:

```bat
python roadscene_staged.py --stage fine --eval-only --eval-split test --data-root "G:\cxj\RoadScence" --pretrained "pre_trained_models\GLUNet_DPED_CityScape_ADE.pth" --output "roadscene_runs\staged_dns_saca"
```

The test command is intentionally shown with `fine` as an example, not a
prediction of the selection. It writes `report_test_selected.json` and
`per_image_test_selected.csv`; do not run it before validation selects the
structure. The script does not claim this new model is better than the prior
SA/CA or DNS+SA/CA runs until the complete validation results exist.

## Local code check

Use an empty output directory and add `--code-check` to the coarse, fine and
joint commands in order. This uses 2 training pairs and 1 validation pair,
verifies the zero-gate start, flow geometry and active gradients, and marks
all resulting checkpoints `code_check_only`. Those checkpoints are rejected
as sources for a formal run.
