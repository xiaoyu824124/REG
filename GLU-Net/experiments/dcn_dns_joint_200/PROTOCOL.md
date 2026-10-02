# RoadScene: 200-epoch joint coarse/fine DCN comparison

## Question and arms

Compare **SA/CA + one 32x32 DCN update** with **DNS + SA/CA + the same
DCN update**. Both start from the independently trained, matched 50-epoch
coarse checkpoints in `roadscene_runs/dns_saca_matched_20` (the directory
name is historical; the checkpoint metadata records 50 epochs). The source
checkpoints were selected in their earlier study using validation coarse EPE.
They are fixed starting points, not claims of coarse-stage convergence.

The **revised recipe (version 2)** first trains only the new zero-initialized
32x32 DCN for 10 epochs at learning rate 1e-4. During epochs 11-200 it
jointly updates the coarse decoder, SA/CA, existing 32x32 decoder and
dilated refiners at 1e-5, while the DCN remains at 1e-4. DNS is also updated
at 1e-5 in the DNS arm. VGG, the later local stages and the pretrained flow
upsampling layers are frozen in both arms. The 32x32 DCN architecture and
training loss are identical. There
is no MIND, extra contrastive loss, additional local module or test-driven
adjustment. This comparison measures which coarse representation works better
**with** DCN after the same joint optimization; it does not isolate the causal
benefit of DCN over a no-DCN continuation.

## Training loss and source distinction

The SmoothFusion keyframe registration path in the local source calls its
packaged pretrained GLU-Net for inference. SmoothFusion's fusion-image
intensity and gradient losses train its fusion module; they do not define a
new registration loss to copy here. CRFT's released `CRFTLoss` supervises
coarse and fine flow with masked componentwise MAE. This study borrows the
**two-scale supervision idea**, while retaining a smooth EPE appropriate to
the RoadScene GT-flow metric:

`L = mean_valid sqrt(||F_final_512 - GT_512||² + 0.01)
   + 0.25 * mean_valid sqrt(||2*upsample(F_coarse_256) - GT_512||² + 0.01)`

The coefficient 0.25 is fixed before the formal run. It is a project choice,
not a CRFT paper coefficient. Both terms use the same 512x512 GT and mask.
No raw IR-visible intensity loss is used because the modalities have
different brightness statistics. AdamW uses the stage-specific learning
rates above and gradient norm clipping at 1.0. The warm-up and low-rate
joint phase were chosen after inspecting the **validation-only failure of
recipe version 1**; the 22-pair test split remains untouched.

## Fixed protocol

- 176 train pairs; 23 validation pairs; 22 test pairs remain locked.
- 200 **new** epochs per arm, batch size 4, seed 2026, 44 updates per epoch,
  8,800 optimizer updates per arm, same shuffled image order each epoch.
- Evaluate and save the zero-update 50-epoch source as **epoch 0** before
  optimizing. A trained checkpoint only replaces it if its validation final
  EPE is strictly lower.
- Within each arm, select the checkpoint with the lowest valid-pixel-weighted
  **final flow EPE at 512x512** on validation. Never select by coarse EPE.
- Both arms use the same base GLU-Net weight, flow direction (visible target
  to IR source), GT, valid mask, precision, evaluator and timing procedure.
- Report four-stage EPE in 512px units, actual first 32-grid window misses,
  EPE on pixels assigned to those missed queries, GT displacement >=64px
  region EPE, each pair's final EPE, CMR@5px, inference time, parameters,
  training throughput and peak allocated/reserved GPU memory.
- `--resume` restores the exact optimizer state, local/coarse parameters and
  epoch order from `latest_*.pth` saved every five epochs. Only `best_*.pth`
  is for evaluation; `latest_*.pth` is for resuming training.

## Server commands (Windows Command Prompt)

Run from `G:\cxj\REG\GLU-Net` after pulling the revised code. Confirm both
starting checkpoints exist under the path below. They are large local
training artifacts and are not committed to Git.

```bat
python roadscene_dcn_dns200.py --data-root "G:\cxj\RoadScence" --pretrained "pre_trained_models\GLUNet_DPED_CityScape_ADE.pth" --attention-checkpoint "roadscene_runs\dns_saca_matched_20\best_attention.pth" --dns-attention-checkpoint "roadscene_runs\dns_saca_matched_20\best_dns_attention.pth" --output "roadscene_runs\dcn_dns_joint_200_v2" --batch-preflight
```

The preflight makes one warm-up and three timed **train-split-only** updates
per arm at batch size 4. It does not select a model. On the local RTX 3060
Laptop GPU (6 GiB), the repeated-update check used approximately 3.6/3.8
GiB allocated for SA/CA+DCN and DNS+SA/CA+DCN. These are **local memory
checks, not A4000 measurements or validation results**. Throughput depends
on data loading and is not inferred from memory utilization alone. On the
server, check its own `batch_preflight.json`. If GPU utilization is low while
memory is low, compare data-loading time and update throughput; more memory
use by itself is not a success criterion.

```bat
python roadscene_dcn_dns200.py --data-root "G:\cxj\RoadScence" --pretrained "pre_trained_models\GLUNet_DPED_CityScape_ADE.pth" --attention-checkpoint "roadscene_runs\dns_saca_matched_20\best_attention.pth" --dns-attention-checkpoint "roadscene_runs\dns_saca_matched_20\best_dns_attention.pth" --output "roadscene_runs\dcn_dns_joint_200_v2" --workers 4
```

After interruption, repeat the formal command with `--resume`. For a
validation-only recheck after training, use `--eval-only` with the same
paths. **Do not resume the original `dcn_dns_joint_200` run with this revised
code.** Its optimizer and phase schedule differ; preserve its files as a
failed-recipe diagnostic and use the new `_v2` directory.

The training outputs are `best_saca_dcn.pth`, `best_dns_saca_dcn.pth`,
`latest_*.pth`, `history_*.json`, `comparison.json` and `per_image.csv`.
The result table is intentionally empty until the complete 200-epoch server
run finishes and the report is transferred back. The 22-pair test split must
not be consulted for choosing a loss weight, checkpoint or branch.

## Current verification status

Recipe version 1 decreased SA/CA+DCN training fine loss from 4.06px at
epoch 1 to 2.81px at epoch 20, but its best validation final EPE over those
epochs was 11.0863px, worse than its 9.7681px epoch-0 starting point.
Local one-epoch stage analysis found the largest deterioration at the
32x32 refinement stage (11.82 -> 15.75px), with outside-first-window final
EPE rising 16.13 -> 22.89px. Training only the new DCN for one diagnostic
epoch instead improved the validation final EPE from 9.77 -> 9.48px for
SA/CA and 7.29 -> 7.10px for DNS+SA/CA. These are small **diagnostics**,
not formal model-selection results. A separate two-epoch diagnostic on the
full 176/23 train/validation split yielded SA/CA+DCN 9.768 -> 9.438 ->
9.686px and DNS+SA/CA+DCN 7.289 -> 7.050 -> 6.533px for epoch 0, DCN-only
epoch 1 and low-rate joint epoch 2, respectively. SA/CA partially rebounded
at epoch 2, so no long-run gain is assumed. This diagnostic used a shortened
warm-up and is marked `code_check_only`; it is not the 200-epoch protocol.
Recipe version 2 also passed the normal two-phase code check on 2 train /
1 validation pairs. The full 200-epoch A4000 run remains pending. A
train-only batch-4 memory preflight previously passed on the local GPU.
