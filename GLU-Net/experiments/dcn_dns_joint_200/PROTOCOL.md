# RoadScene: 200-epoch joint coarse/fine DCN comparison

## Question and arms

Compare **SA/CA + one 32x32 DCN update** with **DNS + SA/CA + the same
DCN update**. Both start from the independently trained, matched 50-epoch
coarse checkpoints in `roadscene_runs/dns_saca_matched_20` (the directory
name is historical; the checkpoint metadata records 50 epochs). The source
checkpoints were selected in their earlier study using validation coarse EPE.
They are fixed starting points, not claims of coarse-stage convergence.

For this study, both arms jointly train the coarse decoder, SA/CA, the 32x32
decoder and its existing dilated refiners, and one zero-initialized deformable
residual update. DNS is additionally trainable in the DNS arm. VGG, the later
local stages and the pretrained flow upsampling layers are frozen in both
arms. The 32x32 DCN architecture and the training loss are identical. There
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
different brightness statistics. AdamW uses learning rate 1e-4 and gradient
norm clipping at 1.0. There is no LR schedule in this comparison.

## Fixed protocol

- 176 train pairs; 23 validation pairs; 22 test pairs remain locked.
- 200 **new** epochs per arm, batch size 4, seed 2026, 44 updates per epoch,
  8,800 optimizer updates per arm, same shuffled image order each epoch.
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

Run from `G:\cxj\REG\GLU-Net` after pulling the new code. Confirm both
starting checkpoints exist under the path below. They are large local
training artifacts and are not committed to Git.

```bat
python roadscene_dcn_dns200.py --data-root "G:\cxj\RoadScence" --pretrained "pre_trained_models\GLUNet_DPED_CityScape_ADE.pth" --attention-checkpoint "roadscene_runs\dns_saca_matched_20\best_attention.pth" --dns-attention-checkpoint "roadscene_runs\dns_saca_matched_20\best_dns_attention.pth" --output "roadscene_runs\dcn_dns_joint_200" --batch-preflight
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
python roadscene_dcn_dns200.py --data-root "G:\cxj\RoadScence" --pretrained "pre_trained_models\GLUNet_DPED_CityScape_ADE.pth" --attention-checkpoint "roadscene_runs\dns_saca_matched_20\best_attention.pth" --dns-attention-checkpoint "roadscene_runs\dns_saca_matched_20\best_dns_attention.pth" --output "roadscene_runs\dcn_dns_joint_200" --workers 4
```

After interruption, repeat the formal command with `--resume`. For a
validation-only recheck after training, use `--eval-only` with the same
paths. Do not use the smoke-check output directory for formal training.

The training outputs are `best_saca_dcn.pth`, `best_dns_saca_dcn.pth`,
`latest_*.pth`, `history_*.json`, `comparison.json` and `per_image.csv`.
The result table is intentionally empty until the complete 200-epoch server
run finishes and the report is transferred back. The 22-pair test split must
not be consulted for choosing a loss weight, checkpoint or branch.

## Current verification status

Local one-epoch smoke check on 2 train / 1 validation pairs verified starting
flow identity, forward/backward, coarse and DCN gradient paths, and both
validation evaluators. It is flagged `code_check_only` and its EPE values
are **not** research measurements. A train-only batch-4 preflight passed on
the local GPU. Full 200-epoch A4000 training and 23-pair formal validation
are pending.
