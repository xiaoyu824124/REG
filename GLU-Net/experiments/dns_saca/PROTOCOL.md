# RoadScene 2D DNS, SA/CA, and DNS → SA → CA

## Paper mechanism and adaptation

Mok et al., *Modality-Agnostic Structural Image Representation Learning for
Deformable Multi-Modality Medical Image Registration*, CVPR 2024, define DNS
on deep features by comparing **neighbours with other neighbours**, excluding
the patch centre. The paper uses direct and dilated 3D six-neighbourhoods,
retains feature channels through the distance computation, then uses feature
squeezing and anatomy-aware contrastive learning. Our 2D adaptation uses
direct/dilated four-neighbourhoods, yielding four adjacent-neighbour
relations per radius (N–E, E–S, S–W, W–N), or eight relations total.

`CoarseDNS.descriptor` produces `[B,512,8,16,16]` on VGG's 16×16 coarse
features. A shared learned `Linear(512,1)` compresses each relation, and a
two-convolution projection restores 512 channels. For both IR and visible:

`F' = F + gamma * Proj(DNS(F))`, with `gamma = 0` initially.

The combined route is VGG → DNS residual → existing self attention → existing
**cross attention** → existing Global Correlation → coarse flow. Encoder,
pretrained flow upsamplers and all local refinement stages remain intact.
This is a DNS-inspired **2D feature experiment**, not a reproduction of the
complete 3D DSIR network, contrastive objective or registration method.

## Matched three-arm study

`roadscene_dns_saca.py` trains three independent arms: SA/CA, DNS alone, and
DNS+SA/CA. **All start from the same original GLU-Net base checkpoint.** The
combined arm does not continue training from the selected SA/CA checkpoint,
so it gets no extra optimization steps. The existing selected SA/CA checkpoint
is used only to verify that inserting a zero-gated DNS reproduces its coarse
flow, correlation and final flow exactly. That check passed locally with
maximum absolute difference **0** at all three outputs.

The original plan assigned each arm the same 176 training pairs, 23 validation pairs, 20 epochs,
batch size 2, seed 2026, AdamW lr 1e-4, image order and `coarse EPE + 2 ×
correlation CE` loss. No contrastive loss, stratified sampling, displacement
weighting or local MIND is included. Within each arm the checkpoint with
minimum validation coarse EPE is retained to match the selected SA/CA
experiment's rule. **Across arms**, the decision considers common-protocol
final EPE at 512px, the actual first 32-grid correlation-window miss rate,
final EPE where GT displacement is ≥64px, paired-image stability and runtime.
Coarse EPE alone cannot select the structure. A combination is worth keeping
only if final flow and large-motion failures improve consistently, without
trading many pairs for a few extreme gains. The 22-pair test remains locked.

The script reports for each arm: coarse EPE in 256px units, the 16-grid flow
upsampled and rescaled then evaluated on the same full 512px GT mask,
first-window outside fraction using the **actual pretrained `deconv4` warp
centre** and ±64px per-axis radius, final 512px EPE, ≥64px GT-region final
EPE, per-pair metrics and synchronized inference time. No registration
module beyond the coarse DNS and pre-existing SA/CA was added.

## Verification and completed server run

On the local RTX 3060, a one-epoch/two-train-pair/one-validation-pair
`--code-check` completed all three arms and the full CuPy refinement path.
The descriptor shape check was `[2,512,8,16,16]` and initial feature
difference was `0.0`. **These smoke-check losses/EPEs are not research
results.** The server subsequently ran all three arms for **50 epochs**,
with 4,400 matched optimization steps per arm. Its output directory retained
the name `dns_saca_matched_20`, but the report and checkpoint metadata all
say 50 epochs. The completed 23-pair validation analysis and archived raw
results are in [`formal_val_50/RESULTS.md`](formal_val_50/RESULTS.md).

The original planned 20-epoch command was:

```powershell
python roadscene_dns_saca.py --data-root 'G:\cxj\RoadScence' --pretrained pre_trained_models\GLUNet_DPED_CityScape_ADE.pth --existing-attention-checkpoint roadscene_runs\phase_matched_20\best_attention.pth --epochs 20 --batch-size 2 --lr 0.0001 --output roadscene_runs\dns_saca_matched_20
```

Outputs: `comparison.json`, `per_image.csv`, per-arm checkpoints and
`progress.json`. Do not use `--code-check` outputs to choose a model.

Paper: https://openaccess.thecvf.com/content/CVPR2024/html/Mok_Modality-Agnostic_Structural_Image_Representation_Learning_for_Deformable_Multi-Modality_Medical_Image_CVPR_2024_paper.html
