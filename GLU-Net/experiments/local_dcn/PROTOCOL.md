# RoadScene: SA/CA with staged deformable local refinement

## Starting point and provenance

The selected coarse path is GLU-Net + self attention + cross attention (SA/CA).
The server training tree is `G:\cxj\REG\GLU-Net`; its RoadScene data root is
`G:\cxj\RoadScence`. The matching local working tree is `E:\REG\GLU-Net`,
while `E:\REG\REG-publish\GLU-Net` is the clean Git publishing checkout.
The base checkpoint is `pre_trained_models/GLUNet_DPED_CityScape_ADE.pth`,
SHA-256 `2eac424ee4c9998cafed7ca60e7181d9f54adcee7e8e71250c5f2c7c0c9e50b1`.
The selected SA/CA checkpoint is `roadscene_runs/phase_matched_20/best_attention.pth`,
SHA-256 `3d5550b78f8492e85c9f3c69a5767ba311b46c7b76732a362b130fd1bf56edbd`.
Previously published baseline/SA/CA, MIND, phase, DNS protocol and
SmoothFusion validation artifacts are retained unchanged. No test result is
read or modified in this study.

## Mechanism

CRFT's fine path uses recurrent flow refinement based on feature mismatch,
with its own attention and spatial geometric transformation. This experiment
borrows the **current-flow → rewarp source features → recompute local evidence
→ predict residual flow → add residual** pattern. It does not transplant
CRFT's SGT, fine attention, feature hierarchy or training loss.

GLU-Net's existing `dc_conv*` names are dilated `nn.Conv2d` layers. The new
`local_dcn32` uses `torchvision.ops.DeformConv2d` with a learned 18-channel
3×3 offset field. The offset is in 32-grid feature pixels and bounded to
±2. It operates on the local 9×9 correlation, current flow, feature
discrepancy and source-sampling validity. The residual head is bounded to
±4 feature pixels per axis per update. Both offset and residual predictors
start at zero, so the added path is initially an identity.

The VGG and SA/CA coarse path, global correlation, `decoder4`, later local
decoders, and pretrained `deconv4`/`deconv2` flow upsamplers stay intact.
All matched arms start from the exact same selected SA/CA plus original GLU
checkpoint. All arms fine-tune `decoder3` and the existing 32-grid *dilated*
refiner for an equal 20 epochs, batch size 2, seed 2026 and learning rate
1e-4. The new DCN arm additionally trains its DCN. This matched local
fine-tuning control is distinct from the already selected SA/CA checkpoint's
historical 11.7238px validation score. Phase 1 reevaluates that **fixed
selected SA/CA** as a descriptive reference and separately trains its
matched local fine-tuning control. Use the matched control for causal DCN
claims; the fixed reference preserves the originally chosen model's result.

## Flow coordinates and masks

- Flow always maps a visible target query to the warped IR source location.
- Coarse 16-grid and first local 32-grid flows are stored in **256-image
  pixels**; the 64- and 128-grid flows are in **512-image pixels**.
- At 32×32, the source VGG feature is warped by `flow32 / 8`, because each
  feature-grid pixel spans eight 256-image pixels. The DCN offsets use
  feature-grid pixels, and its predicted Δflow is converted back to
  256-image pixels before addition to the current flow.
- A synthetic +2 feature-pixel translation checks the source sampling and
  its +16px 256-image representation. A nonzero offset and two nonzero
  residual updates check the actual scale and additive iteration.
- Training final-flow loss and validation EPE use the same native 512×512
  GT and correspondence mask: finite GT with GT-mapped source inside the
  image. Evaluation never masks pixels based on the prediction.
- Four stage EPEs are compared only after interpolation to the same 512×512
  grid; 16/32-grid flow vectors are multiplied by two.
- The first 9×9 correlation window has radius four 32-grid pixels, or
  ±64px per axis at 512px. Window coverage uses the actual pretrained
  `deconv4` warp centre and valid area-pooled GT queries. The 19.4% miss
  rate (4,052/20,851) is from the prior selected SA/CA validation audit;
  every new arm remeasures its own inside/outside EPE and miss rate.

## Stage order and selection

1. **Phase 1:** matched locally fine-tuned SA/CA control and one additional
   32-grid DCN update. Best checkpoint within each arm is the one with the
   lowest valid-pixel-weighted **final 512px validation flow EPE**.
2. **Phase 2:** two shared-weight DCN updates at 32×32, each rewarping IR
   features and recomputing 9×9 correlation. It starts from the same selected
   SA/CA weight, with the same training image order and step count. It is
   allowed only if phase 1 improves final and ≥64px-region EPE, does not
   worsen first-window-outside EPE, improves at least 12/23 pairs, does not
   worsen the five preidentified validation failures on average, and costs
   no more than 1.5× the matched control inference time. It must also beat
   the fixed selected SA/CA reference's final validation EPE.
3. **Later scales:** deliberately not implemented before phase 1 and 2
   validation. The same gate compares phase 2 against phase 1. If it fails,
   no cross-scale DCN branch should be added. This prevents using an untested
   fourth model to select architecture.

Both phases save `best_<arm>.pth`, `history_<arm>.json`, `comparison.json` and
`per_image.csv`. Reports contain four 512px stage EPEs, 32-grid window
inside/outside EPE, ≥64px GT-region EPE, paired image results, synchronized
inference time, peak allocated GPU memory and each 32-grid DCN iteration's
EPE after conversion to 512px units. Selection and gate use only the
176 train and 23 validation pairs. The 22-pair test remains locked.

## Reproduction on the A4000 server

From `G:\cxj\REG\GLU-Net` after updating the source checkout:

```bat
python roadscene_dcn.py --phase 1 --data-root "G:\cxj\RoadScence" --pretrained "pre_trained_models\GLUNet_DPED_CityScape_ADE.pth" --attention-checkpoint "roadscene_runs\phase_matched_20\best_attention.pth" --epochs 20 --batch-size 2 --lr 0.0001 --output "roadscene_runs\dcn_phase1_20"
```

Only if `roadscene_runs\dcn_phase1_20\comparison.json` reports
`phase2_gate.passed=true`:

```bat
python roadscene_dcn.py --phase 2 --data-root "G:\cxj\RoadScence" --pretrained "pre_trained_models\GLUNet_DPED_CityScape_ADE.pth" --attention-checkpoint "roadscene_runs\phase_matched_20\best_attention.pth" --phase1-report "roadscene_runs\dcn_phase1_20\comparison.json" --epochs 20 --batch-size 2 --lr 0.0001 --output "roadscene_runs\dcn_phase2_20"
```

To re-evaluate an individual phase-1 checkpoint on validation:

```bat
python roadscene_dcn.py --phase 1 --eval-only --arm dcn_single --data-root "G:\cxj\RoadScence" --pretrained "pre_trained_models\GLUNet_DPED_CityScape_ADE.pth" --attention-checkpoint "roadscene_runs\phase_matched_20\best_attention.pth" --output "roadscene_runs\dcn_phase1_20"
```

The script never offers a test-split option.

## Verification performed locally

`torchvision.ops.DeformConv2d` executed on the local PyTorch 2.4.0+cu121 GPU.
The synthetic coordinate and offset test passed: +2 grid pixels maps to
+16 image256 pixels; initial offset and Δflow are exactly zero; two
deliberate nonzero updates each add +8 image256 pixels. Initial 16-, 32-
and 64-grid flows matched SA/CA exactly. The final 128-grid comparison
showed up to ~0.0016px difference, while repeated inference of the
unchanged baseline itself varied by ~0.0014px in the CuPy path. Both DCN
arms passed two-pair/one-validation-pair short training, including gradients
through final flow and, on the second optimizer step, into DCN offsets.
These one-epoch smoke-check EPEs are **not** research results. Full 20-epoch
training, all 23 validation pairs, phase gates and the cross-scale decision
have not yet been run. No DCN accuracy benefit is claimed.

Paper: https://openaccess.thecvf.com/content/CVPR2026/html/Liu_CRFT_Consistent-Recurrent_Feature_Flow_Transformer_for_Cross-Modal_Image_Registration_CVPR_2026_paper.html
