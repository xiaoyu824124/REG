# Separate large-motion coarse-match controls

`roadscene_large_motion_dns.py` implements the earlier requested four-arm
study independently of the DNS→SA→CA combination study:

1. Fixed selected GLU-Net+SA/CA reference (20-epoch RoadScene checkpoint).
2. SA/CA with **train-only displacement-stratified sampling**.
3. SA/CA with **train-only large-motion coarse loss weighting**.
4. GLU-Net+DNS **alone**, with neither SA/CA nor contrastive loss.

No arm stacks DNS with the two large-motion interventions. The code reads
only RoadScene train/val. It keeps the base weight, batch size 2, 20 epochs,
AdamW lr 1e-4, 176 draws/epoch (88 updates), full local flow upsamplers and
common validation evaluator. The sampling intervention defines a high
stratum from **training GT only**: at least 75% of valid pixels have
displacement ≥64px at 512px. It draws 75% of each fixed-length epoch from
that stratum, with replacement. The weighting intervention uses normal
shuffle but weights valid 16-grid GT locations with magnitude ≥32px in
256px units twice as much for both coarse flow EPE and correlation CE;
weighted means are normalized. DNS alone uses normal shuffle and unweighted
loss. These are fixed study settings, not settings selected from test data.

For each arm the script reports coarse 256px EPE, coarse full-grid EPE at
512px, actual first-window outside fraction, ≥64px GT region coarse/final
EPE, final flow EPE, per-pair CSV and synchronized inference time. Compare
final flow, window coverage and pair stability jointly. A better coarse EPE
without a better final result does not justify keeping an arm.

Local `--code-check` completed one update and full inference on tiny subsets
for all arms; those numbers are **not validation evidence**. Full training on
the A4000 and the 23-pair validation remain pending. The 22-pair test is
locked until the validation plan has been selected.

From `GLU-Net` on the A4000 server:

```powershell
python roadscene_large_motion_dns.py --data-root 'G:\cxj\RoadScence' --pretrained pre_trained_models\GLUNet_DPED_CityScape_ADE.pth --attention-checkpoint roadscene_runs\phase_matched_20\best_attention.pth --epochs 20 --batch-size 2 --lr 0.0001 --output roadscene_runs\large_motion_dns_matched_20
```

Run this as a **separate study** from `roadscene_dns_saca.py`; do not compare
code-check metrics or treat a DNS-only result as full DSIR.
