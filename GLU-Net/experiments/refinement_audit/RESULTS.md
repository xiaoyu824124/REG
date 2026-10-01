# SA/CA refinement diagnosis on RoadScene validation

**Decision:** Do not add local MIND yet. The largest failures have substantial
coarse/local-search miss rates and remain inaccurate after refinement. The
trained flow upsamplers also shift the local search centres. The current data
do not isolate a local descriptor ambiguity that MIND confidence gating would
address. The 22-pair test split was not opened or used.

## Fixed model and measurement

This is a read-only analysis of the selected SA/CA epoch-20 checkpoint from
the matched phase experiment. Base GLU checkpoint SHA-256:
`2eac424ee4c9998cafed7ca60e7181d9f54adcee7e8e71250c5f2c7c0c9e50b1`;
SA/CA checkpoint SHA-256:
`3d5550b78f8492e85c9f3c69a5767ba311b46c7b76732a362b130fd1bf56edbd`.
There are 23 validation pairs and 5,413,286 valid 512×512 GT pixels. Flow
maps visible target coordinates to warped IR source coordinates.

The four predictions are bilinearly interpolated to 512×512 and compared to
the **same** GT pixels and valid mask. The 16×16 and 32×32 predictions use
256px displacement units, so each is multiplied by 2. The 64×64 and 128×128
predictions already use 512px units. This full-grid coarse EPE is a different
statistic from the previously reported **10.2983 at 256px**, which samples
valid 16×16 coarse GT queries. The earlier formal full EPE was 11.7238;
this separate local replay yields 11.7249, a 0.0011px difference. We retain
the formal number as the selected model result and use the replay consistently
for every diagnostic below.

Local correlation is 9×9 (radius four feature cells per axis). At the 32,
64 and 128 grids the nominal windows are respectively ±64, ±32 and ±16px
per axis in 512px coordinates. A query is outside when either coordinate of
the residual GT displacement relative to the **actual pre-correlation warp
centre** exceeds that radius. GT flow is area averaged to the query grid;
queries with at most 80% valid GT coverage are excluded. This window test is
an approximate reachability check, not a guarantee of a correct match:
the flow decoder can predict a residual outside the correlation window.

## Stage and displacement results

| 512px EPE, same full-resolution valid pixels | Coarse 16 | Local 32 | Local 64 | Final 128 |
|---|---:|---:|---:|---:|
| All, 5,413,286 pixels | 20.63 | 14.13 | 12.92 | 11.72 |
| GT displacement 0–8px, 445,228 | 7.52 | 4.65 | **3.89** | 4.08 |
| 8–16px, 466,996 | 12.60 | 7.43 | 5.67 | 4.77 |
| 16–32px, 482,599 | 17.38 | 9.17 | 7.89 | 7.05 |
| 32–64px, 1,065,618 | 17.79 | 10.21 | 9.09 | 8.09 |
| ≥64px, 2,952,845 | 25.42 | 18.85 | 17.64 | **16.05** |

The 32-grid refinement gives the largest aggregate drop. The final stage
worsens the 0–8px group slightly and worsens four pairs (`000001`, `000006`,
`000009`, `000010`) relative to the 64-grid output. Across all valid pixels,
coarse→32 improves 4,106,335 and worsens 1,306,951; 32→64 improves
3,576,362 and worsens 1,836,923; 64→128 improves 3,280,883 and worsens
2,132,403. These are per-pixel comparisons, not independent samples.

| GT displacement at local query | 32-grid outside ±64px | 64-grid outside ±32px | 128-grid outside ±16px |
|---|---:|---:|---:|
| 0–8px | 1/1,732 (0.1%) | 22/6,901 (0.3%) | 181/27,740 (0.7%) |
| 8–16px | 4/1,795 (0.2%) | 63/7,226 (0.9%) | 1,809/28,869 (6.3%) |
| 16–32px | 19/1,878 (1.0%) | 268/7,495 (3.6%) | 12,529/30,025 (41.7%) |
| 32–64px | 163/4,168 (3.9%) | 726/16,630 (4.4%) | 44,986/66,594 (67.6%) |
| ≥64px | **3,865/11,278 (34.3%)** | 6,845/45,624 (15.0%) | 145,880/183,699 (79.4%) |
| All | 4,052/20,851 (19.4%) | 7,924/83,876 (9.4%) | 205,385/336,927 (61.0%) |

The largest GT motions occupy 54.6% of valid full-resolution pixels and
dominate first-window misses (3,865 of 4,052). For full pixels assigned to
valid 32-grid queries, first-window **inside** pixels have coarse→final
EPE 17.09→9.16; **outside** pixels have 34.47→21.89. Thus local stages can
partly recover misses, but the remaining error is much larger there. The
high 128-grid outside fraction also reflects the trained `deconv2` warp
centre and the narrower window; it cannot be read as a failure rate by
itself.

At the local query grids, the first 32-grid stage lowers mean residual EPE
from 24.09 to 11.43px **inside** its window, and from 102.75 to 25.02px
**outside**. The 64-grid stage lowers it from 9.56 to 8.30px inside, but only
57.15 to 56.56px outside. The last-stage outside group goes from 68.43 to
15.97px, largely correcting its learned upsampler's shifted centre. These
are query-level residuals around each stage's own pre-correlation warp
centre, so they are not the same measure as the full-image stage EPE table.
They show useful local correction where matches are reachable; they do not
establish a dominant local descriptor ambiguity.

The top quartile of visible-image Sobel edge strength (1,353,329 valid
pixels) has 19.64→10.23 coarse→final EPE; the other valid pixels have
20.95→12.22. This simple single-modality edge proxy does not show a broad
high-detail failure. It does not measure IR/visible semantic edge agreement.

## Per-pair failures and trained upsampler check

Only **9/23** pairs have final pair EPE below 5px. The full per-pair stage
table and window rates are in `per_image.csv`. Selected failures:

| Pair | GT displacement p90 | Coarse→32→64→final EPE, 512px | Outside 32-grid window |
|---|---:|---:|---:|
| `000014` | 186.8 | 49.0→53.7→52.2→48.1 | 45.1% |
| `000009` | 97.4 | 44.8→41.4→41.3→43.7 | 31.8% |
| `000015` | 174.8 | 31.7→32.3→31.2→28.8 | 36.3% |
| `000002` | 192.1 | 34.4→32.1→30.8→25.8 | 41.1% |
| `000007` | 186.7 | 27.2→24.1→23.2→22.3 | 39.7% |

Across the 14 pairs at or above 5px final pair EPE, mean first-window miss
rate is 25.7%; across the nine below 5px it is 11.8%. Six of those 14
failures exceed 25% first-window misses. The largest failures are therefore
not well described as small, already-localized residuals. In `000014` the
first local stage actually increases pair EPE. In `000009` the final stage
increases it. This remains an association, not proof of which module caused
each error.

`deconv4` and `deconv2` are **learned** transposed convolutions, although
initialized with a bilinear kernel. Replacing only their warp centres with
ordinary bilinear interpolation in a hypothetical coverage calculation
reduces the first/last outside fractions from 19.4%/61.0% to 3.4%/21.2%.
This does **not** predict an improved model: a separate inference-only probe,
without retraining, gives 512px final EPE 11.72 (original), 27.22 (replace
`deconv4`), 49.48 (replace `deconv2`) and 74.18 (replace both). The frozen
local decoders depend on the pretrained upsampling distribution. The probe
is a diagnostic of this dependency, not a fair trained ablation or a proposed
replacement.

## Next controlled experiment

First improve coarse matching on **training data only**, concentrating on
the ≥64px GT-displacement region without using test pairs: e.g. use
displacement-stratified train sampling or loss weighting of the existing
global correlation/decoder4 supervision. Keep the selected pretrained base,
SA/CA path, split, 20-epoch budget and evaluator fixed. On validation, require
both lower coarse EPE and lower 32-grid first-window miss rate, especially
for ≥64px, then check final EPE and paired failures. A separate matched
experiment can unfreeze `deconv4` and downstream decoders using a final-flow
loss to test propagation mismatch. Change one factor at a time. A local MIND
confidence gate becomes justified only if the difficult pixels are mostly
inside the actual local windows yet the local stages still worsen them.

No local MIND branch was implemented, trained or timed in this diagnosis.
No independent test result was rerun or used for a decision.

## Reproduction

From `GLU-Net` in a CUDA/CuPy environment (set `CUPY_CACHE_DIR` to a writable
directory if needed):

```powershell
python roadscene_refinement_audit.py --data-root E:\datasets\RoadScence --pretrained pre_trained_models\GLUNet_DPED_CityScape_ADE.pth --attention-checkpoint roadscene_runs\phase_matched_20\best_attention.pth --output roadscene_runs\refinement_audit_val
python roadscene_upsampler_probe.py --data-root E:\datasets\RoadScence --pretrained pre_trained_models\GLUNet_DPED_CityScape_ADE.pth --attention-checkpoint roadscene_runs\phase_matched_20\best_attention.pth --output roadscene_runs\refinement_audit_val
```

The scripts hardcode `val`; they have no option to open `test`. The JSON
files include exact denominators and per-pair values. Local previews for
`000009`, `000014`, and `000021` are saved under
`roadscene_runs/refinement_audit_val/heatmaps`; they show the visible target
and coarse/32/64/final error heatmaps (common 0–80px color scale, green to
red). Dataset-derived image previews are excluded from Git.
