# RoadScene phase guide: final validation decision

**Decision: stop the global phase-congruency gate and retain the matched
SA/CA control.** The 22-pair independent test remains locked and was not
used for this decision.

## Evidence and scale

Source: server 20-epoch report copied to
`formal_val/comparison.json` (SHA-256
`979d0956210324f9958be70d1a6385b4cb2a93f8ce30f0ffebd543423847e20b`).
The run used 176 training pairs, 23 validation pairs, original GLU base
weight SHA-256 `2eac424ee4c9998cafed7ca60e7181d9f54adcee7e8e71250c5f2c7c0c9e50b1`,
20 epochs, batch 2, lr 1e-4, seed 2026, identical per-epoch image order,
and minimum validation coarse EPE to choose one checkpoint per arm. The three
checkpoints match their hashes recorded in the report. All three were
evaluated on the same 5,413,286 valid full-resolution pixels and 5,103 valid
coarse pixels. This is a **validation** comparison, not a fresh test result.

| Arm | Selected epoch | Coarse EPE (256px) | Corr Top-1 | Final flow EPE (512px) | Pair AEPE (512px) | Pair CMR@5px | Timed full inference ms/pair | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SA/CA | 20 | 10.2983 | 16.05% | **11.7238** | **12.1963** | 9/23 = 39.13% | 46.98 | 689.85 |
| MIND B+SA/CA | 20 | 10.3964 | 16.28% | 12.4067 | 12.9203 | 9/23 = 39.13% | 51.52 | 757.29 |
| Phase gate+SA/CA | 18 | **10.1631** | 15.93% | 12.4291 | 13.0033 | 9/23 = 39.13% | 53.39 | 785.99 |

The phase gate lowers coarse EPE by 1.31% against SA/CA, yet raises final
EPE by 6.02%, full inference time by 13.64%, and peak allocated memory by
13.94%. It improves final pair AEPE on 10/23 pairs, below the predeclared
16/23 rule. The fixed GT-displacement `64+` pixel bucket worsens from 16.05
to 17.41 final EPE (+8.50%), above the allowed 5% deterioration. It fails
the final-EPE, paired-image, and large-displacement rules. Runtime and memory
are still within the predeclared 1.25× caps, so accuracy is the reason to
stop. Do not tune this branch against the locked test set.

## What an EPE near 10 means here

EPE is the mean Euclidean displacement error **in pixels**, not a percentage.
`10.16` is measured on the 256×256 coarse stage over valid 16×16 query
locations, with the flow expressed in 256px image units. It is roughly
20.3 pixels if only the displacement unit is converted to 512px; that is
**not** the full model's final EPE because local refinements then modify the
flow and the evaluation pixels differ. The phase arm's actual final EPE is
12.43 pixels at 512×512.

As a scale check, predicting zero flow on the local copy of these 23 val
pairs gives 35.50 coarse EPE at 256px and 72.14 final-scale EPE at 512px
using the same GT/mask conventions and matching valid-pixel counts. Thus
registration learns substantial motion. But CMR@5px is only 9/23 pairs for
all three methods, and about 49% of valid pixels are within 5px. The average
also hides hard failures. Whether ~12px is sufficient for downstream fusion
depends on that task's alignment tolerance; this report does not claim
pixel-accurate registration or superiority to external papers/datasets.

## Paired cases and likely mechanism

`formal_val/paired_differences.json` and `formal_val/per_image.csv` hold all
23 paired differences. Against SA/CA, the phase arm improves 10 pairs and
degrades 13. Largest deteriorations in **pair AEPE, 512px**: `000014`
+13.38, `000021` +4.51, `000015` +3.01 pixels. Largest gains: `000009`
-2.54, `000019` -2.44, `000004` -1.39 pixels. Five pairs improve coarse
EPE yet worsen final AEPE.

For `000014`, 184,067 of 205,800 valid pixels have GT displacement >=64px.
The phase arm's coarse EPE rises from 23.80 to 26.88 and final pair AEPE
from 48.14 to 61.52; Top-1 stays 6.63%. The preview shows a large rotated
source region and dense/repeated line structures. The phase maps strongly
respond to many such lines. This is **evidence of a failure case**, while
attributing the error specifically to false structural matches remains a
hypothesis. No filter width, orientation count, gate or learning rate was
retuned after seeing it.

The server run saved same-pair IR/visible structure maps and 16×16 source-bin
correlation maps for val IDs `000002`, `000004`, `000013` under its local
`roadscene_runs/phase_matched_20` directory. Local deterministic replay of
the saved checkpoints generated additional panels for failed `000014` and
improved `000009` under `roadscene_runs/phase_case_review`. These images are
kept outside Git so dataset imagery is not published; the server JSON/CSV in
`formal_val` supplies the table values.

## Cost and next step

Training time for 20 epochs was 158.02 s (SA/CA), 170.77 s (MIND B+SA/CA),
and 168.07 s (phase+SA/CA). Peak allocated training memory was 238.24,
242.07, and 311.35 MiB respectively, using the script's allocated-memory
counter. Do not interpret these as total GPU memory or cross-machine
benchmarks.

Keep SA/CA as the current RoadScene global-stage choice. If studying MIND
again, a **separate** local-refinement experiment is more targeted than
replacing global encoder features: local detail descriptors may address
residual fine alignment. This is a research hypothesis, not a result. The
current local path remains unchanged; no additional module is stacked on
the failed phase branch.
