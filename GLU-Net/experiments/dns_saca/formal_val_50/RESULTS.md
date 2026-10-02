# RoadScene validation: 2D DNS, SA/CA and DNS → SA → CA

## Provenance and scope

This report archives the server results transferred to
`E:\REG\GLU-Net\roadscene_runs\dns_saca_matched_20` on 2026-10-02.
**The directory name says `20`, but the actual report and checkpoints say
50 epochs.** The experiment used 176 train pairs, 23 validation pairs,
batch size 2, learning rate 1e-4, seed 2026 and 88 optimizer steps per
epoch: 4,400 steps per arm. The image-order SHA-256 matches among all
three arms at every epoch. The selected checkpoints are epochs 48, 48 and
49 for SA/CA, DNS and DNS+SA/CA. Each arm was initialized independently
from the same original GLU-Net base weight; no arm continued from the
trained SA/CA checkpoint. The selected SA/CA checkpoint was used only for
the zero-gate identity check, which reports zero differences for coarse
flow, correlation and final flow.

The common base checkpoint SHA-256 is
`2eac424ee4c9998cafed7ca60e7181d9f54adcee7e8e71250c5f2c7c0c9e50b1`.
The pre-existing selected SA/CA checkpoint SHA-256 used for the identity
check is
`3d5550b78f8492e85c9f3c69a5767ba311b46c7b76732a362b130fd1bf56edbd`.
The source report stores the three selected checkpoint SHA-256 values.
The original 22-pair test results were **not** accessed for this comparison.

The within-arm checkpoint rule was *lowest validation coarse EPE*, matching
the previous SA/CA training protocol. Final 512px EPE was evaluated on
those selected checkpoints. This rule is not the final-EPE selection used
by the later, separate DCN study. The 50-epoch arms are internally matched;
they must not be presented as the same-budget experiment as the earlier
20-epoch SA/CA result.

## Validation outcome

All arms used the same 23 pairs and 5,413,286 valid 512px GT pixels.
Coarse EPE is in 256px image units. Coarse-full and final EPE are in 512px
units on the same full-resolution valid mask.

| Arm | Coarse EPE 256px | Coarse-full EPE 512px | Corr Top-1 | First 32-grid window outside | Final EPE 512px | GT ≥64px final EPE | Inference ms/pair | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SA/CA | 8.5717 | 17.2513 | 18.01% | 18.35% (3,826/20,851) | 9.7658 | 13.0847 | 45.83 | 1,005.64 |
| DNS alone | 9.4138 | 18.9038 | 18.75% | 18.93% (3,948/20,851) | 10.8886 | 14.5341 | 45.72 | 993.47 |
| DNS+SA/CA | **6.5352** | **13.0205** | **23.83%** | **17.27% (3,602/20,851)** | **7.2885** | **9.5066** | 47.07 | 1,008.24 |

Against the matched SA/CA arm, DNS alone worsens final EPE by 1.1227px
and improves only 9/23 pair means. DNS+SA/CA lowers final EPE by 2.4773px
(25.4%), lowers ≥64px-region EPE by 3.5781px (27.3%) and removes 224 of
20,851 first-window misses (1.08 percentage points). Runtime rises by
1.24 ms/pair (2.7%) and peak inference allocation by 2.60 MiB.
These timing differences are for this reported device/protocol only.

For valid queries whose GT displacement is ≥64px, the first-window miss
rate falls from 32.52% to 31.49%. This is a modest coverage change compared
with the large flow-EPE reduction. The available report does not isolate
final EPE on *the exact queries outside the first window*; therefore it
cannot establish that DNS solves those missed matches.

## Paired-image stability and failure cases

The combination improves 12/23 pair EPEs and worsens 11/23. Its median
paired change is only **−0.0029px**. The five largest valid-pixel-weighted
improvements (`000009`, `000012`, `000015`, `000021`, `000002`) account for
**94.0%** of the net valid-pixel-weighted EPE reduction. This concentration
means the aggregate gain is real for difficult pairs but not a broad,
uniform improvement.

| Pair | SA/CA final EPE | DNS+SA/CA final EPE | Change |
|---|---:|---:|---:|
| `000009` | 30.49 | 12.45 | −18.04 |
| `000015` | 19.86 | 6.94 | −12.91 |
| `000012` | 20.91 | 9.59 | −11.31 |
| `000021` | 16.08 | 8.76 | −7.32 |
| `000002` | 17.51 | 11.59 | −5.92 |
| `000014` | 33.04 | 27.50 | −5.55 |
| `000016` | 4.26 | 9.46 | **+5.20** |
| `000019` | 4.06 | 5.98 | **+1.92** |

All five preidentified high-error validation cases (`000014`, `000009`,
`000015`, `000002`, `000007`) improve, but `000014` still has 27.50px
pair EPE and its first-window miss fraction rises by 4.01 percentage
points. `000016` becomes a new major failure; its ≥64px-region EPE rises
from 4.61 to 10.80px. In all, 9 pairs with ≥64px GT pixels have worse
large-motion-region EPE despite the aggregate large-motion gain.

The pair-level CMR@5px falls from **10/23 (43.48%)** for SA/CA to
**7/23 (30.43%)** for DNS+SA/CA. Thus lower aggregate/mean EPE coexists
with fewer pairs below the 5px quality threshold. The pair-mean AEPE falls
from 10.0558 to 7.4799px, but this does not remove the stability concern.
Here CMR@5px counts an image pair when its mean valid-pixel EPE at 512×512
is strictly below 5px; it is not a per-pixel success fraction.

## Decision

DNS alone is not supported as a replacement for SA/CA under this
implementation and loss. This is a two-dimensional DNS-inspired feature
experiment without DSIR's anatomy-aware contrastive learning, so the
conclusion does not apply to complete DSIR.

DNS+SA/CA is a promising **large-motion candidate**, especially for the
preidentified difficult cases. It does **not** satisfy the previously
stated requirement of stable final-flow improvement without trading many
pairs for a few extreme gains: 11/23 pairs regress, CMR@5px falls, and
five pairs account for 94% of net gain. Keep the previously selected SA/CA
as the current default and keep the separate DCN study's coarse path fixed
to SA/CA. Do not stack DNS with DCN or use the 22-pair test to resolve this
validation-stage decision.

The raw `comparison.json` and `per_image.csv` in this directory are the
archived validation evidence. Checkpoint binaries remain in the local
`roadscene_runs` directory and are not copied into Git.
