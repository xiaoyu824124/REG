# REG cross-modal registration workspace

- `GLU-Net/`: the GLU-Net baseline and the RoadScene coarse SA/CA experiment.

The CRFT reference stays in a separate official checkout. The new independent
comparison adapter accepts `--crft-root`; its trained RoadScene checkpoint is
currently missing. GLU/MIND experiments run without that external checkout.

Start with [`GLU-Net/ROADSCENE_SA_CA.md`](GLU-Net/ROADSCENE_SA_CA.md). The
baseline checkpoint is included at
`GLU-Net/pre_trained_models/GLUNet_DPED_CityScape_ADE.pth`. Keep RoadScene
images, VTMOT data, and experiment checkpoints outside this Git repository;
supply the RoadScene path on the command line.

The current SA/CA implementation changes only GLU-Net's 16x16 global feature
stage. The local flow stages are unchanged. The included comparison script can
evaluate coarse matching without CuPy and final flow when CuPy/CUDA is ready.

For the frozen-test protocol, unified AEPE/CMR, per-image validation evidence,
fixed MIND routes A/B, matched training and server commands, see
[`GLU-Net/ROADSCENE_CRFT_MIND.md`](GLU-Net/ROADSCENE_CRFT_MIND.md) and
[`GLU-Net/experiments/crft_mind/RESULTS.md`](GLU-Net/experiments/crft_mind/RESULTS.md).
