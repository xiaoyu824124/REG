# REG cross-modal registration workspace

- `GLU-Net/`: the GLU-Net baseline and the RoadScene coarse SA/CA experiment.
- `CRFT-main/`: the CRFT reference implementation used to inspect feature
  interaction, dataset conventions, and evaluation code.

Start with [`GLU-Net/ROADSCENE_SA_CA.md`](GLU-Net/ROADSCENE_SA_CA.md). Keep
RoadScene images, VTMOT data, pretrained weights, and experiment checkpoints
outside this Git repository; supply their paths on the command line.

The current SA/CA implementation changes only GLU-Net's 16x16 global feature
stage. The local flow stages are unchanged. The included comparison script can
evaluate coarse matching without CuPy and final flow when CuPy/CUDA is ready.
