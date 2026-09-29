# REG cross-modal registration workspace

- `GLU-Net/`: the GLU-Net baseline and the RoadScene coarse SA/CA experiment.

The CRFT reference implementation is kept locally for code comparison and is
not required to run this GLU-Net experiment.

Start with [`GLU-Net/ROADSCENE_SA_CA.md`](GLU-Net/ROADSCENE_SA_CA.md). The
baseline checkpoint is included at
`GLU-Net/pre_trained_models/GLUNet_DPED_CityScape_ADE.pth`. Keep RoadScene
images, VTMOT data, and experiment checkpoints outside this Git repository;
supply the RoadScene path on the command line.

The current SA/CA implementation changes only GLU-Net's 16x16 global feature
stage. The local flow stages are unchanged. The included comparison script can
evaluate coarse matching without CuPy and final flow when CuPy/CUDA is ready.
