"""Dataset entry points, imported on demand to avoid unrelated dependencies."""

from importlib import import_module


_MODULES = {
    "KITTI_occ": "KITTI_optical_flow",
    "KITTI_noc": "KITTI_optical_flow",
    "mpi_sintel_clean": "mpisintel",
    "mpi_sintel_final": "mpisintel",
    "HPatchesdataset": "hpatches",
    "DatasetNoGT": "dataset_no_gt",
    "TSS": "TSS",
}
__all__ = tuple(_MODULES)


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module(f".{_MODULES[name]}", __name__), name)
    globals()[name] = value
    return value
