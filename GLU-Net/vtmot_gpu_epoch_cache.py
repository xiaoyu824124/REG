"""Optional, exact-value GPU cache for one VTMOT training stride phase.

Images are still decoded and GT is still built on the CPU. The decoded image
tensors for *one* epoch are kept on the GPU; GT flow and validity masks remain
in host memory to leave room for model activations and gradients. The cache is
released before validation. It never changes the dataset or GT mask.
"""

from __future__ import annotations

import hashlib
import time

import torch
from torch.utils.data import DataLoader, Dataset


FIELDS = ('source_image', 'target_image', 'flow_map',
          'correspondence_mask')
GPU_FIELDS = ('source_image', 'target_image')


class GpuEpochFrames(Dataset):
    already_cuda = True

    def __init__(self, tensors, names):
        self.tensors = tensors
        self.names = names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        return {**{key: tensor[index] for key, tensor in self.tensors.items()},
                'name': self.names[index]}


def build_gpu_epoch_cache(frames, device, num_workers=0, verify=False,
                          reserve_gib=4.0):
    """Return cached copies in original dataset order and inclusive load stats."""
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('GPU epoch cache requires a CUDA-enabled PyTorch')
    if num_workers < 0:
        raise ValueError('num_workers must be nonnegative')
    start = time.perf_counter()
    options = {'num_workers': num_workers, 'pin_memory': True}
    if num_workers:
        options['prefetch_factor'] = 2
    data = DataLoader(frames, batch_size=8, shuffle=False, **options)
    iterator = iter(data)
    first = next(iterator)
    length = len(frames)
    bytes_needed = sum(length * first[key][0].numel() *
                       first[key].element_size() for key in GPU_FIELDS)
    host_bytes = sum(length * first[key][0].numel() *
                     first[key].element_size() for key in FIELDS
                     if key not in GPU_FIELDS)
    torch.cuda.empty_cache()
    cuda_index = (device.index if device.index is not None else
                  torch.cuda.current_device())
    free, total = torch.cuda.mem_get_info(cuda_index)
    reserve = int(reserve_gib * (1024 ** 3))
    if free - bytes_needed < reserve:
        raise MemoryError(
            f'GPU image cache needs {bytes_needed/1024**3:.2f} GiB plus '
            f'{reserve_gib:.1f} GiB reserve, but free VRAM is '
            f'{free/1024**3:.2f} of {total/1024**3:.2f} GiB; '
            'use streaming mode instead')
    tensors = {key: torch.empty((length, *first[key].shape[1:]),
                                dtype=first[key].dtype,
                                device=device if key in GPU_FIELDS else 'cpu')
               for key in FIELDS}
    names = []
    offset = 0

    def add(batch):
        nonlocal offset
        count = len(batch['name'])
        section = slice(offset, offset + count)
        for key in FIELDS:
            tensors[key][section].copy_(batch[key],
                                        non_blocking=key in GPU_FIELDS)
        # A worker may release/reuse its pinned source buffers as soon as this
        # batch leaves scope. Finish H2D copies before accepting another batch.
        torch.cuda.synchronize(device)
        if verify:
            for key in FIELDS:
                if not torch.equal(tensors[key][section],
                                   batch[key].to(tensors[key].device)):
                    raise AssertionError(f'GPU cache changed {key} at {offset}')
        names.extend(batch['name'])
        offset += count

    add(first)
    for batch in iterator:
        add(batch)
    torch.cuda.synchronize(device)
    if offset != length or len(names) != length:
        raise AssertionError('GPU cache has incomplete frame count')
    elapsed = time.perf_counter() - start
    return GpuEpochFrames(tensors, names), {
        'frames': length, 'mode': 'images_gpu_gt_host',
        'bytes': bytes_needed, 'host_bytes': host_bytes,
        'gib': bytes_needed / 1024 ** 3,
        'load_seconds': elapsed,
        'device_name': torch.cuda.get_device_name(cuda_index),
        'gpu_allocated_mib_after_load':
            torch.cuda.memory_allocated(cuda_index) / 1024 ** 2,
        'verified_all_input_tensors': bool(verify)}


def expected_order_sha256(frames, epoch, seed, batch_size):
    """Compute the original DataLoader shuffle order without rereading images."""
    generator = torch.Generator().manual_seed(seed + epoch)
    indices = DataLoader(range(len(frames)), batch_size=batch_size,
                         shuffle=True, generator=generator, num_workers=0)
    digest = hashlib.sha256()
    for batch in indices:
        for index in batch.tolist():
            name, stem = frames.items[index]
            digest.update((f'{name}/{stem}\n').encode())
    return digest.hexdigest()
