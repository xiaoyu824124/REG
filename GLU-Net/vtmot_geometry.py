"""Backward XY flow geometry for VTMOT video registration.

All fields use full 512x512 pixel units. A registration flow maps a visible
target pixel to the infrared source pixel sampled by ``cv2.remap``.
"""

from __future__ import annotations

import cv2
import numpy as np


def grid(hw):
    height, width = hw
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    return np.stack((xx, yy), axis=-1)


def sample(field, coordinates, *, interpolation=cv2.INTER_LINEAR):
    """Sample an HxWxC field at XY coordinates; return values and bounds."""
    height, width = field.shape[:2]
    x, y = coordinates[..., 0], coordinates[..., 1]
    inside = (np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= width - 1)
              & (y >= 0) & (y <= height - 1))
    safe_x = np.nan_to_num(x, nan=-1, posinf=-1, neginf=-1).astype(np.float32)
    safe_y = np.nan_to_num(y, nan=-1, posinf=-1, neginf=-1).astype(np.float32)
    sampled = cv2.remap(field, safe_x, safe_y, interpolation,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return sampled, inside


def compose(first, second, first_valid=None, second_valid=None):
    """A->B then B->C: C(A)=A+first(A)+second(A+first(A))."""
    if first.shape != second.shape or first.ndim != 3 or first.shape[-1] != 2:
        raise ValueError('both flows must be HxWx2 on the same grid')
    coords = grid(first.shape[:2]) + first
    second_at, inside = sample(second, coords)
    valid = inside & np.isfinite(first).all(axis=-1) & np.isfinite(second_at).all(axis=-1)
    if first_valid is not None:
        valid &= first_valid.astype(bool)
    if second_valid is not None:
        sampled_valid, _ = sample(second_valid.astype(np.float32), coords)
        valid &= sampled_valid > .999
    result = first + second_at
    result[~np.isfinite(result).all(axis=-1)] = 0
    return result.astype(np.float32), valid


def propagate(current_vi_to_previous_vi, previous_vi_to_previous_ir,
              previous_ir_to_current_ir, vi_valid=None, reg_valid=None,
              ir_valid=None):
    """Current VI -> previous VI -> previous IR -> current IR."""
    partial, valid1 = compose(current_vi_to_previous_vi,
                              previous_vi_to_previous_ir, vi_valid, reg_valid)
    final, valid2 = compose(partial, previous_ir_to_current_ir, valid1, ir_valid)
    return final, valid2


def backward_warp(image, flow):
    if image.shape[:2] != flow.shape[:2]:
        raise ValueError('image and flow grids differ')
    warped, inside = sample(image, grid(flow.shape[:2]) + flow)
    return warped, inside


def transform_flow(matrix, hw):
    """Projective A->B map as backward XY displacement on the A grid."""
    base = grid(hw)
    xy1 = np.concatenate((base, np.ones((*hw, 1), np.float32)), axis=-1)
    mapped = xy1 @ np.asarray(matrix, np.float64).T
    mapped = mapped[..., :2] / mapped[..., 2:3]
    return (mapped - base).astype(np.float32)


def self_test():
    hw = (80, 96)
    identity = np.eye(3)
    translation = np.array([[1, 0, 4], [0, 1, -3], [0, 0, 1]], np.float64)
    angle = np.deg2rad(7.)
    c, s = np.cos(angle), np.sin(angle)
    center = np.array([[1, 0, -48], [0, 1, -40], [0, 0, 1.]])
    uncenter = np.linalg.inv(center)
    rotation = uncenter @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]]) @ center
    for vi_time, registration, ir_time in (
            (identity, identity, identity),
            (translation, identity, np.linalg.inv(translation)),
            (translation, translation, translation),
            (rotation, translation, np.linalg.inv(rotation))):
        predicted, valid = propagate(transform_flow(vi_time, hw),
                                     transform_flow(registration, hw),
                                     transform_flow(ir_time, hw))
        expected = transform_flow(ir_time @ registration @ vi_time, hw)
        interior = valid & (grid(hw)[..., 0] > 8) & (grid(hw)[..., 0] < 87)
        interior &= (grid(hw)[..., 1] > 8) & (grid(hw)[..., 1] < 71)
        if not interior.any() or np.max(np.abs(predicted[interior] - expected[interior])) > .06:
            raise AssertionError('flow composition order/direction failed')
        if np.any(valid & ~np.isfinite(predicted).all(axis=-1)):
            raise AssertionError('invalid finite mask')
    jump = transform_flow(np.array([[1, 0, 110], [0, 1, 0], [0, 0, 1]]), hw)
    _, valid = propagate(jump, transform_flow(identity, hw), transform_flow(identity, hw))
    if valid.any():
        raise AssertionError('out-of-bounds propagation was not masked')
    return {'identity_translation_rotation': 'passed', 'out_of_bounds': 'passed',
            'flow': 'backward XY, current VI target to current IR source'}


if __name__ == '__main__':
    print(self_test())
