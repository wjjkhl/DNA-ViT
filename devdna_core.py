import math
import os
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


SQRT_HALF = 2.0 ** -0.5
EPS = 1e-8
_DCT_BASIS_CACHE: Dict[Tuple[int, str, int, str], torch.Tensor] = {}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_stage_fractions(stage_text: str) -> List[float]:
    values = [float(x.strip()) for x in stage_text.split(',') if x.strip()]
    if not values:
        raise ValueError('Stage list is empty.')
    values = sorted(set(values))
    if values[-1] != 1.0:
        values.append(1.0)
    if values[0] <= 0.0 or values[-1] > 1.0:
        raise ValueError('Stage fractions must be in (0, 1].')
    return values


def format_bytes(num_bytes: int) -> str:
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB']
    value = float(num_bytes)
    suffix_idx = 0
    while value >= 1024.0 and suffix_idx < len(suffixes) - 1:
        value /= 1024.0
        suffix_idx += 1
    return f'{value:.2f} {suffixes[suffix_idx]}'


def ensure_torchvision_nms_stub() -> None:
    try:
        lib = torch.library.Library('torchvision', 'DEF')
        lib.define('nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor')
    except Exception:
        pass


def stable_argsort(values: torch.Tensor, descending: bool = True) -> torch.Tensor:
    try:
        return torch.argsort(values, dim=0, descending=descending, stable=True)
    except TypeError:
        order = torch.argsort(values.detach().cpu(), dim=0, descending=descending)
        return order.to(device=values.device)


def lexicographic_order(keys: torch.Tensor, descending: bool = True) -> torch.Tensor:
    if keys.ndim != 2:
        raise ValueError('lexicographic_order expects a rank-2 tensor.')
    order = torch.arange(keys.shape[0], device=keys.device)
    for col in reversed(range(keys.shape[1])):
        column = keys.index_select(0, order)[:, col]
        order = order.index_select(0, stable_argsort(column, descending=descending))
    return order


@torch.no_grad()
def deterministic_projection_matrix(
    in_dim: int,
    proj_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    salt: int,
) -> torch.Tensor:
    if proj_dim <= 0:
        raise ValueError('proj_dim must be positive.')
    generator = torch.Generator(device='cpu')
    seed = int((1315423911 ^ (in_dim * 2654435761) ^ (proj_dim * 97531) ^ (salt * 2147483647)) & 0xFFFFFFFF)
    generator.manual_seed(seed)
    matrix = torch.randn((in_dim, proj_dim), generator=generator, dtype=torch.float32)
    matrix = F.normalize(matrix, dim=0, eps=EPS)
    return matrix.to(device=device, dtype=dtype)


@torch.no_grad()
def pca_project(descriptors: torch.Tensor, out_dim: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if descriptors.ndim != 2:
        raise ValueError('pca_project expects a rank-2 tensor.')
    x = descriptors.float()
    mean = x.mean(dim=0, keepdim=True)
    centered = x - mean
    max_rank = min(centered.shape[0], centered.shape[1])
    if max_rank <= 1:
        basis = torch.eye(centered.shape[1], device=x.device, dtype=x.dtype)[:, :1]
        coords = centered @ basis
        return coords, mean, basis
    q = min(out_dim, max_rank)
    _, _, v = torch.pca_lowrank(centered, q=q, center=False, niter=4)
    basis = v[:, :q]
    coords = centered @ basis
    return coords, mean, basis


@torch.no_grad()
def spectral_norm_power_iteration(
    matrix: torch.Tensor,
    num_iters: int = 16,
    eps: float = 1e-8,
) -> float:
    if matrix.numel() == 0:
        return 0.0
    weight = matrix.detach().float()
    rows, cols = weight.shape
    v = torch.randn(cols, device=weight.device, dtype=weight.dtype)
    v = F.normalize(v, dim=0, eps=eps)
    for _ in range(num_iters):
        u = F.normalize(weight @ v, dim=0, eps=eps)
        v = F.normalize(weight.transpose(0, 1) @ u, dim=0, eps=eps)
    sigma = torch.dot(u, weight @ v).abs().item()
    return float(sigma)


@torch.no_grad()
def _prefix_index(dim: int, active_length: int, ndim: int) -> Tuple[slice, ...]:
    index = [slice(None)] * ndim
    index[dim] = slice(0, active_length)
    return tuple(index)


@torch.no_grad()
def _haar_forward_step(x: torch.Tensor, dim: int, active_length: int) -> torch.Tensor:
    if active_length <= 1:
        return x
    out = x.clone()
    prefix = out[_prefix_index(dim, active_length, out.ndim)]
    even_idx = torch.arange(0, active_length, 2, device=out.device)
    odd_idx = torch.arange(1, active_length, 2, device=out.device)
    even = prefix.index_select(dim, even_idx)
    odd = prefix.index_select(dim, odd_idx)
    pair_count = odd.shape[dim]
    low_main = (even.narrow(dim, 0, pair_count) + odd) * SQRT_HALF
    if even.shape[dim] > pair_count:
        tail = even.narrow(dim, pair_count, even.shape[dim] - pair_count)
        low = torch.cat([low_main, tail], dim=dim)
    else:
        low = low_main
    high = (even.narrow(dim, 0, pair_count) - odd) * SQRT_HALF
    transformed = torch.cat([low, high], dim=dim)
    out[_prefix_index(dim, active_length, out.ndim)] = transformed
    return out


@torch.no_grad()
def _haar_inverse_step(x: torch.Tensor, dim: int, active_length: int) -> torch.Tensor:
    if active_length <= 1:
        return x
    out = x.clone()
    prefix = out[_prefix_index(dim, active_length, out.ndim)]
    low_len = (active_length + 1) // 2
    high_len = active_length // 2
    low = prefix.narrow(dim, 0, low_len)
    high = prefix.narrow(dim, low_len, high_len)
    restored_shape = list(prefix.shape)
    restored_shape[dim] = active_length
    restored = prefix.new_empty(restored_shape)
    even_idx = torch.arange(0, active_length, 2, device=out.device)
    odd_idx = torch.arange(1, active_length, 2, device=out.device)
    if high_len > 0:
        even_main = (low.narrow(dim, 0, high_len) + high) * SQRT_HALF
        odd = (low.narrow(dim, 0, high_len) - high) * SQRT_HALF
        restored.index_copy_(dim, even_idx[:high_len], even_main)
        restored.index_copy_(dim, odd_idx, odd)
    if low_len > high_len:
        tail = low.narrow(dim, high_len, low_len - high_len)
        restored.index_copy_(dim, even_idx[high_len:], tail)
    out[_prefix_index(dim, active_length, out.ndim)] = restored
    return out


@torch.no_grad()
def finite_haar_forward(x: torch.Tensor, dims: Sequence[int]) -> Tuple[torch.Tensor, Dict[str, List[int]]]:
    out = x.clone()
    schedules: Dict[str, List[int]] = {}
    for dim in dims:
        active = out.shape[dim]
        lengths: List[int] = []
        while active > 1:
            out = _haar_forward_step(out, dim=dim, active_length=active)
            lengths.append(active)
            active = (active + 1) // 2
        schedules[str(dim)] = lengths
    return out, schedules


@torch.no_grad()
def finite_haar_inverse(x: torch.Tensor, dims: Sequence[int], schedules: Dict[str, List[int]]) -> torch.Tensor:
    out = x.clone()
    for dim in reversed(tuple(dims)):
        lengths = schedules[str(dim)]
        for active in reversed(lengths):
            out = _haar_inverse_step(out, dim=dim, active_length=int(active))
    return out


@torch.no_grad()
def developmental_transform_forward(
    stack: torch.Tensor,
    dims: Sequence[int] = (2, 1, 0),
) -> Tuple[torch.Tensor, Dict[str, List[int]]]:
    if stack.ndim != 3:
        raise ValueError('developmental_transform_forward expects a rank-3 tensor [depth, rows, cols].')
    dims_tuple = tuple(int(dim) for dim in dims)
    if len(dims_tuple) <= 0:
        raise ValueError('developmental_transform_forward requires at least one transform dim.')
    return finite_haar_forward(stack, dims=dims_tuple)


@torch.no_grad()
def developmental_transform_inverse(
    coeff: torch.Tensor,
    schedules: Dict[str, List[int]],
    dims: Sequence[int] = (2, 1, 0),
) -> torch.Tensor:
    if coeff.ndim != 3:
        raise ValueError('developmental_transform_inverse expects a rank-3 tensor [depth, rows, cols].')
    dims_tuple = tuple(int(dim) for dim in dims)
    if len(dims_tuple) <= 0:
        raise ValueError('developmental_transform_inverse requires at least one transform dim.')
    return finite_haar_inverse(coeff, dims=dims_tuple, schedules=schedules)


@torch.no_grad()
def _orthonormal_dct_basis(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    size_i = int(size)
    if size_i <= 0:
        raise ValueError(f'DCT basis size must be positive, got {size!r}.')
    device_index = -1 if device.index is None else int(device.index)
    cache_key = (size_i, str(device.type), device_index, str(dtype))
    cached = _DCT_BASIS_CACHE.get(cache_key, None)
    if cached is not None:
        return cached

    n = torch.arange(size_i, device=device, dtype=torch.float32)
    k = torch.arange(size_i, device=device, dtype=torch.float32).unsqueeze(1)
    basis = torch.cos((math.pi / float(size_i)) * (n + 0.5) * k).to(dtype=dtype)
    scales = torch.full((size_i,), math.sqrt(2.0 / float(size_i)), device=device, dtype=dtype)
    scales[0] = math.sqrt(1.0 / float(size_i))
    basis = basis * scales.unsqueeze(1)
    _DCT_BASIS_CACHE[cache_key] = basis
    return basis


@torch.no_grad()
def dct2_transform_forward(stack: torch.Tensor) -> torch.Tensor:
    if stack.ndim != 3:
        raise ValueError('dct2_transform_forward expects a rank-3 tensor [depth, rows, cols].')
    rows = int(stack.shape[1])
    cols = int(stack.shape[2])
    basis_rows = _orthonormal_dct_basis(rows, device=stack.device, dtype=stack.dtype)
    basis_cols = _orthonormal_dct_basis(cols, device=stack.device, dtype=stack.dtype)
    tmp = torch.matmul(stack, basis_cols.transpose(0, 1))
    coeff = torch.matmul(basis_rows, tmp)
    return coeff


@torch.no_grad()
def dct2_transform_inverse(coeff: torch.Tensor) -> torch.Tensor:
    if coeff.ndim != 3:
        raise ValueError('dct2_transform_inverse expects a rank-3 tensor [depth, rows, cols].')
    rows = int(coeff.shape[1])
    cols = int(coeff.shape[2])
    basis_rows = _orthonormal_dct_basis(rows, device=coeff.device, dtype=coeff.dtype)
    basis_cols = _orthonormal_dct_basis(cols, device=coeff.device, dtype=coeff.dtype)
    tmp = torch.matmul(basis_rows.transpose(0, 1), coeff)
    stack = torch.matmul(tmp, basis_cols)
    return stack


@torch.no_grad()
def mlp_unit_descriptors(
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc1_bias: Optional[torch.Tensor],
    proj_dim: int,
) -> torch.Tensor:
    left = fc1_weight.detach().float()
    right = fc2_weight.detach().float().transpose(0, 1)
    bundle = torch.cat([left, right], dim=1)
    if fc1_bias is not None:
        bundle = torch.cat([bundle, fc1_bias.detach().float().unsqueeze(1)], dim=1)
    norms = bundle.norm(dim=1, keepdim=True)
    means = bundle.mean(dim=1, keepdim=True)
    abs_means = bundle.abs().mean(dim=1, keepdim=True)
    normalized = bundle / norms.clamp_min(EPS)
    anchors = deterministic_projection_matrix(
        in_dim=bundle.shape[1],
        proj_dim=proj_dim,
        device=bundle.device,
        dtype=bundle.dtype,
        salt=17,
    )
    projected = normalized @ anchors
    descriptor = torch.cat([norms, means, abs_means, projected], dim=1)
    return descriptor


@torch.no_grad()
def _fix_row_signs(bundle: torch.Tensor, salt: int) -> Tuple[torch.Tensor, torch.Tensor]:
    anchor = deterministic_projection_matrix(
        in_dim=bundle.shape[1],
        proj_dim=1,
        device=bundle.device,
        dtype=bundle.dtype,
        salt=salt,
    ).squeeze(1)
    score = bundle @ anchor
    signs = torch.where(score >= 0, torch.ones_like(score), -torch.ones_like(score))
    return bundle * signs.unsqueeze(1), signs


@torch.no_grad()
def canonicalize_attention_head_rotation(
    q_head: torch.Tensor,
    k_head: torch.Tensor,
    v_head: torch.Tensor,
    o_head: torch.Tensor,
    q_bias: Optional[torch.Tensor],
    k_bias: Optional[torch.Tensor],
    v_bias: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    qk_bundle = torch.cat([q_head.float(), k_head.float()], dim=1)
    uqk, sqk, _ = torch.linalg.svd(qk_bundle, full_matrices=False)
    rqk = uqk.transpose(0, 1)
    q_rot = rqk @ q_head.float()
    k_rot = rqk @ k_head.float()
    qk_rot = torch.cat([q_rot, k_rot], dim=1)
    qk_rot, qk_signs = _fix_row_signs(qk_rot, salt=101)
    q_rot = qk_rot[:, : q_head.shape[1]]
    k_rot = qk_rot[:, q_head.shape[1] :]
    if q_bias is not None:
        q_bias_rot = (rqk @ q_bias.float()) * qk_signs
    else:
        q_bias_rot = None
    if k_bias is not None:
        k_bias_rot = (rqk @ k_bias.float()) * qk_signs
    else:
        k_bias_rot = None

    vo_bundle = torch.cat([v_head.float(), o_head.float().transpose(0, 1)], dim=1)
    uvo, svo, _ = torch.linalg.svd(vo_bundle, full_matrices=False)
    rvo = uvo.transpose(0, 1)
    v_rot = rvo @ v_head.float()
    o_rot_t = rvo @ o_head.float().transpose(0, 1)
    vo_rot = torch.cat([v_rot, o_rot_t], dim=1)
    vo_rot, vo_signs = _fix_row_signs(vo_rot, salt=211)
    v_rot = vo_rot[:, : v_head.shape[1]]
    o_rot = vo_rot[:, v_head.shape[1] :].transpose(0, 1)
    if v_bias is not None:
        v_bias_rot = (rvo @ v_bias.float()) * vo_signs
    else:
        v_bias_rot = None

    return q_rot, k_rot, v_rot, o_rot, q_bias_rot, k_bias_rot, v_bias_rot, sqk, svo


@torch.no_grad()
def attention_head_descriptor(
    q_head: torch.Tensor,
    k_head: torch.Tensor,
    v_head: torch.Tensor,
    o_head: torch.Tensor,
    sqk: torch.Tensor,
    svo: torch.Tensor,
    q_bias: Optional[torch.Tensor],
    k_bias: Optional[torch.Tensor],
    v_bias: Optional[torch.Tensor],
    proj_dim: int,
) -> torch.Tensor:
    embed_dim = q_head.shape[1]
    anchors = deterministic_projection_matrix(embed_dim, proj_dim, q_head.device, q_head.dtype, salt=37)

    def projected_stats(matrix: torch.Tensor) -> torch.Tensor:
        mean = matrix.mean(dim=0)
        abs_mean = matrix.abs().mean(dim=0)
        return torch.cat([mean @ anchors, abs_mean @ anchors], dim=0)

    norms = torch.tensor([
        q_head.norm(),
        k_head.norm(),
        v_head.norm(),
        o_head.norm(),
    ], device=q_head.device, dtype=q_head.dtype)
    singular = torch.cat([sqk[: min(8, sqk.numel())], svo[: min(8, svo.numel())]], dim=0)
    desc_parts = [
        norms,
        singular,
        projected_stats(q_head),
        projected_stats(k_head),
        projected_stats(v_head),
        projected_stats(o_head.transpose(0, 1)),
    ]
    if q_bias is not None:
        desc_parts.append(q_bias.float())
    if k_bias is not None:
        desc_parts.append(k_bias.float())
    if v_bias is not None:
        desc_parts.append(v_bias.float())
    return torch.cat([part.reshape(-1) for part in desc_parts], dim=0)


@torch.no_grad()
def sensitivity_weighted_target_counts(
    coeff_abs_list: Sequence[torch.Tensor],
    sensitivity_list: Sequence[float],
    total_target: int,
) -> List[int]:
    if len(coeff_abs_list) != len(sensitivity_list):
        raise ValueError('coeff_abs_list and sensitivity_list must have the same length.')
    total_numel = sum(int(x.numel()) for x in coeff_abs_list)
    if total_numel == 0:
        return [0 for _ in coeff_abs_list]
    total_target = max(0, min(int(total_target), total_numel))
    if total_target == 0:
        return [0 for _ in coeff_abs_list]
    if total_target == total_numel:
        return [int(x.numel()) for x in coeff_abs_list]

    scores: List[torch.Tensor] = []
    for coeff_abs, sensitivity in zip(coeff_abs_list, sensitivity_list):
        weight = math.sqrt(max(float(sensitivity), EPS))
        scores.append(coeff_abs.float() * weight)

    max_score = max(float(score.max().item()) for score in scores if score.numel() > 0)
    lo = 0.0
    hi = max_score
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        count = 0
        for score in scores:
            if score.numel() == 0:
                continue
            count += int((score >= mid).sum().item())
        if count > total_target:
            lo = mid
        else:
            hi = mid
    counts = [int((score >= hi).sum().item()) for score in scores]
    deficit = total_target - sum(counts)
    if deficit > 0:
        residuals: List[Tuple[float, int]] = []
        for idx, score in enumerate(scores):
            mask = score < hi
            if int(mask.sum().item()) == 0:
                residuals.append((float('-inf'), idx))
                continue
            value = float(score[mask].max().item())
            residuals.append((value, idx))
        residuals.sort(reverse=True)
        pointer = 0
        while deficit > 0 and residuals[pointer][0] > float('-inf'):
            _, idx = residuals[pointer]
            counts[idx] += 1
            deficit -= 1
            pointer = (pointer + 1) % len(residuals)
    elif deficit < 0:
        overflow = -deficit
        residuals = []
        for idx, score in enumerate(scores):
            mask = score >= hi
            if int(mask.sum().item()) == 0:
                residuals.append((float('inf'), idx))
                continue
            value = float(score[mask].min().item())
            residuals.append((value, idx))
        residuals.sort()
        pointer = 0
        while overflow > 0 and residuals[pointer][0] < float('inf'):
            _, idx = residuals[pointer]
            if counts[idx] > 0:
                counts[idx] -= 1
                overflow -= 1
            pointer = (pointer + 1) % len(residuals)
    return counts
