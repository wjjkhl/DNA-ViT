import concurrent.futures
import csv
import json
import math
import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from devdna_codec import (
    decode_flat_quantized_stack,
    decode_progressive_bitplane_stack,
    deserialize_exact_tensor,
)
from devdna_core import dct2_transform_inverse, developmental_transform_inverse
from devdna_eval import build_timm_model, create_official_imagenet_loader
from devdna_model import ARCHIVE_FORMAT, MANIFEST_NAME, load_gaugedna_archive


STREAMING_RUNTIME_MODES: Tuple[str, ...] = (
    'serial_full_decode',
    'lazy_family_decode',
    'lazy_family_decode_overlap',
    'oracle_ready_to_run',
)
STREAMING_RUNTIME_GROUP_POLICIES: Tuple[str, ...] = (
    'block_attn_mlp',
    'block_full',
    'super_block',
)
STREAMING_RUNTIME_GROUP_POLICY_AUTO = 'auto'


@dataclass
class _RuntimeGroupSpec:
    group_name: str
    kind: str
    block_indices: Tuple[int, ...]
    param_names: Tuple[str, ...]
    blocks: Tuple[nn.Module, ...]


class _ArchiveStreamingReader:
    def __init__(self, archive_path: str):
        self.archive_path = str(archive_path)
        self.zf = zipfile.ZipFile(self.archive_path, mode='r')
        self.manifest = json.loads(self.zf.read(MANIFEST_NAME).decode('utf-8'))
        if self.manifest.get('format') != ARCHIVE_FORMAT:
            raise ValueError(f'Unsupported archive format: {self.manifest.get("format")!r}')

        self.codec_mode = str(self.manifest.get('codec_allocation_mode', 'legacy_global'))
        self.compressed_param_names: List[str] = []
        self._tensor_stack_entries: Dict[str, Dict[str, Any]] = {}
        self._legacy_param_entries: Dict[str, Tuple[Dict[str, Any], int]] = {}

        tensor_stacks = self.manifest.get('tensor_stacks', None)
        if self.codec_mode == 'blockwise_thresholded' or isinstance(tensor_stacks, dict):
            if not isinstance(tensor_stacks, dict) or len(tensor_stacks) <= 0:
                raise KeyError('Archive tensor_stacks is empty; cannot run family-wise streaming runtime.')
            for tensor_name, stack_meta in tensor_stacks.items():
                param_name = str(stack_meta.get('param_name', tensor_name))
                self._tensor_stack_entries[param_name] = stack_meta
                self.compressed_param_names.append(param_name)
        else:
            stacks = self.manifest.get('stacks', None)
            if not isinstance(stacks, dict) or len(stacks) <= 0:
                raise KeyError('Legacy archive is missing non-empty `stacks` metadata.')
            for _stack_key, stack_meta in stacks.items():
                param_names = list(stack_meta.get('param_names', []))
                if len(param_names) <= 0:
                    continue
                for layer_idx, param_name in enumerate(param_names):
                    key = str(param_name)
                    self._legacy_param_entries[key] = (stack_meta, int(layer_idx))
                    self.compressed_param_names.append(key)
        self.compressed_param_names = sorted(list(set(self.compressed_param_names)))
        self.compressed_param_set = set(self.compressed_param_names)

    def close(self) -> None:
        try:
            self.zf.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def load_exact_tensors_into_state(self, state: Dict[str, torch.Tensor]) -> None:
        for tensor_meta in self.manifest.get('exact_tensors', []):
            name = str(tensor_meta['name'])
            state[name] = deserialize_exact_tensor(self.zf, tensor_meta)

    @staticmethod
    def _decode_plane_count(stack_meta: Dict[str, Any], decode_fraction: float, decode_planes: Optional[int]) -> int:
        total_planes = int(stack_meta['code']['num_bitplanes'])
        if decode_planes is not None and int(decode_planes) > 0:
            return min(int(decode_planes), total_planes)
        if float(decode_fraction) >= 1.0:
            return int(total_planes)
        if float(decode_fraction) <= 0.0:
            return 1
        return max(1, min(total_planes, int(math.ceil(float(total_planes) * float(decode_fraction)))))

    def _decode_stack_code(
        self,
        stack_meta: Dict[str, Any],
        decode_fraction: float,
        decode_planes: Optional[int],
    ) -> torch.Tensor:
        code_meta = stack_meta['code']
        code_encoding = str(code_meta.get('encoding', ''))
        if code_encoding == 'flat_sparse_quantized_v1':
            return decode_flat_quantized_stack(
                zf=self.zf,
                meta=code_meta,
                device=torch.device('cpu'),
            )
        planes_to_decode = self._decode_plane_count(
            stack_meta,
            decode_fraction=float(decode_fraction),
            decode_planes=(None if decode_planes is None else int(decode_planes)),
        )
        return decode_progressive_bitplane_stack(
            zf=self.zf,
            meta=code_meta,
            device=torch.device('cpu'),
            max_planes=int(planes_to_decode),
        )

    @staticmethod
    def _inverse_stack_transform(coeff: torch.Tensor, stack_meta: Dict[str, Any]) -> torch.Tensor:
        transform_name = str(stack_meta.get('transform', 'finite_length_haar_3d'))
        if transform_name == 'identity':
            return coeff.cpu()
        if transform_name == 'orthonormal_dct2_2d':
            return dct2_transform_inverse(coeff).cpu()
        if transform_name in {'finite_length_haar_3d', 'finite_length_haar_2d'}:
            schedules = stack_meta.get('schedules', None)
            if schedules is None:
                raise KeyError(
                    'Archive stack entry is missing `schedules` for non-identity transform '
                    f'(`{transform_name}`).'
                )
            dims = (2, 1, 0) if transform_name == 'finite_length_haar_3d' else (2, 1)
            return developmental_transform_inverse(coeff, schedules=schedules, dims=dims).cpu()
        raise ValueError(f'Unsupported transform in archive: {transform_name!r}')

    def decode_param(
        self,
        param_name: str,
        decode_fraction: float = 1.0,
        decode_planes: Optional[int] = None,
    ) -> torch.Tensor:
        key = str(param_name)
        if key in self._tensor_stack_entries:
            stack_meta = self._tensor_stack_entries[key]
            coeff = self._decode_stack_code(stack_meta, decode_fraction=float(decode_fraction), decode_planes=decode_planes)
            stack = self._inverse_stack_transform(coeff, stack_meta=stack_meta)
            if int(stack.shape[0]) != 1:
                raise ValueError(
                    f'Expected stack depth=1 for tensor `{key}`, got shape={tuple(stack.shape)}.'
                )
            return stack[0].contiguous()

        if key in self._legacy_param_entries:
            stack_meta, idx = self._legacy_param_entries[key]
            coeff = self._decode_stack_code(stack_meta, decode_fraction=float(decode_fraction), decode_planes=decode_planes)
            stack = self._inverse_stack_transform(coeff, stack_meta=stack_meta)
            if int(idx) < 0 or int(idx) >= int(stack.shape[0]):
                raise IndexError(f'Legacy stack index out of range for `{key}`: idx={idx}, depth={int(stack.shape[0])}')
            return stack[int(idx)].contiguous()

        raise KeyError(f'Compressed param not found in archive index: {key}')

    def decode_group(
        self,
        param_names: Sequence[str],
        decode_fraction: float = 1.0,
        decode_planes: Optional[int] = None,
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        if len(param_names) <= 0:
            return {}, 0.0
        t0 = time.perf_counter()
        out: Dict[str, torch.Tensor] = {}
        for name in param_names:
            out[str(name)] = self.decode_param(
                param_name=str(name),
                decode_fraction=float(decode_fraction),
                decode_planes=(None if decode_planes is None else int(decode_planes)),
            )
        elapsed = float(time.perf_counter() - t0)
        return out, elapsed


@torch.no_grad()
def _iter_transformer_blocks(model: nn.Module) -> List[Tuple[int, str, nn.Module]]:
    out: List[Tuple[int, str, nn.Module]] = []
    direct_blocks = getattr(model, 'blocks', None)
    if isinstance(direct_blocks, (list, tuple, nn.ModuleList, nn.Sequential)):
        for block_idx, block in enumerate(direct_blocks):
            out.append((int(block_idx), f'blocks.{int(block_idx)}', block))
        if len(out) > 0:
            return out

    layers = getattr(model, 'layers', None)
    if isinstance(layers, (list, tuple, nn.ModuleList, nn.Sequential)):
        global_idx = 0
        for stage_idx, layer in enumerate(layers):
            stage_blocks = getattr(layer, 'blocks', None)
            if not isinstance(stage_blocks, (list, tuple, nn.ModuleList, nn.Sequential)):
                continue
            for stage_block_idx, block in enumerate(stage_blocks):
                out.append((int(global_idx), f'layers.{int(stage_idx)}.blocks.{int(stage_block_idx)}', block))
                global_idx += 1
    if len(out) <= 0:
        raise TypeError(
            'The supplied model does not look like a supported timm transformer backbone. '
            'Expected either `.blocks` or `.layers[*].blocks[*]`.'
        )
    return out


@torch.no_grad()
def _normalize_group_policy(value: Any) -> str:
    policy = str(value if value is not None else '').strip().lower()
    if not policy:
        return 'block_attn_mlp'
    allowed = set(STREAMING_RUNTIME_GROUP_POLICIES)
    allowed.add(STREAMING_RUNTIME_GROUP_POLICY_AUTO)
    if policy not in allowed:
        raise ValueError(
            f'Unsupported streaming runtime group policy: {policy!r}. '
            f'Supported: {[STREAMING_RUNTIME_GROUP_POLICY_AUTO, *STREAMING_RUNTIME_GROUP_POLICIES]}'
        )
    return policy


@torch.no_grad()
def _build_runtime_group_specs(
    model: nn.Module,
    compressed_param_set: Sequence[str],
    group_policy: str = 'block_attn_mlp',
    super_block_k: int = 2,
) -> List[_RuntimeGroupSpec]:
    policy = _normalize_group_policy(group_policy)
    if policy == STREAMING_RUNTIME_GROUP_POLICY_AUTO:
        policy = 'block_attn_mlp'
    compressed = set(str(x) for x in compressed_param_set)
    groups: List[_RuntimeGroupSpec] = []
    block_full_units: List[Tuple[int, nn.Module, Tuple[str, ...]]] = []
    for block_idx, prefix, block in _iter_transformer_blocks(model):
        attn_names_all = (f'{prefix}.attn.qkv.weight', f'{prefix}.attn.proj.weight')
        mlp_names_all = (f'{prefix}.mlp.fc1.weight', f'{prefix}.mlp.fc2.weight')
        attn_names = tuple(name for name in attn_names_all if name in compressed)
        mlp_names = tuple(name for name in mlp_names_all if name in compressed)

        if policy == 'block_attn_mlp':
            # Keep full compute graph coverage even when this block has no compressed params.
            groups.append(
                _RuntimeGroupSpec(
                    group_name=f'block_{int(block_idx)}.attn',
                    kind='attn',
                    block_indices=(int(block_idx),),
                    param_names=attn_names,
                    blocks=(block,),
                )
            )
            groups.append(
                _RuntimeGroupSpec(
                    group_name=f'block_{int(block_idx)}.mlp',
                    kind='mlp',
                    block_indices=(int(block_idx),),
                    param_names=mlp_names,
                    blocks=(block,),
                )
            )
            continue

        full_names = tuple([*attn_names, *mlp_names])
        # Keep full compute graph coverage even when this block has no compressed params.
        block_full_units.append((int(block_idx), block, full_names))
        if policy == 'block_full':
            groups.append(
                _RuntimeGroupSpec(
                    group_name=f'block_{int(block_idx)}.full',
                    kind='block_full',
                    block_indices=(int(block_idx),),
                    param_names=full_names,
                    blocks=(block,),
                )
            )

    if policy == 'super_block':
        if int(super_block_k) <= 0:
            raise ValueError(f'super_block_k must be > 0, got {super_block_k}')
        step = int(super_block_k)
        for start in range(0, len(block_full_units), step):
            chunk = block_full_units[start : start + step]
            if len(chunk) <= 0:
                continue
            block_indices = tuple(int(item[0]) for item in chunk)
            block_modules = tuple(item[1] for item in chunk)
            param_names: List[str] = []
            for _, _block, one_names in chunk:
                param_names.extend(list(one_names))
            groups.append(
                _RuntimeGroupSpec(
                    group_name=f'super_block_{int(block_indices[0])}_{int(block_indices[-1])}',
                    kind='super_block',
                    block_indices=block_indices,
                    param_names=tuple(param_names),
                    blocks=block_modules,
                )
            )

    if len(groups) <= 0:
        raise RuntimeError(
            'No runtime groups resolved from archive/model compressed parameter intersection. '
            f'policy={policy!r}'
        )
    return groups


@torch.no_grad()
def _prepare_batch(batch: Any, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise TypeError(f'Unexpected loader batch type for runtime eval: {type(batch)}')
    images = batch[0]
    target = batch[1]
    if not torch.is_tensor(images):
        images = torch.as_tensor(images)
    if not torch.is_tensor(target):
        target = torch.as_tensor(target)
    images = images.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    return images, target


@torch.no_grad()
def _sync_cuda_if_needed(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def _current_gpu_mem_allocated(device: torch.device) -> int:
    if device.type != 'cuda':
        return 0
    return int(torch.cuda.memory_allocated(device))


@torch.no_grad()
def _forward_vit_stem(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    x = images
    if hasattr(model, 'patch_embed'):
        x = model.patch_embed(x)
    if hasattr(model, '_pos_embed'):
        x = model._pos_embed(x)
    else:
        cls_token = getattr(model, 'cls_token', None)
        if cls_token is not None:
            cls = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls, x), dim=1)
        pos_embed = getattr(model, 'pos_embed', None)
        if pos_embed is not None:
            x = x + pos_embed
        pos_drop = getattr(model, 'pos_drop', None)
        if isinstance(pos_drop, nn.Module):
            x = pos_drop(x)
    patch_drop = getattr(model, 'patch_drop', None)
    if isinstance(patch_drop, nn.Module):
        x = patch_drop(x)
    norm_pre = getattr(model, 'norm_pre', None)
    if isinstance(norm_pre, nn.Module):
        x = norm_pre(x)
    return x


@torch.no_grad()
def _forward_block_attn(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
    norm1 = getattr(block, 'norm1', None)
    if not isinstance(norm1, nn.Module):
        raise AttributeError('Transformer block is missing `norm1` for attn split forward.')
    attn = getattr(block, 'attn', None)
    if not isinstance(attn, nn.Module):
        raise AttributeError('Transformer block is missing `attn` for attn split forward.')
    y = attn(norm1(x))
    ls1 = getattr(block, 'ls1', None)
    if isinstance(ls1, nn.Module):
        y = ls1(y)
    drop_path1 = getattr(block, 'drop_path1', None)
    if isinstance(drop_path1, nn.Module):
        y = drop_path1(y)
    else:
        drop_path = getattr(block, 'drop_path', None)
        if isinstance(drop_path, nn.Module):
            y = drop_path(y)
    return x + y


@torch.no_grad()
def _forward_block_mlp(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
    norm2 = getattr(block, 'norm2', None)
    if not isinstance(norm2, nn.Module):
        raise AttributeError('Transformer block is missing `norm2` for mlp split forward.')
    mlp = getattr(block, 'mlp', None)
    if not isinstance(mlp, nn.Module):
        raise AttributeError('Transformer block is missing `mlp` for mlp split forward.')
    y = mlp(norm2(x))
    ls2 = getattr(block, 'ls2', None)
    if isinstance(ls2, nn.Module):
        y = ls2(y)
    drop_path2 = getattr(block, 'drop_path2', None)
    if isinstance(drop_path2, nn.Module):
        y = drop_path2(y)
    else:
        drop_path = getattr(block, 'drop_path', None)
        if isinstance(drop_path, nn.Module):
            y = drop_path(y)
    return x + y


@torch.no_grad()
def _forward_block_full(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = block(x)
    if not torch.is_tensor(out):
        raise TypeError(f'Unexpected block output type: {type(out)}')
    return out


@torch.no_grad()
def _forward_vit_tail(model: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    x = hidden
    norm = getattr(model, 'norm', None)
    if isinstance(norm, nn.Module):
        x = norm(x)
    if hasattr(model, 'forward_head'):
        return model.forward_head(x, pre_logits=False)
    if x.ndim == 3:
        x = x[:, 0]
    head = getattr(model, 'head', None)
    if isinstance(head, nn.Module):
        return head(x)
    return x


@torch.no_grad()
def _topk_correct_counts(logits: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    if logits.ndim != 2:
        raise ValueError(f'Expected logits rank-2 [N, C], got shape={tuple(logits.shape)}')
    if target.ndim != 1:
        target = target.reshape(-1)
    topk = int(min(5, int(logits.shape[1])))
    _, pred = logits.topk(topk, dim=1, largest=True, sorted=True)
    correct = pred.eq(target.view(-1, 1))
    top1 = float(correct[:, :1].any(dim=1).sum().item())
    top5 = float(correct[:, :topk].any(dim=1).sum().item())
    return top1, top5


class _ResidentMemoryTracker:
    def __init__(self, static_resident_bytes: int, dynamic_param_bytes: Dict[str, int]):
        self.static_resident_bytes = int(max(0, static_resident_bytes))
        self.dynamic_param_bytes = {str(k): int(max(0, v)) for k, v in dynamic_param_bytes.items()}
        self.loaded_params: set = set()
        self.peak_resident_bytes = int(self.static_resident_bytes)

    def _update_peak(self) -> None:
        current = self.current_resident_bytes()
        if current > self.peak_resident_bytes:
            self.peak_resident_bytes = int(current)

    def on_load(self, param_names: Sequence[str]) -> None:
        for name in param_names:
            key = str(name)
            if key in self.dynamic_param_bytes:
                self.loaded_params.add(key)
        self._update_peak()

    def on_release(self, param_names: Sequence[str]) -> None:
        for name in param_names:
            self.loaded_params.discard(str(name))
        self._update_peak()

    def current_resident_bytes(self) -> int:
        dynamic = int(sum(int(self.dynamic_param_bytes.get(name, 0)) for name in self.loaded_params))
        return int(self.static_resident_bytes + dynamic)


@torch.no_grad()
def _load_group_to_device(
    decoded_group: Dict[str, torch.Tensor],
    param_names: Sequence[str],
    param_map: Dict[str, nn.Parameter],
    device: torch.device,
    resident_tracker: _ResidentMemoryTracker,
) -> float:
    if len(param_names) <= 0:
        return 0.0
    _sync_cuda_if_needed(device)
    t0 = time.perf_counter()
    for name in param_names:
        key = str(name)
        if key not in decoded_group:
            raise KeyError(f'Missing decoded tensor for `{key}`.')
        if key not in param_map:
            raise KeyError(f'Model parameter not found for `{key}`.')
        param = param_map[key]
        tensor_cpu = decoded_group[key]
        tensor_device = tensor_cpu.to(device=device, dtype=param.dtype, non_blocking=True)
        param.data = tensor_device
    _sync_cuda_if_needed(device)
    elapsed = float(time.perf_counter() - t0)
    resident_tracker.on_load(param_names)
    return elapsed


@torch.no_grad()
def _release_group_from_device(
    param_names: Sequence[str],
    param_map: Dict[str, nn.Parameter],
    cpu_placeholders: Dict[str, torch.Tensor],
    resident_tracker: _ResidentMemoryTracker,
) -> None:
    if len(param_names) <= 0:
        return
    for name in param_names:
        key = str(name)
        if key not in param_map or key not in cpu_placeholders:
            continue
        param_map[key].data = cpu_placeholders[key]
    resident_tracker.on_release(param_names)


@torch.no_grad()
def _run_group_forward(
    group: _RuntimeGroupSpec,
    hidden: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    autocast_dtype: torch.dtype,
) -> Tuple[torch.Tensor, float]:
    _sync_cuda_if_needed(device)
    t0 = time.perf_counter()
    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=bool(use_amp)):
        if group.kind == 'attn':
            out = _forward_block_attn(group.blocks[0], hidden)
        elif group.kind == 'mlp':
            out = _forward_block_mlp(group.blocks[0], hidden)
        elif group.kind == 'block_full':
            out = _forward_block_full(group.blocks[0], hidden)
        elif group.kind == 'super_block':
            out = hidden
            for one_block in group.blocks:
                out = _forward_block_full(one_block, out)
        else:
            raise ValueError(f'Unknown group kind: {group.kind}')
    _sync_cuda_if_needed(device)
    return out, float(time.perf_counter() - t0)


@torch.no_grad()
def _prepare_lazy_streaming_model(
    reader: _ArchiveStreamingReader,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, nn.Parameter], Dict[str, torch.Tensor], Dict[str, int], int]:
    model_name = str(reader.manifest['model_name'])
    model = build_timm_model(
        model_name=model_name,
        pretrained=False,
        checkpoint_path='',
        pretrained_weight_path='',
    )
    state = model.state_dict()
    reader.load_exact_tensors_into_state(state)

    for name in reader.compressed_param_names:
        if name in state:
            state[name] = torch.zeros_like(state[name], device='cpu')
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    param_map = dict(model.named_parameters())
    cpu_placeholders: Dict[str, torch.Tensor] = {}
    dynamic_param_bytes: Dict[str, int] = {}
    for name in reader.compressed_param_names:
        if name not in param_map:
            continue
        param = param_map[name]
        cpu_placeholder = torch.zeros_like(param.detach().cpu())
        cpu_placeholders[name] = cpu_placeholder
        param.data = cpu_placeholder
        dynamic_param_bytes[name] = int(param.numel() * param.element_size())

    static_resident_bytes = 0
    for name, param in param_map.items():
        if name in reader.compressed_param_set:
            continue
        if param.device.type != device.type:
            continue
        if device.type == 'cuda' and int(param.device.index or 0) != int(device.index or 0):
            continue
        static_resident_bytes += int(param.numel() * param.element_size())

    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return model, param_map, cpu_placeholders, dynamic_param_bytes, int(static_resident_bytes)


@torch.no_grad()
def _evaluate_ready_model_once(
    model: nn.Module,
    loader,
    device: torch.device,
    eval_samples: int,
    warmup_batches: int,
    use_amp: bool,
    autocast_dtype: torch.dtype,
) -> Dict[str, Any]:
    if int(eval_samples) <= 0:
        raise ValueError(f'eval_samples must be > 0, got: {eval_samples}')
    criterion = torch.nn.CrossEntropyLoss().to(device)
    model.eval()

    iterator_box: Dict[str, Any] = {'it': iter(loader)}

    def _next_batch():
        try:
            return next(iterator_box['it'])
        except StopIteration:
            iterator_box['it'] = iter(loader)
            return next(iterator_box['it'])

    with torch.inference_mode():
        for _ in range(int(max(0, warmup_batches))):
            images, _ = _prepare_batch(_next_batch(), device=device)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=bool(use_amp)):
                _ = model(images)
        _sync_cuda_if_needed(device)
        t0 = time.perf_counter()
        seen = 0
        loss_sum = 0.0
        top1_sum = 0.0
        top5_sum = 0.0
        while seen < int(eval_samples):
            images, target = _prepare_batch(_next_batch(), device=device)
            remain = int(eval_samples - seen)
            if int(images.size(0)) > remain:
                images = images[:remain]
                target = target[:remain]
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=bool(use_amp)):
                logits = model(images)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                if isinstance(logits, dict):
                    logits = logits.get('logits', list(logits.values())[0])
                if not torch.is_tensor(logits):
                    raise TypeError(f'Unexpected model output type in ready eval: {type(logits)}')
                loss = criterion(logits.float(), target)
            top1, top5 = _topk_correct_counts(logits.float(), target)
            bs = int(images.size(0))
            seen += bs
            loss_sum += float(loss.item()) * float(bs)
            top1_sum += float(top1)
            top5_sum += float(top5)
        _sync_cuda_if_needed(device)
        infer_seconds = float(time.perf_counter() - t0)

    if seen != int(eval_samples):
        raise RuntimeError(f'Expected eval_samples={eval_samples}, but seen={seen}.')
    return {
        'samples': int(seen),
        'infer_seconds': float(infer_seconds),
        'top1': float(100.0 * top1_sum / max(float(seen), 1.0)),
        'top5': float(100.0 * top5_sum / max(float(seen), 1.0)),
        'loss': float(loss_sum / max(float(seen), 1.0)),
    }


@torch.no_grad()
def _run_streaming_batch(
    mode: str,
    model: nn.Module,
    groups: Sequence[_RuntimeGroupSpec],
    reader: _ArchiveStreamingReader,
    param_map: Dict[str, nn.Parameter],
    cpu_placeholders: Dict[str, torch.Tensor],
    resident_tracker: _ResidentMemoryTracker,
    images: torch.Tensor,
    device: torch.device,
    decode_fraction: float,
    decode_planes: Optional[int],
    prefetch_next_group: bool,
    decoded_group_cache: Optional[Dict[str, Dict[str, torch.Tensor]]],
    use_amp: bool,
    autocast_dtype: torch.dtype,
) -> Dict[str, Any]:
    hidden = _forward_vit_stem(model, images)
    decode_load_seconds_by_group: List[float] = []
    compute_seconds_by_group: List[float] = []
    peak_decode_load_mem = 0
    decoded_group_cache_hits = 0
    decoded_group_cache_misses = 0

    def _decode_group_with_optional_cache(
        group: _RuntimeGroupSpec,
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        nonlocal decoded_group_cache_hits
        nonlocal decoded_group_cache_misses
        if decoded_group_cache is not None and group.group_name in decoded_group_cache:
            decoded_group_cache_hits += 1
            return decoded_group_cache[group.group_name], 0.0
        decoded_group, decode_seconds = reader.decode_group(
            param_names=group.param_names,
            decode_fraction=float(decode_fraction),
            decode_planes=(None if decode_planes is None else int(decode_planes)),
        )
        if decoded_group_cache is not None:
            decoded_group_cache[group.group_name] = decoded_group
        decoded_group_cache_misses += 1
        return decoded_group, float(decode_seconds)

    if mode == 'lazy_family_decode':
        for group in groups:
            decoded_group, decode_seconds = _decode_group_with_optional_cache(group)
            load_seconds = _load_group_to_device(
                decoded_group=decoded_group,
                param_names=group.param_names,
                param_map=param_map,
                device=device,
                resident_tracker=resident_tracker,
            )
            peak_decode_load_mem = max(int(peak_decode_load_mem), int(_current_gpu_mem_allocated(device)))
            decode_load_seconds_by_group.append(float(decode_seconds + load_seconds))
            hidden, compute_seconds = _run_group_forward(
                group=group,
                hidden=hidden,
                device=device,
                use_amp=bool(use_amp),
                autocast_dtype=autocast_dtype,
            )
            compute_seconds_by_group.append(float(compute_seconds))
            _release_group_from_device(
                param_names=group.param_names,
                param_map=param_map,
                cpu_placeholders=cpu_placeholders,
                resident_tracker=resident_tracker,
            )
    elif mode == 'lazy_family_decode_overlap':
        if len(groups) <= 0:
            raise RuntimeError('No runtime groups to execute for overlap mode.')
        current_group = groups[0]
        decoded_group, decode_seconds = _decode_group_with_optional_cache(current_group)
        load_seconds = _load_group_to_device(
            decoded_group=decoded_group,
            param_names=current_group.param_names,
            param_map=param_map,
            device=device,
            resident_tracker=resident_tracker,
        )
        peak_decode_load_mem = max(int(peak_decode_load_mem), int(_current_gpu_mem_allocated(device)))
        decode_load_seconds_by_group.append(float(decode_seconds + load_seconds))

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            for group_idx, group in enumerate(groups):
                future = None
                next_group_from_cache: Optional[_RuntimeGroupSpec] = None
                if bool(prefetch_next_group) and group_idx + 1 < len(groups):
                    next_group = groups[group_idx + 1]
                    if decoded_group_cache is not None and next_group.group_name in decoded_group_cache:
                        next_group_from_cache = next_group
                    else:
                        future = executor.submit(
                            reader.decode_group,
                            next_group.param_names,
                            float(decode_fraction),
                            (None if decode_planes is None else int(decode_planes)),
                        )
                hidden, compute_seconds = _run_group_forward(
                    group=group,
                    hidden=hidden,
                    device=device,
                    use_amp=bool(use_amp),
                    autocast_dtype=autocast_dtype,
                )
                compute_seconds_by_group.append(float(compute_seconds))
                if group_idx + 1 < len(groups):
                    next_group = groups[group_idx + 1]
                    if next_group_from_cache is not None:
                        decoded_group_cache_hits += 1
                        next_decoded = decoded_group_cache[next_group.group_name]
                        next_decode_seconds = 0.0
                    elif future is None:
                        next_decoded, next_decode_seconds = _decode_group_with_optional_cache(next_group)
                    else:
                        next_decoded, next_decode_seconds = future.result()
                        next_decode_seconds = float(next_decode_seconds)
                        if decoded_group_cache is not None:
                            decoded_group_cache[next_group.group_name] = next_decoded
                        decoded_group_cache_misses += 1
                    next_load_seconds = _load_group_to_device(
                        decoded_group=next_decoded,
                        param_names=next_group.param_names,
                        param_map=param_map,
                        device=device,
                        resident_tracker=resident_tracker,
                    )
                    peak_decode_load_mem = max(int(peak_decode_load_mem), int(_current_gpu_mem_allocated(device)))
                    decode_load_seconds_by_group.append(float(next_decode_seconds + next_load_seconds))
                    _release_group_from_device(
                        param_names=group.param_names,
                        param_map=param_map,
                        cpu_placeholders=cpu_placeholders,
                        resident_tracker=resident_tracker,
                    )
                else:
                    _release_group_from_device(
                        param_names=group.param_names,
                        param_map=param_map,
                        cpu_placeholders=cpu_placeholders,
                        resident_tracker=resident_tracker,
                    )
    else:
        raise ValueError(f'Unsupported streaming batch mode: {mode}')

    logits = _forward_vit_tail(model, hidden)
    if not torch.is_tensor(logits):
        raise TypeError(f'Unexpected tail logits type: {type(logits)}')

    total_decode_load = float(sum(float(x) for x in decode_load_seconds_by_group))
    boot_seconds = 0.0
    tail_seconds = 0.0
    hidden_decode_seconds = 0.0
    visible_decode_seconds = total_decode_load
    if mode == 'lazy_family_decode_overlap' and len(decode_load_seconds_by_group) > 0:
        boot_seconds = float(decode_load_seconds_by_group[0])
        tail_seconds = float(compute_seconds_by_group[-1]) if len(compute_seconds_by_group) > 0 else 0.0
        visible = float(boot_seconds)
        for idx in range(1, len(decode_load_seconds_by_group)):
            prev_compute = float(compute_seconds_by_group[idx - 1]) if (idx - 1) < len(compute_seconds_by_group) else 0.0
            curr_decode_load = float(decode_load_seconds_by_group[idx])
            visible += float(max(0.0, curr_decode_load - prev_compute))
        visible_decode_seconds = float(max(0.0, visible))
        hidden_decode_seconds = float(max(0.0, total_decode_load - visible_decode_seconds))

    return {
        'logits': logits,
        'decode_load_seconds_by_group': decode_load_seconds_by_group,
        'compute_seconds_by_group': compute_seconds_by_group,
        'total_decode_load_seconds': float(total_decode_load),
        'boot_seconds': float(boot_seconds),
        'tail_seconds': float(tail_seconds),
        'hidden_decode_seconds': float(hidden_decode_seconds),
        'visible_decode_seconds': float(visible_decode_seconds),
        'peak_decode_load_gpu_mem_bytes': int(peak_decode_load_mem),
        'decoded_group_cache_hits': int(decoded_group_cache_hits),
        'decoded_group_cache_misses': int(decoded_group_cache_misses),
    }


@torch.no_grad()
def _profile_runtime_group_policy(
    policy: str,
    model: nn.Module,
    reader: _ArchiveStreamingReader,
    param_map: Dict[str, nn.Parameter],
    cpu_placeholders: Dict[str, torch.Tensor],
    resident_tracker: _ResidentMemoryTracker,
    next_batch_fn: Callable[[], Any],
    device: torch.device,
    decode_fraction: float,
    decode_planes: Optional[int],
    use_amp: bool,
    autocast_dtype: torch.dtype,
    super_block_k: int,
    auto_profile_batches: int,
) -> Tuple[Dict[str, Any], List[_RuntimeGroupSpec]]:
    groups = _build_runtime_group_specs(
        model=model,
        compressed_param_set=reader.compressed_param_names,
        group_policy=str(policy),
        super_block_k=int(super_block_k),
    )
    profile_rounds = int(max(1, auto_profile_batches))
    total_decode_load_seconds = 0.0
    total_compute_seconds = 0.0
    for _ in range(profile_rounds):
        images, _ = _prepare_batch(next_batch_fn(), device=device)
        one = _run_streaming_batch(
            mode='lazy_family_decode',
            model=model,
            groups=groups,
            reader=reader,
            param_map=param_map,
            cpu_placeholders=cpu_placeholders,
            resident_tracker=resident_tracker,
            images=images,
            device=device,
            decode_fraction=float(decode_fraction),
            decode_planes=(None if decode_planes is None else int(decode_planes)),
            prefetch_next_group=False,
            decoded_group_cache=None,
            use_amp=bool(use_amp),
            autocast_dtype=autocast_dtype,
        )
        total_decode_load_seconds += float(one['total_decode_load_seconds'])
        total_compute_seconds += float(sum(float(x) for x in one.get('compute_seconds_by_group', [])))
    ratio = float(total_compute_seconds / max(total_decode_load_seconds, 1e-12))
    return (
        {
            'policy': str(policy),
            'group_count': int(len(groups)),
            'profile_batches': int(profile_rounds),
            'decode_load_seconds': float(total_decode_load_seconds),
            'compute_seconds': float(total_compute_seconds),
            'compute_decode_ratio': float(ratio),
        },
        groups,
    )


@torch.no_grad()
def _run_lazy_streaming_mode_once(
    mode: str,
    archive_path: str,
    device: torch.device,
    loader,
    eval_samples: int,
    warmup_batches: int,
    decode_fraction: float,
    decode_planes: Optional[int],
    prefetch_next_group: bool,
    group_policy: str,
    super_block_k: int,
    auto_profile_batches: int,
    auto_compute_decode_ratio_target: float,
    cache_decoded_groups: bool,
    reset_decoded_cache_before_timed_eval: bool,
    use_amp: bool,
    autocast_dtype: torch.dtype,
) -> Dict[str, Any]:
    criterion = torch.nn.CrossEntropyLoss().to(device)
    with _ArchiveStreamingReader(archive_path) as reader:
        model, param_map, cpu_placeholders, dynamic_param_bytes, static_resident = _prepare_lazy_streaming_model(
            reader=reader,
            device=device,
        )
        resident_tracker = _ResidentMemoryTracker(
            static_resident_bytes=int(static_resident),
            dynamic_param_bytes=dynamic_param_bytes,
        )
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        iterator_box: Dict[str, Any] = {'it': iter(loader)}

        def _next_batch():
            try:
                return next(iterator_box['it'])
            except StopIteration:
                iterator_box['it'] = iter(loader)
                return next(iterator_box['it'])

        requested_policy = _normalize_group_policy(group_policy)
        effective_super_block_k = int(max(1, int(super_block_k)))
        auto_profile: List[Dict[str, Any]] = []
        selected_policy = str(requested_policy)
        auto_selection_reason = 'manual_policy'
        auto_profile_seconds = 0.0
        if requested_policy == STREAMING_RUNTIME_GROUP_POLICY_AUTO:
            auto_profile_start = time.perf_counter()
            target_ratio = float(max(0.0, float(auto_compute_decode_ratio_target)))
            selected_groups: Optional[List[_RuntimeGroupSpec]] = None
            candidate_groups_map: Dict[str, List[_RuntimeGroupSpec]] = {}
            for candidate in STREAMING_RUNTIME_GROUP_POLICIES:
                one_profile, candidate_groups = _profile_runtime_group_policy(
                    policy=str(candidate),
                    model=model,
                    reader=reader,
                    param_map=param_map,
                    cpu_placeholders=cpu_placeholders,
                    resident_tracker=resident_tracker,
                    next_batch_fn=_next_batch,
                    device=device,
                    decode_fraction=float(decode_fraction),
                    decode_planes=(None if decode_planes is None else int(decode_planes)),
                    use_amp=bool(use_amp),
                    autocast_dtype=autocast_dtype,
                    super_block_k=int(effective_super_block_k),
                    auto_profile_batches=int(auto_profile_batches),
                )
                auto_profile.append(one_profile)
                candidate_groups_map[str(candidate)] = candidate_groups
                if selected_groups is None and float(one_profile['compute_decode_ratio']) >= float(target_ratio):
                    selected_policy = str(candidate)
                    selected_groups = candidate_groups
                    auto_selection_reason = (
                        f'first_candidate_meeting_target_ratio>={float(target_ratio):.6f}'
                    )
            if selected_groups is None:
                # If no candidate reaches the target ratio, choose the empirically best
                # profile instead of always falling back to the coarsest granularity.
                best_profile = sorted(
                    auto_profile,
                    key=lambda item: (
                        float(item.get('compute_decode_ratio', 0.0)),
                        -float(item.get('decode_load_seconds', 0.0)),
                    ),
                    reverse=True,
                )[0]
                selected_policy = str(best_profile.get('policy', STREAMING_RUNTIME_GROUP_POLICIES[-1]))
                selected_groups = candidate_groups_map.get(str(selected_policy), None)
                if selected_groups is None:
                    selected_groups = _build_runtime_group_specs(
                        model=model,
                        compressed_param_set=reader.compressed_param_names,
                        group_policy=str(selected_policy),
                        super_block_k=int(effective_super_block_k),
                    )
                auto_selection_reason = (
                    'no_candidate_meets_target_ratio, choose_best_profile_by_ratio_then_decode_load'
                )
            groups = selected_groups
            auto_profile_seconds = float(time.perf_counter() - auto_profile_start)
        else:
            groups = _build_runtime_group_specs(
                model=model,
                compressed_param_set=reader.compressed_param_names,
                group_policy=str(requested_policy),
                super_block_k=int(effective_super_block_k),
            )

        # Ensure metric/eval sample selection is comparable with non-auto runs.
        # Auto profiling above consumes a few batches from the loader iterator.
        iterator_box['it'] = iter(loader)

        decoded_group_cache: Optional[Dict[str, Dict[str, torch.Tensor]]] = (
            {} if bool(cache_decoded_groups) else None
        )

        with torch.inference_mode():
            for _ in range(int(max(0, warmup_batches))):
                warm_images, _ = _prepare_batch(_next_batch(), device=device)
                _ = _run_streaming_batch(
                    mode=mode,
                    model=model,
                    groups=groups,
                    reader=reader,
                    param_map=param_map,
                    cpu_placeholders=cpu_placeholders,
                    resident_tracker=resident_tracker,
                    images=warm_images,
                    device=device,
                    decode_fraction=float(decode_fraction),
                    decode_planes=(None if decode_planes is None else int(decode_planes)),
                    prefetch_next_group=bool(prefetch_next_group),
                    decoded_group_cache=decoded_group_cache,
                    use_amp=bool(use_amp),
                    autocast_dtype=autocast_dtype,
                )

            if decoded_group_cache is not None and bool(reset_decoded_cache_before_timed_eval):
                decoded_group_cache.clear()

            seen = 0
            loss_sum = 0.0
            top1_sum = 0.0
            top5_sum = 0.0
            total_decode_load_seconds = 0.0
            total_boot_seconds = 0.0
            total_tail_seconds = 0.0
            total_hidden_decode_seconds = 0.0
            total_visible_decode_seconds = 0.0
            peak_decode_load_gpu_mem_bytes = 0
            total_cache_hits = 0
            total_cache_misses = 0

            _sync_cuda_if_needed(device)
            t0 = time.perf_counter()
            while seen < int(eval_samples):
                images, target = _prepare_batch(_next_batch(), device=device)
                remain = int(eval_samples - seen)
                if int(images.size(0)) > remain:
                    images = images[:remain]
                    target = target[:remain]
                run = _run_streaming_batch(
                    mode=mode,
                    model=model,
                    groups=groups,
                    reader=reader,
                    param_map=param_map,
                    cpu_placeholders=cpu_placeholders,
                    resident_tracker=resident_tracker,
                    images=images,
                    device=device,
                    decode_fraction=float(decode_fraction),
                    decode_planes=(None if decode_planes is None else int(decode_planes)),
                    prefetch_next_group=bool(prefetch_next_group),
                    decoded_group_cache=decoded_group_cache,
                    use_amp=bool(use_amp),
                    autocast_dtype=autocast_dtype,
                )
                logits = run['logits'].float()
                loss = criterion(logits, target)
                top1, top5 = _topk_correct_counts(logits=logits, target=target)
                bs = int(images.size(0))
                seen += bs
                loss_sum += float(loss.item()) * float(bs)
                top1_sum += float(top1)
                top5_sum += float(top5)
                total_decode_load_seconds += float(run['total_decode_load_seconds'])
                total_boot_seconds += float(run['boot_seconds'])
                total_tail_seconds += float(run['tail_seconds'])
                total_hidden_decode_seconds += float(run['hidden_decode_seconds'])
                total_visible_decode_seconds += float(run['visible_decode_seconds'])
                total_cache_hits += int(run.get('decoded_group_cache_hits', 0))
                total_cache_misses += int(run.get('decoded_group_cache_misses', 0))
                peak_decode_load_gpu_mem_bytes = max(
                    int(peak_decode_load_gpu_mem_bytes),
                    int(run['peak_decode_load_gpu_mem_bytes']),
                )
            _sync_cuda_if_needed(device)
            e2e_seconds = float(time.perf_counter() - t0)

    if seen != int(eval_samples):
        raise RuntimeError(f'Expected eval_samples={eval_samples}, but seen={seen}.')
    peak_gpu = int(torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' else 0
    return {
        'samples': int(seen),
        'e2e_seconds': float(e2e_seconds),
        'top1': float(100.0 * top1_sum / max(float(seen), 1.0)),
        'top5': float(100.0 * top5_sum / max(float(seen), 1.0)),
        'loss': float(loss_sum / max(float(seen), 1.0)),
        'total_decode_load_seconds': float(total_decode_load_seconds),
        'boot_seconds': float(total_boot_seconds),
        'tail_seconds': float(total_tail_seconds),
        'hidden_decode_seconds': float(total_hidden_decode_seconds),
        'visible_decode_seconds_estimated': float(total_visible_decode_seconds),
        'peak_gpu_mem_bytes': int(peak_gpu),
        'peak_decode_load_gpu_mem_bytes': int(peak_decode_load_gpu_mem_bytes),
        'peak_model_resident_gpu_mem_bytes': int(resident_tracker.peak_resident_bytes),
        'group_policy_used': str(selected_policy),
        'group_count': int(len(groups)),
        'super_block_k': int(effective_super_block_k),
        'decoded_group_cache_enabled': bool(cache_decoded_groups),
        'decoded_group_cache_entries': int(len(decoded_group_cache) if decoded_group_cache is not None else 0),
        'decoded_group_cache_hits': int(total_cache_hits),
        'decoded_group_cache_misses': int(total_cache_misses),
        'auto_group_policy_profile': auto_profile,
        'auto_group_policy_selection_reason': str(auto_selection_reason),
        'auto_group_policy_profile_seconds': float(auto_profile_seconds),
        'reset_decoded_cache_before_timed_eval': bool(reset_decoded_cache_before_timed_eval),
    }


@torch.no_grad()
def _mean(values: Sequence[float]) -> float:
    if len(values) <= 0:
        return 0.0
    return float(sum(float(v) for v in values) / float(len(values)))


@torch.no_grad()
def _max_int(values: Sequence[int]) -> int:
    if len(values) <= 0:
        return 0
    return int(max(int(v) for v in values))


@torch.no_grad()
def _aggregate_repeat_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    first_row = rows[0] if len(rows) > 0 else {}
    return {
        'samples': int(first_row.get('samples', 0)) if len(rows) > 0 else 0,
        'e2e_seconds': _mean([float(item.get('e2e_seconds', 0.0)) for item in rows]),
        'infer_seconds': _mean([float(item.get('infer_seconds', 0.0)) for item in rows]),
        'top1': _mean([float(item.get('top1', 0.0)) for item in rows]),
        'top5': _mean([float(item.get('top5', 0.0)) for item in rows]),
        'loss': _mean([float(item.get('loss', 0.0)) for item in rows]),
        'total_decode_load_seconds': _mean([float(item.get('total_decode_load_seconds', 0.0)) for item in rows]),
        'boot_seconds': _mean([float(item.get('boot_seconds', 0.0)) for item in rows]),
        'tail_seconds': _mean([float(item.get('tail_seconds', 0.0)) for item in rows]),
        'hidden_decode_seconds': _mean([float(item.get('hidden_decode_seconds', 0.0)) for item in rows]),
        'visible_decode_seconds_estimated': _mean([float(item.get('visible_decode_seconds_estimated', 0.0)) for item in rows]),
        'peak_gpu_mem_bytes': _max_int([int(item.get('peak_gpu_mem_bytes', 0)) for item in rows]),
        'peak_decode_load_gpu_mem_bytes': _max_int(
            [int(item.get('peak_decode_load_gpu_mem_bytes', 0)) for item in rows]
        ),
        'peak_model_resident_gpu_mem_bytes': _max_int(
            [int(item.get('peak_model_resident_gpu_mem_bytes', 0)) for item in rows]
        ),
        'decoded_group_cache_hits': _mean([float(item.get('decoded_group_cache_hits', 0.0)) for item in rows]),
        'decoded_group_cache_misses': _mean([float(item.get('decoded_group_cache_misses', 0.0)) for item in rows]),
        'decoded_group_cache_entries': _max_int([int(item.get('decoded_group_cache_entries', 0)) for item in rows]),
        'auto_group_policy_profile_seconds': _mean(
            [float(item.get('auto_group_policy_profile_seconds', 0.0)) for item in rows]
        ),
        'group_policy_used': str(first_row.get('group_policy_used', '')),
        'auto_group_policy_selection_reason': str(first_row.get('auto_group_policy_selection_reason', '')),
    }


@torch.no_grad()
def _normalize_runtime_modes(value: Any) -> List[str]:
    if isinstance(value, str):
        tokens = [item.strip() for item in value.replace('\n', ',').split(',') if item.strip()]
    elif isinstance(value, (list, tuple)):
        tokens = [str(item).strip() for item in value if str(item).strip()]
    else:
        tokens = []
    if len(tokens) <= 0:
        return list(STREAMING_RUNTIME_MODES)
    if len(tokens) == 1 and tokens[0].lower() == 'all':
        return list(STREAMING_RUNTIME_MODES)
    allowed = set(STREAMING_RUNTIME_MODES)
    resolved: List[str] = []
    seen = set()
    invalid: List[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        if token not in allowed:
            invalid.append(token)
            continue
        resolved.append(token)
    if len(invalid) > 0:
        raise ValueError(
            f'Unsupported streaming runtime modes: {invalid}. Supported: {list(STREAMING_RUNTIME_MODES)}'
        )
    for mode in STREAMING_RUNTIME_MODES:
        if mode not in resolved:
            resolved.append(mode)
    return resolved


@torch.no_grad()
def _write_result_csv(path: str, rows: Sequence[Dict[str, Any]]) -> str:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'runtime_mode',
        'e2e_latency_ms_per_sample',
        'infer_only_latency_ms_per_sample',
        'visible_decode_overhead_ms_per_sample',
        'decode_overhead_hiding_ratio',
        'boot_seconds',
        'tail_seconds',
        'hidden_decode_seconds',
        'visible_decode_seconds',
        'auto_group_policy_profile_overhead_ms_per_sample',
        'group_policy_used',
        'peak_gpu_mem_bytes',
        'peak_decode_load_gpu_mem_bytes',
        'peak_model_resident_gpu_mem_bytes',
        'memory_reduction_ratio',
        'top1',
        'top5',
    ]
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, None) for name in fieldnames})
    return str(output_path)


@torch.no_grad()
def run_streaming_runtime_familywise_experiment(args, device: torch.device) -> Dict[str, Any]:
    archive_path = str(args.archive_path)
    if not os.path.isfile(archive_path):
        raise FileNotFoundError(f'Archive not found: {archive_path}')

    runtime_modes = _normalize_runtime_modes(getattr(args, 'streaming_runtime_modes', None))
    eval_batch_size = int(getattr(args, 'streaming_runtime_eval_batch_size', args.batch_size))
    eval_samples = int(getattr(args, 'streaming_runtime_eval_samples', 1024))
    warmup_batches = int(getattr(args, 'streaming_runtime_warmup_batches', 5))
    repeats = int(getattr(args, 'streaming_runtime_repeats', 3))
    if eval_samples <= 0:
        raise ValueError(f'streaming_runtime_eval_samples must be > 0, got {eval_samples}')
    if eval_batch_size <= 0:
        raise ValueError(f'streaming_runtime_eval_batch_size must be > 0, got {eval_batch_size}')
    if repeats <= 0:
        raise ValueError(f'streaming_runtime_repeats must be > 0, got {repeats}')

    with _ArchiveStreamingReader(archive_path) as reader_for_meta:
        model_name = str(reader_for_meta.manifest['model_name'])
        manifest_for_report = dict(reader_for_meta.manifest)
    archive_bytes = int(os.path.getsize(archive_path))
    payload_mib = float(archive_bytes / (1024.0 * 1024.0))

    reference_model = build_timm_model(
        model_name=model_name,
        pretrained=False,
        checkpoint_path='',
        pretrained_weight_path='',
    )
    loader, data_config = create_official_imagenet_loader(
        model=reference_model,
        data_root=args.data_root,
        split=args.split,
        batch_size=int(eval_batch_size),
        workers=int(args.workers),
        class_map=args.class_map,
        pin_memory=bool(args.pin_memory),
        crop_pct=args.crop_pct,
        interpolation=args.interpolation,
        no_prefetcher=bool(args.no_prefetcher),
        device=device,
        use_test_size=True,
        enforce_official_preprocess=True,
    )
    del reference_model

    use_amp = bool(device.type == 'cuda')
    autocast_dtype = torch.bfloat16 if str(args.amp_dtype) == 'bf16' else torch.float16
    decode_fraction = float(getattr(args, 'decode_fraction', 1.0))
    decode_planes_arg = int(getattr(args, 'decode_planes', -1))
    decode_planes: Optional[int] = None if decode_planes_arg <= 0 else int(decode_planes_arg)
    raw_group_policy = getattr(args, 'streaming_runtime_group_policy', '')
    if not str(raw_group_policy).strip():
        raw_group_policy = getattr(args, 'streaming_runtime_logical_scheduler_unit', 'block_attn_mlp')
    group_policy = _normalize_group_policy(raw_group_policy)
    super_block_k = int(getattr(args, 'streaming_runtime_super_block_k', 2))
    auto_profile_batches = int(getattr(args, 'streaming_runtime_auto_profile_batches', 1))
    auto_compute_decode_ratio_target = float(
        getattr(args, 'streaming_runtime_auto_compute_decode_ratio_target', 1.15)
    )
    cache_decoded_groups = bool(getattr(args, 'streaming_runtime_cache_decoded_groups', True))
    reset_decoded_cache_before_timed_eval = bool(
        getattr(args, 'streaming_runtime_reset_decoded_cache_before_timed_eval', True)
    )
    drop_first_repeat_for_aggregation = bool(
        getattr(args, 'streaming_runtime_drop_first_repeat_for_aggregation', True)
    )
    if int(super_block_k) <= 0:
        raise ValueError(f'streaming_runtime_super_block_k must be > 0, got {super_block_k}')
    if int(auto_profile_batches) <= 0:
        raise ValueError(
            f'streaming_runtime_auto_profile_batches must be > 0, got {auto_profile_batches}'
        )
    if float(auto_compute_decode_ratio_target) < 0.0:
        raise ValueError(
            'streaming_runtime_auto_compute_decode_ratio_target must be >= 0, '
            f'got {auto_compute_decode_ratio_target}'
        )

    mode_repeat_rows: Dict[str, List[Dict[str, Any]]] = {mode: [] for mode in runtime_modes}

    if 'oracle_ready_to_run' in runtime_modes:
        print('[streaming_runtime] running mode=oracle_ready_to_run ...', flush=True)
        oracle_model, _oracle_manifest = load_gaugedna_archive(
            archive_path=archive_path,
            model_builder=lambda name: build_timm_model(
                model_name=name,
                pretrained=False,
                checkpoint_path='',
                pretrained_weight_path='',
            ),
            device=device,
            decode_fraction=1.0,
            decode_planes=None,
        )
        for repeat_idx in range(int(repeats)):
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            one = _evaluate_ready_model_once(
                model=oracle_model,
                loader=loader,
                device=device,
                eval_samples=int(eval_samples),
                warmup_batches=int(warmup_batches),
                use_amp=bool(use_amp),
                autocast_dtype=autocast_dtype,
            )
            peak_gpu = int(torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' else 0
            resident_peak = int(
                sum(
                    int(p.numel() * p.element_size())
                    for p in oracle_model.parameters()
                    if p.device.type == device.type and (device.type != 'cuda' or int(p.device.index or 0) == int(device.index or 0))
                )
            )
            mode_repeat_rows['oracle_ready_to_run'].append(
                {
                    'samples': int(one['samples']),
                    'infer_seconds': float(one['infer_seconds']),
                    'e2e_seconds': float(one['infer_seconds']),
                    'top1': float(one['top1']),
                    'top5': float(one['top5']),
                    'loss': float(one['loss']),
                    'peak_gpu_mem_bytes': int(peak_gpu),
                    'peak_decode_load_gpu_mem_bytes': 0,
                    'peak_model_resident_gpu_mem_bytes': int(resident_peak),
                    'total_decode_load_seconds': 0.0,
                    'boot_seconds': 0.0,
                    'tail_seconds': 0.0,
                    'hidden_decode_seconds': 0.0,
                    'visible_decode_seconds_estimated': 0.0,
                    'repeat_index': int(repeat_idx),
                }
            )
        del oracle_model

    if 'serial_full_decode' in runtime_modes:
        print('[streaming_runtime] running mode=serial_full_decode ...', flush=True)
        for repeat_idx in range(int(repeats)):
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            decode_t0 = time.perf_counter()
            serial_model, _serial_manifest = load_gaugedna_archive(
                archive_path=archive_path,
                model_builder=lambda name: build_timm_model(
                    model_name=name,
                    pretrained=False,
                    checkpoint_path='',
                    pretrained_weight_path='',
                ),
                device=device,
                decode_fraction=1.0,
                decode_planes=None,
            )
            _sync_cuda_if_needed(device)
            decode_load_seconds = float(time.perf_counter() - decode_t0)
            peak_decode_load_mem = int(torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' else 0
            one = _evaluate_ready_model_once(
                model=serial_model,
                loader=loader,
                device=device,
                eval_samples=int(eval_samples),
                warmup_batches=int(warmup_batches),
                use_amp=bool(use_amp),
                autocast_dtype=autocast_dtype,
            )
            peak_gpu = int(torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' else 0
            resident_peak = int(
                sum(
                    int(p.numel() * p.element_size())
                    for p in serial_model.parameters()
                    if p.device.type == device.type and (device.type != 'cuda' or int(p.device.index or 0) == int(device.index or 0))
                )
            )
            mode_repeat_rows['serial_full_decode'].append(
                {
                    'samples': int(one['samples']),
                    'infer_seconds': float(one['infer_seconds']),
                    'e2e_seconds': float(decode_load_seconds + float(one['infer_seconds'])),
                    'top1': float(one['top1']),
                    'top5': float(one['top5']),
                    'loss': float(one['loss']),
                    'peak_gpu_mem_bytes': int(peak_gpu),
                    'peak_decode_load_gpu_mem_bytes': int(peak_decode_load_mem),
                    'peak_model_resident_gpu_mem_bytes': int(resident_peak),
                    'total_decode_load_seconds': float(decode_load_seconds),
                    'boot_seconds': 0.0,
                    'tail_seconds': 0.0,
                    'hidden_decode_seconds': 0.0,
                    'visible_decode_seconds_estimated': float(decode_load_seconds),
                    'repeat_index': int(repeat_idx),
                }
            )
            del serial_model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    if 'lazy_family_decode' in runtime_modes:
        print('[streaming_runtime] running mode=lazy_family_decode ...', flush=True)
        for repeat_idx in range(int(repeats)):
            one = _run_lazy_streaming_mode_once(
                mode='lazy_family_decode',
                archive_path=archive_path,
                device=device,
                loader=loader,
                eval_samples=int(eval_samples),
                warmup_batches=int(warmup_batches),
                decode_fraction=float(decode_fraction),
                decode_planes=(None if decode_planes is None else int(decode_planes)),
                prefetch_next_group=False,
                group_policy=str(group_policy),
                super_block_k=int(super_block_k),
                auto_profile_batches=int(auto_profile_batches),
                auto_compute_decode_ratio_target=float(auto_compute_decode_ratio_target),
                cache_decoded_groups=bool(cache_decoded_groups),
                reset_decoded_cache_before_timed_eval=bool(reset_decoded_cache_before_timed_eval),
                use_amp=bool(use_amp),
                autocast_dtype=autocast_dtype,
            )
            one['repeat_index'] = int(repeat_idx)
            mode_repeat_rows['lazy_family_decode'].append(one)
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    if 'lazy_family_decode_overlap' in runtime_modes:
        print('[streaming_runtime] running mode=lazy_family_decode_overlap ...', flush=True)
        prefetch_next_group = bool(getattr(args, 'streaming_runtime_prefetch_next_group', True))
        for repeat_idx in range(int(repeats)):
            one = _run_lazy_streaming_mode_once(
                mode='lazy_family_decode_overlap',
                archive_path=archive_path,
                device=device,
                loader=loader,
                eval_samples=int(eval_samples),
                warmup_batches=int(warmup_batches),
                decode_fraction=float(decode_fraction),
                decode_planes=(None if decode_planes is None else int(decode_planes)),
                prefetch_next_group=bool(prefetch_next_group),
                group_policy=str(group_policy),
                super_block_k=int(super_block_k),
                auto_profile_batches=int(auto_profile_batches),
                auto_compute_decode_ratio_target=float(auto_compute_decode_ratio_target),
                cache_decoded_groups=bool(cache_decoded_groups),
                reset_decoded_cache_before_timed_eval=bool(reset_decoded_cache_before_timed_eval),
                use_amp=bool(use_amp),
                autocast_dtype=autocast_dtype,
            )
            one['repeat_index'] = int(repeat_idx)
            mode_repeat_rows['lazy_family_decode_overlap'].append(one)
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    aggregated: Dict[str, Dict[str, Any]] = {}
    for mode in runtime_modes:
        raw_rows = list(mode_repeat_rows.get(mode, []))
        used_rows = list(raw_rows)
        dropped_repeat_indices: List[int] = []
        if bool(drop_first_repeat_for_aggregation) and len(raw_rows) >= 2:
            first_repeat_idx = int(raw_rows[0].get('repeat_index', 0))
            dropped_repeat_indices.append(first_repeat_idx)
            used_rows = raw_rows[1:]
        aggregated[mode] = _aggregate_repeat_metrics(used_rows)
        aggregated[mode]['aggregation_num_raw_repeats'] = int(len(raw_rows))
        aggregated[mode]['aggregation_num_used_repeats'] = int(len(used_rows))
        aggregated[mode]['aggregation_dropped_repeat_indices'] = [int(x) for x in dropped_repeat_indices]
        aggregated[mode]['aggregation_used_repeat_indices'] = [
            int(item.get('repeat_index', idx)) for idx, item in enumerate(used_rows)
        ]
        samples = int(aggregated[mode].get('samples', 0))
        e2e_seconds = float(aggregated[mode].get('e2e_seconds', 0.0))
        total_decode_load_seconds = float(aggregated[mode].get('total_decode_load_seconds', 0.0))
        visible_decode_seconds_raw = float(
            aggregated[mode].get('visible_decode_seconds_estimated', total_decode_load_seconds)
        )
        if mode == 'serial_full_decode':
            visible_decode_seconds_raw = float(total_decode_load_seconds)
        if mode == 'oracle_ready_to_run':
            visible_decode_seconds_raw = 0.0
        visible_decode_seconds = float(min(max(0.0, visible_decode_seconds_raw), max(0.0, e2e_seconds)))
        hidden_decode_seconds = float(max(0.0, total_decode_load_seconds - visible_decode_seconds))
        infer_only_seconds = float(max(0.0, e2e_seconds - visible_decode_seconds))
        aggregated[mode]['e2e_latency_ms_per_sample'] = float((e2e_seconds * 1000.0) / max(float(samples), 1.0))
        aggregated[mode]['infer_only_latency_ms_per_sample_internal'] = float(
            (infer_only_seconds * 1000.0) / max(float(samples), 1.0)
        )
        aggregated[mode]['visible_decode_overhead_ms_per_sample_internal'] = float(
            (visible_decode_seconds * 1000.0) / max(float(samples), 1.0)
        )
        aggregated[mode]['visible_decode_seconds_internal'] = float(visible_decode_seconds)
        aggregated[mode]['hidden_decode_seconds_internal'] = float(hidden_decode_seconds)

    lazy_visible_overhead_ms = float(
        aggregated.get('lazy_family_decode', {}).get('visible_decode_overhead_ms_per_sample_internal', 0.0)
    )
    serial_visible_overhead_ms = float(
        aggregated.get('serial_full_decode', {}).get('visible_decode_overhead_ms_per_sample_internal', 0.0)
    )
    baseline_visible_overhead_ms = float(
        lazy_visible_overhead_ms if lazy_visible_overhead_ms > 0.0 else serial_visible_overhead_ms
    )
    serial_peak_resident = int(aggregated.get('serial_full_decode', {}).get('peak_model_resident_gpu_mem_bytes', 0))
    eps = 1e-12

    ordered_modes = list(STREAMING_RUNTIME_MODES)
    results_rows: List[Dict[str, Any]] = []
    for mode in ordered_modes:
        if mode not in aggregated:
            continue
        item = aggregated[mode]
        samples = int(item.get('samples', eval_samples))
        e2e_latency_ms = float(item.get('e2e_latency_ms_per_sample', 0.0))
        infer_only_latency_ms = float(item.get('infer_only_latency_ms_per_sample_internal', 0.0))
        visible_overhead_ms = float(item.get('visible_decode_overhead_ms_per_sample_internal', 0.0))
        visible_decode_seconds = float(item.get('visible_decode_seconds_internal', 0.0))
        hidden_decode_seconds = float(item.get('hidden_decode_seconds_internal', 0.0))
        auto_profile_seconds = float(item.get('auto_group_policy_profile_seconds', 0.0))
        auto_profile_overhead_ms = float((auto_profile_seconds * 1000.0) / max(float(samples), 1.0))
        decode_hiding_ratio = 0.0
        if mode == 'lazy_family_decode_overlap':
            decode_hiding_ratio = float(
                1.0 - (visible_overhead_ms / (baseline_visible_overhead_ms + float(eps)))
            )
        peak_model_resident = int(item.get('peak_model_resident_gpu_mem_bytes', 0))
        memory_reduction_ratio = 0.0
        if mode in ('lazy_family_decode', 'lazy_family_decode_overlap'):
            memory_reduction_ratio = float(1.0 - (float(peak_model_resident) / (float(serial_peak_resident) + float(eps))))

        results_rows.append(
            {
                'runtime_mode': mode,
                'e2e_latency_ms_per_sample': float(e2e_latency_ms),
                'infer_only_latency_ms_per_sample': float(infer_only_latency_ms),
                'visible_decode_overhead_ms_per_sample': float(visible_overhead_ms),
                'decode_overhead_hiding_ratio': float(decode_hiding_ratio),
                'boot_seconds': float(item.get('boot_seconds', 0.0)) if mode == 'lazy_family_decode_overlap' else 0.0,
                'tail_seconds': float(item.get('tail_seconds', 0.0)) if mode == 'lazy_family_decode_overlap' else 0.0,
                'hidden_decode_seconds': float(hidden_decode_seconds),
                'visible_decode_seconds': float(visible_decode_seconds),
                'auto_group_policy_profile_seconds': float(auto_profile_seconds),
                'auto_group_policy_profile_overhead_ms_per_sample': float(auto_profile_overhead_ms),
                'group_policy_used': str(item.get('group_policy_used', '')),
                'auto_group_policy_selection_reason': str(item.get('auto_group_policy_selection_reason', '')),
                'peak_gpu_mem_bytes': int(item.get('peak_gpu_mem_bytes', 0)),
                'peak_decode_load_gpu_mem_bytes': int(item.get('peak_decode_load_gpu_mem_bytes', 0)),
                'peak_model_resident_gpu_mem_bytes': int(peak_model_resident),
                'memory_reduction_ratio': float(memory_reduction_ratio),
                'top1': float(item.get('top1', 0.0)),
                'top5': float(item.get('top5', 0.0)),
                'cross_entropy_loss': float(item.get('loss', 0.0)),
                'samples': int(samples),
            }
        )

    runtime_scheduling_table = [
        {
            'runtime_mode': row['runtime_mode'],
            'e2e_latency_ms_per_sample': row['e2e_latency_ms_per_sample'],
            'infer_only_latency_ms_per_sample': row['infer_only_latency_ms_per_sample'],
            'visible_decode_overhead_ms_per_sample': row['visible_decode_overhead_ms_per_sample'],
            'decode_overhead_hiding_ratio': row['decode_overhead_hiding_ratio'],
            'boot_seconds': row['boot_seconds'],
            'tail_seconds': row['tail_seconds'],
            'hidden_decode_seconds': row['hidden_decode_seconds'],
            'visible_decode_seconds': row['visible_decode_seconds'],
            'auto_group_policy_profile_overhead_ms_per_sample': row['auto_group_policy_profile_overhead_ms_per_sample'],
            'group_policy_used': row['group_policy_used'],
            'top1': row['top1'],
        }
        for row in results_rows
    ]
    memory_table = [
        {
            'runtime_mode': row['runtime_mode'],
            'peak_gpu_mem_bytes': row['peak_gpu_mem_bytes'],
            'peak_decode_load_gpu_mem_bytes': row['peak_decode_load_gpu_mem_bytes'],
            'peak_model_resident_gpu_mem_bytes': row['peak_model_resident_gpu_mem_bytes'],
            'memory_reduction_ratio': row['memory_reduction_ratio'],
            'top1': row['top1'],
        }
        for row in results_rows
        if row['runtime_mode'] in ('serial_full_decode', 'lazy_family_decode', 'lazy_family_decode_overlap')
    ]

    output_json = str(
        getattr(args, 'streaming_runtime_output_json', 'outputs/streaming_runtime_familywise_results.json')
    )
    output_csv = str(
        getattr(args, 'streaming_runtime_output_csv', 'outputs/streaming_runtime_familywise_results.csv')
    )
    Path(output_json).expanduser().parent.mkdir(parents=True, exist_ok=True)
    saved_csv = _write_result_csv(output_csv, results_rows)

    payload = {
        'protocol': {
            'mode': 'eval_streaming_runtime_familywise',
            'archive_invariant': bool(getattr(args, 'streaming_runtime_archive_invariant', True)),
            'runtime_unit': 'family/tensor primitive + dynamic group policy',
            'logical_scheduler_unit': str(getattr(args, 'streaming_runtime_logical_scheduler_unit', 'block_attn_mlp')),
            'group_policy': str(group_policy),
            'group_policy_candidates': list(STREAMING_RUNTIME_GROUP_POLICIES),
            'super_block_k': int(super_block_k),
            'auto_profile_batches': int(auto_profile_batches),
            'auto_compute_decode_ratio_target': float(auto_compute_decode_ratio_target),
            'cache_decoded_groups': bool(cache_decoded_groups),
            'reset_decoded_cache_before_timed_eval': bool(reset_decoded_cache_before_timed_eval),
            'drop_first_repeat_for_aggregation': bool(drop_first_repeat_for_aggregation),
            'primitive_unit': str(getattr(args, 'streaming_runtime_primitive_unit', 'tensor_or_family_entry')),
            'model': str(model_name),
            'archive_path': str(archive_path),
            'archive_bytes': int(archive_bytes),
            'payload_mib': float(payload_mib),
            'eval_samples': int(eval_samples),
            'batch_size': int(eval_batch_size),
            'warmup_batches': int(warmup_batches),
            'repeats': int(repeats),
            'decode_fraction': float(decode_fraction),
            'decode_planes': None if decode_planes is None else int(decode_planes),
            'runtime_modes': ordered_modes,
            'metric_definitions': {
                'infer_only_latency_ms_per_sample': 'e2e_latency_ms_per_sample - visible_decode_overhead_ms_per_sample (same mode)',
                'visible_decode_overhead_ms_per_sample': 'amortized visible decode/load time estimated from per-batch decode timeline',
                'decode_overhead_hiding_ratio': '1 - overlap_visible_decode_overhead / baseline_visible_decode_overhead',
                'decode_overhead_hiding_ratio_baseline': (
                    'lazy_family_decode' if lazy_visible_overhead_ms > 0.0 else 'serial_full_decode'
                ),
                'repeat_aggregation_rule': (
                    'drop first repeat when repeats>=2' if bool(drop_first_repeat_for_aggregation) else 'use all repeats'
                ),
            },
            'data_config': data_config,
        },
        'results': results_rows,
        'tables': {
            'runtime_scheduling_comparison': runtime_scheduling_table,
            'memory_comparison': memory_table,
        },
        'mode_details': {
            mode: {
                'aggregate': aggregated.get(mode, {}),
                'runs': mode_repeat_rows.get(mode, []),
            }
            for mode in runtime_modes
        },
        'archive_manifest': manifest_for_report,
        'outputs': {
            'result_json': output_json,
            'result_csv': saved_csv,
        },
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f'Saved streaming runtime json to: {output_json}', flush=True)
    print(f'Saved streaming runtime csv to: {saved_csv}', flush=True)
    return payload
