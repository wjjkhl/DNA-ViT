import copy
import csv
import io
import json
import math
import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from devdna_codec import (
    decode_flat_quantized_stack,
    decode_progressive_bitplane_stack,
    deserialize_exact_tensor,
    encode_flat_quantized_stack,
    encode_progressive_bitplane_stack,
    serialize_exact_tensor,
)
from devdna_core import (
    attention_head_descriptor,
    canonicalize_attention_head_rotation,
    dct2_transform_forward,
    dct2_transform_inverse,
    developmental_transform_forward,
    developmental_transform_inverse,
    format_bytes,
    lexicographic_order,
    mlp_unit_descriptors,
    pca_project,
    sensitivity_weighted_target_counts,
    spectral_norm_power_iteration,
)


ARCHIVE_FORMAT = 'gaugedna_vit_archive_v2'
MANIFEST_NAME = 'gaugedna_manifest.json'
COMPRESSED_FAMILIES = ('attn_qkv', 'attn_proj', 'mlp_fc1', 'mlp_fc2')
FAMILY_PARAM_SUFFIX = {
    'attn_qkv': '.attn.qkv.weight',
    'attn_proj': '.attn.proj.weight',
    'mlp_fc1': '.mlp.fc1.weight',
    'mlp_fc2': '.mlp.fc2.weight',
}


def _build_temp_archive_path(target_path: str) -> str:
    parent = os.path.dirname(target_path)
    base = os.path.basename(target_path)
    return os.path.join(parent, f'.{base}.tmp.{os.getpid()}.{time.time_ns()}')


@dataclass
class FamilyCompressionStats:
    name: str
    dense_bytes: int
    target_nonzero: int
    actual_nonzero: int
    sensitivity: float
    quant_bits: int


@dataclass
class CompressionSummary:
    archive_path: str
    archive_bytes: int
    dense_state_bytes: int
    compressed_weight_dense_bytes: int
    compressed_scope_fp32_bytes: int
    exact_tensor_dense_bytes: int
    family_stats: List[FamilyCompressionStats]
    gauge_report: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'archive_path': self.archive_path,
            'archive_bytes': int(self.archive_bytes),
            'archive_human': format_bytes(self.archive_bytes),
            'dense_state_bytes': int(self.dense_state_bytes),
            'dense_state_human': format_bytes(self.dense_state_bytes),
            'compressed_weight_dense_bytes': int(self.compressed_weight_dense_bytes),
            'compressed_weight_dense_human': format_bytes(self.compressed_weight_dense_bytes),
            'compressed_scope_fp32_bytes': int(self.compressed_scope_fp32_bytes),
            'compressed_scope_fp32_human': format_bytes(self.compressed_scope_fp32_bytes),
            # Alias for downstream scripts expecting "baseline" wording.
            'baseline_fp32_scope_bytes': int(self.compressed_scope_fp32_bytes),
            'baseline_fp32_scope_human': format_bytes(self.compressed_scope_fp32_bytes),
            'exact_tensor_dense_bytes': int(self.exact_tensor_dense_bytes),
            'exact_tensor_dense_human': format_bytes(self.exact_tensor_dense_bytes),
            'family_stats': [
                {
                    'name': stat.name,
                    'dense_bytes': int(stat.dense_bytes),
                    'dense_human': format_bytes(stat.dense_bytes),
                    'target_nonzero': int(stat.target_nonzero),
                    'actual_nonzero': int(stat.actual_nonzero),
                    'sensitivity': float(stat.sensitivity),
                    'quant_bits': int(stat.quant_bits),
                }
                for stat in self.family_stats
            ],
            'gauge_report': self.gauge_report,
        }


@torch.no_grad()
def _is_identity_module(module: nn.Module) -> bool:
    return module.__class__.__name__ == 'Identity'


@torch.no_grad()
def _iter_supported_transformer_blocks(model: nn.Module) -> List[Dict[str, Any]]:
    block_entries: List[Dict[str, Any]] = []

    direct_blocks = getattr(model, 'blocks', None)
    if isinstance(direct_blocks, (list, tuple, nn.ModuleList, nn.Sequential)):
        for block_idx, block in enumerate(direct_blocks):
            block_entries.append(
                {
                    'block_index': int(block_idx),
                    'stage_index': -1,
                    'stage_block_index': int(block_idx),
                    'param_prefix': f'blocks.{block_idx}',
                    'block': block,
                }
            )
        if len(block_entries) > 0:
            return block_entries

    layers = getattr(model, 'layers', None)
    if isinstance(layers, (list, tuple, nn.ModuleList, nn.Sequential)):
        global_index = 0
        for stage_idx, layer in enumerate(layers):
            stage_blocks = getattr(layer, 'blocks', None)
            if not isinstance(stage_blocks, (list, tuple, nn.ModuleList, nn.Sequential)):
                continue
            for stage_block_idx, block in enumerate(stage_blocks):
                block_entries.append(
                    {
                        'block_index': int(global_index),
                        'stage_index': int(stage_idx),
                        'stage_block_index': int(stage_block_idx),
                        'param_prefix': f'layers.{stage_idx}.blocks.{stage_block_idx}',
                        'block': block,
                    }
                )
                global_index += 1
    if len(block_entries) <= 0:
        raise TypeError(
            'The supplied model does not look like a supported timm transformer backbone. '
            'Expected either `.blocks` (ViT/DeiT-like) or `.layers[*].blocks[*]` (Swin-like).'
        )
    return block_entries


@torch.no_grad()
def _validate_supported_transformer_model(model: nn.Module, enable_attn_rotation_gauge: bool) -> None:
    for entry in _iter_supported_transformer_blocks(model):
        block_idx = int(entry['block_index'])
        block_path = str(entry['param_prefix'])
        block = entry['block']
        if not hasattr(block, 'attn') or not hasattr(block, 'mlp'):
            raise TypeError(f'Block {block_idx} ({block_path}) is missing .attn or .mlp and is not supported.')
        attn = block.attn
        mlp = block.mlp
        if not hasattr(attn, 'qkv') or not hasattr(attn, 'proj'):
            raise TypeError(f'Block {block_idx} ({block_path}) attention is missing qkv/proj linear layers.')
        if not hasattr(mlp, 'fc1') or not hasattr(mlp, 'fc2'):
            raise TypeError(f'Block {block_idx} ({block_path}) mlp is missing fc1/fc2 linear layers.')
        if enable_attn_rotation_gauge:
            if hasattr(attn, 'q_norm') and not _is_identity_module(attn.q_norm):
                raise NotImplementedError(
                    f'Block {block_idx} ({block_path}) uses non-identity q_norm. Exact attention rotation gauge requires q_norm to be Identity. '
                    'Disable attention rotation gauge with --disable-attn-rotation-gauge if needed.'
                )
            if hasattr(attn, 'k_norm') and not _is_identity_module(attn.k_norm):
                raise NotImplementedError(
                    f'Block {block_idx} ({block_path}) uses non-identity k_norm. Exact attention rotation gauge requires k_norm to be Identity. '
                    'Disable attention rotation gauge with --disable-attn-rotation-gauge if needed.'
                )
            if hasattr(attn, 'norm') and not _is_identity_module(attn.norm):
                raise NotImplementedError(
                    f'Block {block_idx} ({block_path}) uses non-identity attention.norm. '
                    'Exact attention rotation gauge targets standard timm attention blocks with Identity norm.'
                )
        if hasattr(mlp, 'norm') and not _is_identity_module(mlp.norm):
            raise NotImplementedError(
                f'Block {block_idx} ({block_path}) uses non-identity mlp.norm. '
                'This implementation targets standard timm transformer MLP blocks with Identity norm.'
            )


@torch.no_grad()
def _validate_supported_vit(model: nn.Module, enable_attn_rotation_gauge: bool) -> None:
    # Backward-compatible alias used by existing call sites.
    _validate_supported_transformer_model(model, enable_attn_rotation_gauge=enable_attn_rotation_gauge)


@torch.no_grad()
def _copy_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


@torch.no_grad()
def _iter_head_linear_weight_names(model: nn.Module) -> List[str]:
    out: List[str] = []
    for attr_name in ('head', 'head_dist'):
        module = getattr(model, attr_name, None)
        if isinstance(module, nn.Linear):
            out.append(f'{attr_name}.weight')
            continue
        # Swin/timm ClassifierHead keeps the actual linear layer under `.fc`.
        if hasattr(module, 'fc'):
            fc_module = getattr(module, 'fc', None)
            if isinstance(fc_module, nn.Linear):
                out.append(f'{attr_name}.fc.weight')
    return out


@torch.no_grad()
def _compressed_weight_names(model: nn.Module, include_head_linear: bool = False) -> List[str]:
    names: List[str] = []
    for entry in _iter_supported_transformer_blocks(model):
        prefix = str(entry['param_prefix'])
        names.extend(
            [
                f'{prefix}.attn.qkv.weight',
                f'{prefix}.attn.proj.weight',
                f'{prefix}.mlp.fc1.weight',
                f'{prefix}.mlp.fc2.weight',
            ]
        )
    if include_head_linear:
        names.extend(_iter_head_linear_weight_names(model))
    return names


@torch.no_grad()
def _verify_mlp_gauge(block: nn.Module, fc1: torch.Tensor, fc2: torch.Tensor, fc1_bias: Optional[torch.Tensor], fc2_bias: Optional[torch.Tensor], device: torch.device) -> float:
    test_module = copy.deepcopy(block.mlp).to(device).eval()
    ref_module = block.mlp.to(device).eval()
    dtype = ref_module.fc1.weight.dtype
    test_module.fc1.weight.copy_(fc1.to(device=device, dtype=dtype))
    test_module.fc2.weight.copy_(fc2.to(device=device, dtype=dtype))
    if test_module.fc1.bias is not None and fc1_bias is not None:
        test_module.fc1.bias.copy_(fc1_bias.to(device=device, dtype=dtype))
    if test_module.fc2.bias is not None and fc2_bias is not None:
        test_module.fc2.bias.copy_(fc2_bias.to(device=device, dtype=dtype))
    x = torch.randn((2, 197, ref_module.fc1.weight.shape[1]), device=device, dtype=dtype)
    y_ref = ref_module(x)
    y_test = test_module(x)
    return float((y_ref - y_test).abs().max().item())


@torch.no_grad()
def _verify_attn_gauge(
    block: nn.Module,
    qkv_weight: torch.Tensor,
    qkv_bias: Optional[torch.Tensor],
    proj_weight: torch.Tensor,
    proj_bias: Optional[torch.Tensor],
    relative_position_bias_table: Optional[torch.Tensor],
    logit_scale: Optional[torch.Tensor],
    device: torch.device,
) -> float:
    num_tokens = 197
    window_area = getattr(block.attn, 'window_area', None)
    if isinstance(window_area, (int, np.integer)) and int(window_area) > 0:
        num_tokens = int(window_area)
    else:
        window_size = getattr(block.attn, 'window_size', None)
        if isinstance(window_size, (tuple, list)):
            if len(window_size) >= 2 and int(window_size[0]) > 0 and int(window_size[1]) > 0:
                num_tokens = int(window_size[0]) * int(window_size[1])
            elif len(window_size) == 1 and int(window_size[0]) > 0:
                num_tokens = int(window_size[0]) * int(window_size[0])
        elif isinstance(window_size, (int, np.integer)) and int(window_size) > 0:
            num_tokens = int(window_size) * int(window_size)

    test_module = copy.deepcopy(block.attn).to(device).eval()
    ref_module = block.attn.to(device).eval()
    dtype = ref_module.qkv.weight.dtype
    test_module.qkv.weight.copy_(qkv_weight.to(device=device, dtype=dtype))
    test_module.proj.weight.copy_(proj_weight.to(device=device, dtype=dtype))
    if test_module.qkv.bias is not None and qkv_bias is not None:
        test_module.qkv.bias.copy_(qkv_bias.to(device=device, dtype=dtype))
    if test_module.proj.bias is not None and proj_bias is not None:
        test_module.proj.bias.copy_(proj_bias.to(device=device, dtype=dtype))
    if hasattr(test_module, 'relative_position_bias_table'):
        if relative_position_bias_table is not None:
            target = test_module.relative_position_bias_table
            target_dtype = target.dtype
            target.copy_(relative_position_bias_table.to(device=device, dtype=target_dtype))
    if hasattr(test_module, 'logit_scale'):
        if logit_scale is not None:
            target = test_module.logit_scale
            target_dtype = target.dtype
            target.copy_(logit_scale.to(device=device, dtype=target_dtype))
    x = torch.randn((2, int(num_tokens), ref_module.proj.weight.shape[0]), device=device, dtype=dtype)
    y_ref = ref_module(x)
    y_test = test_module(x)
    return float((y_ref - y_test).abs().max().item())


@torch.no_grad()
def _canonicalize_attention_blocks(
    model: nn.Module,
    canonical_state: Dict[str, torch.Tensor],
    device: torch.device,
    gauge_proj_dim: int,
    gauge_pca_dim: int,
    enable_attn_rotation_gauge: bool,
    verify_gauge: bool,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str], List[str], Dict[str, Any]]:
    head_records: List[Dict[str, Any]] = []
    all_descriptors: List[torch.Tensor] = []

    for entry in _iter_supported_transformer_blocks(model):
        block = entry['block']
        block_idx = int(entry['block_index'])
        param_prefix = str(entry['param_prefix'])
        attn = block.attn
        qkv_weight = attn.qkv.weight.detach().to(device=device, dtype=torch.float32)
        qkv_bias = None if attn.qkv.bias is None else attn.qkv.bias.detach().to(device=device, dtype=torch.float32)
        proj_weight = attn.proj.weight.detach().to(device=device, dtype=torch.float32)
        proj_bias = None if attn.proj.bias is None else attn.proj.bias.detach().to(device=device, dtype=torch.float32)
        rel_pos_bias = None
        if hasattr(attn, 'relative_position_bias_table') and torch.is_tensor(attn.relative_position_bias_table):
            rel_pos_bias = attn.relative_position_bias_table.detach().to(device=device)
        logit_scale = None
        if hasattr(attn, 'logit_scale') and torch.is_tensor(attn.logit_scale):
            logit_scale = attn.logit_scale.detach().to(device=device)

        embed_dim = proj_weight.shape[0]
        num_heads = int(attn.num_heads)
        head_dim = embed_dim // num_heads
        qkv = qkv_weight.view(3, num_heads, head_dim, embed_dim)
        proj_heads = proj_weight.view(embed_dim, num_heads, head_dim).permute(1, 0, 2)

        if qkv_bias is not None:
            qkv_bias_heads = qkv_bias.view(3, num_heads, head_dim)
        else:
            qkv_bias_heads = None

        q_heads: List[torch.Tensor] = []
        k_heads: List[torch.Tensor] = []
        v_heads: List[torch.Tensor] = []
        o_heads: List[torch.Tensor] = []
        qb_heads: List[Optional[torch.Tensor]] = []
        kb_heads: List[Optional[torch.Tensor]] = []
        vb_heads: List[Optional[torch.Tensor]] = []
        desc_heads: List[torch.Tensor] = []

        for head_idx in range(num_heads):
            q_head = qkv[0, head_idx]
            k_head = qkv[1, head_idx]
            v_head = qkv[2, head_idx]
            o_head = proj_heads[head_idx]
            q_bias_head = None if qkv_bias_heads is None else qkv_bias_heads[0, head_idx]
            k_bias_head = None if qkv_bias_heads is None else qkv_bias_heads[1, head_idx]
            v_bias_head = None if qkv_bias_heads is None else qkv_bias_heads[2, head_idx]

            if enable_attn_rotation_gauge:
                q_rot, k_rot, v_rot, o_rot, qb_rot, kb_rot, vb_rot, sqk, svo = canonicalize_attention_head_rotation(
                    q_head=q_head,
                    k_head=k_head,
                    v_head=v_head,
                    o_head=o_head,
                    q_bias=q_bias_head,
                    k_bias=k_bias_head,
                    v_bias=v_bias_head,
                )
            else:
                q_rot = q_head.float()
                k_rot = k_head.float()
                v_rot = v_head.float()
                o_rot = o_head.float()
                qb_rot = None if q_bias_head is None else q_bias_head.float()
                kb_rot = None if k_bias_head is None else k_bias_head.float()
                vb_rot = None if v_bias_head is None else v_bias_head.float()
                sqk = torch.linalg.svdvals(torch.cat([q_rot, k_rot], dim=1))
                svo = torch.linalg.svdvals(torch.cat([v_rot, o_rot.transpose(0, 1)], dim=1))

            descriptor = attention_head_descriptor(
                q_head=q_rot,
                k_head=k_rot,
                v_head=v_rot,
                o_head=o_rot,
                sqk=sqk,
                svo=svo,
                q_bias=qb_rot,
                k_bias=kb_rot,
                v_bias=vb_rot,
                proj_dim=gauge_proj_dim,
            )
            q_heads.append(q_rot)
            k_heads.append(k_rot)
            v_heads.append(v_rot)
            o_heads.append(o_rot)
            qb_heads.append(qb_rot)
            kb_heads.append(kb_rot)
            vb_heads.append(vb_rot)
            desc_heads.append(descriptor)
            all_descriptors.append(descriptor.unsqueeze(0))

        head_records.append(
            {
                'block_idx': block_idx,
                'q_heads': q_heads,
                'k_heads': k_heads,
                'v_heads': v_heads,
                'o_heads': o_heads,
                'qb_heads': qb_heads,
                'kb_heads': kb_heads,
                'vb_heads': vb_heads,
                'desc': torch.stack(desc_heads, dim=0),
                'proj_bias': proj_bias,
                'embed_dim': embed_dim,
                'head_dim': head_dim,
                'num_heads': num_heads,
                'block_module': block,
                'param_prefix': param_prefix,
                'rel_pos_bias': rel_pos_bias,
                'logit_scale': logit_scale,
            }
        )

    all_desc = torch.cat(all_descriptors, dim=0)
    all_coords, mean, basis = pca_project(all_desc, out_dim=gauge_pca_dim)

    qkv_stack: List[torch.Tensor] = []
    proj_stack: List[torch.Tensor] = []
    qkv_names: List[str] = []
    proj_names: List[str] = []
    verify_diffs: List[float] = []
    cursor = 0

    for record in head_records:
        num_heads = int(record['num_heads'])
        coords = all_coords[cursor : cursor + num_heads]
        cursor += num_heads
        order_keys = torch.cat([coords, record['desc'][:, : min(8, record['desc'].shape[1])]], dim=1)
        order = lexicographic_order(order_keys, descending=True)
        order_list = [int(x) for x in order.detach().cpu().tolist()]

        q_sorted = torch.stack([record['q_heads'][i] for i in order_list], dim=0)
        k_sorted = torch.stack([record['k_heads'][i] for i in order_list], dim=0)
        v_sorted = torch.stack([record['v_heads'][i] for i in order_list], dim=0)
        o_sorted = torch.stack([record['o_heads'][i] for i in order_list], dim=0)
        qkv_weight = torch.stack([q_sorted, k_sorted, v_sorted], dim=0).reshape(3 * record['embed_dim'], record['embed_dim'])
        proj_weight = o_sorted.permute(1, 0, 2).reshape(record['embed_dim'], record['embed_dim'])

        if record['qb_heads'][0] is not None:
            q_bias = torch.stack([record['qb_heads'][i] for i in order_list], dim=0)
            k_bias = torch.stack([record['kb_heads'][i] for i in order_list], dim=0)
            v_bias = torch.stack([record['vb_heads'][i] for i in order_list], dim=0)
            qkv_bias = torch.stack([q_bias, k_bias, v_bias], dim=0).reshape(3 * record['embed_dim'])
        else:
            qkv_bias = None

        block_idx = int(record['block_idx'])
        param_prefix = str(record['param_prefix'])
        rel_pos_bias_canon = None
        if record.get('rel_pos_bias', None) is not None:
            rel_pos_bias = record['rel_pos_bias']
            if rel_pos_bias.ndim == 2 and int(rel_pos_bias.shape[1]) == int(record['num_heads']):
                rel_pos_bias_canon = rel_pos_bias.index_select(1, order)
            else:
                rel_pos_bias_canon = rel_pos_bias
        logit_scale_canon = None
        if record.get('logit_scale', None) is not None:
            raw_logit_scale = record['logit_scale']
            if raw_logit_scale.ndim >= 1 and int(raw_logit_scale.shape[0]) == int(record['num_heads']):
                logit_scale_canon = raw_logit_scale.index_select(0, order)
            else:
                logit_scale_canon = raw_logit_scale

        qkv_name = f'{param_prefix}.attn.qkv.weight'
        proj_name = f'{param_prefix}.attn.proj.weight'
        canonical_state[qkv_name] = qkv_weight.detach().cpu()
        canonical_state[proj_name] = proj_weight.detach().cpu()
        qkv_bias_name = f'{param_prefix}.attn.qkv.bias'
        if qkv_bias is not None and qkv_bias_name in canonical_state:
            canonical_state[qkv_bias_name] = qkv_bias.detach().cpu()
        rel_pos_bias_name = f'{param_prefix}.attn.relative_position_bias_table'
        if rel_pos_bias_canon is not None and rel_pos_bias_name in canonical_state:
            canonical_state[rel_pos_bias_name] = rel_pos_bias_canon.detach().cpu()
        logit_scale_name = f'{param_prefix}.attn.logit_scale'
        if logit_scale_canon is not None and logit_scale_name in canonical_state:
            canonical_state[logit_scale_name] = logit_scale_canon.detach().cpu()
        qkv_stack.append(qkv_weight.detach().cpu())
        proj_stack.append(proj_weight.detach().cpu())
        qkv_names.append(qkv_name)
        proj_names.append(proj_name)

        if verify_gauge:
            diff = _verify_attn_gauge(
                block=record['block_module'],
                qkv_weight=qkv_weight,
                qkv_bias=qkv_bias,
                proj_weight=proj_weight,
                proj_bias=record['proj_bias'],
                relative_position_bias_table=rel_pos_bias_canon,
                logit_scale=logit_scale_canon,
                device=device,
            )
            verify_diffs.append(diff)

    gauge_meta: Dict[str, Any] = {
        'attention_descriptor_mean_shape': list(mean.shape),
        'attention_descriptor_basis_shape': list(basis.shape),
        'attention_verify_max_abs_diffs': verify_diffs,
        'attention_verify_max_abs_diff': float(max(verify_diffs) if verify_diffs else 0.0),
        'attention_rotation_gauge_enabled': bool(enable_attn_rotation_gauge),
    }
    return qkv_stack, proj_stack, qkv_names, proj_names, gauge_meta


@torch.no_grad()
def _canonicalize_mlp_blocks(
    model: nn.Module,
    canonical_state: Dict[str, torch.Tensor],
    device: torch.device,
    gauge_proj_dim: int,
    gauge_pca_dim: int,
    verify_gauge: bool,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str], List[str], Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    all_descriptors: List[torch.Tensor] = []

    for entry in _iter_supported_transformer_blocks(model):
        block = entry['block']
        block_idx = int(entry['block_index'])
        param_prefix = str(entry['param_prefix'])
        mlp = block.mlp
        fc1_weight = mlp.fc1.weight.detach().to(device=device, dtype=torch.float32)
        fc2_weight = mlp.fc2.weight.detach().to(device=device, dtype=torch.float32)
        fc1_bias = None if mlp.fc1.bias is None else mlp.fc1.bias.detach().to(device=device, dtype=torch.float32)
        fc2_bias = None if mlp.fc2.bias is None else mlp.fc2.bias.detach().to(device=device, dtype=torch.float32)
        descriptor = mlp_unit_descriptors(fc1_weight=fc1_weight, fc2_weight=fc2_weight, fc1_bias=fc1_bias, proj_dim=gauge_proj_dim)
        records.append(
            {
                'block_idx': block_idx,
                'fc1_weight': fc1_weight,
                'fc2_weight': fc2_weight,
                'fc1_bias': fc1_bias,
                'fc2_bias': fc2_bias,
                'desc': descriptor,
                'block_module': block,
                'param_prefix': param_prefix,
            }
        )
        all_descriptors.append(descriptor)

    all_desc = torch.cat(all_descriptors, dim=0)
    all_coords, mean, basis = pca_project(all_desc, out_dim=gauge_pca_dim)

    fc1_stack: List[torch.Tensor] = []
    fc2_stack: List[torch.Tensor] = []
    fc1_names: List[str] = []
    fc2_names: List[str] = []
    verify_diffs: List[float] = []
    cursor = 0

    for record in records:
        hidden = record['desc'].shape[0]
        coords = all_coords[cursor : cursor + hidden]
        cursor += hidden
        keys = torch.cat([coords, record['desc'][:, : min(8, record['desc'].shape[1])]], dim=1)
        order = lexicographic_order(keys, descending=True)
        fc1_canon = record['fc1_weight'].index_select(0, order)
        fc2_canon = record['fc2_weight'].index_select(1, order)
        fc1_bias_canon = None if record['fc1_bias'] is None else record['fc1_bias'].index_select(0, order)
        block_idx = int(record['block_idx'])
        param_prefix = str(record['param_prefix'])
        fc1_name = f'{param_prefix}.mlp.fc1.weight'
        fc2_name = f'{param_prefix}.mlp.fc2.weight'
        canonical_state[fc1_name] = fc1_canon.detach().cpu()
        canonical_state[fc2_name] = fc2_canon.detach().cpu()
        fc1_bias_name = f'{param_prefix}.mlp.fc1.bias'
        if fc1_bias_canon is not None and fc1_bias_name in canonical_state:
            canonical_state[fc1_bias_name] = fc1_bias_canon.detach().cpu()
        fc1_stack.append(fc1_canon.detach().cpu())
        fc2_stack.append(fc2_canon.detach().cpu())
        fc1_names.append(fc1_name)
        fc2_names.append(fc2_name)

        if verify_gauge:
            diff = _verify_mlp_gauge(
                block=record['block_module'],
                fc1=fc1_canon,
                fc2=fc2_canon,
                fc1_bias=fc1_bias_canon,
                fc2_bias=record['fc2_bias'],
                device=device,
            )
            verify_diffs.append(diff)

    gauge_meta: Dict[str, Any] = {
        'mlp_descriptor_mean_shape': list(mean.shape),
        'mlp_descriptor_basis_shape': list(basis.shape),
        'mlp_verify_max_abs_diffs': verify_diffs,
        'mlp_verify_max_abs_diff': float(max(verify_diffs) if verify_diffs else 0.0),
    }
    return fc1_stack, fc2_stack, fc1_names, fc2_names, gauge_meta


@torch.no_grad()
def canonicalize_vit_state_dict(
    model: nn.Module,
    device: torch.device,
    gauge_proj_dim: int,
    gauge_pca_dim: int,
    enable_attn_rotation_gauge: bool,
    verify_gauge: bool,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[str]], Dict[str, Any]]:
    _validate_supported_vit(model, enable_attn_rotation_gauge=enable_attn_rotation_gauge)
    canonical_state = _copy_state_dict(model)

    qkv_stack, proj_stack, qkv_names, proj_names, attn_meta = _canonicalize_attention_blocks(
        model=model,
        canonical_state=canonical_state,
        device=device,
        gauge_proj_dim=gauge_proj_dim,
        gauge_pca_dim=gauge_pca_dim,
        enable_attn_rotation_gauge=enable_attn_rotation_gauge,
        verify_gauge=verify_gauge,
    )
    fc1_stack, fc2_stack, fc1_names, fc2_names, mlp_meta = _canonicalize_mlp_blocks(
        model=model,
        canonical_state=canonical_state,
        device=device,
        gauge_proj_dim=gauge_proj_dim,
        gauge_pca_dim=gauge_pca_dim,
        verify_gauge=verify_gauge,
    )

    stacks: Dict[str, List[torch.Tensor]] = {
        'attn_qkv': qkv_stack,
        'attn_proj': proj_stack,
        'mlp_fc1': fc1_stack,
        'mlp_fc2': fc2_stack,
    }
    stack_names = {
        'attn_qkv': qkv_names,
        'attn_proj': proj_names,
        'mlp_fc1': fc1_names,
        'mlp_fc2': fc2_names,
    }
    gauge_report = {'attention': attn_meta, 'mlp': mlp_meta}
    return canonical_state, stacks, stack_names, gauge_report


@torch.no_grad()
def _build_identity_vit_stacks(
    model: nn.Module,
    state: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, List[str]]]:
    bundles = _iter_vit_block_bundle_names(model)
    stacks: Dict[str, List[torch.Tensor]] = {
        'attn_qkv': [],
        'attn_proj': [],
        'mlp_fc1': [],
        'mlp_fc2': [],
    }
    stack_names: Dict[str, List[str]] = {
        'attn_qkv': [],
        'attn_proj': [],
        'mlp_fc1': [],
        'mlp_fc2': [],
    }
    for bundle in bundles:
        for family in COMPRESSED_FAMILIES:
            param_name = str(bundle[family])
            if param_name not in state:
                raise KeyError(f'Missing required parameter for compression family `{family}`: {param_name}')
            stacks[family].append(state[param_name].detach().cpu())
            stack_names[family].append(param_name)
    return stacks, stack_names


@torch.no_grad()
def _global_proportional_target_counts(numel_list: Sequence[int], total_target: int) -> List[int]:
    if len(numel_list) <= 0:
        return []
    counts = [max(0, int(x)) for x in numel_list]
    total_numel = int(sum(counts))
    if total_numel <= 0 or total_target <= 0:
        return [0 for _ in counts]
    total_target = max(0, min(int(total_target), total_numel))
    if total_target == total_numel:
        return list(counts)

    raw = [float(n) * float(total_target) / float(total_numel) for n in counts]
    base = [int(math.floor(x)) for x in raw]
    base = [min(max(0, x), counts[idx]) for idx, x in enumerate(base)]
    remaining = int(total_target - sum(base))
    if remaining > 0:
        order = sorted(
            range(len(counts)),
            key=lambda idx: (raw[idx] - float(base[idx]), counts[idx]),
            reverse=True,
        )
        for idx in order:
            if remaining <= 0:
                break
            capacity = int(counts[idx] - base[idx])
            if capacity <= 0:
                continue
            delta = min(capacity, remaining)
            base[idx] += int(delta)
            remaining -= int(delta)
    elif remaining < 0:
        overflow = -remaining
        order = sorted(
            range(len(counts)),
            key=lambda idx: (raw[idx] - float(base[idx]), counts[idx]),
        )
        for idx in order:
            if overflow <= 0:
                break
            removable = min(int(base[idx]), overflow)
            if removable <= 0:
                continue
            base[idx] -= int(removable)
            overflow -= int(removable)
    return [int(x) for x in base]


@torch.no_grad()
def _normalize_target_counts_from_raw(
    raw_targets: Sequence[float],
    numel_list: Sequence[int],
    total_target: int,
) -> List[int]:
    if len(raw_targets) != len(numel_list):
        raise ValueError('raw_targets and numel_list must have the same length.')
    if len(numel_list) <= 0:
        return []
    capacities = [max(0, int(x)) for x in numel_list]
    total_capacity = int(sum(capacities))
    total_target = max(0, min(int(total_target), total_capacity))
    if total_target <= 0:
        return [0 for _ in capacities]
    if total_target >= total_capacity:
        return [int(x) for x in capacities]

    clamped = [
        float(min(max(float(raw_targets[idx]), 0.0), float(capacities[idx])))
        for idx in range(len(capacities))
    ]
    base = [int(math.floor(value)) for value in clamped]
    base = [min(max(0, int(base[idx])), int(capacities[idx])) for idx in range(len(base))]
    diff = int(total_target - sum(base))
    if diff > 0:
        order = sorted(
            range(len(base)),
            key=lambda idx: (clamped[idx] - float(base[idx]), capacities[idx] - base[idx], idx),
            reverse=True,
        )
        for idx in order:
            if diff <= 0:
                break
            room = int(capacities[idx] - base[idx])
            if room <= 0:
                continue
            delta = min(room, diff)
            base[idx] += int(delta)
            diff -= int(delta)
        if diff > 0:
            for idx in sorted(range(len(base)), key=lambda i: (capacities[i] - base[i], i), reverse=True):
                if diff <= 0:
                    break
                room = int(capacities[idx] - base[idx])
                if room <= 0:
                    continue
                delta = min(room, diff)
                base[idx] += int(delta)
                diff -= int(delta)
    elif diff < 0:
        overflow = -diff
        order = sorted(
            range(len(base)),
            key=lambda idx: (clamped[idx] - float(base[idx]), base[idx], idx),
        )
        for idx in order:
            if overflow <= 0:
                break
            removable = min(int(base[idx]), int(overflow))
            if removable <= 0:
                continue
            base[idx] -= int(removable)
            overflow -= int(removable)
        if overflow > 0:
            for idx in sorted(range(len(base)), key=lambda i: (base[i], i)):
                if overflow <= 0:
                    break
                removable = min(int(base[idx]), int(overflow))
                if removable <= 0:
                    continue
                base[idx] -= int(removable)
                overflow -= int(removable)
    return [int(x) for x in base]


@torch.no_grad()
def _apply_min_keep_ratio_floor(
    targets: Sequence[int],
    numel_list: Sequence[int],
    total_target: int,
    min_keep_ratio: float,
) -> Tuple[List[int], Dict[str, Any]]:
    if len(targets) != len(numel_list):
        raise ValueError('targets and numel_list must have the same length.')
    ratio = float(min_keep_ratio)
    capacities = [max(0, int(x)) for x in numel_list]
    total_capacity = int(sum(capacities))
    total_target = max(0, min(int(total_target), total_capacity))
    if ratio <= 0.0 or len(capacities) <= 0:
        normalized = _normalize_target_counts_from_raw(
            raw_targets=[float(int(x)) for x in targets],
            numel_list=capacities,
            total_target=total_target,
        )
        return normalized, {
            'enabled': False,
            'requested_min_keep_ratio': float(ratio),
            'effective_min_keep_ratio': 0.0,
            'requested_floor_sum': 0,
            'effective_floor_sum': 0,
            'floor_tensor_count': 0,
            'floor_infeasible': False,
        }

    ratio = float(max(0.0, min(1.0, ratio)))
    floor_targets = [min(capacities[idx], int(math.floor(ratio * float(capacities[idx])))) for idx in range(len(capacities))]
    requested_floor_sum = int(sum(floor_targets))
    floor_infeasible = bool(requested_floor_sum > total_target)
    if floor_infeasible and total_target > 0:
        floor_targets = _global_proportional_target_counts(numel_list=floor_targets, total_target=total_target)
    elif total_target <= 0:
        floor_targets = [0 for _ in capacities]
    effective_floor_sum = int(sum(floor_targets))

    adjusted = [
        min(capacities[idx], max(int(targets[idx]), int(floor_targets[idx])))
        for idx in range(len(capacities))
    ]
    current_sum = int(sum(adjusted))
    if current_sum > total_target:
        overflow = int(current_sum - total_target)
        order = sorted(
            range(len(adjusted)),
            key=lambda idx: (adjusted[idx] - floor_targets[idx], adjusted[idx], idx),
            reverse=True,
        )
        for idx in order:
            if overflow <= 0:
                break
            removable = int(adjusted[idx] - floor_targets[idx])
            if removable <= 0:
                continue
            delta = min(removable, overflow)
            adjusted[idx] -= int(delta)
            overflow -= int(delta)
    elif current_sum < total_target:
        deficit = int(total_target - current_sum)
        order = sorted(
            range(len(adjusted)),
            key=lambda idx: (capacities[idx] - adjusted[idx], capacities[idx], idx),
            reverse=True,
        )
        for idx in order:
            if deficit <= 0:
                break
            room = int(capacities[idx] - adjusted[idx])
            if room <= 0:
                continue
            delta = min(room, deficit)
            adjusted[idx] += int(delta)
            deficit -= int(delta)

    normalized = _normalize_target_counts_from_raw(
        raw_targets=[float(x) for x in adjusted],
        numel_list=capacities,
        total_target=total_target,
    )
    floor_tensor_count = int(sum(1 for idx in range(len(normalized)) if int(normalized[idx]) <= int(floor_targets[idx])))
    return normalized, {
        'enabled': True,
        'requested_min_keep_ratio': float(min_keep_ratio),
        'effective_min_keep_ratio': float(ratio),
        'requested_floor_sum': int(requested_floor_sum),
        'effective_floor_sum': int(effective_floor_sum),
        'floor_tensor_count': int(floor_tensor_count),
        'floor_infeasible': bool(floor_infeasible),
    }


@torch.no_grad()
def _resolve_fixed_budget_allocation_mode(
    allocation_mode: str,
    use_sensitivity_allocation: bool,
) -> str:
    mode_raw = str(allocation_mode or '').strip().lower()
    alias_map = {
        '': 'sensitivity',
        'legacy': 'sensitivity',
        'legacy_sensitivity': 'sensitivity',
        'sensitivity_weighted': 'sensitivity',
        'equal_keep_ratio': 'uniform',
        'proportional': 'uniform',
        'size_proportional': 'uniform',
    }
    mode = alias_map.get(mode_raw, mode_raw)
    valid_modes = {
        'sensitivity',
        'uniform',
        'hybrid_uniform_sensitivity',
        'sensitivity_with_floor',
    }
    if mode not in valid_modes:
        raise ValueError(
            f'Unsupported `allocation_mode`={allocation_mode!r} for fixed_global_keep_ratio; '
            f'expected one of {sorted(valid_modes)}.'
        )
    if not bool(use_sensitivity_allocation):
        return 'uniform'
    return mode


@torch.no_grad()
def _allocate_fixed_budget_targets(
    coeff_abs_list: Sequence[torch.Tensor],
    sensitivity_list: Sequence[float],
    total_target: int,
    allocation_mode: str,
    allocation_gamma: float,
    allocation_sens_beta: float,
    allocation_min_tensor_keep_ratio: float,
) -> Tuple[List[int], Dict[str, Any]]:
    if len(coeff_abs_list) != len(sensitivity_list):
        raise ValueError('coeff_abs_list and sensitivity_list must have the same length.')
    numel_list = [int(coeff.numel()) for coeff in coeff_abs_list]
    total_numel = int(sum(numel_list))
    total_target = max(0, min(int(total_target), int(total_numel)))
    mode = str(allocation_mode)
    if total_numel <= 0:
        return [0 for _ in coeff_abs_list], {
            'mode': mode,
            'gamma': float(allocation_gamma),
            'sens_beta': float(allocation_sens_beta),
            'min_tensor_keep_ratio': float(allocation_min_tensor_keep_ratio),
            'total_target': int(total_target),
            'total_numel': int(total_numel),
            'floor': {'enabled': False},
        }
    if not math.isfinite(float(allocation_sens_beta)) or float(allocation_sens_beta) < 0.0:
        raise ValueError(
            f'`allocation_sens_beta` must be a non-negative finite float, got {allocation_sens_beta!r}.'
        )
    if not math.isfinite(float(allocation_gamma)) or not (0.0 <= float(allocation_gamma) <= 1.0):
        raise ValueError(f'`allocation_gamma` must be in [0, 1], got {allocation_gamma!r}.')
    if not math.isfinite(float(allocation_min_tensor_keep_ratio)) or not (
        0.0 <= float(allocation_min_tensor_keep_ratio) <= 1.0
    ):
        raise ValueError(
            '`allocation_min_tensor_keep_ratio` must be in [0, 1], '
            f'got {allocation_min_tensor_keep_ratio!r}.'
        )

    uniform_targets = _global_proportional_target_counts(
        numel_list=numel_list,
        total_target=int(total_target),
    )
    sens_beta = float(allocation_sens_beta)
    sensitivity_adjusted = [float(max(float(s), 1e-12) ** sens_beta) for s in sensitivity_list]
    sensitivity_targets = sensitivity_weighted_target_counts(
        coeff_abs_list=coeff_abs_list,
        sensitivity_list=sensitivity_adjusted,
        total_target=int(total_target),
    )

    if mode == 'uniform':
        targets = [int(x) for x in uniform_targets]
    elif mode == 'sensitivity':
        targets = [int(x) for x in sensitivity_targets]
    elif mode == 'hybrid_uniform_sensitivity':
        gamma = float(allocation_gamma)
        raw_targets = [
            (1.0 - gamma) * float(uniform_targets[idx]) + gamma * float(sensitivity_targets[idx])
            for idx in range(len(numel_list))
        ]
        targets = _normalize_target_counts_from_raw(
            raw_targets=raw_targets,
            numel_list=numel_list,
            total_target=int(total_target),
        )
    elif mode == 'sensitivity_with_floor':
        targets = [int(x) for x in sensitivity_targets]
    else:
        raise ValueError(f'Unsupported allocation mode: {mode}')

    floor_info: Dict[str, Any] = {'enabled': False}
    if float(allocation_min_tensor_keep_ratio) > 0.0 and mode in {'hybrid_uniform_sensitivity', 'sensitivity_with_floor'}:
        targets, floor_info = _apply_min_keep_ratio_floor(
            targets=targets,
            numel_list=numel_list,
            total_target=int(total_target),
            min_keep_ratio=float(allocation_min_tensor_keep_ratio),
        )
    else:
        targets = _normalize_target_counts_from_raw(
            raw_targets=[float(x) for x in targets],
            numel_list=numel_list,
            total_target=int(total_target),
        )
    return targets, {
        'mode': mode,
        'gamma': float(allocation_gamma),
        'sens_beta': float(allocation_sens_beta),
        'min_tensor_keep_ratio': float(allocation_min_tensor_keep_ratio),
        'total_target': int(total_target),
        'total_numel': int(total_numel),
        'uniform_targets_sum': int(sum(int(x) for x in uniform_targets)),
        'sensitivity_targets_sum': int(sum(int(x) for x in sensitivity_targets)),
        'floor': floor_info,
    }


@torch.no_grad()
def _estimate_family_sensitivities(stacks: Dict[str, List[torch.Tensor]], power_iters: int) -> Dict[str, float]:
    attn_proj_norms = []
    mlp_fc2_norms = []
    for proj_weight in stacks.get('attn_proj', []):
        attn_proj_norms.append(spectral_norm_power_iteration(proj_weight.float(), num_iters=power_iters))
    for fc2_weight in stacks.get('mlp_fc2', []):
        mlp_fc2_norms.append(spectral_norm_power_iteration(fc2_weight.float(), num_iters=power_iters))

    attn_proj_sensitivity = 1.0
    mlp_fc2_sensitivity = 1.0
    attn_qkv_sensitivity = float(np.mean(np.square(attn_proj_norms))) if attn_proj_norms else 1.0
    mlp_fc1_sensitivity = float(np.mean(np.square(mlp_fc2_norms))) if mlp_fc2_norms else 1.0
    return {
        'attn_qkv': max(attn_qkv_sensitivity, 1e-8),
        'attn_proj': max(attn_proj_sensitivity, 1e-8),
        'mlp_fc1': max(mlp_fc1_sensitivity, 1e-8),
        'mlp_fc2': max(mlp_fc2_sensitivity, 1e-8),
    }


@torch.no_grad()
def _dense_state_num_bytes(state: Dict[str, torch.Tensor]) -> int:
    return int(sum(t.numel() * t.element_size() for t in state.values()))


@torch.no_grad()
def _stack_dense_bytes(stack: torch.Tensor) -> int:
    return int(stack.numel() * stack.element_size())


@torch.no_grad()
def _strip_family_param_suffix(param_name: str, family_name: str) -> str:
    if family_name not in FAMILY_PARAM_SUFFIX:
        raise KeyError(f'Unknown family_name: {family_name}')
    suffix = str(FAMILY_PARAM_SUFFIX[family_name])
    if not str(param_name).endswith(suffix):
        raise ValueError(f'Parameter name `{param_name}` does not match expected suffix `{suffix}` for family `{family_name}`.')
    return str(param_name)[: -len(suffix)]


@torch.no_grad()
def _build_shape_grouped_family_pair(
    tensors_a: Sequence[torch.Tensor],
    names_a: Sequence[str],
    family_a: str,
    tensors_b: Sequence[torch.Tensor],
    names_b: Sequence[str],
    family_b: str,
) -> List[Dict[str, Any]]:
    if len(tensors_a) != len(names_a):
        raise ValueError(f'Length mismatch for {family_a}: tensors={len(tensors_a)} names={len(names_a)}')
    if len(tensors_b) != len(names_b):
        raise ValueError(f'Length mismatch for {family_b}: tensors={len(tensors_b)} names={len(names_b)}')
    if len(tensors_a) != len(tensors_b):
        raise ValueError(
            f'Family pair length mismatch: {family_a}={len(tensors_a)} vs {family_b}={len(tensors_b)}. '
            'Each block must contribute both tensors of the bundle.'
        )

    groups: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Dict[str, Any]] = {}
    group_order: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
    for tensor_a, name_a, tensor_b, name_b in zip(tensors_a, names_a, tensors_b, names_b):
        prefix_a = _strip_family_param_suffix(str(name_a), family_name=family_a)
        prefix_b = _strip_family_param_suffix(str(name_b), family_name=family_b)
        if prefix_a != prefix_b:
            raise ValueError(
                'Family pair tensor order mismatch. '
                f'Expected aligned bundle prefixes but got `{prefix_a}` and `{prefix_b}`.'
            )
        key = (
            tuple(int(x) for x in tensor_a.shape),
            tuple(int(x) for x in tensor_b.shape),
        )
        if key not in groups:
            groups[key] = {
                'tensors_a': [],
                'tensors_b': [],
                'param_names_a': [],
                'param_names_b': [],
            }
            group_order.append(key)
        groups[key]['tensors_a'].append(tensor_a)
        groups[key]['tensors_b'].append(tensor_b)
        groups[key]['param_names_a'].append(str(name_a))
        groups[key]['param_names_b'].append(str(name_b))

    grouped: List[Dict[str, Any]] = []
    for group_idx, key in enumerate(group_order):
        item = groups[key]
        grouped.append(
            {
                'group_index': int(group_idx),
                'shape_a': list(key[0]),
                'shape_b': list(key[1]),
                'stack_a': torch.stack(item['tensors_a'], dim=0),
                'stack_b': torch.stack(item['tensors_b'], dim=0),
                'param_names_a': list(item['param_names_a']),
                'param_names_b': list(item['param_names_b']),
            }
        )
    return grouped


@torch.no_grad()
def compress_vit_to_archive(
    model: nn.Module,
    save_path: str,
    model_name: str,
    mlp_keep_ratio: float,
    attn_keep_ratio: float,
    quant_bits: int,
    device: torch.device,
    gauge_proj_dim: int = 16,
    gauge_pca_dim: int = 8,
    sensitivity_power_iters: int = 16,
    enable_attn_rotation_gauge: bool = True,
    verify_gauge: bool = True,
    include_head_linear: bool = False,
    enable_canonicalization: bool = True,
    enable_haar_transform: bool = True,
    stack_transform: str = 'haar3d',
    use_sensitivity_allocation: bool = True,
    use_progressive_archive: bool = True,
    fixed_global_keep_ratio: Optional[float] = None,
    allocation_mode: str = 'sensitivity',
    allocation_gamma: float = 0.35,
    allocation_sens_beta: float = 1.0,
    allocation_min_tensor_keep_ratio: float = 0.0,
) -> CompressionSummary:
    model.eval()
    if bool(enable_canonicalization):
        canonical_state, stacks, stack_names, gauge_report = canonicalize_vit_state_dict(
            model=model,
            device=device,
            gauge_proj_dim=gauge_proj_dim,
            gauge_pca_dim=gauge_pca_dim,
            enable_attn_rotation_gauge=enable_attn_rotation_gauge,
            verify_gauge=verify_gauge,
        )
    else:
        canonical_state = _copy_state_dict(model)
        stacks, stack_names = _build_identity_vit_stacks(model=model, state=canonical_state)
        gauge_report = {
            'canonicalization_applied': False,
            'attention_rotation_gauge_enabled': False,
            'verify_gauge': False,
        }
    dense_state_bytes = _dense_state_num_bytes(canonical_state)
    compressed_weight_names = set(_compressed_weight_names(model, include_head_linear=bool(include_head_linear)))
    compressed_weight_dense_bytes = int(sum(canonical_state[name].numel() * canonical_state[name].element_size() for name in compressed_weight_names))
    compressed_scope_fp32_bytes = int(sum(canonical_state[name].numel() * 4 for name in compressed_weight_names))
    exact_tensor_dense_bytes = dense_state_bytes - compressed_weight_dense_bytes
    sensitivities = _estimate_family_sensitivities(stacks=stacks, power_iters=sensitivity_power_iters)
    attn_pair_groups = _build_shape_grouped_family_pair(
        tensors_a=stacks['attn_qkv'],
        names_a=stack_names['attn_qkv'],
        family_a='attn_qkv',
        tensors_b=stacks['attn_proj'],
        names_b=stack_names['attn_proj'],
        family_b='attn_proj',
    )
    mlp_pair_groups = _build_shape_grouped_family_pair(
        tensors_a=stacks['mlp_fc1'],
        names_a=stack_names['mlp_fc1'],
        family_a='mlp_fc1',
        tensors_b=stacks['mlp_fc2'],
        names_b=stack_names['mlp_fc2'],
        family_b='mlp_fc2',
    )

    save_path = str(save_path)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    family_stats: List[FamilyCompressionStats] = []
    manifest: Dict[str, Any] = {
        'format': ARCHIVE_FORMAT,
        'model_name': str(model_name),
        'mlp_keep_ratio': float(mlp_keep_ratio),
        'attn_keep_ratio': float(attn_keep_ratio),
        'quant_bits': int(quant_bits),
        'gauge_proj_dim': int(gauge_proj_dim),
        'gauge_pca_dim': int(gauge_pca_dim),
        'sensitivity_power_iters': int(sensitivity_power_iters),
        'enable_attn_rotation_gauge': bool(enable_attn_rotation_gauge),
        'verify_gauge': bool(verify_gauge),
        'include_head_linear': bool(include_head_linear),
        'enable_canonicalization': bool(enable_canonicalization),
        'enable_haar_transform': bool(enable_haar_transform),
        'stack_transform': str(stack_transform),
        'use_sensitivity_allocation': bool(use_sensitivity_allocation),
        'archive_serializer': 'progressive_bitplane' if bool(use_progressive_archive) else 'flat_sparse_quantized_v1',
        'fixed_global_keep_ratio': None if fixed_global_keep_ratio is None else float(fixed_global_keep_ratio),
        'allocation_mode': str(allocation_mode),
        'allocation_gamma': float(allocation_gamma),
        'allocation_sens_beta': float(allocation_sens_beta),
        'allocation_min_tensor_keep_ratio': float(allocation_min_tensor_keep_ratio),
        'dense_state_bytes': int(dense_state_bytes),
        'compressed_weight_dense_bytes': int(compressed_weight_dense_bytes),
        'compressed_scope_fp32_bytes': int(compressed_scope_fp32_bytes),
        'exact_tensor_dense_bytes': int(exact_tensor_dense_bytes),
        'gauge_report': gauge_report,
        'stack_grouping': {
            'attn_pair_groups': [
                {
                    'group_index': int(item['group_index']),
                    'attn_qkv_shape': list(item['shape_a']),
                    'attn_proj_shape': list(item['shape_b']),
                    'stack_depth': int(len(item['param_names_a'])),
                }
                for item in attn_pair_groups
            ],
            'mlp_pair_groups': [
                {
                    'group_index': int(item['group_index']),
                    'mlp_fc1_shape': list(item['shape_a']),
                    'mlp_fc2_shape': list(item['shape_b']),
                    'stack_depth': int(len(item['param_names_a'])),
                }
                for item in mlp_pair_groups
            ],
        },
        'stacks': {},
        'exact_tensors': [],
    }

    tmp_save_path = _build_temp_archive_path(save_path)
    try:
        with zipfile.ZipFile(tmp_save_path, mode='w', compression=zipfile.ZIP_STORED) as zf:
            exact_idx = 0
            for name, tensor in canonical_state.items():
                if name in compressed_weight_names:
                    continue
                meta = serialize_exact_tensor(zf, entry_name=f'exact/{exact_idx:05d}.bin', tensor=tensor.detach().cpu(), codec='lzma')
                meta['name'] = name
                manifest['exact_tensors'].append(meta)
                exact_idx += 1

            def _zip_key(raw_key: str) -> str:
                return str(raw_key).replace('::', '__')

            transform_alias_map = {
                'haar3d': 'haar3d',
                'finite_length_haar_3d': 'haar3d',
                '3d': 'haar3d',
                'haar2d': 'haar2d',
                'finite_length_haar_2d': 'haar2d',
                '2d': 'haar2d',
                'alt_domain': 'alt_domain_dct2',
                'dct2': 'alt_domain_dct2',
                'alt_domain_dct2': 'alt_domain_dct2',
                'identity': 'identity',
                'none': 'identity',
            }
            transform_mode_raw = str(stack_transform or 'haar3d').strip().lower()
            transform_mode = transform_alias_map.get(transform_mode_raw, transform_mode_raw)
            valid_transform_modes = {'haar3d', 'haar2d', 'alt_domain_dct2', 'identity'}
            if transform_mode not in valid_transform_modes:
                raise ValueError(
                    f'Unsupported `stack_transform`={stack_transform!r}; '
                    f'expected one of {sorted(valid_transform_modes)}.'
                )
            if not bool(enable_haar_transform):
                transform_mode = 'identity'
            manifest['stack_transform_effective'] = str(transform_mode)

            def _transform_stack(stack: torch.Tensor) -> Tuple[torch.Tensor, Optional[List[Dict[str, Any]]], str]:
                stack_dev = stack.to(device=device, dtype=torch.float32).contiguous()
                if transform_mode == 'identity':
                    return stack_dev, None, 'identity'
                if transform_mode == 'haar3d':
                    coeff, schedules = developmental_transform_forward(stack_dev, dims=(2, 1, 0))
                    return coeff, schedules, 'finite_length_haar_3d'
                if transform_mode == 'haar2d':
                    coeff, schedules = developmental_transform_forward(stack_dev, dims=(2, 1))
                    return coeff, schedules, 'finite_length_haar_2d'
                if transform_mode == 'alt_domain_dct2':
                    coeff = dct2_transform_forward(stack_dev)
                    return coeff, None, 'orthonormal_dct2_2d'
                raise RuntimeError(f'Unexpected transform_mode: {transform_mode}')

            def _encode_coeff(prefix: str, coeff: torch.Tensor, target_nonzero: int) -> Dict[str, Any]:
                coeff_cpu = coeff.detach().cpu().contiguous()
                if bool(use_progressive_archive):
                    return encode_progressive_bitplane_stack(
                        zf=zf,
                        prefix=prefix,
                        coeff=coeff_cpu,
                        target_nonzero=int(target_nonzero),
                        quant_bits=quant_bits,
                        codec='lzma',
                    )
                return encode_flat_quantized_stack(
                    zf=zf,
                    prefix=prefix,
                    coeff=coeff_cpu,
                    target_nonzero=int(target_nonzero),
                    quant_bits=quant_bits,
                    codec='lzma',
                )

            def _write_stack_entry(
                stack_key: str,
                family: str,
                param_names: Sequence[str],
                stack: torch.Tensor,
                coeff: torch.Tensor,
                schedules: Optional[List[Dict[str, Any]]],
                transform_name: str,
                target_nonzero: int,
                sensitivity_value: float,
                shape_group_index: Optional[int] = None,
                shape_group_a: Optional[Sequence[int]] = None,
                shape_group_b: Optional[Sequence[int]] = None,
            ) -> None:
                _target = max(0, min(int(target_nonzero), int(coeff.numel())))
                meta = _encode_coeff(
                    prefix=f'stacks/{_zip_key(stack_key)}',
                    coeff=coeff,
                    target_nonzero=_target,
                )
                stack_meta: Dict[str, Any] = {
                    'family': str(family),
                    'param_names': [str(x) for x in param_names],
                    'stack_shape': list(stack.shape),
                    'transform': str(transform_name),
                    'sensitivity': float(sensitivity_value),
                    'code': meta,
                }
                if shape_group_index is not None:
                    stack_meta['shape_group_index'] = int(shape_group_index)
                if shape_group_a is not None:
                    stack_meta['shape_group_a'] = [int(x) for x in shape_group_a]
                if shape_group_b is not None:
                    stack_meta['shape_group_b'] = [int(x) for x in shape_group_b]
                if schedules is not None:
                    stack_meta['schedules'] = schedules
                manifest['stacks'][str(stack_key)] = stack_meta
                family_stats.append(
                    FamilyCompressionStats(
                        name=str(stack_key),
                        dense_bytes=_stack_dense_bytes(stack),
                        target_nonzero=int(meta['target_nonzero']),
                        actual_nonzero=int(meta['actual_nonzero']),
                        sensitivity=float(sensitivity_value),
                        quant_bits=int(quant_bits),
                    )
                )
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            def _append_pair_entries(
                prefix_a: str,
                prefix_b: str,
                grouped_pairs: Sequence[Dict[str, Any]],
                output_entries: List[Dict[str, Any]],
            ) -> None:
                if len(grouped_pairs) <= 0:
                    raise ValueError(f'No grouped stacks found for family pair ({prefix_a}, {prefix_b}).')
                num_groups = int(len(grouped_pairs))
                for group_item in grouped_pairs:
                    group_idx = int(group_item['group_index'])
                    stack_key_a = str(prefix_a) if num_groups == 1 else f'{prefix_a}::shape_group_{group_idx}'
                    stack_key_b = str(prefix_b) if num_groups == 1 else f'{prefix_b}::shape_group_{group_idx}'
                    output_entries.append(
                        {
                            'stack_key': stack_key_a,
                            'family': str(prefix_a),
                            'param_names': list(group_item['param_names_a']),
                            'stack': group_item['stack_a'],
                            'sensitivity': float(sensitivities.get(prefix_a, 1.0)),
                            'shape_group_index': int(group_idx),
                            'shape_group_a': list(group_item['shape_a']),
                            'shape_group_b': list(group_item['shape_b']),
                            'pair_key': (str(prefix_a), int(group_idx)),
                        }
                    )
                    output_entries.append(
                        {
                            'stack_key': stack_key_b,
                            'family': str(prefix_b),
                            'param_names': list(group_item['param_names_b']),
                            'stack': group_item['stack_b'],
                            'sensitivity': float(sensitivities.get(prefix_b, 1.0)),
                            'shape_group_index': int(group_idx),
                            'shape_group_a': list(group_item['shape_a']),
                            'shape_group_b': list(group_item['shape_b']),
                            'pair_key': (str(prefix_a), int(group_idx)),
                        }
                    )

            stack_entries: List[Dict[str, Any]] = []
            _append_pair_entries('attn_qkv', 'attn_proj', grouped_pairs=attn_pair_groups, output_entries=stack_entries)
            _append_pair_entries('mlp_fc1', 'mlp_fc2', grouped_pairs=mlp_pair_groups, output_entries=stack_entries)
            if bool(include_head_linear):
                for head_name in _iter_head_linear_weight_names(model):
                    if head_name in canonical_state and head_name in compressed_weight_names:
                        stack_entries.append(
                            {
                                'stack_key': f'head::{head_name}',
                                'family': 'head',
                                'param_names': [str(head_name)],
                                'stack': canonical_state[str(head_name)].unsqueeze(0),
                                'sensitivity': 1.0,
                                'shape_group_index': None,
                                'shape_group_a': None,
                                'shape_group_b': None,
                                'pair_key': None,
                            }
                        )
            if len(stack_entries) <= 0:
                raise RuntimeError('No stacks were collected for compression.')

            if fixed_global_keep_ratio is not None:
                ratio = float(fixed_global_keep_ratio)
                if ratio < 0.0 or ratio > 1.0:
                    raise ValueError(f'`fixed_global_keep_ratio` must be in [0, 1], got {ratio}.')
                effective_allocation_mode = _resolve_fixed_budget_allocation_mode(
                    allocation_mode=str(allocation_mode),
                    use_sensitivity_allocation=bool(use_sensitivity_allocation),
                )
                manifest['allocation_mode_effective'] = str(effective_allocation_mode)

                coeff_entries: List[Dict[str, Any]] = []
                for entry in stack_entries:
                    coeff, schedules, transform_name = _transform_stack(entry['stack'])
                    coeff_cpu = coeff.detach().cpu().contiguous()
                    coeff_entries.append(
                        {
                            **entry,
                            'coeff': coeff_cpu,
                            'schedules': schedules,
                            'transform_name': str(transform_name),
                        }
                    )
                    del coeff
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

                total_coeff_numel = int(sum(int(entry['coeff'].numel()) for entry in coeff_entries))
                total_keep_budget = int(math.floor(ratio * float(total_coeff_numel)))
                total_keep_budget = max(0, min(total_keep_budget, total_coeff_numel))
                manifest['global_coeff_numel'] = int(total_coeff_numel)
                manifest['global_keep_budget'] = int(total_keep_budget)
                targets, allocation_detail = _allocate_fixed_budget_targets(
                    coeff_abs_list=[entry['coeff'].abs() for entry in coeff_entries],
                    sensitivity_list=[float(entry['sensitivity']) for entry in coeff_entries],
                    total_target=int(total_keep_budget),
                    allocation_mode=str(effective_allocation_mode),
                    allocation_gamma=float(allocation_gamma),
                    allocation_sens_beta=float(allocation_sens_beta),
                    allocation_min_tensor_keep_ratio=float(allocation_min_tensor_keep_ratio),
                )
                manifest['allocation'] = {
                    'mode': str(allocation_detail.get('mode', effective_allocation_mode)),
                    'gamma': float(allocation_detail.get('gamma', allocation_gamma)),
                    'sens_beta': float(allocation_detail.get('sens_beta', allocation_sens_beta)),
                    'min_tensor_keep_ratio': float(
                        allocation_detail.get('min_tensor_keep_ratio', allocation_min_tensor_keep_ratio)
                    ),
                    'total_numel': int(allocation_detail.get('total_numel', total_coeff_numel)),
                    'total_target': int(allocation_detail.get('total_target', total_keep_budget)),
                    'floor': allocation_detail.get('floor', {'enabled': False}),
                }
                if int(sum(int(x) for x in targets)) != int(total_keep_budget):
                    raise RuntimeError(
                        'Internal allocation error: per-stack keep targets do not sum to total keep budget '
                        f'({sum(int(x) for x in targets)} vs {total_keep_budget}).'
                    )

                for entry, target in zip(coeff_entries, targets):
                    stack_key = str(entry['stack_key'])
                    meta = _encode_coeff(
                        prefix=f'stacks/{_zip_key(stack_key)}',
                        coeff=entry['coeff'],
                        target_nonzero=int(target),
                    )
                    stack_meta: Dict[str, Any] = {
                        'family': str(entry['family']),
                        'param_names': [str(x) for x in entry['param_names']],
                        'stack_shape': list(entry['stack'].shape),
                        'transform': str(entry['transform_name']),
                        'sensitivity': float(entry['sensitivity']),
                        'code': meta,
                    }
                    if entry.get('shape_group_index', None) is not None:
                        stack_meta['shape_group_index'] = int(entry['shape_group_index'])
                    if entry.get('shape_group_a', None) is not None:
                        stack_meta['shape_group_a'] = [int(x) for x in entry['shape_group_a']]
                    if entry.get('shape_group_b', None) is not None:
                        stack_meta['shape_group_b'] = [int(x) for x in entry['shape_group_b']]
                    if entry.get('schedules', None) is not None:
                        stack_meta['schedules'] = entry['schedules']
                    manifest['stacks'][stack_key] = stack_meta
                    family_stats.append(
                        FamilyCompressionStats(
                            name=stack_key,
                            dense_bytes=_stack_dense_bytes(entry['stack']),
                            target_nonzero=int(meta['target_nonzero']),
                            actual_nonzero=int(meta['actual_nonzero']),
                            sensitivity=float(entry['sensitivity']),
                            quant_bits=int(quant_bits),
                        )
                    )
            else:
                def _compress_pair_with_ratio(
                    prefix_a: str,
                    prefix_b: str,
                    grouped_pairs: Sequence[Dict[str, Any]],
                    total_keep_ratio: float,
                ) -> None:
                    if len(grouped_pairs) <= 0:
                        raise ValueError(f'No grouped stacks found for family pair ({prefix_a}, {prefix_b}).')
                    num_groups = int(len(grouped_pairs))
                    for group_item in grouped_pairs:
                        group_idx = int(group_item['group_index'])
                        stack_a = group_item['stack_a']
                        stack_b = group_item['stack_b']
                        coeff_a, schedules_a, transform_name_a = _transform_stack(stack_a)
                        coeff_b, schedules_b, transform_name_b = _transform_stack(stack_b)
                        total_target = int(round(float(total_keep_ratio) * float(coeff_a.numel() + coeff_b.numel())))
                        total_target = max(0, min(total_target, int(coeff_a.numel() + coeff_b.numel())))
                        if bool(use_sensitivity_allocation):
                            target_a, target_b = sensitivity_weighted_target_counts(
                                coeff_abs_list=[coeff_a.abs(), coeff_b.abs()],
                                sensitivity_list=[sensitivities[prefix_a], sensitivities[prefix_b]],
                                total_target=total_target,
                            )
                        else:
                            target_a, target_b = _global_proportional_target_counts(
                                numel_list=[int(coeff_a.numel()), int(coeff_b.numel())],
                                total_target=int(total_target),
                            )

                        stack_key_a = str(prefix_a) if num_groups == 1 else f'{prefix_a}::shape_group_{group_idx}'
                        stack_key_b = str(prefix_b) if num_groups == 1 else f'{prefix_b}::shape_group_{group_idx}'
                        _write_stack_entry(
                            stack_key=stack_key_a,
                            family=prefix_a,
                            param_names=group_item['param_names_a'],
                            stack=stack_a,
                            coeff=coeff_a,
                            schedules=schedules_a,
                            transform_name=str(transform_name_a),
                            target_nonzero=int(target_a),
                            sensitivity_value=float(sensitivities[prefix_a]),
                            shape_group_index=int(group_idx),
                            shape_group_a=group_item['shape_a'],
                            shape_group_b=group_item['shape_b'],
                        )
                        _write_stack_entry(
                            stack_key=stack_key_b,
                            family=prefix_b,
                            param_names=group_item['param_names_b'],
                            stack=stack_b,
                            coeff=coeff_b,
                            schedules=schedules_b,
                            transform_name=str(transform_name_b),
                            target_nonzero=int(target_b),
                            sensitivity_value=float(sensitivities[prefix_b]),
                            shape_group_index=int(group_idx),
                            shape_group_a=group_item['shape_a'],
                            shape_group_b=group_item['shape_b'],
                        )
                        del coeff_a, coeff_b

                _compress_pair_with_ratio(
                    'attn_qkv',
                    'attn_proj',
                    grouped_pairs=attn_pair_groups,
                    total_keep_ratio=float(attn_keep_ratio),
                )
                _compress_pair_with_ratio(
                    'mlp_fc1',
                    'mlp_fc2',
                    grouped_pairs=mlp_pair_groups,
                    total_keep_ratio=float(mlp_keep_ratio),
                )
                if bool(include_head_linear):
                    for head_name in _iter_head_linear_weight_names(model):
                        if head_name in canonical_state and head_name in compressed_weight_names:
                            stack = canonical_state[str(head_name)].unsqueeze(0)
                            coeff, schedules, transform_name = _transform_stack(stack)
                            target = int(round(float(mlp_keep_ratio) * float(coeff.numel())))
                            target = max(0, min(target, int(coeff.numel())))
                            _write_stack_entry(
                                stack_key=f'head::{head_name}',
                                family='head',
                                param_names=[str(head_name)],
                                stack=stack,
                                coeff=coeff,
                                schedules=schedules,
                                transform_name=str(transform_name),
                                target_nonzero=int(target),
                                sensitivity_value=1.0,
                            )
                            del coeff

            zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2, ensure_ascii=False).encode('utf-8'))
        os.replace(tmp_save_path, save_path)
    finally:
        if os.path.exists(tmp_save_path):
            try:
                os.remove(tmp_save_path)
            except OSError:
                pass

    archive_bytes = int(os.path.getsize(save_path))
    return CompressionSummary(
        archive_path=save_path,
        archive_bytes=archive_bytes,
        dense_state_bytes=dense_state_bytes,
        compressed_weight_dense_bytes=compressed_weight_dense_bytes,
        compressed_scope_fp32_bytes=compressed_scope_fp32_bytes,
        exact_tensor_dense_bytes=exact_tensor_dense_bytes,
        family_stats=family_stats,
        gauge_report=gauge_report,
    )


@torch.no_grad()
def compress_vit_to_archive_blockwise_thresholded(
    model: nn.Module,
    save_path: str,
    model_name: str,
    quant_bits: int,
    device: torch.device,
    sensitivity_report_path: str,
    target_archive_mb: float,
    target_archive_tolerance_ratio: float = 0.005,
    target_archive_max_iters: int = 26,
    lambda_probe_start: float = 1e-8,
    sens_weight_ce: float = 0.5,
    sens_weight_logit: float = 0.3,
    sens_weight_byte: float = 0.2,
    importance_alpha: float = 2.0,
    importance_quantile_q: float = 0.95,
    sens_beta: float = 1.0,
    min_tensor_keep_ratio: float = 0.01,
    gauge_proj_dim: int = 16,
    gauge_pca_dim: int = 8,
    enable_attn_rotation_gauge: bool = True,
    verify_gauge: bool = True,
    include_head_linear: bool = False,
) -> Tuple[CompressionSummary, Dict[str, Any]]:
    def _progress(message: str) -> None:
        print(f'[blockwise_thresholded] {message}', flush=True)

    _progress('Start compression pipeline.')
    if float(target_archive_mb) <= 0.0:
        raise ValueError('`target_archive_mb` must be > 0 when codec_allocation_mode=blockwise_thresholded.')
    if int(target_archive_max_iters) <= 0:
        raise ValueError('`target_archive_max_iters` must be > 0 when codec_allocation_mode=blockwise_thresholded.')
    if not math.isfinite(float(lambda_probe_start)) or float(lambda_probe_start) <= 0.0:
        raise ValueError(f'`lambda_probe_start` must be a positive finite float, got: {lambda_probe_start!r}')
    if not math.isfinite(float(importance_alpha)) or float(importance_alpha) <= 0.0:
        raise ValueError(f'`importance_alpha` must be a positive finite float, got: {importance_alpha!r}')
    if not math.isfinite(float(importance_quantile_q)) or not (0.0 < float(importance_quantile_q) <= 1.0):
        raise ValueError(
            f'`importance_quantile_q` must be in (0, 1], got: {importance_quantile_q!r}'
        )
    if not math.isfinite(float(sens_beta)) or float(sens_beta) < 0.0:
        raise ValueError(f'`sens_beta` must be a non-negative finite float, got: {sens_beta!r}')
    _progress(
        'Score formula: score = (1 + '
        f'{float(sens_beta):.6g}'
        f' * sensitivity) * (|c| / (Q_{float(importance_quantile_q):.6g} + eps))^{float(importance_alpha):.6g}'
    )
    _progress(f'Load sensitivity report: {sensitivity_report_path}')
    sensitivity_report = load_blockwise_sensitivity_report(sensitivity_report_path)
    sensitivity_scores = build_blockwise_sensitivity_scores(
        sensitivity_report=sensitivity_report,
        sens_weight_ce=float(sens_weight_ce),
        sens_weight_logit=float(sens_weight_logit),
        sens_weight_byte=float(sens_weight_byte),
    )
    _progress(f'Built blockwise sensitivity scores for {len(sensitivity_scores["units"])} units.')
    _validate_blockwise_sensitivity_alignment(
        model=model,
        model_name=str(model_name),
        sensitivity_report=sensitivity_report,
        block_sensitivity_scores=sensitivity_scores['block_scores'],
        sensitivity_report_path=str(sensitivity_report_path),
    )
    _progress('Sensitivity report alignment check passed.')

    model.eval()
    _progress('Canonicalize model state with gauge...')
    canonical_state, _, _, gauge_report = canonicalize_vit_state_dict(
        model=model,
        device=device,
        gauge_proj_dim=gauge_proj_dim,
        gauge_pca_dim=gauge_pca_dim,
        enable_attn_rotation_gauge=enable_attn_rotation_gauge,
        verify_gauge=verify_gauge,
    )
    dense_state_bytes = _dense_state_num_bytes(canonical_state)
    compressed_weight_names = _compressed_weight_names(model, include_head_linear=bool(include_head_linear))
    compressed_name_set = set(compressed_weight_names)
    compressed_weight_dense_bytes = int(sum(canonical_state[name].numel() * canonical_state[name].element_size() for name in compressed_name_set))
    compressed_scope_fp32_bytes = int(sum(canonical_state[name].numel() * 4 for name in compressed_name_set))
    exact_tensor_dense_bytes = int(dense_state_bytes - compressed_weight_dense_bytes)

    tensor_entries = _iter_vit_target_tensors(model, include_head_linear=bool(include_head_linear))
    _progress(f'Prepare transform scores for {len(tensor_entries)} target tensors...')
    preferred_score_device = device if device.type == 'cuda' else torch.device('cpu')
    try:
        tensor_records = compute_tensor_transform_scores(
            canonical_state=canonical_state,
            tensor_entries=tensor_entries,
            unit_sensitivity_scores=sensitivity_scores['unit_scores'],
            compute_device=preferred_score_device,
            progress_callback=_progress,
            importance_alpha=float(importance_alpha),
            importance_quantile_q=float(importance_quantile_q),
            sens_beta=float(sens_beta),
        )
    except RuntimeError as exc:
        if preferred_score_device.type == 'cuda' and 'out of memory' in str(exc).lower():
            _progress('CUDA OOM during score preparation; fallback to CPU score computation.')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            tensor_records = compute_tensor_transform_scores(
                canonical_state=canonical_state,
                tensor_entries=tensor_entries,
                unit_sensitivity_scores=sensitivity_scores['unit_scores'],
                compute_device=torch.device('cpu'),
                progress_callback=_progress,
                importance_alpha=float(importance_alpha),
                importance_quantile_q=float(importance_quantile_q),
                sens_beta=float(sens_beta),
            )
        else:
            raise
    lambda_upper = 0.0
    for record in tensor_records:
        score = record['score']
        if int(score.numel()) <= 0:
            continue
        local_max = float(score.max().item())
        if math.isfinite(local_max):
            lambda_upper = max(lambda_upper, local_max)
    _progress(f'Lambda search upper bound prepared: {lambda_upper:.6g}')

    def _evaluate_lambda(lambda_value: float) -> Dict[str, Any]:
        return _encode_blockwise_thresholded_archive(
            canonical_state=canonical_state,
            compressed_weight_names=compressed_weight_names,
            tensor_records=tensor_records,
            model_name=model_name,
            quant_bits=int(quant_bits),
            gauge_report=gauge_report,
            dense_state_bytes=dense_state_bytes,
            compressed_weight_dense_bytes=compressed_weight_dense_bytes,
            compressed_scope_fp32_bytes=compressed_scope_fp32_bytes,
            exact_tensor_dense_bytes=exact_tensor_dense_bytes,
            lambda_value=float(lambda_value),
            min_tensor_keep_ratio=float(min_tensor_keep_ratio),
            save_path=None,
            stage_label='lambda_probe',
            progress_callback=None,
            collect_tensor_stats=False,
        )

    _progress('Start lambda search with real archive-byte budget...')
    lambda_search = search_lambda_for_target_mb(
        encoder_fn=_evaluate_lambda,
        target_archive_mb=float(target_archive_mb),
        target_archive_tolerance_ratio=float(target_archive_tolerance_ratio),
        lambda_lower=0.0,
        lambda_upper=float(lambda_upper),
        lambda_probe_start=float(lambda_probe_start),
        max_iters=int(target_archive_max_iters),
        progress_callback=_progress,
        retain_eval_result=False,
    )
    lambda_final = float(lambda_search['lambda_final'])
    _progress(
        f'Lambda search done: lambda_final={lambda_final:.6g}, '
        f'actual={int(lambda_search["actual_archive_bytes"])} bytes, '
        f'target={int(lambda_search["target_archive_bytes"])} bytes'
    )

    save_path = str(save_path)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    _progress(f'Write final archive: {save_path}')
    final_encode = _encode_blockwise_thresholded_archive(
        canonical_state=canonical_state,
        compressed_weight_names=compressed_weight_names,
        tensor_records=tensor_records,
        model_name=model_name,
        quant_bits=int(quant_bits),
        gauge_report=gauge_report,
        dense_state_bytes=dense_state_bytes,
        compressed_weight_dense_bytes=compressed_weight_dense_bytes,
        compressed_scope_fp32_bytes=compressed_scope_fp32_bytes,
        exact_tensor_dense_bytes=exact_tensor_dense_bytes,
        lambda_value=float(lambda_final),
        min_tensor_keep_ratio=float(min_tensor_keep_ratio),
        save_path=save_path,
        stage_label='final_encode',
        progress_callback=_progress,
    )
    archive_bytes = int(final_encode['archive_bytes'])
    tensor_stats: List[Dict[str, Any]] = list(final_encode['tensor_stats'])
    floor_applied_tensor_count = 0
    min_target_keep_ratio_observed: Optional[float] = None
    min_actual_keep_ratio_observed: Optional[float] = None
    if len(tensor_stats) > 0:
        floor_applied_tensor_count = int(sum(1 for item in tensor_stats if bool(item.get('floor_applied', False))))
        min_target_keep_ratio_observed = float(min(float(item.get('target_keep_ratio', 0.0)) for item in tensor_stats))
        min_actual_keep_ratio_observed = float(min(float(item.get('keep_ratio', 0.0)) for item in tensor_stats))
    if float(min_tensor_keep_ratio) > 0.0:
        if floor_applied_tensor_count <= 0:
            observed_text = (
                f'{float(min_target_keep_ratio_observed):.4f}'
                if min_target_keep_ratio_observed is not None
                else 'n/a'
            )
            _progress(
                f'Note: min_tensor_keep_ratio={float(min_tensor_keep_ratio):.4f} did not bind at lambda_final; '
                f'observed min target keep ratio={observed_text}.'
            )
        else:
            observed_text = (
                f'{float(min_target_keep_ratio_observed):.4f}'
                if min_target_keep_ratio_observed is not None
                else 'n/a'
            )
            _progress(
                f'min_tensor_keep_ratio bound {floor_applied_tensor_count}/{len(tensor_stats)} tensors '
                f'(observed min target keep ratio={observed_text}).'
            )
    family_stats: List[FamilyCompressionStats] = []
    for stat in tensor_stats:
        family_stats.append(
            FamilyCompressionStats(
                name=str(stat['param_name']),
                dense_bytes=int(stat['tensor_dense_bytes']),
                target_nonzero=int(stat['target_nonzero']),
                actual_nonzero=int(stat['actual_nonzero']),
                sensitivity=float(stat['sensitivity']),
                quant_bits=int(quant_bits),
            )
        )

    block_rows = compute_bundle_keep_ratios(
        tensor_stats=tensor_stats,
        block_sensitivity_scores=sensitivity_scores['block_scores'],
    )

    # Ensure report fields remain stable and machine-readable.
    allocation_report: Dict[str, Any] = {
        'mode': 'compress',
        'model': str(model_name),
        'codec_allocation_mode': 'blockwise_thresholded',
        'include_head_linear': bool(include_head_linear),
        'compressed_scope_fp32_bytes': int(compressed_scope_fp32_bytes),
        'compressed_scope_fp32_human': format_bytes(int(compressed_scope_fp32_bytes)),
        'baseline_fp32_scope_bytes': int(compressed_scope_fp32_bytes),
        'baseline_fp32_scope_human': format_bytes(int(compressed_scope_fp32_bytes)),
        'target_archive_mb': float(lambda_search['target_archive_mb']),
        'actual_archive_mb': float(archive_bytes / (1024.0 * 1024.0)),
        'target_archive_bytes': int(lambda_search['target_archive_bytes']),
        'actual_archive_bytes': int(archive_bytes),
        'lambda_final': float(lambda_final),
        'quant_bits': int(quant_bits),
        'min_tensor_keep_ratio': float(min_tensor_keep_ratio),
        'floor_applied_tensor_count': int(floor_applied_tensor_count),
        'floor_applied_tensor_ratio': float(floor_applied_tensor_count / max(float(len(tensor_stats)), 1.0)),
        'min_target_keep_ratio_observed': min_target_keep_ratio_observed,
        'min_actual_keep_ratio_observed': min_actual_keep_ratio_observed,
        'lambda_probe_start': float(lambda_probe_start),
        'sens_weight_ce': float(sens_weight_ce),
        'sens_weight_logit': float(sens_weight_logit),
        'sens_weight_byte': float(sens_weight_byte),
        'importance_alpha': float(importance_alpha),
        'importance_quantile_q': float(importance_quantile_q),
        'sens_beta': float(sens_beta),
        'blocks': block_rows,
        'tensor_stats': tensor_stats,
        'sensitivity_units': sensitivity_scores['units'],
        'lambda_search': {
            'target_archive_mb': float(lambda_search['target_archive_mb']),
            'target_archive_bytes': int(lambda_search['target_archive_bytes']),
            'tolerance_bytes': int(lambda_search['tolerance_bytes']),
            'max_iters': int(target_archive_max_iters),
            'probe_start': float(lambda_search.get('probe_start', lambda_probe_start)),
            'converged': bool(lambda_search['converged']),
            'iterations': int(lambda_search['iterations']),
            'eval_trace': lambda_search['eval_trace'],
        },
    }

    summary = CompressionSummary(
        archive_path=save_path,
        archive_bytes=int(archive_bytes),
        dense_state_bytes=int(dense_state_bytes),
        compressed_weight_dense_bytes=int(compressed_weight_dense_bytes),
        compressed_scope_fp32_bytes=int(compressed_scope_fp32_bytes),
        exact_tensor_dense_bytes=int(exact_tensor_dense_bytes),
        family_stats=family_stats,
        gauge_report=gauge_report,
    )
    _progress(f'Compression finished: archive_bytes={archive_bytes}')
    return summary, allocation_report


@torch.no_grad()
def is_gaugedna_archive(path: str) -> bool:
    try:
        with zipfile.ZipFile(path, mode='r') as zf:
            if MANIFEST_NAME not in zf.namelist():
                return False
            manifest = json.loads(zf.read(MANIFEST_NAME).decode('utf-8'))
            return manifest.get('format') == ARCHIVE_FORMAT
    except Exception:
        return False


@torch.no_grad()
def inspect_gaugedna_archive(path: str) -> Dict[str, Any]:
    with zipfile.ZipFile(path, mode='r') as zf:
        manifest = json.loads(zf.read(MANIFEST_NAME).decode('utf-8'))
    manifest['archive_path'] = str(path)
    manifest['archive_bytes'] = int(os.path.getsize(path))
    return manifest


@torch.no_grad()
def _decode_plane_count(stack_meta: Dict[str, Any], decode_fraction: float, decode_planes: Optional[int]) -> int:
    total_planes = int(stack_meta['code']['num_bitplanes'])
    if decode_planes is not None and decode_planes > 0:
        return min(int(decode_planes), total_planes)
    if decode_fraction >= 1.0:
        return total_planes
    if decode_fraction <= 0.0:
        return 1
    return max(1, min(total_planes, int(math.ceil(total_planes * decode_fraction))))


@torch.no_grad()
def load_gaugedna_archive(
    archive_path: str,
    model_builder: Callable[[str], nn.Module],
    device: torch.device,
    decode_fraction: float = 1.0,
    decode_planes: Optional[int] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    with zipfile.ZipFile(archive_path, mode='r') as zf:
        manifest = json.loads(zf.read(MANIFEST_NAME).decode('utf-8'))
        if manifest.get('format') != ARCHIVE_FORMAT:
            raise ValueError(f'Unsupported archive format: {manifest.get("format")}')
        model = model_builder(str(manifest['model_name']))
        state = model.state_dict()

        def _decode_stack_code(stack_meta: Dict[str, Any]) -> torch.Tensor:
            code_meta = stack_meta['code']
            code_encoding = str(code_meta.get('encoding', ''))
            if code_encoding == 'flat_sparse_quantized_v1':
                return decode_flat_quantized_stack(
                    zf=zf,
                    meta=code_meta,
                    device=torch.device('cpu'),
                )
            planes_to_decode = _decode_plane_count(stack_meta, decode_fraction=decode_fraction, decode_planes=decode_planes)
            return decode_progressive_bitplane_stack(
                zf=zf,
                meta=code_meta,
                device=torch.device('cpu'),
                max_planes=planes_to_decode,
            )

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

        for tensor_meta in manifest['exact_tensors']:
            state[str(tensor_meta['name'])] = deserialize_exact_tensor(zf, tensor_meta)

        codec_mode = str(manifest.get('codec_allocation_mode', 'legacy_global'))
        if codec_mode == 'blockwise_thresholded' or 'tensor_stacks' in manifest:
            tensor_stacks = manifest.get('tensor_stacks', None)
            if not isinstance(tensor_stacks, dict) or len(tensor_stacks) == 0:
                raise KeyError('blockwise_thresholded archive must contain a non-empty `tensor_stacks` dict.')
            for tensor_name in sorted(tensor_stacks.keys()):
                stack_meta = tensor_stacks[tensor_name]
                coeff = _decode_stack_code(stack_meta)
                stack = _inverse_stack_transform(coeff, stack_meta=stack_meta)
                if int(stack.shape[0]) != 1:
                    raise ValueError(
                        f'Expected tensor stack depth=1 for blockwise_thresholded entry `{tensor_name}`, got shape={tuple(stack.shape)}'
                    )
                param_name = str(stack_meta.get('param_name', tensor_name))
                state[param_name] = stack[0]
        else:
            stacks_manifest = manifest.get('stacks', {})
            if not isinstance(stacks_manifest, dict) or len(stacks_manifest) == 0:
                raise KeyError('Legacy archive is missing non-empty `stacks` metadata.')
            for family_name in sorted(stacks_manifest.keys()):
                stack_meta = stacks_manifest[family_name]
                coeff = _decode_stack_code(stack_meta)
                stack = _inverse_stack_transform(coeff, stack_meta=stack_meta)
                for layer_idx, param_name in enumerate(stack_meta['param_names']):
                    state[param_name] = stack[layer_idx]

        model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    if 'compressed_scope_fp32_bytes' not in manifest:
        try:
            include_head_linear = bool(manifest.get('include_head_linear', False))
            compressed_names = _compressed_weight_names(model, include_head_linear=include_head_linear)
            manifest['compressed_scope_fp32_bytes'] = int(sum(state[name].numel() * 4 for name in compressed_names if name in state))
        except Exception:
            pass
    if 'compressed_scope_fp32_bytes' in manifest:
        manifest['baseline_fp32_scope_bytes'] = int(manifest['compressed_scope_fp32_bytes'])
        manifest['baseline_fp32_scope_human'] = format_bytes(int(manifest['compressed_scope_fp32_bytes']))
    manifest['archive_path'] = str(archive_path)
    manifest['archive_bytes'] = int(os.path.getsize(archive_path))
    manifest['decode_fraction'] = float(decode_fraction)
    manifest['decode_planes'] = None if decode_planes is None else int(decode_planes)
    return model, manifest


@torch.no_grad()
def _iter_vit_block_bundle_names(model: nn.Module) -> List[Dict[str, Any]]:
    bundles: List[Dict[str, Any]] = []
    for entry in _iter_supported_transformer_blocks(model):
        block_idx = int(entry['block_index'])
        stage_idx = int(entry['stage_index'])
        stage_block_idx = int(entry['stage_block_index'])
        prefix = str(entry['param_prefix'])
        bundles.append(
            {
                'block_index': int(block_idx),
                'stage_index': int(stage_idx),
                'stage_block_index': int(stage_block_idx),
                'block_path': str(prefix),
                'attn_qkv': f'{prefix}.attn.qkv.weight',
                'attn_proj': f'{prefix}.attn.proj.weight',
                'mlp_fc1': f'{prefix}.mlp.fc1.weight',
                'mlp_fc2': f'{prefix}.mlp.fc2.weight',
            }
        )
    return bundles


@torch.no_grad()
def _build_reference_model_from_state(model_name: str, canonical_state: Dict[str, torch.Tensor], device: torch.device) -> nn.Module:
    from devdna_eval import build_timm_model

    model = build_timm_model(
        model_name=model_name,
        pretrained=False,
        checkpoint_path='',
        pretrained_weight_path='',
    )
    model.load_state_dict(canonical_state, strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def _compute_bundle_relative_distortion(
    reference_tensors: Sequence[torch.Tensor],
    reconstructed_tensors: Sequence[torch.Tensor],
    eps: float = 1e-12,
) -> float:
    if len(reference_tensors) != len(reconstructed_tensors):
        raise ValueError('reference_tensors and reconstructed_tensors must have the same length.')
    numerator = 0.0
    denominator = 0.0
    for ref_tensor, recon_tensor in zip(reference_tensors, reconstructed_tensors):
        ref = ref_tensor.detach().float()
        recon = recon_tensor.detach().float()
        numerator += float((recon - ref).pow(2).sum().item())
        denominator += float(ref.pow(2).sum().item())
    return float(numerator / (denominator + eps))


@torch.no_grad()
def _compute_block_heuristic_scores(
    canonical_state: Dict[str, torch.Tensor],
    bundle_names: Dict[str, Any],
    sensitivity_power_iters: int,
) -> Dict[str, float]:
    attn_proj = canonical_state[str(bundle_names['attn_proj'])].float()
    mlp_fc2 = canonical_state[str(bundle_names['mlp_fc2'])].float()
    attn_score = spectral_norm_power_iteration(attn_proj, num_iters=sensitivity_power_iters) ** 2
    mlp_score = spectral_norm_power_iteration(mlp_fc2, num_iters=sensitivity_power_iters) ** 2
    return {
        'attn': float(attn_score),
        'mlp': float(mlp_score),
    }


@torch.no_grad()
def _rankdata_average(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    order = np.argsort(arr, kind='mergesort')
    ranks = np.zeros((n,), dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = rank
        i = j + 1
    return ranks


@torch.no_grad()
def _require_finite_float(value: Any, field_name: str, context: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f'Invalid numeric value for {field_name} in {context}: {value!r}') from exc
    if not math.isfinite(out):
        raise ValueError(f'Non-finite value for {field_name} in {context}: {value!r}')
    return float(out)


@torch.no_grad()
def _rank_normalize_01(values: Sequence[float]) -> List[float]:
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    if n <= 0:
        return []
    if n == 1:
        return [0.0]
    ranks = _rankdata_average(arr.tolist())
    normalized = (ranks - 1.0) / max(float(n - 1), 1.0)
    return [float(x) for x in normalized]


@torch.no_grad()
def _emit_progress(progress_callback: Optional[Callable[[str], None]], message: str) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(str(message))
    except Exception:
        # Progress logging must never break compression flow.
        return


@torch.no_grad()
def load_blockwise_sensitivity_report(path: str) -> Dict[str, Any]:
    if not path or not str(path).strip():
        raise ValueError('`sensitivity_report` is required for codec_allocation_mode=blockwise_thresholded.')
    report_path = Path(path).expanduser()
    if not report_path.is_file():
        raise FileNotFoundError(f'Sensitivity report not found: {report_path}')
    with open(report_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f'Sensitivity report root must be a dict, got: {type(data)}')
    blocks = data.get('blocks', None)
    if not isinstance(blocks, list) or len(blocks) == 0:
        raise KeyError('Sensitivity report must contain a non-empty `blocks` list.')
    return data


@torch.no_grad()
def build_blockwise_sensitivity_scores(
    sensitivity_report: Dict[str, Any],
    sens_weight_ce: float = 0.5,
    sens_weight_logit: float = 0.3,
    sens_weight_byte: float = 0.2,
) -> Dict[str, Any]:
    blocks = sensitivity_report.get('blocks', None)
    if not isinstance(blocks, list) or len(blocks) == 0:
        raise KeyError('Sensitivity report must contain a non-empty `blocks` list.')

    weights = {
        'sens_weight_ce': float(sens_weight_ce),
        'sens_weight_logit': float(sens_weight_logit),
        'sens_weight_byte': float(sens_weight_byte),
    }

    unit_rows: List[Dict[str, Any]] = []
    for block_item in blocks:
        if not isinstance(block_item, dict):
            raise TypeError(f'Each block entry must be a dict, got: {type(block_item)}')
        if 'block_index' in block_item:
            block_index = int(block_item['block_index'])
        elif 'block' in block_item:
            block_index = int(block_item['block'])
        else:
            raise KeyError('Each block entry must contain `block_index` (or fallback `block`).')

        for bundle_type in ('attn', 'mlp'):
            bundle = block_item.get(bundle_type, None)
            if not isinstance(bundle, dict):
                raise KeyError(f'Block {block_index} is missing bundle `{bundle_type}` in sensitivity report.')
            context = f'block={block_index}, bundle={bundle_type}'
            delta_ce = _require_finite_float(bundle.get('delta_ce', None), 'delta_ce', context)
            rel_logit_mse = _require_finite_float(bundle.get('rel_logit_mse', None), 'rel_logit_mse', context)
            sensitivity_byte = _require_finite_float(bundle.get('sensitivity_byte', None), 'sensitivity_byte', context)
            unit_rows.append(
                {
                    'block_index': int(block_index),
                    'bundle_type': str(bundle_type),
                    'unit_key': f'block_{int(block_index)}.{bundle_type}',
                    'delta_ce': float(delta_ce),
                    'rel_logit_mse': float(rel_logit_mse),
                    'sensitivity_byte': float(sensitivity_byte),
                }
            )

    head_units = sensitivity_report.get('head_units', None)
    if isinstance(head_units, list):
        for head_idx, head_item in enumerate(head_units):
            if not isinstance(head_item, dict):
                continue
            unit_key = str(head_item.get('unit_key', f'head_unit_{head_idx}'))
            context = f'head_unit={unit_key}'
            delta_ce = _require_finite_float(head_item.get('delta_ce', None), 'delta_ce', context)
            rel_logit_mse = _require_finite_float(head_item.get('rel_logit_mse', None), 'rel_logit_mse', context)
            sensitivity_byte = _require_finite_float(head_item.get('sensitivity_byte', None), 'sensitivity_byte', context)
            unit_rows.append(
                {
                    'block_index': int(head_item.get('block_index', -1)),
                    'bundle_type': 'head',
                    'unit_key': unit_key,
                    'delta_ce': float(delta_ce),
                    'rel_logit_mse': float(rel_logit_mse),
                    'sensitivity_byte': float(sensitivity_byte),
                }
            )

    def _bundle_order(bundle_type: str) -> int:
        if bundle_type == 'attn':
            return 0
        if bundle_type == 'mlp':
            return 1
        if bundle_type == 'head':
            return 2
        return 99

    unit_rows.sort(
        key=lambda x: (
            int(x['block_index']),
            _bundle_order(str(x['bundle_type'])),
            str(x['unit_key']),
        )
    )
    ce_values = [float(row['delta_ce']) for row in unit_rows]
    logit_values = [float(row['rel_logit_mse']) for row in unit_rows]
    byte_values = [float(row['sensitivity_byte']) for row in unit_rows]
    ranks_ce = _rank_normalize_01(ce_values)
    ranks_logit = _rank_normalize_01(logit_values)
    ranks_byte = _rank_normalize_01(byte_values)

    unit_scores: Dict[str, float] = {}
    block_scores: Dict[int, Dict[str, float]] = {}
    for idx, row in enumerate(unit_rows):
        r_ce = float(ranks_ce[idx])
        r_logit = float(ranks_logit[idx])
        r_byte = float(ranks_byte[idx])
        sens = (
            float(weights['sens_weight_ce']) * r_ce
            + float(weights['sens_weight_logit']) * r_logit
            + float(weights['sens_weight_byte']) * r_byte
        )
        row['rank_ce'] = float(r_ce)
        row['rank_logit'] = float(r_logit)
        row['rank_byte'] = float(r_byte)
        row['sens'] = float(sens)
        unit_scores[str(row['unit_key'])] = float(sens)
        block_index = int(row['block_index'])
        block_scores.setdefault(block_index, {})
        block_scores[block_index][str(row['bundle_type'])] = float(sens)

    return {
        'weights': weights,
        'units': unit_rows,
        'unit_scores': unit_scores,
        'block_scores': block_scores,
    }


@torch.no_grad()
def _validate_blockwise_sensitivity_alignment(
    model: nn.Module,
    model_name: str,
    sensitivity_report: Dict[str, Any],
    block_sensitivity_scores: Dict[int, Dict[str, float]],
    sensitivity_report_path: str,
) -> None:
    expected_block_indices = sorted({int(entry['block_index']) for entry in _iter_supported_transformer_blocks(model)})
    expected_set = set(expected_block_indices)

    provided_block_indices = sorted(
        int(block_idx)
        for block_idx, score_item in block_sensitivity_scores.items()
        if isinstance(score_item, dict) and 'attn' in score_item and 'mlp' in score_item
    )
    provided_set = set(provided_block_indices)

    missing_blocks = sorted(expected_set - provided_set)
    extra_blocks = sorted(provided_set - expected_set)

    reported_model = str(sensitivity_report.get('model', '')).strip()
    target_model = str(model_name).strip()

    if not missing_blocks and not extra_blocks and (
        not reported_model or reported_model == target_model
    ):
        return

    message_lines: List[str] = [
        'Sensitivity report is not aligned with the current compression model.',
        f'  report_path={str(sensitivity_report_path)}',
    ]
    if reported_model:
        message_lines.append(f'  report_model={reported_model}')
    message_lines.append(f'  target_model={target_model}')
    message_lines.append(
        f'  expected_blocks={len(expected_block_indices)} '
        f'[{expected_block_indices[0] if expected_block_indices else "n/a"}..{expected_block_indices[-1] if expected_block_indices else "n/a"}]'
    )
    message_lines.append(
        f'  report_blocks={len(provided_block_indices)} '
        f'[{provided_block_indices[0] if provided_block_indices else "n/a"}..{provided_block_indices[-1] if provided_block_indices else "n/a"}]'
    )
    if missing_blocks:
        message_lines.append(
            f'  missing_blocks_in_report(count={len(missing_blocks)}): '
            + ', '.join(str(x) for x in missing_blocks[:24])
        )
    if extra_blocks:
        message_lines.append(
            f'  extra_blocks_in_report(count={len(extra_blocks)}): '
            + ', '.join(str(x) for x in extra_blocks[:24])
        )
    message_lines.append(
        'Use a sensitivity report generated by `mode=analyze_sensitivity` with the same `--model` and checkpoint.'
    )
    raise ValueError('\n'.join(message_lines))


@torch.no_grad()
def _iter_vit_target_tensors(model: nn.Module, include_head_linear: bool = False) -> List[Dict[str, Any]]:
    targets: List[Dict[str, Any]] = []
    for bundle in _iter_vit_block_bundle_names(model):
        block_idx = int(bundle['block_index'])
        stage_idx = int(bundle.get('stage_index', -1))
        stage_block_idx = int(bundle.get('stage_block_index', block_idx))
        block_path = str(bundle.get('block_path', f'blocks.{block_idx}'))
        targets.extend(
            [
                {
                    'block_index': int(block_idx),
                    'stage_index': int(stage_idx),
                    'stage_block_index': int(stage_block_idx),
                    'block_path': str(block_path),
                    'bundle_type': 'attn',
                    'unit_key': f'block_{block_idx}.attn',
                    'tensor_role': 'qkv',
                    'param_name': str(bundle['attn_qkv']),
                },
                {
                    'block_index': int(block_idx),
                    'stage_index': int(stage_idx),
                    'stage_block_index': int(stage_block_idx),
                    'block_path': str(block_path),
                    'bundle_type': 'attn',
                    'unit_key': f'block_{block_idx}.attn',
                    'tensor_role': 'proj',
                    'param_name': str(bundle['attn_proj']),
                },
                {
                    'block_index': int(block_idx),
                    'stage_index': int(stage_idx),
                    'stage_block_index': int(stage_block_idx),
                    'block_path': str(block_path),
                    'bundle_type': 'mlp',
                    'unit_key': f'block_{block_idx}.mlp',
                    'tensor_role': 'fc1',
                    'param_name': str(bundle['mlp_fc1']),
                },
                {
                    'block_index': int(block_idx),
                    'stage_index': int(stage_idx),
                    'stage_block_index': int(stage_block_idx),
                    'block_path': str(block_path),
                    'bundle_type': 'mlp',
                    'unit_key': f'block_{block_idx}.mlp',
                    'tensor_role': 'fc2',
                    'param_name': str(bundle['mlp_fc2']),
                },
            ]
        )
    if bool(include_head_linear):
        for head_weight_name in _iter_head_linear_weight_names(model):
            targets.append(
                {
                    'block_index': -1,
                    'stage_index': -1,
                    'stage_block_index': -1,
                    'block_path': 'head',
                    'bundle_type': 'head',
                    'unit_key': f'head::{head_weight_name}',
                    'tensor_role': 'head',
                    'param_name': str(head_weight_name),
                }
            )
    return targets


@torch.no_grad()
def compute_tensor_transform_scores(
    canonical_state: Dict[str, torch.Tensor],
    tensor_entries: Sequence[Dict[str, Any]],
    unit_sensitivity_scores: Dict[str, float],
    eps: float = 1e-12,
    compute_device: Optional[torch.device] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    importance_alpha: float = 2.0,
    importance_quantile_q: float = 0.95,
    sens_beta: float = 1.0,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    total_entries = int(len(tensor_entries))
    work_device = torch.device('cpu') if compute_device is None else torch.device(compute_device)
    _emit_progress(progress_callback, f'Compute tensor scores on device: {work_device}')
    if not math.isfinite(float(importance_alpha)) or float(importance_alpha) <= 0.0:
        raise ValueError(f'`importance_alpha` must be a positive finite float, got: {importance_alpha!r}')
    if not math.isfinite(float(importance_quantile_q)) or not (0.0 < float(importance_quantile_q) <= 1.0):
        raise ValueError(
            f'`importance_quantile_q` must be in (0, 1], got: {importance_quantile_q!r}'
        )
    if not math.isfinite(float(sens_beta)) or float(sens_beta) < 0.0:
        raise ValueError(f'`sens_beta` must be a non-negative finite float, got: {sens_beta!r}')
    if len(unit_sensitivity_scores) > 0:
        default_sens = float(np.mean([float(x) for x in unit_sensitivity_scores.values()]))
    else:
        default_sens = 0.0
    warned_missing_unit_keys: set = set()
    for idx, entry in enumerate(tensor_entries, start=1):
        param_name = str(entry['param_name'])
        unit_key = str(entry['unit_key'])
        if param_name not in canonical_state:
            raise KeyError(f'Missing parameter in canonical state: {param_name}')
        if unit_key in unit_sensitivity_scores:
            sens = float(unit_sensitivity_scores[unit_key])
        else:
            sens = float(default_sens)
            if unit_key not in warned_missing_unit_keys:
                warned_missing_unit_keys.add(unit_key)
                _emit_progress(
                    progress_callback,
                    f'Warning: missing sensitivity score for unit `{unit_key}`; fallback sens={sens:.6g}.',
                )

        tensor = canonical_state[param_name].detach()
        if tensor.device != work_device:
            tensor = tensor.to(device=work_device, dtype=torch.float32, non_blocking=(work_device.type == 'cuda'))
        else:
            tensor = tensor.float()
        tensor = tensor.contiguous()
        coeff, schedules = developmental_transform_forward(tensor.unsqueeze(0))
        abs_coeff = coeff.abs()
        flat_abs = abs_coeff.reshape(-1)
        if int(flat_abs.numel()) <= 0:
            scale = 1.0
        else:
            try:
                scale = float(torch.quantile(flat_abs, float(importance_quantile_q)).item())
            except Exception:
                scale = float(np.percentile(flat_abs.detach().cpu().numpy(), float(importance_quantile_q) * 100.0))
            if not math.isfinite(scale) or scale <= 0.0:
                scale = float(flat_abs.max().item())
            if not math.isfinite(scale) or scale <= 0.0:
                scale = 1.0
        # Importance_t,j = (|c_t,j| / (Q_q + eps))^alpha
        importance = (abs_coeff / float(scale + eps)).pow(float(importance_alpha))
        sens_weight = float(1.0 + float(sens_beta) * float(sens))
        if not math.isfinite(sens_weight) or sens_weight <= 0.0:
            sens_weight = float(eps)
        # Score_t,j = (1 + beta * Sens(u(t))) * Importance_t,j
        score = importance * float(sens_weight)
        records.append(
            {
                'block_index': int(entry['block_index']),
                'bundle_type': str(entry['bundle_type']),
                'unit_key': str(unit_key),
                'tensor_role': str(entry['tensor_role']),
                'param_name': str(param_name),
                'tensor_dense_bytes': int(canonical_state[param_name].numel() * canonical_state[param_name].element_size()),
                'sensitivity': float(sens),
                'sensitivity_beta': float(sens_beta),
                'sensitivity_weight_after_beta': float(sens_weight),
                'importance_alpha': float(importance_alpha),
                'importance_quantile_q': float(importance_quantile_q),
                'scale_quantile': float(scale),
                # Backward-compatible alias used by older reports.
                'scale_p95': float(scale),
                'coeff': coeff.detach().to(device=torch.device('cpu')).contiguous(),
                'schedules': schedules,
                'score': score.detach().to(device=torch.device('cpu')).contiguous(),
            }
        )
        del tensor, coeff, abs_coeff, flat_abs, importance, score
        if idx == 1 or idx == total_entries or (idx % 4) == 0:
            _emit_progress(progress_callback, f'Prepared tensor transform score {idx}/{total_entries}: {param_name}')
    return records


@torch.no_grad()
def _build_threshold_keep_mask(score: torch.Tensor, lambda_value: float, min_tensor_keep_ratio: float) -> torch.Tensor:
    flat_score = score.detach().reshape(-1)
    numel = int(flat_score.numel())
    if numel <= 0:
        return torch.zeros_like(score, dtype=torch.bool)
    keep = flat_score >= float(lambda_value)
    min_ratio = float(max(0.0, min(1.0, min_tensor_keep_ratio)))
    min_keep = int(math.ceil(min_ratio * float(numel)))
    min_keep = max(0, min(min_keep, numel))
    current_keep = int(keep.sum().item())
    if current_keep < min_keep:
        topk_indices = torch.topk(flat_score, k=min_keep, largest=True, sorted=False).indices
        keep = torch.zeros_like(keep, dtype=torch.bool)
        keep[topk_indices] = True
    return keep.reshape(score.shape)


@torch.no_grad()
def _encode_blockwise_thresholded_archive(
    canonical_state: Dict[str, torch.Tensor],
    compressed_weight_names: Sequence[str],
    tensor_records: Sequence[Dict[str, Any]],
    model_name: str,
    quant_bits: int,
    gauge_report: Dict[str, Any],
    dense_state_bytes: int,
    compressed_weight_dense_bytes: int,
    compressed_scope_fp32_bytes: int,
    exact_tensor_dense_bytes: int,
    lambda_value: float,
    min_tensor_keep_ratio: float,
    save_path: Optional[str],
    stage_label: str = 'encode',
    progress_callback: Optional[Callable[[str], None]] = None,
    collect_tensor_stats: bool = True,
) -> Dict[str, Any]:
    compressed_name_set = set(str(x) for x in compressed_weight_names)
    save_to_disk = bool(save_path)
    tmp_save_path: Optional[str] = None
    if save_to_disk:
        save_path = str(save_path)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        tmp_save_path = _build_temp_archive_path(save_path)
        zip_target: Any = tmp_save_path
        buffer = None
    else:
        buffer = io.BytesIO()
        zip_target = buffer

    manifest: Dict[str, Any] = {
        'format': ARCHIVE_FORMAT,
        'model_name': str(model_name),
        'codec_allocation_mode': 'blockwise_thresholded',
        'quant_bits': int(quant_bits),
        'include_head_linear': bool(any(str(item.get('bundle_type', '')) == 'head' for item in tensor_records)),
        'dense_state_bytes': int(dense_state_bytes),
        'compressed_weight_dense_bytes': int(compressed_weight_dense_bytes),
        'compressed_scope_fp32_bytes': int(compressed_scope_fp32_bytes),
        'exact_tensor_dense_bytes': int(exact_tensor_dense_bytes),
        'gauge_report': gauge_report,
        'lambda': float(lambda_value),
        'min_tensor_keep_ratio': float(min_tensor_keep_ratio),
        'stacks': {},
        'tensor_stacks': {},
        'exact_tensors': [],
    }
    tensor_stats: List[Dict[str, Any]] = []
    min_ratio_clamped = float(max(0.0, min(1.0, min_tensor_keep_ratio)))
    floor_applied_tensors = 0
    min_target_keep_ratio_observed = 1.0
    min_actual_keep_ratio_observed = 1.0
    _emit_progress(
        progress_callback,
        f'[{stage_label}] Start archive encode: tensors={len(tensor_records)}, lambda={float(lambda_value):.6g}, min_keep={float(min_tensor_keep_ratio):.4f}',
    )

    try:
        with zipfile.ZipFile(zip_target, mode='w', compression=zipfile.ZIP_STORED) as zf:
            exact_idx = 0
            for name, tensor in canonical_state.items():
                if name in compressed_name_set:
                    continue
                meta = serialize_exact_tensor(zf, entry_name=f'exact/{exact_idx:05d}.bin', tensor=tensor.detach().cpu(), codec='lzma')
                meta['name'] = str(name)
                manifest['exact_tensors'].append(meta)
                exact_idx += 1

            total_tensor_records = int(len(tensor_records))
            for record_idx, record in enumerate(tensor_records, start=1):
                # `coeff` and `score` are already prepared as CPU contiguous tensors
                # in `compute_tensor_transform_scores`; avoid redundant per-iter copies.
                coeff = record['coeff']
                score = record['score']
                total_coeff = int(coeff.numel())
                threshold_keep_before_floor = 0
                min_keep_floor_count = 0
                floor_applied = False
                if total_coeff > 0:
                    flat_score = score.detach().reshape(-1)
                    threshold_keep_before_floor = int((flat_score >= float(lambda_value)).sum().item())
                    min_keep_floor_count = int(math.ceil(min_ratio_clamped * float(total_coeff)))
                    min_keep_floor_count = max(0, min(min_keep_floor_count, total_coeff))
                    floor_applied = bool(threshold_keep_before_floor < min_keep_floor_count)
                keep_mask = _build_threshold_keep_mask(
                    score=score,
                    lambda_value=float(lambda_value),
                    min_tensor_keep_ratio=float(min_tensor_keep_ratio),
                )
                masked_coeff = coeff * keep_mask.to(dtype=coeff.dtype)
                target_nonzero = int(keep_mask.sum().item())
                target_keep_ratio = float(target_nonzero / max(float(total_coeff), 1.0))
                safe_name = str(record['param_name']).replace('.', '_')
                meta = encode_progressive_bitplane_stack(
                    zf=zf,
                    prefix=f'tensor_stacks/{safe_name}',
                    coeff=masked_coeff,
                    target_nonzero=int(target_nonzero),
                    quant_bits=int(quant_bits),
                    codec='lzma',
                )
                actual_nonzero = int(meta['actual_nonzero'])
                keep_ratio = float(actual_nonzero / max(float(total_coeff), 1.0))
                min_target_keep_ratio_observed = min(min_target_keep_ratio_observed, target_keep_ratio)
                min_actual_keep_ratio_observed = min(min_actual_keep_ratio_observed, keep_ratio)
                if floor_applied:
                    floor_applied_tensors += 1
                tensor_entry = {
                    'param_name': str(record['param_name']),
                    'stack_shape': list(coeff.shape),
                    'transform': 'finite_length_haar_3d',
                    'schedules': record['schedules'],
                    'block_index': int(record['block_index']),
                    'bundle_type': str(record['bundle_type']),
                    'tensor_role': str(record['tensor_role']),
                    'unit_key': str(record['unit_key']),
                    'sensitivity': float(record['sensitivity']),
                    'sensitivity_beta': float(record.get('sensitivity_beta', 1.0)),
                    'sensitivity_weight_after_beta': float(record.get('sensitivity_weight_after_beta', 1.0 + float(record['sensitivity']))),
                    'importance_alpha': float(record.get('importance_alpha', 2.0)),
                    'importance_quantile_q': float(record.get('importance_quantile_q', 0.95)),
                    'scale_quantile': float(record.get('scale_quantile', record.get('scale_p95', 1.0))),
                    'scale_p95': float(record.get('scale_p95', record.get('scale_quantile', 1.0))),
                    'lambda': float(lambda_value),
                    'threshold_keep_before_floor': int(threshold_keep_before_floor),
                    'min_keep_floor_count': int(min_keep_floor_count),
                    'floor_applied': bool(floor_applied),
                    'target_nonzero': int(target_nonzero),
                    'target_keep_ratio': float(target_keep_ratio),
                    'actual_nonzero': int(actual_nonzero),
                    'total_coeff': int(total_coeff),
                    'keep_ratio': float(keep_ratio),
                    'code': meta,
                }
                manifest['tensor_stacks'][str(record['param_name'])] = tensor_entry
                if collect_tensor_stats:
                    tensor_stats.append(
                        {
                            'param_name': str(record['param_name']),
                            'block_index': int(record['block_index']),
                            'bundle_type': str(record['bundle_type']),
                            'tensor_role': str(record['tensor_role']),
                            'unit_key': str(record['unit_key']),
                            'sensitivity': float(record['sensitivity']),
                            'sensitivity_beta': float(record.get('sensitivity_beta', 1.0)),
                            'sensitivity_weight_after_beta': float(record.get('sensitivity_weight_after_beta', 1.0 + float(record['sensitivity']))),
                            'importance_alpha': float(record.get('importance_alpha', 2.0)),
                            'importance_quantile_q': float(record.get('importance_quantile_q', 0.95)),
                            'tensor_dense_bytes': int(record['tensor_dense_bytes']),
                            'threshold_keep_before_floor': int(threshold_keep_before_floor),
                            'min_keep_floor_count': int(min_keep_floor_count),
                            'floor_applied': bool(floor_applied),
                            'target_nonzero': int(target_nonzero),
                            'target_keep_ratio': float(target_keep_ratio),
                            'actual_nonzero': int(actual_nonzero),
                            'total_coeff': int(total_coeff),
                            'keep_ratio': float(keep_ratio),
                        }
                    )
                if record_idx == 1 or record_idx == total_tensor_records or (record_idx % 4) == 0:
                    _emit_progress(
                        progress_callback,
                        f'[{stage_label}] Encoded tensor {record_idx}/{total_tensor_records}: {record["param_name"]} '
                        f'(keep={keep_ratio:.4f}, nz={actual_nonzero}/{total_coeff}, floor={"yes" if floor_applied else "no"})',
                    )

            manifest['min_tensor_keep_floor_stats'] = {
                'requested_ratio': float(min_ratio_clamped),
                'tensors_total': int(total_tensor_records),
                'floor_applied_tensors': int(floor_applied_tensors),
                'floor_applied_ratio': float(floor_applied_tensors / max(float(total_tensor_records), 1.0)),
                'min_target_keep_ratio_observed': (
                    float(min_target_keep_ratio_observed) if total_tensor_records > 0 else None
                ),
                'min_actual_keep_ratio_observed': (
                    float(min_actual_keep_ratio_observed) if total_tensor_records > 0 else None
                ),
            }
            zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2, ensure_ascii=False).encode('utf-8'))
        if save_to_disk and tmp_save_path is not None and save_path is not None:
            os.replace(tmp_save_path, save_path)
    finally:
        if save_to_disk and tmp_save_path is not None and os.path.exists(tmp_save_path):
            try:
                os.remove(tmp_save_path)
            except OSError:
                pass

    if save_to_disk:
        archive_bytes = int(os.path.getsize(str(save_path)))
    else:
        # `getvalue()` copies the whole buffer; `tell()` gives exact size without copy.
        archive_bytes = int(buffer.tell()) if buffer is not None else 0
    return {
        'archive_bytes': int(archive_bytes),
        'manifest': manifest,
        'tensor_stats': tensor_stats,
    }


@torch.no_grad()
def search_lambda_for_target_mb(
    encoder_fn: Callable[[float], Dict[str, Any]],
    target_archive_mb: float,
    target_archive_tolerance_ratio: float = 0.005,
    lambda_lower: float = 0.0,
    lambda_upper: float = 1.0,
    lambda_probe_start: float = 1e-8,
    max_iters: int = 26,
    progress_callback: Optional[Callable[[str], None]] = None,
    retain_eval_result: bool = True,
) -> Dict[str, Any]:
    target_mb = float(target_archive_mb)
    if target_mb <= 0.0:
        raise ValueError('`target_archive_mb` must be > 0 for blockwise_thresholded mode.')
    target_bytes = int(round(target_mb * 1024.0 * 1024.0))
    tol_ratio = float(max(0.0, target_archive_tolerance_ratio))
    tolerance_bytes = max(1, int(round(float(target_bytes) * tol_ratio)))
    probe_start = float(lambda_probe_start)
    if not math.isfinite(probe_start) or probe_start <= 0.0:
        raise ValueError(f'`lambda_probe_start` must be a positive finite float, got: {lambda_probe_start!r}')

    lo = float(max(0.0, lambda_lower))
    hi_hint = float(max(lo, lambda_upper))
    max_bisect_iters = max(1, int(max_iters))
    cache: Dict[str, Dict[str, Any]] = {}
    eval_trace: List[Dict[str, Any]] = []

    def _eval(lambda_value: float) -> Dict[str, Any]:
        key = f'{float(lambda_value):.12g}'
        if key in cache:
            return cache[key]
        start_t = time.perf_counter()
        result = encoder_fn(float(lambda_value))
        elapsed_sec = float(max(0.0, time.perf_counter() - start_t))
        archive_bytes = int(result['archive_bytes'])
        stored_result = result if retain_eval_result else {'archive_bytes': int(archive_bytes)}
        merged = {
            'lambda': float(lambda_value),
            'archive_bytes': int(archive_bytes),
            'elapsed_sec': float(elapsed_sec),
            'result': stored_result,
        }
        cache[key] = merged
        eval_trace.append(
            {
                'lambda': float(lambda_value),
                'archive_bytes': int(archive_bytes),
                'elapsed_sec': float(elapsed_sec),
            }
        )
        _emit_progress(
            progress_callback,
            f'lambda_eval: lambda={float(lambda_value):.6g}, archive_bytes={int(archive_bytes)}, target_bytes={target_bytes}, '
            f'elapsed={elapsed_sec:.3f}s',
        )
        return merged

    def _finalize(chosen_eval: Dict[str, Any], converged_flag: bool, iterations_done: int) -> Dict[str, Any]:
        return {
            'lambda_final': float(chosen_eval['lambda']),
            'target_archive_mb': float(target_mb),
            'target_archive_bytes': int(target_bytes),
            'actual_archive_mb': float(int(chosen_eval['archive_bytes']) / (1024.0 * 1024.0)),
            'actual_archive_bytes': int(chosen_eval['archive_bytes']),
            'tolerance_bytes': int(tolerance_bytes),
            'probe_start': float(probe_start),
            'converged': bool(converged_flag),
            'iterations': int(iterations_done),
            'eval_trace': eval_trace,
            'result': chosen_eval['result'],
        }

    _emit_progress(
        progress_callback,
        f'lambda_search: target_mb={target_mb:.4f}, target_bytes={target_bytes}, tolerance_bytes={tolerance_bytes}, '
        f'lambda_lower={lo:.6g}, lambda_upper_hint={hi_hint:.6g}, lambda_probe_start={probe_start:.6g}',
    )
    lo_eval = _eval(lo)
    best_eval = lo_eval

    if abs(int(lo_eval['archive_bytes']) - target_bytes) <= tolerance_bytes:
        return _finalize(lo_eval, converged_flag=True, iterations_done=0)

    # bytes(lambda) is expected to be non-increasing: larger lambda keeps fewer coefficients.
    # If lower bound already undershoots target, return lower-bound candidate.
    if int(lo_eval['archive_bytes']) < target_bytes:
        converged = abs(int(lo_eval['archive_bytes']) - target_bytes) <= tolerance_bytes
        return _finalize(lo_eval, converged_flag=converged, iterations_done=0)

    # Step 1: probe true upper bound by exponentially increasing lambda
    # until first bytes(lambda) <= target_bytes.
    probe_cap = float(hi_hint) if (math.isfinite(hi_hint) and hi_hint > 0.0) else float('inf')
    probe_lambda = float(probe_start if lo <= 0.0 else max(probe_start, lo * 2.0))
    if math.isfinite(probe_cap):
        probe_lambda = float(min(probe_lambda, probe_cap))
    if probe_lambda <= lo and math.isfinite(probe_cap) and probe_cap > lo:
        probe_lambda = float(probe_cap)

    hi_eval: Optional[Dict[str, Any]] = None
    hi = float(lo)
    max_probe_steps = max(16, max_bisect_iters * 6)
    _emit_progress(
        progress_callback,
        f'lambda_probe_hi: start={probe_lambda:.6g}, cap={("inf" if not math.isfinite(probe_cap) else f"{probe_cap:.6g}")}, '
        f'max_probe_steps={max_probe_steps}',
    )

    for probe_step in range(max_probe_steps):
        if probe_lambda <= lo:
            break
        current_eval = _eval(probe_lambda)
        if abs(int(current_eval['archive_bytes']) - target_bytes) < abs(int(best_eval['archive_bytes']) - target_bytes):
            best_eval = current_eval
        _emit_progress(
            progress_callback,
            f'lambda_probe_hi_iter {probe_step + 1}/{max_probe_steps}: '
            f'lambda={probe_lambda:.6g}, bytes={int(current_eval["archive_bytes"])}',
        )
        if int(current_eval['archive_bytes']) <= target_bytes:
            hi = float(probe_lambda)
            hi_eval = current_eval
            break
        lo = float(probe_lambda)
        lo_eval = current_eval
        if math.isfinite(probe_cap) and probe_lambda >= probe_cap:
            break
        next_lambda = float(probe_lambda * 2.0)
        if (not math.isfinite(next_lambda)) or next_lambda <= probe_lambda:
            break
        if math.isfinite(probe_cap):
            next_lambda = float(min(next_lambda, probe_cap))
        if next_lambda <= probe_lambda:
            break
        probe_lambda = next_lambda

    if hi_eval is None:
        # Could not bracket target (or target is infeasible); return best explored point.
        converged = abs(int(best_eval['archive_bytes']) - target_bytes) <= tolerance_bytes
        _emit_progress(
            progress_callback,
            'lambda_probe_hi: failed to find bytes(lambda) <= target; return best explored candidate.',
        )
        return _finalize(best_eval, converged_flag=converged, iterations_done=0)

    if abs(int(hi_eval['archive_bytes']) - target_bytes) < abs(int(best_eval['archive_bytes']) - target_bytes):
        best_eval = hi_eval
    _emit_progress(
        progress_callback,
        f'lambda_probe_hi: bracket ready, lo={lo:.6g} (bytes={int(lo_eval["archive_bytes"])}), '
        f'hi={hi:.6g} (bytes={int(hi_eval["archive_bytes"])})',
    )

    converged = False
    iterations_done = 0
    for step in range(max_bisect_iters):
        iterations_done = step + 1
        lo_bytes = int(lo_eval['archive_bytes'])
        hi_bytes = int(hi_eval['archive_bytes'])
        if lo_bytes == hi_bytes:
            # Plateau: no further split can improve byte target.
            break
        if lo_bytes >= target_bytes >= hi_bytes and (lo_bytes - hi_bytes) <= tolerance_bytes:
            lo_gap = abs(lo_bytes - target_bytes)
            hi_gap = abs(hi_bytes - target_bytes)
            candidate = lo_eval if lo_gap <= hi_gap else hi_eval
            if abs(int(candidate['archive_bytes']) - target_bytes) < abs(int(best_eval['archive_bytes']) - target_bytes):
                best_eval = candidate
            converged = True
            break
        mid = 0.5 * (lo + hi)
        if not math.isfinite(mid) or mid <= lo or mid >= hi:
            break
        mid_eval = _eval(mid)
        _emit_progress(
            progress_callback,
            f'lambda_search_iter {iterations_done}/{max_iters}: mid={mid:.6g}, bytes={int(mid_eval["archive_bytes"])}, '
            f'best_lambda={float(best_eval["lambda"]):.6g}, best_bytes={int(best_eval["archive_bytes"])}',
        )
        if abs(int(mid_eval['archive_bytes']) - target_bytes) < abs(int(best_eval['archive_bytes']) - target_bytes):
            best_eval = mid_eval
        if abs(int(mid_eval['archive_bytes']) - target_bytes) <= tolerance_bytes:
            best_eval = mid_eval
            converged = True
            break
        if int(mid_eval['archive_bytes']) > target_bytes:
            lo = float(mid)
            lo_eval = mid_eval
        else:
            hi = float(mid)
            hi_eval = mid_eval

    return _finalize(
        best_eval,
        converged_flag=bool(converged or (abs(int(best_eval['archive_bytes']) - target_bytes) <= tolerance_bytes)),
        iterations_done=int(iterations_done),
    )


@torch.no_grad()
def compute_bundle_keep_ratios(
    tensor_stats: Sequence[Dict[str, Any]],
    block_sensitivity_scores: Dict[int, Dict[str, float]],
) -> List[Dict[str, Any]]:
    role_order = ('qkv', 'proj', 'fc1', 'fc2')
    by_block: Dict[int, Dict[str, Any]] = {}
    for item in tensor_stats:
        block_index = int(item['block_index'])
        role = str(item['tensor_role'])
        if block_index < 0:
            # Keep-ratio diagnostics are defined for ViT blocks only.
            # Optional extra tensors (for example classifier head) are excluded here.
            continue
        if role not in role_order:
            # Unknown roles are ignored to keep diagnostics stable when optional
            # compression scopes add new tensor families.
            continue
        row = by_block.setdefault(block_index, {'block_index': int(block_index), 'roles': {}})
        row['roles'][role] = {
            'keep_ratio': float(item['keep_ratio']),
            'actual_nonzero': int(item['actual_nonzero']),
            'total_coeff': int(item['total_coeff']),
            'param_name': str(item['param_name']),
        }

    rows: List[Dict[str, Any]] = []
    for block_index in sorted(by_block.keys()):
        block_item = by_block[block_index]
        roles = block_item['roles']
        for role in role_order:
            if role not in roles:
                raise KeyError(f'Block {block_index} missing tensor role `{role}` for keep-ratio diagnostics.')
        sens_block = block_sensitivity_scores.get(int(block_index), None)
        if not isinstance(sens_block, dict) or 'attn' not in sens_block or 'mlp' not in sens_block:
            raise KeyError(f'Missing block sensitivity score for block {block_index}.')
        kept_attn = int(roles['qkv']['actual_nonzero'] + roles['proj']['actual_nonzero'])
        total_attn = int(roles['qkv']['total_coeff'] + roles['proj']['total_coeff'])
        kept_mlp = int(roles['fc1']['actual_nonzero'] + roles['fc2']['actual_nonzero'])
        total_mlp = int(roles['fc1']['total_coeff'] + roles['fc2']['total_coeff'])
        rows.append(
            {
                'block_index': int(block_index),
                'sens_attn': float(sens_block['attn']),
                'sens_mlp': float(sens_block['mlp']),
                'kr_qkv': float(roles['qkv']['keep_ratio']),
                'kr_proj': float(roles['proj']['keep_ratio']),
                'kr_fc1': float(roles['fc1']['keep_ratio']),
                'kr_fc2': float(roles['fc2']['keep_ratio']),
                'kept_qkv': int(roles['qkv']['actual_nonzero']),
                'kept_proj': int(roles['proj']['actual_nonzero']),
                'kept_fc1': int(roles['fc1']['actual_nonzero']),
                'kept_fc2': int(roles['fc2']['actual_nonzero']),
                'total_qkv': int(roles['qkv']['total_coeff']),
                'total_proj': int(roles['proj']['total_coeff']),
                'total_fc1': int(roles['fc1']['total_coeff']),
                'total_fc2': int(roles['fc2']['total_coeff']),
                'kr_attn': float(kept_attn / max(float(total_attn), 1.0)),
                'kr_mlp': float(kept_mlp / max(float(total_mlp), 1.0)),
            }
        )
    return rows


@torch.no_grad()
def write_allocation_report(report: Dict[str, Any], output_path: str) -> str:
    if not output_path:
        raise ValueError('allocation report output path is empty.')
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return str(path)


@torch.no_grad()
def write_allocation_csv(block_rows: Sequence[Dict[str, Any]], output_path: str) -> str:
    if not output_path:
        raise ValueError('allocation csv output path is empty.')
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'block_index',
        'sens_attn',
        'sens_mlp',
        'kr_attn',
        'kr_mlp',
        'kr_qkv',
        'kr_proj',
        'kr_fc1',
        'kr_fc2',
        'kept_qkv',
        'total_qkv',
        'kept_proj',
        'total_proj',
        'kept_fc1',
        'total_fc1',
        'kept_fc2',
        'total_fc2',
    ]
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in block_rows:
            writer.writerow({key: row.get(key, None) for key in fieldnames})
    return str(path)


@torch.no_grad()
def _compress_reconstruct_local_bundle(
    bundle_type: str,
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    tensor_names: Tuple[str, str],
    keep_ratio: float,
    quant_bits: int,
    sensitivity_power_iters: int,
    allocation_strategy: str = 'equal_keep_ratio',
) -> Dict[str, Any]:
    if bundle_type not in ('attn', 'mlp'):
        raise ValueError(f'Unsupported bundle_type: {bundle_type}')
    if allocation_strategy not in ('equal_keep_ratio', 'sensitivity_weighted'):
        raise ValueError(
            f'Unsupported allocation_strategy: {allocation_strategy}. '
            f'Expected one of: equal_keep_ratio, sensitivity_weighted.'
        )
    keep_ratio = float(max(0.0, min(1.0, keep_ratio)))

    stack_a = tensor_a.detach().float().unsqueeze(0).contiguous()
    stack_b = tensor_b.detach().float().unsqueeze(0).contiguous()
    coeff_a, schedules_a = developmental_transform_forward(stack_a)
    coeff_b, schedules_b = developmental_transform_forward(stack_b)

    numel_a = int(coeff_a.numel())
    numel_b = int(coeff_b.numel())
    sensitivity_a = spectral_norm_power_iteration(tensor_b.detach().float(), num_iters=sensitivity_power_iters) ** 2
    sensitivity_b = 1.0
    if allocation_strategy == 'equal_keep_ratio':
        # Same keep ratio for each tensor inside the bundle.
        target_a = int(round(keep_ratio * float(numel_a)))
        target_b = int(round(keep_ratio * float(numel_b)))
        target_a = max(0, min(target_a, numel_a))
        target_b = max(0, min(target_b, numel_b))
        total_target = int(target_a + target_b)
    else:
        # Sensitivity-weighted split inside the bundle with a shared total budget.
        total_target = int(round(keep_ratio * float(numel_a + numel_b)))
        target_a, target_b = sensitivity_weighted_target_counts(
            coeff_abs_list=[coeff_a.abs(), coeff_b.abs()],
            sensitivity_list=[sensitivity_a, sensitivity_b],
            total_target=total_target,
        )
        target_a = max(0, min(int(target_a), numel_a))
        target_b = max(0, min(int(target_b), numel_b))

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode='w', compression=zipfile.ZIP_STORED) as zf:
        meta_a = encode_progressive_bitplane_stack(
            zf=zf,
            prefix=f'{bundle_type}/{tensor_names[0]}',
            coeff=coeff_a.detach().cpu(),
            target_nonzero=int(target_a),
            quant_bits=int(quant_bits),
            codec='lzma',
        )
        meta_b = encode_progressive_bitplane_stack(
            zf=zf,
            prefix=f'{bundle_type}/{tensor_names[1]}',
            coeff=coeff_b.detach().cpu(),
            target_nonzero=int(target_b),
            quant_bits=int(quant_bits),
            codec='lzma',
        )
    local_encoded_bytes = int(len(buffer.getvalue()))

    with zipfile.ZipFile(io.BytesIO(buffer.getvalue()), mode='r') as zf:
        decoded_a = decode_progressive_bitplane_stack(
            zf=zf,
            meta=meta_a,
            device=torch.device('cpu'),
            max_planes=None,
        )
        decoded_b = decode_progressive_bitplane_stack(
            zf=zf,
            meta=meta_b,
            device=torch.device('cpu'),
            max_planes=None,
        )

    recon_a = developmental_transform_inverse(decoded_a, schedules=schedules_a)[0].to(dtype=tensor_a.dtype).cpu().contiguous()
    recon_b = developmental_transform_inverse(decoded_b, schedules=schedules_b)[0].to(dtype=tensor_b.dtype).cpu().contiguous()

    return {
        'reconstructed': {
            tensor_names[0]: recon_a,
            tensor_names[1]: recon_b,
        },
        'local_encoded_bytes': int(local_encoded_bytes),
        'local_codec': {
            'bundle_type': bundle_type,
            'allocation_strategy': str(allocation_strategy),
            'quant_bits': int(quant_bits),
            'keep_ratio': float(keep_ratio),
            'total_target_nonzero': int(total_target),
            'target_nonzero_per_tensor': {
                tensor_names[0]: int(target_a),
                tensor_names[1]: int(target_b),
            },
            'effective_keep_ratio_per_tensor': {
                tensor_names[0]: float(target_a / max(float(numel_a), 1.0)),
                tensor_names[1]: float(target_b / max(float(numel_b), 1.0)),
            },
            'sensitivities': {
                tensor_names[0]: float(sensitivity_a),
                tensor_names[1]: float(sensitivity_b),
            },
            'encode_meta': {
                tensor_names[0]: meta_a,
                tensor_names[1]: meta_b,
            },
            'schedules': {
                tensor_names[0]: schedules_a,
                tensor_names[1]: schedules_b,
            },
        },
    }


@torch.no_grad()
def _compress_reconstruct_local_tensor(
    tensor: torch.Tensor,
    tensor_name: str,
    keep_ratio: float,
    quant_bits: int,
) -> Dict[str, Any]:
    keep_ratio = float(max(0.0, min(1.0, keep_ratio)))
    name_safe = str(tensor_name).replace('/', '_').replace('\\', '_')

    stack = tensor.detach().float().unsqueeze(0).contiguous()
    coeff, schedules = developmental_transform_forward(stack)
    numel = int(coeff.numel())
    target_nonzero = int(round(keep_ratio * float(numel)))
    target_nonzero = max(0, min(target_nonzero, numel))

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode='w', compression=zipfile.ZIP_STORED) as zf:
        meta = encode_progressive_bitplane_stack(
            zf=zf,
            prefix=f'tensor/{name_safe}',
            coeff=coeff.detach().cpu(),
            target_nonzero=int(target_nonzero),
            quant_bits=int(quant_bits),
            codec='lzma',
        )
    local_encoded_bytes = int(len(buffer.getvalue()))

    with zipfile.ZipFile(io.BytesIO(buffer.getvalue()), mode='r') as zf:
        decoded = decode_progressive_bitplane_stack(
            zf=zf,
            meta=meta,
            device=torch.device('cpu'),
            max_planes=None,
        )

    recon = developmental_transform_inverse(decoded, schedules=schedules)[0].to(dtype=tensor.dtype).cpu().contiguous()
    return {
        'reconstructed': recon,
        'local_encoded_bytes': int(local_encoded_bytes),
        'local_codec': {
            'bundle_type': 'single_tensor',
            'tensor_name': str(tensor_name),
            'quant_bits': int(quant_bits),
            'keep_ratio': float(keep_ratio),
            'target_nonzero': int(target_nonzero),
            'effective_keep_ratio': float(target_nonzero / max(float(numel), 1.0)),
            'encode_meta': meta,
            'schedules': schedules,
        },
    }


@torch.no_grad()
def _safe_float(value: Any, default: float = float('nan')) -> float:
    try:
        result = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(result):
        return float(default)
    return float(result)


@torch.no_grad()
def _write_sensitivity_csv(
    blocks_report: Sequence[Dict[str, Any]],
    csv_output_path: str,
    allocation: str,
    head_units: Optional[Sequence[Dict[str, Any]]] = None,
) -> None:
    Path(csv_output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'block_index',
        'bundle_type',
        'allocation',
        'unit_key',
        'param_name',
        'param_count',
        'dense_fp32_bytes',
        'local_encoded_bytes',
        'saved_bytes',
        'delta_ce',
        'delta_top1',
        'rel_logit_mse',
        'relative_weight_distortion',
        'sensitivity_dist',
        'sensitivity_byte',
        'heuristic_score',
    ]
    with open(csv_output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for block_item in blocks_report:
            block_index = int(block_item.get('block', block_item.get('block_index', -1)))
            for bundle_type in ('attn', 'mlp'):
                bundle = block_item.get(bundle_type, {})
                writer.writerow(
                    {
                        'block_index': int(block_index),
                        'bundle_type': str(bundle_type),
                        'allocation': str(allocation),
                        'unit_key': '',
                        'param_name': '',
                        'param_count': int(bundle.get('param_count', 0)),
                        'dense_fp32_bytes': int(bundle.get('dense_fp32_bytes', 0)),
                        'local_encoded_bytes': int(bundle.get('local_encoded_bytes', 0)),
                        'saved_bytes': int(bundle.get('saved_bytes', 0)),
                        'delta_ce': _safe_float(bundle.get('delta_ce', float('nan'))),
                        'delta_top1': _safe_float(bundle.get('delta_top1', float('nan'))),
                        'rel_logit_mse': _safe_float(bundle.get('rel_logit_mse', float('nan'))),
                        'relative_weight_distortion': _safe_float(bundle.get('relative_weight_distortion', float('nan'))),
                        'sensitivity_dist': _safe_float(bundle.get('sensitivity_dist', float('nan'))),
                        'sensitivity_byte': _safe_float(bundle.get('sensitivity_byte', float('nan'))),
                        'heuristic_score': _safe_float(bundle.get('heuristic_score', float('nan'))),
                    }
                )
        for unit in (head_units or []):
            writer.writerow(
                {
                    'block_index': int(unit.get('block_index', -1)),
                    'bundle_type': 'head',
                    'allocation': str(allocation),
                    'unit_key': str(unit.get('unit_key', '')),
                    'param_name': str(unit.get('param_name', '')),
                    'param_count': int(unit.get('param_count', 0)),
                    'dense_fp32_bytes': int(unit.get('dense_fp32_bytes', 0)),
                    'local_encoded_bytes': int(unit.get('local_encoded_bytes', 0)),
                    'saved_bytes': int(unit.get('saved_bytes', 0)),
                    'delta_ce': _safe_float(unit.get('delta_ce', float('nan'))),
                    'delta_top1': _safe_float(unit.get('delta_top1', float('nan'))),
                    'rel_logit_mse': _safe_float(unit.get('rel_logit_mse', float('nan'))),
                    'relative_weight_distortion': _safe_float(unit.get('relative_weight_distortion', float('nan'))),
                    'sensitivity_dist': _safe_float(unit.get('sensitivity_dist', float('nan'))),
                    'sensitivity_byte': _safe_float(unit.get('sensitivity_byte', float('nan'))),
                    'heuristic_score': _safe_float(unit.get('heuristic_score', float('nan'))),
                }
            )


@torch.no_grad()
def analyze_vit_block_sensitivity(
    model: nn.Module,
    model_name: str,
    data_root: str,
    split: str,
    batch_size: int,
    workers: int,
    class_map: str,
    pin_memory: bool,
    crop_pct: Optional[float],
    interpolation: str,
    no_prefetcher: bool,
    device: torch.device,
    sensitivity_calib_size: int,
    sensitivity_keep_ratio: float,
    sensitivity_quant_bits: int,
    sensitivity_allocation: str,
    gauge_proj_dim: int,
    gauge_pca_dim: int,
    sensitivity_power_iters: int,
    enable_attn_rotation_gauge: bool,
    verify_gauge: bool,
    print_freq: int,
    include_head_linear: bool = False,
    sensitivity_csv_output: Optional[str] = None,
) -> Dict[str, Any]:
    from devdna_eval import create_official_imagenet_loader, evaluate_for_sensitivity

    eps = 1e-12
    # The main experiment is expected to run with allocation=equal_keep_ratio so that
    # bundle sensitivity is measured without heuristic intra-bundle budget reallocation.
    # sensitivity_weighted is kept as an optional codec-as-is control condition.
    model.eval()
    canonical_state, _, _, gauge_report = canonicalize_vit_state_dict(
        model=model,
        device=device,
        gauge_proj_dim=gauge_proj_dim,
        gauge_pca_dim=gauge_pca_dim,
        enable_attn_rotation_gauge=enable_attn_rotation_gauge,
        verify_gauge=verify_gauge,
    )
    bundle_entries = _iter_vit_block_bundle_names(model)
    eval_model = _build_reference_model_from_state(model_name=model_name, canonical_state=canonical_state, device=device)
    loader, data_config = create_official_imagenet_loader(
        model=eval_model,
        data_root=data_root,
        split=split,
        batch_size=batch_size,
        workers=workers,
        class_map=class_map,
        pin_memory=pin_memory,
        crop_pct=crop_pct,
        interpolation=interpolation,
        no_prefetcher=no_prefetcher,
        device=device,
        use_test_size=True,
        enforce_official_preprocess=True,
    )

    head_weight_names: List[str] = _iter_head_linear_weight_names(model) if bool(include_head_linear) else []
    total_bundle_evals = int(len(bundle_entries) * 2)
    total_head_evals = int(len(head_weight_names))
    total_perturb_evals = int(total_bundle_evals + total_head_evals)
    print(
        "[analyze_sensitivity] start: "
        f"model={model_name}, blocks={len(bundle_entries)}, "
        f"calib_size={int(sensitivity_calib_size)}, keep_ratio={float(sensitivity_keep_ratio):.6g}, "
        f"quant_bits={int(sensitivity_quant_bits)}, allocation={str(sensitivity_allocation)}, "
        f"perturb_evals={total_perturb_evals}",
        flush=True,
    )
    print(
        f"[analyze_sensitivity] reference eval (0/{total_perturb_evals}) started...",
        flush=True,
    )
    reference_start_time = time.perf_counter()
    reference_eval = evaluate_for_sensitivity(
        model=eval_model,
        loader=loader,
        device=device,
        max_samples=int(sensitivity_calib_size),
        print_freq=print_freq,
        reference_logits_cpu=None,
    )
    print(
        "[analyze_sensitivity] reference eval finished: "
        f"top1={float(reference_eval['top1']):.4f}, loss={float(reference_eval['loss']):.6f}, "
        f"elapsed={float(time.perf_counter() - reference_start_time):.1f}s",
        flush=True,
    )
    reference_logits_cpu = reference_eval.pop('reference_logits_cpu')
    if int(reference_eval['samples']) != int(sensitivity_calib_size):
        raise RuntimeError(
            f'Reference samples mismatch: expected {sensitivity_calib_size}, got {reference_eval["samples"]}'
        )

    blocks_report: List[Dict[str, Any]] = []
    head_units: List[Dict[str, Any]] = []
    bundle_eval_index = 0

    for entry in bundle_entries:
        block_idx = int(entry['block_index'])
        heuristics = _compute_block_heuristic_scores(
            canonical_state=canonical_state,
            bundle_names=entry,
            sensitivity_power_iters=sensitivity_power_iters,
        )
        block_item: Dict[str, Any] = {'block': block_idx, 'block_index': block_idx}

        for bundle_type in ('attn', 'mlp'):
            bundle_eval_index += 1
            bundle_start_time = time.perf_counter()
            print(
                "[analyze_sensitivity] "
                f"perturb eval ({bundle_eval_index}/{total_perturb_evals}) "
                f"block={block_idx}, bundle={bundle_type} started...",
                flush=True,
            )
            if bundle_type == 'attn':
                name_a = str(entry['attn_qkv'])
                name_b = str(entry['attn_proj'])
                tensor_names = ('qkv', 'proj')
            else:
                name_a = str(entry['mlp_fc1'])
                name_b = str(entry['mlp_fc2'])
                tensor_names = ('fc1', 'fc2')

            ref_a = canonical_state[name_a]
            ref_b = canonical_state[name_b]
            local_result = _compress_reconstruct_local_bundle(
                bundle_type=bundle_type,
                tensor_a=ref_a,
                tensor_b=ref_b,
                tensor_names=tensor_names,
                keep_ratio=sensitivity_keep_ratio,
                quant_bits=sensitivity_quant_bits,
                sensitivity_power_iters=sensitivity_power_iters,
                allocation_strategy=sensitivity_allocation,
            )
            perturbed_state = {name: tensor.detach().cpu().clone() for name, tensor in canonical_state.items()}
            recon_a = local_result['reconstructed'][tensor_names[0]]
            recon_b = local_result['reconstructed'][tensor_names[1]]
            perturbed_state[name_a] = recon_a
            perturbed_state[name_b] = recon_b

            eval_model.load_state_dict(perturbed_state, strict=True)
            perturbed_eval = evaluate_for_sensitivity(
                model=eval_model,
                loader=loader,
                device=device,
                max_samples=int(sensitivity_calib_size),
                print_freq=0,
                reference_logits_cpu=reference_logits_cpu,
            )
            if int(perturbed_eval['samples']) != int(sensitivity_calib_size):
                raise RuntimeError(
                    f'Perturbed samples mismatch: expected {sensitivity_calib_size}, got {perturbed_eval["samples"]}'
                )

            delta_ce = float(perturbed_eval['loss'] - reference_eval['loss'])
            delta_top1 = float(reference_eval['top1'] - perturbed_eval['top1'])
            delta_top5 = float(reference_eval['top5'] - perturbed_eval['top5'])
            rel_logit_mse = float(perturbed_eval['rel_logit_mse'])
            relative_weight_distortion = _compute_bundle_relative_distortion(
                reference_tensors=[ref_a, ref_b],
                reconstructed_tensors=[recon_a, recon_b],
                eps=eps,
            )
            param_count = int(ref_a.numel() + ref_b.numel())
            dense_fp32_bytes = int(4 * param_count)
            local_encoded_bytes = int(local_result['local_encoded_bytes'])
            saved_bytes = int(dense_fp32_bytes - local_encoded_bytes)

            bundle_report = {
                'bundle_type': str(bundle_type),
                'param_names': [name_a, name_b],
                'param_count': int(param_count),
                'dense_fp32_bytes': int(dense_fp32_bytes),
                'local_encoded_bytes': int(local_encoded_bytes),
                'saved_bytes': int(saved_bytes),
                'heuristic_score': float(heuristics[bundle_type]),
                'delta_ce': float(delta_ce),
                'delta_top1': float(delta_top1),
                'delta_top5': float(delta_top5),
                'rel_logit_mse': float(rel_logit_mse),
                'relative_weight_distortion': float(relative_weight_distortion),
                'sensitivity_dist': float(delta_ce / (relative_weight_distortion + eps)),
                'sensitivity_byte': float(delta_ce / (float(saved_bytes) + eps)),
                'local_codec': local_result['local_codec'],
            }
            block_item[bundle_type] = bundle_report


            if device.type == 'cuda':
                torch.cuda.empty_cache()
            print(
                "[analyze_sensitivity] "
                f"perturb eval ({bundle_eval_index}/{total_perturb_evals}) "
                f"block={block_idx}, bundle={bundle_type} done: "
                f"delta_ce={float(delta_ce):.6f}, delta_top1={float(delta_top1):.4f}, "
                f"rel_logit_mse={float(rel_logit_mse):.6f}, "
                f"elapsed={float(time.perf_counter() - bundle_start_time):.1f}s",
                flush=True,
            )

        blocks_report.append(block_item)

    if bool(include_head_linear):
        for head_weight_name in head_weight_names:
            bundle_eval_index += 1
            head_start_time = time.perf_counter()
            print(
                "[analyze_sensitivity] "
                f"perturb eval ({bundle_eval_index}/{total_perturb_evals}) "
                f"head={head_weight_name} started...",
                flush=True,
            )
            if head_weight_name not in canonical_state:
                continue
            ref_head = canonical_state[head_weight_name]
            local_result = _compress_reconstruct_local_tensor(
                tensor=ref_head,
                tensor_name=head_weight_name,
                keep_ratio=sensitivity_keep_ratio,
                quant_bits=sensitivity_quant_bits,
            )
            perturbed_state = {name: tensor.detach().cpu().clone() for name, tensor in canonical_state.items()}
            recon_head = local_result['reconstructed']
            perturbed_state[head_weight_name] = recon_head

            eval_model.load_state_dict(perturbed_state, strict=True)
            perturbed_eval = evaluate_for_sensitivity(
                model=eval_model,
                loader=loader,
                device=device,
                max_samples=int(sensitivity_calib_size),
                print_freq=0,
                reference_logits_cpu=reference_logits_cpu,
            )
            if int(perturbed_eval['samples']) != int(sensitivity_calib_size):
                raise RuntimeError(
                    f'Perturbed samples mismatch: expected {sensitivity_calib_size}, got {perturbed_eval["samples"]}'
                )

            delta_ce = float(perturbed_eval['loss'] - reference_eval['loss'])
            delta_top1 = float(reference_eval['top1'] - perturbed_eval['top1'])
            delta_top5 = float(reference_eval['top5'] - perturbed_eval['top5'])
            rel_logit_mse = float(perturbed_eval['rel_logit_mse'])
            relative_weight_distortion = _compute_bundle_relative_distortion(
                reference_tensors=[ref_head],
                reconstructed_tensors=[recon_head],
                eps=eps,
            )
            param_count = int(ref_head.numel())
            dense_fp32_bytes = int(4 * param_count)
            local_encoded_bytes = int(local_result['local_encoded_bytes'])
            saved_bytes = int(dense_fp32_bytes - local_encoded_bytes)
            heuristic_score = float(
                spectral_norm_power_iteration(ref_head.float(), num_iters=sensitivity_power_iters) ** 2
            )
            unit_key = f'head::{head_weight_name}'

            head_report = {
                'block_index': -1,
                'bundle_type': 'head',
                'unit_key': unit_key,
                'param_name': str(head_weight_name),
                'param_count': int(param_count),
                'dense_fp32_bytes': int(dense_fp32_bytes),
                'local_encoded_bytes': int(local_encoded_bytes),
                'saved_bytes': int(saved_bytes),
                'heuristic_score': float(heuristic_score),
                'delta_ce': float(delta_ce),
                'delta_top1': float(delta_top1),
                'delta_top5': float(delta_top5),
                'rel_logit_mse': float(rel_logit_mse),
                'relative_weight_distortion': float(relative_weight_distortion),
                'sensitivity_dist': float(delta_ce / (relative_weight_distortion + eps)),
                'sensitivity_byte': float(delta_ce / (float(saved_bytes) + eps)),
                'local_codec': local_result['local_codec'],
            }
            head_units.append(head_report)

            if device.type == 'cuda':
                torch.cuda.empty_cache()
            print(
                "[analyze_sensitivity] "
                f"perturb eval ({bundle_eval_index}/{total_perturb_evals}) "
                f"head={head_weight_name} done: "
                f"delta_ce={float(delta_ce):.6f}, delta_top1={float(delta_top1):.4f}, "
                f"rel_logit_mse={float(rel_logit_mse):.6f}, "
                f"elapsed={float(time.perf_counter() - head_start_time):.1f}s",
                flush=True,
            )


    notes: List[str] = [
        'Main experiment must use allocation=equal_keep_ratio.',
        'sensitivity_weighted is codec-as-is control only.',
    ]

    csv_artifact: Optional[str] = None
    if sensitivity_csv_output:
        try:
            _write_sensitivity_csv(
                blocks_report=blocks_report,
                csv_output_path=str(sensitivity_csv_output),
                allocation=str(sensitivity_allocation),
                head_units=head_units,
            )
            csv_artifact = str(sensitivity_csv_output)
        except Exception as exc:
            warning = f'CSV generation failed: {exc}'
            print(f'Warning: {warning}')
            notes.append(warning)


    summary = {
        'num_blocks': int(len(blocks_report)),
        'notes': notes,
        'include_head_linear': bool(include_head_linear),
        'num_head_units': int(len(head_units)),
    }

    return {
        'mode': 'analyze_sensitivity',
        'model': str(model_name),
        'allocation': str(sensitivity_allocation),
        'include_head_linear': bool(include_head_linear),
        'device': str(device),
        'calibration': {
            'data_root': str(data_root),
            'split': str(split),
            'samples': int(sensitivity_calib_size),
            'batch_size': int(batch_size),
            'workers': int(workers),
            'fp32_eval': True,
            'amp_enabled': False,
            'data_config': data_config,
        },
        'compression_recipe': {
            'keep_ratio': float(sensitivity_keep_ratio),
            'quant_bits': int(sensitivity_quant_bits),
            'allocation': str(sensitivity_allocation),
            'codec_consistent_local_reconstruction': True,
        },
        'gauge': {
            'enabled': bool(enable_attn_rotation_gauge),
            'report': gauge_report,
            'gauge_proj_dim': int(gauge_proj_dim),
            'gauge_pca_dim': int(gauge_pca_dim),
            'attention_rotation_gauge_enabled': bool(enable_attn_rotation_gauge),
            'verify_gauge': bool(verify_gauge),
            'gauge_report': gauge_report,
        },
        'reference': {
            'ce': float(reference_eval['loss']),
            'top1': float(reference_eval['top1']),
            'top5': float(reference_eval['top5']),
            'samples': int(reference_eval['samples']),
        },
        'blocks': blocks_report,
        'head_units': head_units,
        'artifacts': {
            'csv': csv_artifact,
        },
        'summary': summary,
    }
