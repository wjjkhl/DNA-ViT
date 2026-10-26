import json
import lzma
import math
import zipfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


TENSOR_DTYPE_TO_NAME: Dict[torch.dtype, str] = {
    torch.float16: 'float16',
    torch.float32: 'float32',
    torch.float64: 'float64',
    torch.int8: 'int8',
    torch.uint8: 'uint8',
    torch.int16: 'int16',
    torch.int32: 'int32',
    torch.int64: 'int64',
    torch.bool: 'bool',
}
NAME_TO_TENSOR_DTYPE: Dict[str, torch.dtype] = {v: k for k, v in TENSOR_DTYPE_TO_NAME.items()}
NAME_TO_NUMPY_DTYPE: Dict[str, Any] = {
    'float16': np.float16,
    'float32': np.float32,
    'float64': np.float64,
    'int8': np.int8,
    'uint8': np.uint8,
    'int16': np.int16,
    'int32': np.int32,
    'int64': np.int64,
    'bool': np.bool_,
}


def _write_entry(zf: zipfile.ZipFile, entry_name: str, payload: bytes) -> None:
    zf.writestr(entry_name, payload)


def _pack_bool_sequence(bits: torch.Tensor) -> bytes:
    flat = bits.detach().reshape(-1).to(dtype=torch.uint8, device='cpu').numpy()
    packed = np.packbits(flat, bitorder='little')
    return packed.tobytes(order='C')


def _unpack_bool_sequence(data: bytes, count: int, device: torch.device) -> torch.Tensor:
    if count == 0:
        return torch.zeros((0,), dtype=torch.bool, device=device)
    array = np.frombuffer(data, dtype=np.uint8)
    unpacked = np.unpackbits(array, bitorder='little')[:count]
    return torch.from_numpy(unpacked.astype(np.bool_)).to(device=device)


def _compress_bytes(payload: bytes, codec: str) -> bytes:
    if codec == 'lzma':
        return lzma.compress(payload, preset=6)
    if codec == 'raw':
        return payload
    raise ValueError(f'Unsupported codec: {codec}')


def _decompress_bytes(payload: bytes, codec: str) -> bytes:
    if codec == 'lzma':
        return lzma.decompress(payload)
    if codec == 'raw':
        return payload
    raise ValueError(f'Unsupported codec: {codec}')


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    array = tensor.detach().cpu().contiguous().numpy()
    return array.tobytes(order='C')


def bytes_to_tensor(data: bytes, shape: Sequence[int], dtype_name: str) -> torch.Tensor:
    np_dtype = NAME_TO_NUMPY_DTYPE[dtype_name]
    array = np.frombuffer(data, dtype=np_dtype).copy()
    if shape:
        array = array.reshape(tuple(int(x) for x in shape))
    tensor = torch.from_numpy(array)
    return tensor.to(dtype=NAME_TO_TENSOR_DTYPE[dtype_name])


def serialize_exact_tensor(
    zf: zipfile.ZipFile,
    entry_name: str,
    tensor: torch.Tensor,
    codec: str = 'lzma',
) -> Dict[str, Any]:
    payload = _compress_bytes(tensor_to_bytes(tensor), codec=codec)
    _write_entry(zf, entry_name, payload)
    return {
        'entry': entry_name,
        'shape': list(tensor.shape),
        'dtype': TENSOR_DTYPE_TO_NAME[tensor.dtype],
        'codec': codec,
    }


def deserialize_exact_tensor(zf: zipfile.ZipFile, meta: Dict[str, Any]) -> torch.Tensor:
    payload = zf.read(meta['entry'])
    raw = _decompress_bytes(payload, codec=str(meta['codec']))
    return bytes_to_tensor(raw, meta['shape'], str(meta['dtype']))


@torch.no_grad()
def choose_quant_step(coeff_abs: torch.Tensor, target_nonzero: int, qmax: int) -> float:
    flat = coeff_abs.detach().reshape(-1).float()
    if flat.numel() == 0:
        return 1.0
    max_abs = float(flat.max().item())
    if max_abs == 0.0:
        return 1.0
    target_nonzero = max(0, min(int(target_nonzero), flat.numel()))
    min_step = max(max_abs / max(float(qmax), 1.0), 1e-12)
    if target_nonzero <= 0:
        return max_abs * 2.0 + 1e-12
    if target_nonzero >= flat.numel():
        return min_step
    kth_smallest = flat.numel() - target_nonzero + 1
    tau = float(torch.kthvalue(flat, kth_smallest).values.item())
    return max(min_step, 2.0 * tau + 1e-12)


@torch.no_grad()
def encode_progressive_bitplane_stack(
    zf: zipfile.ZipFile,
    prefix: str,
    coeff: torch.Tensor,
    target_nonzero: int,
    quant_bits: int,
    codec: str = 'lzma',
) -> Dict[str, Any]:
    if coeff.ndim != 3:
        raise ValueError('encode_progressive_bitplane_stack expects a rank-3 tensor [depth, rows, cols].')
    if quant_bits < 2 or quant_bits > 16:
        raise ValueError('quant_bits must be in [2, 16].')

    coeff_cpu = coeff.detach().float().cpu().contiguous()
    qmax = (1 << (quant_bits - 1)) - 1
    step = choose_quant_step(coeff_cpu.abs(), target_nonzero=target_nonzero, qmax=qmax)
    quant = torch.clamp(torch.round(coeff_cpu / step), min=-qmax, max=qmax).to(torch.int32)
    abs_quant = quant.abs().reshape(-1)
    sign_quant = (quant.reshape(-1) < 0)
    numel = abs_quant.numel()
    num_bitplanes = int(math.ceil(math.log2(qmax + 1)))
    significant = torch.zeros((numel,), dtype=torch.bool)
    planes: List[Dict[str, Any]] = []

    for plane in reversed(range(num_bitplanes)):
        candidate_mask = ~significant
        candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
        candidate_values = abs_quant[candidate_indices]
        sig_bits = ((candidate_values >> plane) & 1).to(torch.bool)

        prev_indices = torch.nonzero(significant, as_tuple=False).flatten()
        refine_bits = ((abs_quant[prev_indices] >> plane) & 1).to(torch.bool)
        new_indices = candidate_indices[sig_bits]
        sign_bits = sign_quant[new_indices]

        plane_meta: Dict[str, Any] = {
            'bit': int(plane),
            'candidate_count': int(candidate_indices.numel()),
            'prev_significant_count': int(prev_indices.numel()),
            'new_significant_count': int(new_indices.numel()),
            'sig_entry': None,
            'refine_entry': None,
            'sign_entry': None,
            'codec': codec,
        }

        if candidate_indices.numel() > 0:
            entry = f'{prefix}/plane_{plane:02d}_sig.bin'
            _write_entry(zf, entry, _compress_bytes(_pack_bool_sequence(sig_bits), codec=codec))
            plane_meta['sig_entry'] = entry
        if prev_indices.numel() > 0:
            entry = f'{prefix}/plane_{plane:02d}_refine.bin'
            _write_entry(zf, entry, _compress_bytes(_pack_bool_sequence(refine_bits), codec=codec))
            plane_meta['refine_entry'] = entry
        if new_indices.numel() > 0:
            entry = f'{prefix}/plane_{plane:02d}_sign.bin'
            _write_entry(zf, entry, _compress_bytes(_pack_bool_sequence(sign_bits), codec=codec))
            plane_meta['sign_entry'] = entry

        significant[new_indices] = True
        planes.append(plane_meta)

    actual_nonzero = int((quant != 0).sum().item())
    return {
        'shape': list(coeff.shape),
        'step': float(step),
        'quant_bits': int(quant_bits),
        'qmax': int(qmax),
        'num_bitplanes': int(num_bitplanes),
        'target_nonzero': int(target_nonzero),
        'actual_nonzero': int(actual_nonzero),
        'codec': codec,
        'planes': planes,
    }


@torch.no_grad()
def decode_progressive_bitplane_stack(
    zf: zipfile.ZipFile,
    meta: Dict[str, Any],
    device: torch.device,
    max_planes: Optional[int] = None,
) -> torch.Tensor:
    shape = tuple(int(x) for x in meta['shape'])
    numel = int(np.prod(shape))
    num_bitplanes = int(meta['num_bitplanes'])
    planes = meta['planes']
    if max_planes is None or max_planes <= 0:
        planes_to_decode = num_bitplanes
    else:
        planes_to_decode = min(int(max_planes), num_bitplanes)

    abs_quant = torch.zeros((numel,), dtype=torch.int32, device=device)
    sign_quant = torch.zeros((numel,), dtype=torch.bool, device=device)
    significant = torch.zeros((numel,), dtype=torch.bool, device=device)

    for plane_offset, plane_meta in enumerate(planes):
        if plane_offset >= planes_to_decode:
            break
        bit = int(plane_meta['bit'])
        candidate_mask = ~significant
        candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
        prev_indices = torch.nonzero(significant, as_tuple=False).flatten()

        if plane_meta['sig_entry'] is not None:
            try:
                payload = _decompress_bytes(zf.read(plane_meta['sig_entry']), codec=str(plane_meta['codec']))
            except zipfile.BadZipFile as exc:
                raise zipfile.BadZipFile(
                    f'Failed to read zip entry `{plane_meta["sig_entry"]}` while decoding bit-plane {bit}. '
                    'Archive may be corrupted/truncated, or overwritten concurrently.'
                ) from exc
            sig_bits = _unpack_bool_sequence(payload, int(plane_meta['candidate_count']), device=device)
        else:
            sig_bits = torch.zeros((int(plane_meta['candidate_count']),), dtype=torch.bool, device=device)

        if plane_meta['refine_entry'] is not None:
            try:
                payload = _decompress_bytes(zf.read(plane_meta['refine_entry']), codec=str(plane_meta['codec']))
            except zipfile.BadZipFile as exc:
                raise zipfile.BadZipFile(
                    f'Failed to read zip entry `{plane_meta["refine_entry"]}` while decoding bit-plane {bit}. '
                    'Archive may be corrupted/truncated, or overwritten concurrently.'
                ) from exc
            refine_bits = _unpack_bool_sequence(payload, int(plane_meta['prev_significant_count']), device=device)
            abs_quant[prev_indices] += refine_bits.to(dtype=torch.int32) << bit

        new_indices = candidate_indices[sig_bits]
        if new_indices.numel() > 0:
            abs_quant[new_indices] += 1 << bit
            if plane_meta['sign_entry'] is not None:
                try:
                    payload = _decompress_bytes(zf.read(plane_meta['sign_entry']), codec=str(plane_meta['codec']))
                except zipfile.BadZipFile as exc:
                    raise zipfile.BadZipFile(
                        f'Failed to read zip entry `{plane_meta["sign_entry"]}` while decoding bit-plane {bit}. '
                        'Archive may be corrupted/truncated, or overwritten concurrently.'
                    ) from exc
                sign_bits = _unpack_bool_sequence(payload, int(plane_meta['new_significant_count']), device=device)
                sign_quant[new_indices] = sign_bits
            significant[new_indices] = True

    signed = abs_quant.to(dtype=torch.float32)
    signed[sign_quant] = -signed[sign_quant]
    coeff = signed.reshape(shape) * float(meta['step'])
    return coeff


@torch.no_grad()
def encode_flat_quantized_stack(
    zf: zipfile.ZipFile,
    prefix: str,
    coeff: torch.Tensor,
    target_nonzero: int,
    quant_bits: int,
    codec: str = 'lzma',
) -> Dict[str, Any]:
    if coeff.ndim != 3:
        raise ValueError('encode_flat_quantized_stack expects a rank-3 tensor [depth, rows, cols].')
    if quant_bits < 2 or quant_bits > 16:
        raise ValueError('quant_bits must be in [2, 16].')

    coeff_cpu = coeff.detach().float().cpu().contiguous()
    qmax = (1 << (quant_bits - 1)) - 1
    step = choose_quant_step(coeff_cpu.abs(), target_nonzero=target_nonzero, qmax=qmax)
    quant = torch.clamp(torch.round(coeff_cpu / step), min=-qmax, max=qmax).to(torch.int32)
    flat_quant = quant.reshape(-1)
    kept_indices = torch.nonzero(flat_quant != 0, as_tuple=False).flatten().to(dtype=torch.int64)
    kept_values = flat_quant[kept_indices].to(dtype=torch.int32).contiguous()

    index_meta = serialize_exact_tensor(
        zf=zf,
        entry_name=f'{prefix}/kept_indices.bin',
        tensor=kept_indices,
        codec=codec,
    )
    value_meta = serialize_exact_tensor(
        zf=zf,
        entry_name=f'{prefix}/quantized_values.bin',
        tensor=kept_values,
        codec=codec,
    )
    actual_nonzero = int(kept_indices.numel())
    return {
        'encoding': 'flat_sparse_quantized_v1',
        'shape': list(coeff.shape),
        'numel': int(flat_quant.numel()),
        'step': float(step),
        'quant_bits': int(quant_bits),
        'qmax': int(qmax),
        'target_nonzero': int(target_nonzero),
        'actual_nonzero': int(actual_nonzero),
        'codec': codec,
        'kept_indices': index_meta,
        'quantized_values': value_meta,
    }


@torch.no_grad()
def decode_flat_quantized_stack(
    zf: zipfile.ZipFile,
    meta: Dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    shape = tuple(int(x) for x in meta['shape'])
    numel = int(np.prod(shape))
    indices = deserialize_exact_tensor(zf, meta['kept_indices']).to(dtype=torch.int64, device=device).reshape(-1)
    values = deserialize_exact_tensor(zf, meta['quantized_values']).to(dtype=torch.int32, device=device).reshape(-1)
    if int(indices.numel()) != int(values.numel()):
        raise ValueError(
            'Corrupted flat sparse stack: `kept_indices` and `quantized_values` length mismatch '
            f'({indices.numel()} vs {values.numel()}).'
        )
    if int(indices.numel()) > 0:
        min_idx = int(indices.min().item())
        max_idx = int(indices.max().item())
        if min_idx < 0 or max_idx >= numel:
            raise ValueError(
                'Corrupted flat sparse stack: index out of bounds '
                f'(min={min_idx}, max={max_idx}, numel={numel}).'
            )
    flat_quant = torch.zeros((numel,), dtype=torch.int32, device=device)
    if int(indices.numel()) > 0:
        flat_quant[indices] = values
    coeff = flat_quant.to(dtype=torch.float32).reshape(shape) * float(meta['step'])
    return coeff
