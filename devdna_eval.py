import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import os
import torch

from devdna_core import ensure_torchvision_nms_stub


@dataclass
class EvalResult:
    top1: float
    top5: float
    loss: float
    rmse: float
    samples: int
    throughput: float
    ms_per_sample: float
    peak_gpu_mem_bytes: int


@torch.no_grad()
def _extract_logits_from_model_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output

    if isinstance(output, dict):
        for key in ('logits', 'output', 'out', 'pred', 'preds'):
            value = output.get(key, None)
            if torch.is_tensor(value):
                return value
        for value in output.values():
            if torch.is_tensor(value):
                return value
        for value in output.values():
            try:
                return _extract_logits_from_model_output(value)
            except Exception:
                continue
        raise TypeError('Model output dict does not contain tensor logits.')

    if isinstance(output, (list, tuple)):
        tensor_items = [item for item in output if torch.is_tensor(item)]
        if len(tensor_items) == 1:
            return tensor_items[0]
        if len(tensor_items) >= 2:
            ref_shape = tuple(tensor_items[0].shape)
            if all(tuple(item.shape) == ref_shape for item in tensor_items):
                merged = torch.stack([item.to(dtype=torch.float32) for item in tensor_items], dim=0).mean(dim=0)
                return merged.to(dtype=tensor_items[0].dtype)
            return tensor_items[0]
        for item in output:
            try:
                return _extract_logits_from_model_output(item)
            except Exception:
                continue
        raise TypeError('Model output list/tuple does not contain tensor logits.')

    raise TypeError(f'Unsupported model output type for logits extraction: {type(output)}')


def import_timm():
    ensure_torchvision_nms_stub()
    import timm  # type: ignore
    from timm.data import create_dataset, create_loader  # type: ignore
    try:
        from timm.data import create_transform  # type: ignore
    except Exception:
        create_transform = None
    from timm.utils import AverageMeter, accuracy  # type: ignore
    try:
        from timm.data import resolve_model_data_config as resolve_data_config  # type: ignore
    except Exception:
        from timm.data import resolve_data_config  # type: ignore
    return timm, create_dataset, create_loader, resolve_data_config, AverageMeter, accuracy, create_transform


@torch.no_grad()
def _normalize_state_dict_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(state_dict, dict) or not state_dict:
        return {}
    normalized = {str(key): value for key, value in state_dict.items()}
    keys = list(normalized.keys())
    if keys and all(str(key).startswith('module.') for key in keys):
        return {str(key)[7:]: value for key, value in normalized.items()}
    return normalized


@torch.no_grad()
def _clean_tensor_state_dict(state_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    normalized = _normalize_state_dict_keys(state_dict)
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in normalized.items():
        if torch.is_tensor(value):
            cleaned[str(key)] = value
    return cleaned


@torch.no_grad()
def _enumerate_state_dict_candidates(payload: Any) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    if not isinstance(payload, dict):
        return []
    candidates: List[Tuple[str, Dict[str, torch.Tensor]]] = []
    if 'state_dict' in payload and isinstance(payload['state_dict'], dict):
        state_dict = _clean_tensor_state_dict(payload['state_dict'])
        if state_dict:
            candidates.append(('state_dict', state_dict))
    if 'model' in payload and isinstance(payload['model'], dict):
        state_dict = _clean_tensor_state_dict(payload['model'])
        if state_dict:
            candidates.append(('model', state_dict))
    payload_state_dict = _clean_tensor_state_dict(payload)
    if payload_state_dict:
        candidates.append(('payload', payload_state_dict))

    dedup: List[Tuple[str, Dict[str, torch.Tensor]]] = []
    seen = set()
    for source_name, state_dict in candidates:
        if source_name in seen:
            continue
        dedup.append((source_name, state_dict))
        seen.add(source_name)
    return dedup


@torch.no_grad()
def _state_dict_match_stats(state_dict: Dict[str, torch.Tensor], model_state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    shape_match = 0
    shape_mismatch = 0
    unexpected_keys = 0
    for key, value in state_dict.items():
        target = model_state_dict.get(key, None)
        if target is None:
            unexpected_keys += 1
            continue
        if tuple(target.shape) == tuple(value.shape):
            shape_match += 1
        else:
            shape_mismatch += 1
    missing_keys = 0
    for key in model_state_dict.keys():
        if key not in state_dict:
            missing_keys += 1
    return {
        'shape_match': int(shape_match),
        'shape_mismatch': int(shape_mismatch),
        'missing_keys': int(missing_keys),
        'unexpected_keys': int(unexpected_keys),
    }


@torch.no_grad()
def _state_dict_match_score(stats: Dict[str, int]) -> Tuple[int, int, int, int]:
    return (
        int(stats['shape_match']),
        -int(stats['shape_mismatch']),
        -int(stats['missing_keys']),
        -int(stats['unexpected_keys']),
    )


@torch.no_grad()
def _shape_tuple(value: Any) -> Optional[Tuple[int, ...]]:
    shape = getattr(value, 'shape', None)
    if shape is None:
        return None
    try:
        return tuple(int(x) for x in shape)
    except Exception:
        return None


@torch.no_grad()
def _collect_shape_mismatches(
    state_dict: Dict[str, torch.Tensor],
    model_state_dict: Dict[str, torch.Tensor],
    limit: int = 10,
) -> List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]]:
    mismatches: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
    for key, value in state_dict.items():
        target = model_state_dict.get(key, None)
        if target is None:
            continue
        ckpt_shape = _shape_tuple(value)
        model_shape = _shape_tuple(target)
        if ckpt_shape is None or model_shape is None:
            continue
        if ckpt_shape != model_shape:
            mismatches.append((str(key), ckpt_shape, model_shape))
            if len(mismatches) >= int(limit):
                break
    return mismatches


@torch.no_grad()
def _is_swin_model_name(model_name: str) -> bool:
    return 'swin' in str(model_name).lower()


@torch.no_grad()
def _remap_swin_downsample_keys(state_dict: Dict[str, torch.Tensor], shift: int) -> Optional[Dict[str, torch.Tensor]]:
    if int(shift) == 0:
        return None
    remapped: Dict[str, torch.Tensor] = {}
    changed = False
    for key, value in state_dict.items():
        if (
            isinstance(key, str)
            and key.startswith('layers.')
            and '.downsample.' in key
        ):
            parts = key.split('.')
            if len(parts) >= 4 and parts[0] == 'layers' and parts[1].isdigit() and parts[2] == 'downsample':
                dst_layer = int(parts[1]) + int(shift)
                if dst_layer < 0:
                    changed = True
                    continue
                parts[1] = str(dst_layer)
                new_key = '.'.join(parts)
                remapped[new_key] = value
                changed = changed or (new_key != key)
                continue
        remapped[key] = value
    if not changed:
        return None
    return remapped


@torch.no_grad()
def _enumerate_state_dict_variants(
    model_name: str,
    source_name: str,
    state_dict: Dict[str, torch.Tensor],
) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    variants: List[Tuple[str, Dict[str, torch.Tensor]]] = [(str(source_name), state_dict)]
    if not _is_swin_model_name(model_name):
        return variants
    for shift in (1, -1):
        remapped = _remap_swin_downsample_keys(state_dict, shift=shift)
        if remapped is not None:
            variants.append((f'{source_name}|swin_downsample_shift{shift:+d}', remapped))
    return variants


@torch.no_grad()
def _select_best_state_dict_candidate(
    model_name: str,
    model: torch.nn.Module,
    candidates: List[Tuple[str, Dict[str, torch.Tensor]]],
) -> Tuple[str, Dict[str, torch.Tensor], Dict[str, int], int]:
    model_state = model.state_dict()
    scored: List[Tuple[str, Dict[str, torch.Tensor], Dict[str, int]]] = []
    for source_name, candidate_state_dict in candidates:
        for variant_source, variant_state in _enumerate_state_dict_variants(
            model_name=model_name,
            source_name=source_name,
            state_dict=candidate_state_dict,
        ):
            stats = _state_dict_match_stats(variant_state, model_state)
            scored.append((variant_source, variant_state, stats))
    if not scored:
        raise ValueError('No valid state_dict candidate variant found.')
    scored.sort(key=lambda item: _state_dict_match_score(item[2]), reverse=True)
    best_source, best_state_dict, best_stats = scored[0]
    return best_source, best_state_dict, best_stats, len(scored)


@torch.no_grad()
def _apply_known_key_aliases(
    state_dict: Dict[str, torch.Tensor],
    model_state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    remapped: Dict[str, torch.Tensor] = dict(state_dict)
    notes: List[str] = []

    def _move_key(src: str, dst: str) -> None:
        if src in remapped and dst in model_state_dict and dst not in remapped:
            remapped[dst] = remapped.pop(src)
            notes.append(f'{src}->{dst}')

    _move_key('head.weight', 'head.fc.weight')
    _move_key('head.bias', 'head.fc.bias')
    _move_key('head.fc.weight', 'head.weight')
    _move_key('head.fc.bias', 'head.bias')
    return remapped, notes


@torch.no_grad()
def _load_checkpoint_into_model(
    model: torch.nn.Module,
    checkpoint_path: str,
    model_name: str = '',
    strict: bool = False,
) -> torch.nn.Module:
    if not checkpoint_path:
        return model
    path = str(checkpoint_path)
    selected_source = 'single'
    candidates: List[Tuple[str, Dict[str, torch.Tensor]]]
    if not torch.jit.is_scripting() and path.lower().endswith('.safetensors'):
        try:
            from safetensors.torch import load_file as safetensors_load_file  # type: ignore
        except Exception as exc:
            raise ImportError(
                'Loading .safetensors weights requires `safetensors` package. '
                'Please install it with `pip install safetensors`.'
            ) from exc
        state_dict = _clean_tensor_state_dict(safetensors_load_file(path, device='cpu'))
        candidates = [('safetensors', state_dict)] if state_dict else []
    else:
        payload = torch.load(path, map_location='cpu', weights_only=False)
        candidates = _enumerate_state_dict_candidates(payload)
    if not candidates:
        raise ValueError(f'No tensor state_dict candidate found in checkpoint: {checkpoint_path}')

    effective_model_name = str(model_name).strip() or str(model.__class__.__name__)
    selected_source, state_dict, best_stats, num_scored = _select_best_state_dict_candidate(
        model_name=effective_model_name,
        model=model,
        candidates=candidates,
    )
    if num_scored > 1:
        print(
            'Checkpoint has multiple state_dict candidates/variants; '
            f'selected `{selected_source}` by shape match: '
            f'match={best_stats["shape_match"]}, '
            f'shape_mismatch={best_stats["shape_mismatch"]}, '
            f'missing={best_stats["missing_keys"]}, '
            f'unexpected={best_stats["unexpected_keys"]}',
            flush=True,
        )

    model_state = model.state_dict()
    state_dict, alias_notes = _apply_known_key_aliases(state_dict, model_state)
    if alias_notes:
        print('Applied checkpoint key aliases: ' + ', '.join(alias_notes), flush=True)
    match_stats = _state_dict_match_stats(state_dict, model_state)
    shape_mismatches = _collect_shape_mismatches(state_dict, model_state, limit=10)

    timm_version = 'unknown'
    timm_path = 'unknown'
    try:
        timm, *_ = import_timm()
        timm_version = str(getattr(timm, '__version__', 'unknown'))
        timm_path = str(getattr(timm, '__file__', 'unknown'))
    except Exception:
        pass

    try:
        incompatible = model.load_state_dict(state_dict, strict=bool(strict))
    except RuntimeError as exc:
        mismatch_text = '\n'.join(
            f'  - {key}: ckpt={ckpt_shape} vs model={model_shape}'
            for key, ckpt_shape, model_shape in shape_mismatches
        )
        if not mismatch_text:
            mismatch_text = '  (no overlapping tensor-shape mismatch preview)'
        raise RuntimeError(
            'Failed to load local pretrained checkpoint due to shape mismatch.\n'
            f'model_name={effective_model_name}\n'
            f'checkpoint={path}\n'
            f'candidate_source={selected_source}\n'
            f'timm_version={timm_version}\n'
            f'timm_path={timm_path}\n'
            f'shape_match={match_stats["shape_match"]}, '
            f'shape_mismatch={match_stats["shape_mismatch"]}, '
            f'missing={match_stats["missing_keys"]}, '
            f'unexpected={match_stats["unexpected_keys"]}\n'
            'First shape mismatches:\n'
            f'{mismatch_text}\n'
            'Likely cause: the current timm model definition does not match this checkpoint architecture.'
        ) from exc

    missing_keys = list(getattr(incompatible, 'missing_keys', []))
    unexpected_keys = list(getattr(incompatible, 'unexpected_keys', []))
    if missing_keys or unexpected_keys:
        print(
            'Warning: local pretrained loaded with non-strict match; '
            f'missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}, file={path}',
            flush=True,
        )
        if missing_keys:
            print('  Missing key sample: ' + ', '.join(missing_keys[:12]), flush=True)
        if unexpected_keys:
            print('  Unexpected key sample: ' + ', '.join(unexpected_keys[:12]), flush=True)
    else:
        print(f'Loaded local pretrained weights: {path}', flush=True)
    return model


@torch.no_grad()
def build_timm_model(
    model_name: str,
    pretrained: bool,
    checkpoint_path: str = '',
    pretrained_weight_path: str = '',
) -> torch.nn.Module:
    timm, *_ = import_timm()
    local_weight_path = str(checkpoint_path or pretrained_weight_path).strip()
    if pretrained and not local_weight_path:
        raise ValueError(
            'Offline local pretrained weights are required. '
            'Please provide `--pretrained-weight-path` (supports .safetensors).'
        )
    model = timm.create_model(model_name, pretrained=False)
    if local_weight_path:
        if not os.path.isfile(local_weight_path):
            raise FileNotFoundError(f'Local weight file not found: {local_weight_path}')
        model = _load_checkpoint_into_model(model, local_weight_path, model_name=str(model_name), strict=False)
    return model


@torch.no_grad()
def _resolve_data_config_for_model(model: torch.nn.Module, use_test_size: bool = False) -> Dict[str, Any]:
    _, _, _, resolve_data_config, _, _, _ = import_timm()
    try:
        data_config = resolve_data_config(model=model, use_test_size=bool(use_test_size))
    except TypeError:
        try:
            data_config = resolve_data_config(args={}, model=model, use_test_size=bool(use_test_size))
        except TypeError:
            try:
                data_config = resolve_data_config({}, model=model)
            except TypeError:
                data_config = resolve_data_config(model=model)
    return dict(data_config)


@torch.no_grad()
def create_official_imagenet_loader(
    model: torch.nn.Module,
    data_root: str,
    split: str,
    batch_size: int,
    workers: int,
    class_map: str = '',
    pin_memory: bool = True,
    crop_pct: Optional[float] = None,
    interpolation: str = '',
    no_prefetcher: bool = False,
    device: Optional[torch.device] = None,
    use_test_size: bool = True,
    enforce_official_preprocess: bool = True,
):
    _, create_dataset, create_loader, _, _, _, _ = import_timm()
    data_config = _resolve_data_config_for_model(model=model, use_test_size=bool(use_test_size))

    if bool(enforce_official_preprocess):
        if crop_pct is not None or str(interpolation).strip():
            print(
                '[devdna_eval] enforce_official_preprocess=True: ignore custom crop_pct/interpolation '
                'and use timm-resolved preprocess config.',
                flush=True,
            )
    else:
        if crop_pct is not None:
            data_config['crop_pct'] = crop_pct
        if interpolation:
            data_config['interpolation'] = interpolation

    dataset_kwargs = dict(
        name='',
        root=data_root,
        split=split,
        is_training=False,
        batch_size=batch_size,
    )
    if class_map:
        dataset_kwargs['class_map'] = class_map
    try:
        dataset = create_dataset(search_split=True, **dataset_kwargs)
    except TypeError:
        try:
            dataset = create_dataset(**dataset_kwargs)
        except TypeError:
            dataset = create_dataset('', data_root, split=split, is_training=False)

    loader_kwargs = dict(
        dataset=dataset,
        input_size=data_config['input_size'],
        batch_size=batch_size,
        is_training=False,
        use_prefetcher=not no_prefetcher,
        interpolation=data_config.get('interpolation', 'bicubic'),
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=workers,
        crop_pct=data_config.get('crop_pct', 1.0),
        pin_memory=pin_memory,
    )
    if device is not None:
        loader_kwargs['device'] = device
    try:
        loader = create_loader(**loader_kwargs)
    except TypeError:
        loader_kwargs.pop('device', None)
        loader = create_loader(**loader_kwargs)
    return loader, data_config


@torch.no_grad()
def build_activation_calibration_loader(
    model: torch.nn.Module,
    data_root: str,
    source: str = 'val',
    num_images: int = 128,
    seed: int = 3,
    batch_size: int = 256,
    workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    crop_pct: Optional[float] = None,
    interpolation: str = '',
    use_test_size: bool = True,
    enforce_official_preprocess: bool = True,
):
    from torch.utils.data import DataLoader, Subset
    from torchvision.datasets import ImageFolder

    _, _, _, _, _, _, create_transform = import_timm()
    if create_transform is None:
        raise RuntimeError('timm.data.create_transform is not available; cannot build activation calibration loader.')

    source_norm = str(source).strip().lower()
    if source_norm not in ('train', 'val'):
        raise ValueError(f'Unsupported activation calibration source={source!r}, expected "train" or "val".')

    source_root = Path(str(data_root)).expanduser() / source_norm
    if not source_root.is_dir():
        raise FileNotFoundError(f'Activation calibration source directory not found: {source_root}')

    data_config = _resolve_data_config_for_model(model=model, use_test_size=bool(use_test_size))
    if bool(enforce_official_preprocess):
        if crop_pct is not None or str(interpolation).strip():
            print(
                '[devdna_eval] enforce_official_preprocess=True for calibration: '
                'ignore custom crop_pct/interpolation and use timm-resolved preprocess config.',
                flush=True,
            )
    else:
        if crop_pct is not None:
            data_config['crop_pct'] = float(crop_pct)
        if interpolation:
            data_config['interpolation'] = str(interpolation)

    transform = create_transform(
        input_size=data_config['input_size'],
        is_training=False,
        interpolation=data_config.get('interpolation', 'bicubic'),
        mean=data_config['mean'],
        std=data_config['std'],
        crop_pct=data_config.get('crop_pct', 1.0),
    )

    dataset = ImageFolder(str(source_root), transform=transform)
    num_images = max(1, min(int(num_images), len(dataset)))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    indices = torch.randperm(len(dataset), generator=generator)[:num_images].tolist()
    subset = Subset(dataset, indices)

    dataloader_kwargs: Dict[str, Any] = {
        'batch_size': max(1, int(batch_size)),
        'shuffle': False,
        'num_workers': max(0, int(workers)),
        'pin_memory': bool(pin_memory),
        'persistent_workers': bool(int(workers) > 0 and bool(persistent_workers)),
    }
    if int(workers) > 0:
        dataloader_kwargs['prefetch_factor'] = 2
    loader = DataLoader(subset, **dataloader_kwargs)
    return loader, source_norm, indices, data_config


@torch.no_grad()
def maybe_wrap_dataparallel(model: torch.nn.Module) -> torch.nn.Module:
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return torch.nn.DataParallel(model)
    return model


@torch.no_grad()
def evaluate_with_official_logic(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    amp_dtype: str = 'fp16',
    print_freq: int = 20,
    use_amp: Optional[bool] = None,
) -> EvalResult:
    _, _, _, _, AverageMeter, accuracy, _ = import_timm()
    criterion = torch.nn.CrossEntropyLoss().to(device)
    model.eval()
    model.to(device)

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    if use_amp is None:
        use_amp = device.type == 'cuda'
    else:
        use_amp = bool(use_amp) and device.type == 'cuda'
    autocast_dtype = torch.bfloat16 if amp_dtype == 'bf16' else torch.float16

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    total_samples = 0
    rmse_sse = 0.0
    rmse_count = 0

    for batch_idx, (images, target) in enumerate(loader):
        if not torch.is_tensor(images):
            images = torch.as_tensor(images)
        if not torch.is_tensor(target):
            target = torch.as_tensor(target)
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
            output = _extract_logits_from_model_output(model(images))
            loss = criterion(output, target)
        probs = torch.softmax(output.float(), dim=1)
        one_hot = torch.zeros_like(probs).scatter_(1, target.view(-1, 1), 1.0)
        rmse_sse += float((probs - one_hot).pow(2).sum().item())
        rmse_count += int(probs.numel())

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(float(loss.item()), images.size(0))
        top1.update(float(prec1.item()), images.size(0))
        top5.update(float(prec5.item()), images.size(0))
        total_samples += images.size(0)
        batch_time.update(time.time() - end)
        end = time.time()

        if print_freq > 0 and batch_idx % print_freq == 0:
            images_per_sec = images.size(0) / max(batch_time.val, 1e-12)
            print(
                f'Test: [{batch_idx}/{len(loader)}]  '
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                f'Rate {images_per_sec:.2f}/s  '
                f'Loss {losses.val:.4f} ({losses.avg:.4f})  '
                f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})  '
                f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'
            )

    throughput = total_samples / max(batch_time.sum, 1e-12)
    ms_per_sample = 1000.0 / max(throughput, 1e-12)
    rmse = math.sqrt(rmse_sse / max(float(rmse_count), 1.0))
    peak_mem = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
    return EvalResult(
        top1=float(top1.avg),
        top5=float(top5.avg),
        loss=float(losses.avg),
        rmse=float(rmse),
        samples=int(total_samples),
        throughput=float(throughput),
        ms_per_sample=float(ms_per_sample),
        peak_gpu_mem_bytes=int(peak_mem),
    )


@torch.no_grad()
def run_activation_calibration(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    controller,
) -> None:
    model.eval()
    controller.begin_calibration()
    with torch.inference_mode():
        for images, _labels in dataloader:
            if not torch.is_tensor(images):
                images = torch.as_tensor(images)
            images = images.to(device, non_blocking=True)
            model(images)
    controller.end_calibration()


@torch.no_grad()
def evaluate_for_sensitivity(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    max_samples: int,
    print_freq: int = 20,
    reference_logits_cpu: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    _, _, _, _, AverageMeter, accuracy, _ = import_timm()
    criterion = torch.nn.CrossEntropyLoss().to(device)
    model.eval()
    model.to(device)
    if max_samples <= 0:
        raise ValueError(f'max_samples must be positive, got {max_samples}.')

    collect_reference_logits = reference_logits_cpu is None
    if not collect_reference_logits:
        if reference_logits_cpu.ndim != 2:
            raise ValueError('reference_logits_cpu must be rank-2 [N, C].')
        if int(reference_logits_cpu.shape[0]) < int(max_samples):
            raise ValueError(
                f'reference_logits_cpu has {reference_logits_cpu.shape[0]} rows but max_samples={max_samples}.'
            )

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    total_samples = 0
    rel_logit_mse_sum = 0.0
    eps = 1e-12
    logits_cpu_chunks = []

    # Sensitivity analysis is intentionally evaluated in pure fp32 without autocast.
    # This avoids AMP numerical noise from masking small perturbation effects.
    for batch_idx, (images, target) in enumerate(loader):
        if total_samples >= max_samples:
            break
        if not torch.is_tensor(images):
            images = torch.as_tensor(images)
        if not torch.is_tensor(target):
            target = torch.as_tensor(target)
        remaining = int(max_samples - total_samples)
        if images.size(0) > remaining:
            images = images[:remaining]
            target = target[:remaining]
        images = images.to(device, non_blocking=True).float()
        target = target.to(device, non_blocking=True)

        output = _extract_logits_from_model_output(model(images)).float()
        loss = criterion(output, target)
        k5 = min(5, int(output.shape[1]))
        if k5 <= 1:
            prec1 = accuracy(output, target, topk=(1,))[0]
            prec5 = prec1
        else:
            prec1, prec5 = accuracy(output, target, topk=(1, k5))
        losses.update(float(loss.item()), images.size(0))
        top1.update(float(prec1.item()), images.size(0))
        top5.update(float(prec5.item()), images.size(0))

        output_cpu = output.detach().cpu()
        if collect_reference_logits:
            logits_cpu_chunks.append(output_cpu)
        else:
            start = int(total_samples)
            stop = int(total_samples + images.size(0))
            ref_logits = reference_logits_cpu[start:stop].float()
            if ref_logits.shape != output_cpu.shape:
                raise RuntimeError(
                    f'Logits shape mismatch at samples [{start}:{stop}]: '
                    f'perturbed={tuple(output_cpu.shape)} vs reference={tuple(ref_logits.shape)}'
                )
            numerator = (output_cpu - ref_logits).pow(2).sum(dim=1)
            denominator = ref_logits.pow(2).sum(dim=1) + eps
            rel_logit_mse_sum += float((numerator / denominator).sum().item())

        total_samples += int(images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if print_freq > 0 and batch_idx % print_freq == 0:
            images_per_sec = images.size(0) / max(batch_time.val, 1e-12)
            print(
                f'SensitivityEval: [{total_samples}/{max_samples}]  '
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                f'Rate {images_per_sec:.2f}/s  '
                f'Loss {losses.val:.4f} ({losses.avg:.4f})  '
                f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})  '
                f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'
            )

    if total_samples != int(max_samples):
        raise RuntimeError(
            f'Not enough samples from loader: expected exactly {max_samples}, got {total_samples}.'
        )

    throughput = total_samples / max(batch_time.sum, 1e-12)
    ms_per_sample = 1000.0 / max(throughput, 1e-12)
    peak_mem = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0

    result: Dict[str, Any] = {
        'loss': float(losses.avg),
        'top1': float(top1.avg),
        'top5': float(top5.avg),
        'samples': int(total_samples),
        'throughput': float(throughput),
        'ms_per_sample': float(ms_per_sample),
        'peak_gpu_mem_bytes': int(peak_mem),
    }
    if collect_reference_logits:
        result['reference_logits_cpu'] = torch.cat(logits_cpu_chunks, dim=0).contiguous()
    else:
        result['rel_logit_mse'] = float(rel_logit_mse_sum / max(float(total_samples), 1.0))
    return result
