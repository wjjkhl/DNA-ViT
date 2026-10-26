import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


def _extract_cli_value(argv: Sequence[str], flag: str) -> str:
    for idx, token in enumerate(argv):
        if token == flag:
            if idx + 1 < len(argv):
                return str(argv[idx + 1]).strip()
            return ''
        if token.startswith(f'{flag}='):
            return str(token.split('=', 1)[1]).strip()
    return ''


def _load_yaml_config(path: str) -> Dict[str, Any]:
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f'Config file not found: {config_path}')
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise ImportError(
            'Loading YAML config requires `pyyaml`. Please install with `pip install pyyaml`.'
        ) from exc
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f'Config root must be a mapping/dict, got: {type(data)}')
    return data


def _bootstrap_cuda_visible_devices_from_config() -> None:
    cuda_devices = _extract_cli_value(sys.argv, '--cuda-devices')
    if cuda_devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices
        return
    config_path = _extract_cli_value(sys.argv, '--config')
    if not config_path:
        return
    try:
        config = _load_yaml_config(config_path)
    except Exception:
        return
    value = config.get('cuda_devices', None)
    if value is not None and str(value).strip():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(value).strip()


def _get_nested(config: Dict[str, Any], *keys: str) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _config_to_arg_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}

    def assign(arg_name: str, value: Any) -> None:
        if value is not None:
            defaults[arg_name] = value

    top_level_keys = [
        'mode',
        'seed',
        'cuda_devices',
        'device',
        'amp_dtype',
        'print_freq',
        'codec_allocation_mode',
    ]
    for key in top_level_keys:
        assign(key.replace('-', '_'), config.get(key, None))

    model_cfg = config.get('model', None)
    if isinstance(model_cfg, str):
        assign('model', model_cfg)
    else:
        assign('model', _get_nested(config, 'model', 'name'))
    assign('pretrained', _get_nested(config, 'model', 'pretrained'))
    assign('pretrained_weight_path', _get_nested(config, 'model', 'pretrained_weight_path'))
    assign('checkpoint', _get_nested(config, 'model', 'checkpoint'))

    assign('data_root', _get_nested(config, 'data', 'data_root'))
    assign('split', _get_nested(config, 'data', 'split'))
    assign('class_map', _get_nested(config, 'data', 'class_map'))
    assign('batch_size', _get_nested(config, 'data', 'batch_size'))
    assign('workers', _get_nested(config, 'data', 'workers'))
    assign('pin_memory', _get_nested(config, 'data', 'pin_memory'))
    assign('no_prefetcher', _get_nested(config, 'data', 'no_prefetcher'))
    assign('crop_pct', _get_nested(config, 'data', 'crop_pct'))
    assign('interpolation', _get_nested(config, 'data', 'interpolation'))

    assign('mlp_keep_ratio', _get_nested(config, 'compression', 'mlp_keep_ratio'))
    assign('attn_keep_ratio', _get_nested(config, 'compression', 'attn_keep_ratio'))
    assign('quant_bits', _get_nested(config, 'compression', 'quant_bits'))
    assign('codec_allocation_mode', _get_nested(config, 'compression', 'codec_allocation_mode'))
    assign('sensitivity_report', _get_nested(config, 'compression', 'sensitivity_report'))
    assign('target_archive_mb', _get_nested(config, 'compression', 'target_archive_mb'))
    assign('target_archive_tolerance_ratio', _get_nested(config, 'compression', 'target_archive_tolerance_ratio'))
    assign('target_archive_max_iters', _get_nested(config, 'compression', 'target_archive_max_iters'))
    assign('lambda_probe_start', _get_nested(config, 'compression', 'lambda_probe_start'))
    assign('sens_weight_ce', _get_nested(config, 'compression', 'sens_weight_ce'))
    assign('sens_weight_logit', _get_nested(config, 'compression', 'sens_weight_logit'))
    assign('sens_weight_byte', _get_nested(config, 'compression', 'sens_weight_byte'))
    importance_alpha_cfg = _get_nested(config, 'compression', 'importance_alpha')
    if importance_alpha_cfg is None:
        # Friendly short alias for YAML experiments.
        importance_alpha_cfg = _get_nested(config, 'compression', 'alpha')
    assign('importance_alpha', importance_alpha_cfg)
    importance_quantile_q_cfg = _get_nested(config, 'compression', 'importance_quantile_q')
    if importance_quantile_q_cfg is None:
        # Friendly short alias for YAML experiments.
        importance_quantile_q_cfg = _get_nested(config, 'compression', 'q')
    assign('importance_quantile_q', importance_quantile_q_cfg)
    sens_beta_cfg = _get_nested(config, 'compression', 'sens_beta')
    if sens_beta_cfg is None:
        # Backward compatibility: previous key was `compression.sens_exponent`.
        sens_beta_cfg = _get_nested(config, 'compression', 'sens_exponent')
    assign('sens_beta', sens_beta_cfg)
    assign('min_tensor_keep_ratio', _get_nested(config, 'compression', 'min_tensor_keep_ratio'))
    assign('gauge_proj_dim', _get_nested(config, 'compression', 'gauge_proj_dim'))
    assign('gauge_pca_dim', _get_nested(config, 'compression', 'gauge_pca_dim'))
    assign('sensitivity_power_iters', _get_nested(config, 'compression', 'sensitivity_power_iters'))
    assign('disable_attn_rotation_gauge', _get_nested(config, 'compression', 'disable_attn_rotation_gauge'))
    assign('disable_gauge_verify', _get_nested(config, 'compression', 'disable_gauge_verify'))

    assign('sensitivity_calib_size', _get_nested(config, 'sensitivity', 'calib_size'))
    assign('sensitivity_keep_ratio', _get_nested(config, 'sensitivity', 'keep_ratio'))
    assign('sensitivity_quant_bits', _get_nested(config, 'sensitivity', 'quant_bits'))
    assign('sensitivity_allocation', _get_nested(config, 'sensitivity', 'allocation'))
    assign('sensitivity_output', _get_nested(config, 'sensitivity', 'output'))
    assign('sensitivity_csv_output', _get_nested(config, 'sensitivity', 'csv_output'))

    assign('decode_fraction', _get_nested(config, 'decode', 'fraction'))
    assign('decode_planes', _get_nested(config, 'decode', 'planes'))
    assign('include_head_linear', _get_nested(config, 'scope', 'include_head_linear'))


    assign('streaming_runtime_enabled', _get_nested(config, 'streaming_runtime', 'enabled'))
    assign('streaming_runtime_archive_invariant', _get_nested(config, 'streaming_runtime', 'archive_invariant'))
    assign('streaming_runtime_mode', _get_nested(config, 'streaming_runtime', 'mode'))
    assign('streaming_runtime_logical_scheduler_unit', _get_nested(config, 'streaming_runtime', 'logical_scheduler_unit'))
    assign('streaming_runtime_group_policy', _get_nested(config, 'streaming_runtime', 'group_policy'))
    assign('streaming_runtime_super_block_k', _get_nested(config, 'streaming_runtime', 'super_block_k'))
    assign('streaming_runtime_auto_profile_batches', _get_nested(config, 'streaming_runtime', 'auto_profile_batches'))
    assign(
        'streaming_runtime_auto_compute_decode_ratio_target',
        _get_nested(config, 'streaming_runtime', 'auto_compute_decode_ratio_target'),
    )
    assign('streaming_runtime_cache_decoded_groups', _get_nested(config, 'streaming_runtime', 'cache_decoded_groups'))
    assign('streaming_runtime_primitive_unit', _get_nested(config, 'streaming_runtime', 'primitive_unit'))
    assign('streaming_runtime_modes', _get_nested(config, 'streaming_runtime', 'modes'))
    assign('streaming_runtime_eval_batch_size', _get_nested(config, 'streaming_runtime', 'eval_batch_size'))
    assign('streaming_runtime_eval_samples', _get_nested(config, 'streaming_runtime', 'eval_samples'))
    assign('streaming_runtime_warmup_batches', _get_nested(config, 'streaming_runtime', 'warmup_batches'))
    assign('streaming_runtime_repeats', _get_nested(config, 'streaming_runtime', 'repeats'))
    assign(
        'streaming_runtime_reset_decoded_cache_before_timed_eval',
        _get_nested(config, 'streaming_runtime', 'reset_decoded_cache_before_timed_eval'),
    )
    assign(
        'streaming_runtime_drop_first_repeat_for_aggregation',
        _get_nested(config, 'streaming_runtime', 'drop_first_repeat_for_aggregation'),
    )
    assign('streaming_runtime_overlap_enabled', _get_nested(config, 'streaming_runtime', 'overlap', 'enabled'))
    assign('streaming_runtime_double_buffering', _get_nested(config, 'streaming_runtime', 'overlap', 'double_buffering'))
    assign('streaming_runtime_prefetch_next_group', _get_nested(config, 'streaming_runtime', 'overlap', 'prefetch_next_group'))
    assign('streaming_runtime_track_peak_gpu_mem', _get_nested(config, 'streaming_runtime', 'memory', 'track_peak_gpu_mem'))
    assign(
        'streaming_runtime_track_peak_decode_load_gpu_mem',
        _get_nested(config, 'streaming_runtime', 'memory', 'track_peak_decode_load_gpu_mem'),
    )
    assign(
        'streaming_runtime_track_peak_model_resident_gpu_mem',
        _get_nested(config, 'streaming_runtime', 'memory', 'track_peak_model_resident_gpu_mem'),
    )
    assign('streaming_runtime_output_json', _get_nested(config, 'streaming_runtime', 'outputs', 'result_json'))
    assign('streaming_runtime_output_csv', _get_nested(config, 'streaming_runtime', 'outputs', 'result_csv'))

    assign('activation_quant_enabled', _get_nested(config, 'eval', 'activation_quant', 'enabled'))
    assign('activation_quant_bit', _get_nested(config, 'eval', 'activation_quant', 'bit'))
    assign('activation_quant_scheme', _get_nested(config, 'eval', 'activation_quant', 'scheme'))
    assign('activation_observer_type', _get_nested(config, 'eval', 'activation_quant', 'observer', 'type'))
    assign('activation_observer_percentile', _get_nested(config, 'eval', 'activation_quant', 'observer', 'percentile'))
    assign('activation_quant_eps', _get_nested(config, 'eval', 'activation_quant', 'eps'))
    assign('activation_max_observer_samples', _get_nested(config, 'eval', 'activation_quant', 'observer', 'max_samples'))
    assign(
        'activation_observer_samples_per_update',
        _get_nested(config, 'eval', 'activation_quant', 'observer', 'samples_per_update'),
    )
    assign('activation_linear_inputs', _get_nested(config, 'eval', 'activation_quant', 'scope', 'linear_inputs'))
    assign('activation_matmul_qk_inputs', _get_nested(config, 'eval', 'activation_quant', 'scope', 'matmul_qk_inputs'))
    assign('activation_matmul_av_inputs', _get_nested(config, 'eval', 'activation_quant', 'scope', 'matmul_av_inputs'))
    assign(
        'activation_include_head_linear_input',
        _get_nested(config, 'eval', 'activation_quant', 'scope', 'include_head_linear_input'),
    )
    assign(
        'activation_skip_layernorm_outputs',
        _get_nested(config, 'eval', 'activation_quant', 'scope', 'skip_layernorm_outputs'),
    )
    assign('activation_calibration_source', _get_nested(config, 'eval', 'activation_quant', 'calibration', 'source'))
    assign('activation_calibration_num_images', _get_nested(config, 'eval', 'activation_quant', 'calibration', 'num_images'))
    assign('activation_calibration_seed', _get_nested(config, 'eval', 'activation_quant', 'calibration', 'seed'))
    assign('activation_calibration_batch_size', _get_nested(config, 'eval', 'activation_quant', 'calibration', 'batch_size'))
    assign('activation_calibration_workers', _get_nested(config, 'eval', 'activation_quant', 'calibration', 'num_workers'))
    assign('activation_calibration_pin_memory', _get_nested(config, 'eval', 'activation_quant', 'calibration', 'pin_memory'))
    assign(
        'activation_calibration_persistent_workers',
        _get_nested(config, 'eval', 'activation_quant', 'calibration', 'persistent_workers'),
    )


    assign('archive_path', _get_nested(config, 'paths', 'archive_path'))
    assign('save_report', _get_nested(config, 'paths', 'save_report'))
    assign('save_decoded_checkpoint', _get_nested(config, 'paths', 'save_decoded_checkpoint'))
    assign('allocation_report_output', _get_nested(config, 'paths', 'allocation_report_output'))
    assign('allocation_csv_output', _get_nested(config, 'paths', 'allocation_csv_output'))

    # Allow direct arg-style keys at config root as an escape hatch.
    for arg_name in [
        'pretrained',
        'pretrained_weight_path',
        'checkpoint',
        'archive_path',
        'save_report',
        'save_decoded_checkpoint',
        'data_root',
        'split',
        'class_map',
        'batch_size',
        'workers',
        'pin_memory',
        'no_prefetcher',
        'crop_pct',
        'interpolation',
        'mlp_keep_ratio',
        'attn_keep_ratio',
        'quant_bits',
        'codec_allocation_mode',
        'sensitivity_report',
        'target_archive_mb',
        'target_archive_tolerance_ratio',
        'target_archive_max_iters',
        'lambda_probe_start',
        'sens_weight_ce',
        'sens_weight_logit',
        'sens_weight_byte',
        'importance_alpha',
        'importance_quantile_q',
        'sens_beta',
        'min_tensor_keep_ratio',
        'gauge_proj_dim',
        'gauge_pca_dim',
        'sensitivity_power_iters',
        'disable_attn_rotation_gauge',
        'disable_gauge_verify',
        'sensitivity_calib_size',
        'sensitivity_keep_ratio',
        'sensitivity_quant_bits',
        'sensitivity_allocation',
        'sensitivity_output',
        'sensitivity_csv_output',
        'decode_fraction',
        'decode_planes',
        'include_head_linear',
        'activation_quant_enabled',
        'activation_quant_bit',
        'activation_quant_scheme',
        'activation_observer_type',
        'activation_observer_percentile',
        'activation_quant_eps',
        'activation_max_observer_samples',
        'activation_observer_samples_per_update',
        'activation_linear_inputs',
        'activation_matmul_qk_inputs',
        'activation_matmul_av_inputs',
        'activation_include_head_linear_input',
        'activation_skip_layernorm_outputs',
        'activation_calibration_source',
        'activation_calibration_num_images',
        'activation_calibration_seed',
        'activation_calibration_batch_size',
        'activation_calibration_workers',
        'activation_calibration_pin_memory',
        'activation_calibration_persistent_workers',
        'allocation_report_output',
        'allocation_csv_output',
        'streaming_runtime_enabled',
        'streaming_runtime_archive_invariant',
        'streaming_runtime_mode',
        'streaming_runtime_logical_scheduler_unit',
        'streaming_runtime_group_policy',
        'streaming_runtime_super_block_k',
        'streaming_runtime_auto_profile_batches',
        'streaming_runtime_auto_compute_decode_ratio_target',
        'streaming_runtime_cache_decoded_groups',
        'streaming_runtime_primitive_unit',
        'streaming_runtime_modes',
        'streaming_runtime_eval_batch_size',
        'streaming_runtime_eval_samples',
        'streaming_runtime_warmup_batches',
        'streaming_runtime_repeats',
        'streaming_runtime_reset_decoded_cache_before_timed_eval',
        'streaming_runtime_drop_first_repeat_for_aggregation',
        'streaming_runtime_overlap_enabled',
        'streaming_runtime_double_buffering',
        'streaming_runtime_prefetch_next_group',
        'streaming_runtime_track_peak_gpu_mem',
        'streaming_runtime_track_peak_decode_load_gpu_mem',
        'streaming_runtime_track_peak_model_resident_gpu_mem',
        'streaming_runtime_output_json',
        'streaming_runtime_output_csv',
    ]:
        if arg_name in config:
            value = config[arg_name]
            if isinstance(value, dict):
                continue
            assign(arg_name, value)
    if 'sens_beta' not in config and 'sens_exponent' in config:
        value = config['sens_exponent']
        if not isinstance(value, dict):
            assign('sens_beta', value)
    if 'importance_alpha' not in config and 'alpha' in config:
        value = config['alpha']
        if not isinstance(value, dict):
            assign('importance_alpha', value)
    if 'importance_quantile_q' not in config and 'q' in config:
        value = config['q']
        if not isinstance(value, dict):
            assign('importance_quantile_q', value)
    return defaults


def _parse_args_with_config(parser: argparse.ArgumentParser):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', type=str, default='')
    pre_args, _ = pre_parser.parse_known_args()
    config_dict: Dict[str, Any] = {}
    if pre_args.config:
        config_dict = _load_yaml_config(pre_args.config)
        parser.set_defaults(**_config_to_arg_defaults(config_dict))
    args = parser.parse_args()
    return args, config_dict


_bootstrap_cuda_visible_devices_from_config()

from devdna_core import (
    ensure_torchvision_nms_stub,
    format_bytes,
    set_global_seed,
)
try:
    from devdna_activation_fake_quant import ActivationFakeQuantController
    _ACTIVATION_QUANT_IMPORT_ERROR: Optional[str] = None
except Exception as _aq_import_exc:
    ActivationFakeQuantController = None  # type: ignore[assignment]
    _ACTIVATION_QUANT_IMPORT_ERROR = str(_aq_import_exc)
from devdna_eval import (
    build_activation_calibration_loader,
    build_timm_model,
    create_official_imagenet_loader,
    evaluate_with_official_logic,
    maybe_wrap_dataparallel,
    run_activation_calibration,
)
from devdna_streaming_runtime import run_streaming_runtime_familywise_experiment
from devdna_model import (
    ARCHIVE_FORMAT,
    analyze_vit_block_sensitivity,
    compress_vit_to_archive_blockwise_thresholded,
    compress_vit_to_archive,
    inspect_gaugedna_archive,
    is_gaugedna_archive,
    load_gaugedna_archive,
    write_allocation_csv,
    write_allocation_report,
)


DEFAULT_DATA_ROOT = '/home/wangjunjie/wjjcode/Neurips/dasp_vit/imagenet'


def _compact_archive_manifest_for_report(manifest: Dict[str, Any]) -> Dict[str, Any]:
    compact = dict(manifest)
    removed_detail_sections = False

    exact_tensors = compact.pop('exact_tensors', None)
    if isinstance(exact_tensors, list):
        compact['exact_tensors_count'] = int(len(exact_tensors))
        removed_detail_sections = True

    tensor_stacks = compact.pop('tensor_stacks', None)
    if isinstance(tensor_stacks, dict):
        compact['tensor_stacks_count'] = int(len(tensor_stacks))
        removed_detail_sections = True

    stacks = compact.pop('stacks', None)
    if isinstance(stacks, dict):
        compact['stacks_count'] = int(len(stacks))
        removed_detail_sections = True

    if removed_detail_sections:
        compact['layer_details_omitted'] = True
    return compact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='GaugeDNA / DevDNA storage-side compression for timm ViT models.')
    parser.add_argument('--config', type=str, default='')
    parser.add_argument(
        '--mode',
        type=str,
        default='compress_eval',
        choices=[
            'compress',
            'eval',
            'compress_eval',
            'inspect',
            'analyze_sensitivity',
            'eval_pretrained',
            'eval_streaming_runtime_familywise',
        ],
    )
    parser.add_argument('--model', type=str, default='vit_base_patch16_224')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--pretrained-weight-path', type=str, default='')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--archive-path', type=str, default='outputs/gaugedna_vit.zip')
    parser.add_argument('--save-report', type=str, default='outputs/gaugedna_report.json')
    parser.add_argument('--save-decoded-checkpoint', type=str, default='')

    parser.add_argument('--data-root', type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--class-map', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--pin-memory', action='store_true')
    parser.add_argument('--no-prefetcher', action='store_true')
    parser.add_argument('--crop-pct', type=float, default=None)
    parser.add_argument('--interpolation', type=str, default='')

    parser.add_argument('--cuda-devices', type=str, default='0')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--amp-dtype', type=str, default='fp16', choices=['fp16', 'bf16'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print-freq', type=int, default=20)

    parser.add_argument('--mlp-keep-ratio', type=float, default=0.35)
    parser.add_argument('--attn-keep-ratio', type=float, default=0.35)
    parser.add_argument('--quant-bits', type=int, default=12)
    parser.add_argument(
        '--codec-allocation-mode',
        type=str,
        default='blockwise_thresholded',
        choices=['legacy_global', 'blockwise_thresholded'],
    )
    parser.add_argument('--sensitivity-report', type=str, default='')
    parser.add_argument('--target-archive-mb', type=float, default=0.0)
    parser.add_argument('--target-archive-tolerance-ratio', type=float, default=0.0001)
    parser.add_argument('--target-archive-max-iters', type=int, default=40)
    parser.add_argument('--lambda-probe-start', type=float, default=1e-2)
    parser.add_argument('--sens-weight-ce', type=float, default=0.0)
    parser.add_argument('--sens-weight-logit', type=float, default=0.45)
    parser.add_argument('--sens-weight-byte', type=float, default=0.55)
    parser.add_argument('--importance-alpha', '--alpha', dest='importance_alpha', type=float, default=2.0)
    parser.add_argument('--importance-quantile-q', '--q', dest='importance_quantile_q', type=float, default=0.95)
    parser.add_argument('--sens-beta', '--sens-exponent', dest='sens_beta', type=float, default=1.9)
    parser.add_argument('--min-tensor-keep-ratio', type=float, default=0.0)
    parser.add_argument('--allocation-report-output', type=str, default='')
    parser.add_argument('--allocation-csv-output', type=str, default='')
    parser.add_argument('--gauge-proj-dim', type=int, default=16)
    parser.add_argument('--gauge-pca-dim', type=int, default=8)
    parser.add_argument('--sensitivity-power-iters', type=int, default=16)
    parser.add_argument('--disable-attn-rotation-gauge', action='store_true')
    parser.add_argument('--disable-gauge-verify', action='store_true')
    parser.add_argument('--sensitivity-calib-size', type=int, default=1024)
    parser.add_argument('--sensitivity-keep-ratio', type=float, default=0.35)
    parser.add_argument('--sensitivity-quant-bits', type=int, default=12)
    parser.add_argument(
        '--sensitivity-allocation',
        type=str,
        default='equal_keep_ratio',
        choices=['equal_keep_ratio', 'sensitivity_weighted'],
    )
    parser.add_argument('--sensitivity-output', type=str, default='outputs/sensitivity_report.json')
    parser.add_argument('--sensitivity-csv-output', type=str, default='outputs/sensitivity_table.csv')


    parser.add_argument('--streaming-runtime-enabled', dest='streaming_runtime_enabled', action='store_true')
    parser.add_argument('--streaming-runtime-disabled', dest='streaming_runtime_enabled', action='store_false')
    parser.add_argument('--streaming-runtime-archive-invariant', dest='streaming_runtime_archive_invariant', action='store_true')
    parser.add_argument(
        '--streaming-runtime-no-archive-invariant',
        dest='streaming_runtime_archive_invariant',
        action='store_false',
    )
    parser.add_argument('--streaming-runtime-mode', type=str, default='familywise_runtime')
    parser.add_argument('--streaming-runtime-logical-scheduler-unit', type=str, default='block_attn_mlp')
    parser.add_argument(
        '--streaming-runtime-group-policy',
        type=str,
        default='auto',
        choices=['auto', 'block_attn_mlp', 'block_full', 'super_block'],
    )
    parser.add_argument('--streaming-runtime-super-block-k', type=int, default=2)
    parser.add_argument('--streaming-runtime-auto-profile-batches', type=int, default=1)
    parser.add_argument('--streaming-runtime-auto-compute-decode-ratio-target', type=float, default=1.15)
    parser.add_argument(
        '--streaming-runtime-cache-decoded-groups',
        dest='streaming_runtime_cache_decoded_groups',
        action='store_true',
    )
    parser.add_argument(
        '--streaming-runtime-no-cache-decoded-groups',
        dest='streaming_runtime_cache_decoded_groups',
        action='store_false',
    )
    parser.add_argument('--streaming-runtime-primitive-unit', type=str, default='tensor_or_family_entry')
    parser.add_argument(
        '--streaming-runtime-modes',
        type=str,
        default='serial_full_decode,lazy_family_decode,lazy_family_decode_overlap,oracle_ready_to_run',
    )
    parser.add_argument('--streaming-runtime-eval-batch-size', type=int, default=128)
    parser.add_argument('--streaming-runtime-eval-samples', type=int, default=1024)
    parser.add_argument('--streaming-runtime-warmup-batches', type=int, default=5)
    parser.add_argument('--streaming-runtime-repeats', type=int, default=3)
    parser.add_argument(
        '--streaming-runtime-reset-decoded-cache-before-timed-eval',
        dest='streaming_runtime_reset_decoded_cache_before_timed_eval',
        action='store_true',
    )
    parser.add_argument(
        '--streaming-runtime-no-reset-decoded-cache-before-timed-eval',
        dest='streaming_runtime_reset_decoded_cache_before_timed_eval',
        action='store_false',
    )
    parser.add_argument(
        '--streaming-runtime-drop-first-repeat-for-aggregation',
        dest='streaming_runtime_drop_first_repeat_for_aggregation',
        action='store_true',
    )
    parser.add_argument(
        '--streaming-runtime-no-drop-first-repeat-for-aggregation',
        dest='streaming_runtime_drop_first_repeat_for_aggregation',
        action='store_false',
    )
    parser.add_argument('--streaming-runtime-overlap-enabled', dest='streaming_runtime_overlap_enabled', action='store_true')
    parser.add_argument(
        '--streaming-runtime-overlap-disabled',
        dest='streaming_runtime_overlap_enabled',
        action='store_false',
    )
    parser.add_argument('--streaming-runtime-double-buffering', dest='streaming_runtime_double_buffering', action='store_true')
    parser.add_argument(
        '--streaming-runtime-no-double-buffering',
        dest='streaming_runtime_double_buffering',
        action='store_false',
    )
    parser.add_argument('--streaming-runtime-prefetch-next-group', dest='streaming_runtime_prefetch_next_group', action='store_true')
    parser.add_argument(
        '--streaming-runtime-no-prefetch-next-group',
        dest='streaming_runtime_prefetch_next_group',
        action='store_false',
    )
    parser.add_argument('--streaming-runtime-track-peak-gpu-mem', dest='streaming_runtime_track_peak_gpu_mem', action='store_true')
    parser.add_argument(
        '--streaming-runtime-no-track-peak-gpu-mem',
        dest='streaming_runtime_track_peak_gpu_mem',
        action='store_false',
    )
    parser.add_argument(
        '--streaming-runtime-track-peak-decode-load-gpu-mem',
        dest='streaming_runtime_track_peak_decode_load_gpu_mem',
        action='store_true',
    )
    parser.add_argument(
        '--streaming-runtime-no-track-peak-decode-load-gpu-mem',
        dest='streaming_runtime_track_peak_decode_load_gpu_mem',
        action='store_false',
    )
    parser.add_argument(
        '--streaming-runtime-track-peak-model-resident-gpu-mem',
        dest='streaming_runtime_track_peak_model_resident_gpu_mem',
        action='store_true',
    )
    parser.add_argument(
        '--streaming-runtime-no-track-peak-model-resident-gpu-mem',
        dest='streaming_runtime_track_peak_model_resident_gpu_mem',
        action='store_false',
    )
    parser.add_argument(
        '--streaming-runtime-output-json',
        type=str,
        default='outputs/streaming_runtime_familywise_results.json',
    )
    parser.add_argument(
        '--streaming-runtime-output-csv',
        type=str,
        default='outputs/streaming_runtime_familywise_results.csv',
    )

    parser.add_argument('--decode-fraction', type=float, default=1.0)
    parser.add_argument('--decode-planes', type=int, default=-1)
    parser.add_argument('--include-head-linear', dest='include_head_linear', action='store_true')
    parser.add_argument('--exclude-head-linear', dest='include_head_linear', action='store_false')

    parser.add_argument('--activation-quant-enabled', dest='activation_quant_enabled', action='store_true')
    parser.add_argument('--activation-quant-disabled', dest='activation_quant_enabled', action='store_false')
    parser.add_argument('--activation-quant-bit', type=int, default=8)
    parser.add_argument('--activation-quant-scheme', type=str, default='symmetric', choices=['symmetric', 'asymmetric'])
    parser.add_argument('--activation-observer-type', type=str, default='percentile', choices=['minmax', 'percentile'])
    parser.add_argument('--activation-observer-percentile', type=float, default=0.999)
    parser.add_argument('--activation-quant-eps', type=float, default=1.0e-8)
    parser.add_argument('--activation-max-observer-samples', type=int, default=262144)
    parser.add_argument('--activation-observer-samples-per-update', type=int, default=8192)

    parser.add_argument('--activation-linear-inputs', dest='activation_linear_inputs', action='store_true')
    parser.add_argument('--no-activation-linear-inputs', dest='activation_linear_inputs', action='store_false')
    parser.add_argument('--activation-matmul-qk-inputs', dest='activation_matmul_qk_inputs', action='store_true')
    parser.add_argument('--no-activation-matmul-qk-inputs', dest='activation_matmul_qk_inputs', action='store_false')
    parser.add_argument('--activation-matmul-av-inputs', dest='activation_matmul_av_inputs', action='store_true')
    parser.add_argument('--no-activation-matmul-av-inputs', dest='activation_matmul_av_inputs', action='store_false')
    parser.add_argument('--activation-include-head-linear-input', dest='activation_include_head_linear_input', action='store_true')
    parser.add_argument('--activation-exclude-head-linear-input', dest='activation_include_head_linear_input', action='store_false')
    parser.add_argument('--activation-skip-layernorm-outputs', dest='activation_skip_layernorm_outputs', action='store_true')
    parser.add_argument('--activation-no-skip-layernorm-outputs', dest='activation_skip_layernorm_outputs', action='store_false')

    parser.add_argument('--activation-calibration-source', type=str, default='val', choices=['train', 'val'])
    parser.add_argument('--activation-calibration-num-images', type=int, default=128)
    parser.add_argument('--activation-calibration-seed', type=int, default=3)
    parser.add_argument('--activation-calibration-batch-size', type=int, default=256)
    parser.add_argument('--activation-calibration-workers', type=int, default=4)
    parser.add_argument('--activation-calibration-pin-memory', dest='activation_calibration_pin_memory', action='store_true')
    parser.add_argument('--activation-calibration-no-pin-memory', dest='activation_calibration_pin_memory', action='store_false')
    parser.add_argument(
        '--activation-calibration-persistent-workers',
        dest='activation_calibration_persistent_workers',
        action='store_true',
    )
    parser.add_argument(
        '--activation-calibration-no-persistent-workers',
        dest='activation_calibration_persistent_workers',
        action='store_false',
    )

    parser.set_defaults(
        include_head_linear=False,
        activation_quant_enabled=False,
        activation_linear_inputs=True,
        activation_matmul_qk_inputs=True,
        activation_matmul_av_inputs=True,
        activation_include_head_linear_input=False,
        activation_skip_layernorm_outputs=True,
        activation_calibration_pin_memory=True,
        activation_calibration_persistent_workers=True,
        streaming_runtime_enabled=True,
        streaming_runtime_archive_invariant=True,
        streaming_runtime_overlap_enabled=True,
        streaming_runtime_double_buffering=True,
        streaming_runtime_prefetch_next_group=True,
        streaming_runtime_cache_decoded_groups=True,
        streaming_runtime_reset_decoded_cache_before_timed_eval=True,
        streaming_runtime_drop_first_repeat_for_aggregation=True,
        streaming_runtime_track_peak_gpu_mem=True,
        streaming_runtime_track_peak_decode_load_gpu_mem=True,
        streaming_runtime_track_peak_model_resident_gpu_mem=True,
    )
    return parser


@torch.no_grad()
def create_device(device_text: str) -> torch.device:
    if device_text == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is requested but no CUDA device is available.')
        return torch.device('cuda:0')
    return torch.device(device_text)


@torch.no_grad()
def build_or_load_model(args, device: torch.device):
    build_meta: Dict[str, Any] = {
        'archive_decode_seconds': None,
        'loaded_from_archive': False,
    }
    if not isinstance(args.model, str) or not args.model.strip():
        raise TypeError(
            f'`model` must be a non-empty string, but got {type(args.model)} value={args.model!r}. '
            'If using --config, set `model.name: <timm_model_name>`.'
        )
    if args.mode == 'eval' and args.archive_path and is_gaugedna_archive(args.archive_path):
        decode_start_time = time.perf_counter()
        model, archive_manifest = load_gaugedna_archive(
            archive_path=args.archive_path,
            model_builder=lambda model_name: build_timm_model(
                model_name=model_name,
                pretrained=False,
                checkpoint_path='',
                pretrained_weight_path='',
            ),
            device=device,
            decode_fraction=float(args.decode_fraction),
            decode_planes=(None if args.decode_planes <= 0 else int(args.decode_planes)),
        )
        build_meta['archive_decode_seconds'] = float(time.perf_counter() - decode_start_time)
        build_meta['loaded_from_archive'] = True
        return model, archive_manifest, build_meta

    if args.mode == 'inspect':
        return None, inspect_gaugedna_archive(args.archive_path), build_meta

    model = build_timm_model(
        model_name=args.model,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint,
        pretrained_weight_path=args.pretrained_weight_path,
    )
    model.to(device)
    model.eval()
    return model, None, build_meta


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, args, device: torch.device) -> Dict[str, Any]:
    # Always align evaluation preprocessing with timm official defaults for the selected model.
    use_test_size = True
    activation_quant_enabled = bool(getattr(args, 'activation_quant_enabled', False))
    if activation_quant_enabled:
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print('[run_devdna_vit] activation_quant enabled: disable DataParallel for stable observer statistics.')
        eval_model = model
    else:
        eval_model = maybe_wrap_dataparallel(model)
    model_for_loader = model.module if isinstance(eval_model, torch.nn.DataParallel) else model

    loader, data_config = create_official_imagenet_loader(
        model=model_for_loader,
        data_root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        workers=args.workers,
        class_map=args.class_map,
        pin_memory=args.pin_memory,
        crop_pct=args.crop_pct,
        interpolation=args.interpolation,
        no_prefetcher=args.no_prefetcher,
        device=device,
        use_test_size=use_test_size,
        enforce_official_preprocess=True,
    )
    activation_controller: Optional[Any] = None
    activation_summary: Optional[Dict[str, Any]] = None
    activation_calibration_meta: Optional[Dict[str, Any]] = None
    use_amp_eval = bool(device.type == 'cuda')
    try:
        if activation_quant_enabled:
            if ActivationFakeQuantController is None:
                raise ModuleNotFoundError(
                    'Activation quantization is enabled, but `devdna_activation_fake_quant.py` is unavailable. '
                    f'Original import error: {_ACTIVATION_QUANT_IMPORT_ERROR}. '
                    'Please make sure this file exists in the project root, or disable activation quantization '
                    '(set `eval.activation_quant.enabled: false` or pass `--activation-quant-disabled`).'
                )
            activation_controller = ActivationFakeQuantController(
                model=model_for_loader,
                bit=int(args.activation_quant_bit),
                scheme=str(args.activation_quant_scheme),
                observer_type=str(args.activation_observer_type),
                percentile=float(args.activation_observer_percentile),
                eps=float(args.activation_quant_eps),
                linear_inputs=bool(args.activation_linear_inputs),
                matmul_qk_inputs=bool(args.activation_matmul_qk_inputs),
                matmul_av_inputs=bool(args.activation_matmul_av_inputs),
                include_head_linear_input=bool(args.activation_include_head_linear_input),
                skip_layernorm_outputs=bool(args.activation_skip_layernorm_outputs),
                max_observer_samples=int(args.activation_max_observer_samples),
                observer_samples_per_update=int(args.activation_observer_samples_per_update),
            )
            calib_loader, calib_source, _calib_indices, calib_data_config = build_activation_calibration_loader(
                model=model_for_loader,
                data_root=args.data_root,
                source=str(args.activation_calibration_source),
                num_images=int(args.activation_calibration_num_images),
                seed=int(args.activation_calibration_seed),
                batch_size=int(args.activation_calibration_batch_size),
                workers=int(args.activation_calibration_workers),
                pin_memory=bool(args.activation_calibration_pin_memory),
                persistent_workers=bool(args.activation_calibration_persistent_workers),
                crop_pct=args.crop_pct,
                interpolation=args.interpolation,
                use_test_size=use_test_size,
                enforce_official_preprocess=True,
            )
            print(
                '[run_devdna_vit] activation calibration: '
                f'source={calib_source}, samples={len(calib_loader.dataset)}, '
                f'batch_size={calib_loader.batch_size}',
                flush=True,
            )
            run_activation_calibration(
                model=eval_model,
                dataloader=calib_loader,
                device=device,
                controller=activation_controller,
            )
            activation_summary = activation_controller.summary()
            activation_calibration_meta = {
                'source': str(calib_source),
                'samples': int(len(calib_loader.dataset)),
                'batch_size': int(calib_loader.batch_size),
                'seed': int(args.activation_calibration_seed),
                'data_config': calib_data_config,
            }
            use_amp_eval = False

        result = evaluate_with_official_logic(
            model=eval_model,
            loader=loader,
            device=device,
            amp_dtype=args.amp_dtype,
            print_freq=args.print_freq,
            use_amp=bool(use_amp_eval),
        )
    finally:
        if activation_controller is not None:
            activation_controller.restore()

    output: Dict[str, Any] = {
        'top1': result.top1,
        'top5': result.top5,
        'loss': result.loss,
        'rmse': result.rmse,
        'samples': result.samples,
        'throughput': result.throughput,
        'ms_per_sample': result.ms_per_sample,
        'peak_gpu_mem_bytes': result.peak_gpu_mem_bytes,
        'peak_gpu_mem_human': format_bytes(result.peak_gpu_mem_bytes),
        'data_config': data_config,
        'amp_enabled': bool(use_amp_eval),
        'use_test_size': bool(use_test_size),
        'preprocess_profile': 'timm_validate_official_like',
    }
    if activation_summary is not None:
        output['activation_quant'] = activation_summary
    if activation_calibration_meta is not None:
        output['activation_calibration'] = activation_calibration_meta
    return output


@torch.no_grad()
def maybe_save_dense_checkpoint(model: torch.nn.Module, path: str) -> Optional[int]:
    if not path:
        return None
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return int(os.path.getsize(path))


def _build_decode_induced_e2e_latency_report(
    decode_seconds: float,
    eval_samples: int,
) -> Dict[str, Any]:
    decode_seconds = float(max(0.0, decode_seconds))
    output: Dict[str, Any] = {
        'decode_induced_e2e_inference_latency_seconds': float(decode_seconds),
        'decode_induced_e2e_inference_latency_ms': float(decode_seconds * 1000.0),
        'definition': 'extra end-to-end waiting time introduced by decode before inference (first request, no overlap)',
    }
    if int(eval_samples) > 0:
        output['decode_induced_e2e_inference_latency_ms_per_sample_amortized'] = float(
            (decode_seconds * 1000.0) / float(eval_samples)
        )
    else:
        output['decode_induced_e2e_inference_latency_ms_per_sample_amortized'] = None
    return output


@torch.no_grad()
def main() -> None:
    ensure_torchvision_nms_stub()
    parser = build_parser()
    args, loaded_config = _parse_args_with_config(parser)
    if getattr(args, 'cuda_devices', ''):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_devices)
    set_global_seed(args.seed)
    device = create_device(args.device)
    print(
        f'[run_devdna_vit] mode={args.mode}, codec_allocation_mode={getattr(args, "codec_allocation_mode", "blockwise_thresholded")}, '
        f'model={args.model}, device={device}',
        flush=True,
    )

    report: Dict[str, Any] = {
        'config': str(args.config),
        'config_loaded': bool(bool(getattr(args, 'config', ''))),
        'config_keys': sorted(list(loaded_config.keys())) if loaded_config else [],
        'mode': args.mode,
        'model': args.model,
        'device': str(device),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        'archive_format': ARCHIVE_FORMAT,
        'pretrained': bool(args.pretrained),
        'pretrained_weight_path': args.pretrained_weight_path,
        'checkpoint': args.checkpoint,
        'data_root': args.data_root,
        'split': args.split,
        'batch_size': args.batch_size,
        'workers': args.workers,
        'mlp_keep_ratio': args.mlp_keep_ratio,
        'attn_keep_ratio': args.attn_keep_ratio,
        'quant_bits': args.quant_bits,
        'codec_allocation_mode': args.codec_allocation_mode,
        'sensitivity_report': args.sensitivity_report,
        'target_archive_mb': args.target_archive_mb,
        'target_archive_tolerance_ratio': args.target_archive_tolerance_ratio,
        'target_archive_max_iters': args.target_archive_max_iters,
        'lambda_probe_start': args.lambda_probe_start,
        'sens_weight_ce': args.sens_weight_ce,
        'sens_weight_logit': args.sens_weight_logit,
        'sens_weight_byte': args.sens_weight_byte,
        'importance_alpha': args.importance_alpha,
        'importance_quantile_q': args.importance_quantile_q,
        'sens_beta': args.sens_beta,
        'min_tensor_keep_ratio': args.min_tensor_keep_ratio,
        'include_head_linear': bool(args.include_head_linear),
        'allocation_report_output': args.allocation_report_output,
        'allocation_csv_output': args.allocation_csv_output,
        'gauge_proj_dim': args.gauge_proj_dim,
        'gauge_pca_dim': args.gauge_pca_dim,
        'sensitivity_calib_size': args.sensitivity_calib_size,
        'sensitivity_keep_ratio': args.sensitivity_keep_ratio,
        'sensitivity_quant_bits': args.sensitivity_quant_bits,
        'sensitivity_allocation': args.sensitivity_allocation,
        'sensitivity_output': args.sensitivity_output,
        'sensitivity_csv_output': args.sensitivity_csv_output,
        'decode_fraction': args.decode_fraction,
        'decode_planes': None if args.decode_planes <= 0 else args.decode_planes,
        'activation_quant_enabled': bool(args.activation_quant_enabled),
        'activation_quant_bit': int(args.activation_quant_bit),
        'activation_quant_scheme': str(args.activation_quant_scheme),
        'activation_observer_type': str(args.activation_observer_type),
        'activation_observer_percentile': float(args.activation_observer_percentile),
        'activation_quant_eps': float(args.activation_quant_eps),
        'activation_max_observer_samples': int(args.activation_max_observer_samples),
        'activation_observer_samples_per_update': int(args.activation_observer_samples_per_update),
        'activation_linear_inputs': bool(args.activation_linear_inputs),
        'activation_matmul_qk_inputs': bool(args.activation_matmul_qk_inputs),
        'activation_matmul_av_inputs': bool(args.activation_matmul_av_inputs),
        'activation_include_head_linear_input': bool(args.activation_include_head_linear_input),
        'activation_skip_layernorm_outputs': bool(args.activation_skip_layernorm_outputs),
        'activation_calibration_source': str(args.activation_calibration_source),
        'activation_calibration_num_images': int(args.activation_calibration_num_images),
        'activation_calibration_seed': int(args.activation_calibration_seed),
        'activation_calibration_batch_size': int(args.activation_calibration_batch_size),
        'activation_calibration_workers': int(args.activation_calibration_workers),
        'activation_calibration_pin_memory': bool(args.activation_calibration_pin_memory),
        'activation_calibration_persistent_workers': bool(args.activation_calibration_persistent_workers),
        'streaming_runtime_enabled': bool(args.streaming_runtime_enabled),
        'streaming_runtime_archive_invariant': bool(args.streaming_runtime_archive_invariant),
        'streaming_runtime_mode': str(args.streaming_runtime_mode),
        'streaming_runtime_logical_scheduler_unit': str(args.streaming_runtime_logical_scheduler_unit),
        'streaming_runtime_group_policy': str(args.streaming_runtime_group_policy),
        'streaming_runtime_super_block_k': int(args.streaming_runtime_super_block_k),
        'streaming_runtime_auto_profile_batches': int(args.streaming_runtime_auto_profile_batches),
        'streaming_runtime_auto_compute_decode_ratio_target': float(args.streaming_runtime_auto_compute_decode_ratio_target),
        'streaming_runtime_cache_decoded_groups': bool(args.streaming_runtime_cache_decoded_groups),
        'streaming_runtime_primitive_unit': str(args.streaming_runtime_primitive_unit),
        'streaming_runtime_modes': str(args.streaming_runtime_modes),
        'streaming_runtime_eval_batch_size': int(args.streaming_runtime_eval_batch_size),
        'streaming_runtime_eval_samples': int(args.streaming_runtime_eval_samples),
        'streaming_runtime_warmup_batches': int(args.streaming_runtime_warmup_batches),
        'streaming_runtime_repeats': int(args.streaming_runtime_repeats),
        'streaming_runtime_reset_decoded_cache_before_timed_eval': bool(args.streaming_runtime_reset_decoded_cache_before_timed_eval),
        'streaming_runtime_drop_first_repeat_for_aggregation': bool(args.streaming_runtime_drop_first_repeat_for_aggregation),
        'streaming_runtime_overlap_enabled': bool(args.streaming_runtime_overlap_enabled),
        'streaming_runtime_double_buffering': bool(args.streaming_runtime_double_buffering),
        'streaming_runtime_prefetch_next_group': bool(args.streaming_runtime_prefetch_next_group),
        'streaming_runtime_track_peak_gpu_mem': bool(args.streaming_runtime_track_peak_gpu_mem),
        'streaming_runtime_track_peak_decode_load_gpu_mem': bool(args.streaming_runtime_track_peak_decode_load_gpu_mem),
        'streaming_runtime_track_peak_model_resident_gpu_mem': bool(args.streaming_runtime_track_peak_model_resident_gpu_mem),
        'streaming_runtime_output_json': str(args.streaming_runtime_output_json),
        'streaming_runtime_output_csv': str(args.streaming_runtime_output_csv),
    }
    run_start_time = time.perf_counter()
    distribution_start_time: Optional[float] = None
    timing_report: Dict[str, Any] = {}

    if args.mode == 'inspect':
        manifest = inspect_gaugedna_archive(args.archive_path)
        report['archive'] = manifest
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if args.save_report:
            Path(args.save_report).parent.mkdir(parents=True, exist_ok=True)
            with open(args.save_report, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        return


    if args.mode == 'eval_streaming_runtime_familywise':
        streaming_runtime_report = run_streaming_runtime_familywise_experiment(args=args, device=device)
        report['streaming_runtime'] = streaming_runtime_report
        report['timing'] = {'total_runtime_seconds': float(time.perf_counter() - run_start_time)}
        if args.save_report:
            Path(args.save_report).parent.mkdir(parents=True, exist_ok=True)
            with open(args.save_report, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f'Saved report to: {args.save_report}')
        else:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        return


    distribution_start_time = time.perf_counter()
    model_prepare_start_time = time.perf_counter()
    model, archive_meta, model_build_meta = build_or_load_model(args, device)
    timing_report['initial_model_prepare_seconds'] = float(time.perf_counter() - model_prepare_start_time)
    if model_build_meta.get('archive_decode_seconds', None) is not None:
        timing_report['archive_decode_seconds'] = float(model_build_meta['archive_decode_seconds'])

    if args.mode == 'analyze_sensitivity':
        sensitivity_report = analyze_vit_block_sensitivity(
            model=model,
            model_name=args.model,
            data_root=args.data_root,
            split=args.split,
            batch_size=int(args.batch_size),
            workers=int(args.workers),
            class_map=args.class_map,
            pin_memory=bool(args.pin_memory),
            crop_pct=args.crop_pct,
            interpolation=args.interpolation,
            no_prefetcher=bool(args.no_prefetcher),
            device=device,
            sensitivity_calib_size=int(args.sensitivity_calib_size),
            sensitivity_keep_ratio=float(args.sensitivity_keep_ratio),
            sensitivity_quant_bits=int(args.sensitivity_quant_bits),
            sensitivity_allocation=str(args.sensitivity_allocation),
            gauge_proj_dim=int(args.gauge_proj_dim),
            gauge_pca_dim=int(args.gauge_pca_dim),
            sensitivity_power_iters=int(args.sensitivity_power_iters),
            enable_attn_rotation_gauge=not args.disable_attn_rotation_gauge,
            verify_gauge=not args.disable_gauge_verify,
            include_head_linear=bool(args.include_head_linear),
            print_freq=int(args.print_freq),
            sensitivity_csv_output=args.sensitivity_csv_output,
        )
        print(json.dumps(sensitivity_report, indent=2, ensure_ascii=False))
        if args.sensitivity_output:
            Path(args.sensitivity_output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.sensitivity_output, 'w', encoding='utf-8') as f:
                json.dump(sensitivity_report, f, indent=2, ensure_ascii=False)
            print(f'Saved sensitivity report to: {args.sensitivity_output}')
        if args.save_report:
            # Allowed by design: in sensitivity mode, save_report may store the same
            # JSON payload as sensitivity_output for compatibility with existing scripts.
            Path(args.save_report).parent.mkdir(parents=True, exist_ok=True)
            with open(args.save_report, 'w', encoding='utf-8') as f:
                json.dump(sensitivity_report, f, indent=2, ensure_ascii=False)
            print(f'Saved report to: {args.save_report}')
        return

    if args.mode in ('compress', 'compress_eval'):
        print('[run_devdna_vit] Enter compression path...', flush=True)
        compression_start_time = time.perf_counter()
        allocation_report: Optional[Dict[str, Any]] = None
        if str(args.codec_allocation_mode) == 'blockwise_thresholded':
            print('[run_devdna_vit] Use blockwise_thresholded allocation mode.', flush=True)
            summary, allocation_report = compress_vit_to_archive_blockwise_thresholded(
                model=model,
                save_path=args.archive_path,
                model_name=args.model,
                quant_bits=int(args.quant_bits),
                device=device,
                sensitivity_report_path=args.sensitivity_report,
                target_archive_mb=float(args.target_archive_mb),
                target_archive_tolerance_ratio=float(args.target_archive_tolerance_ratio),
                target_archive_max_iters=int(args.target_archive_max_iters),
                lambda_probe_start=float(args.lambda_probe_start),
                sens_weight_ce=float(args.sens_weight_ce),
                sens_weight_logit=float(args.sens_weight_logit),
                sens_weight_byte=float(args.sens_weight_byte),
                importance_alpha=float(args.importance_alpha),
                importance_quantile_q=float(args.importance_quantile_q),
                sens_beta=float(args.sens_beta),
                min_tensor_keep_ratio=float(args.min_tensor_keep_ratio),
                gauge_proj_dim=int(args.gauge_proj_dim),
                gauge_pca_dim=int(args.gauge_pca_dim),
                enable_attn_rotation_gauge=not args.disable_attn_rotation_gauge,
                verify_gauge=not args.disable_gauge_verify,
                include_head_linear=bool(args.include_head_linear),
            )
        else:
            print('[run_devdna_vit] Use legacy_global allocation mode.', flush=True)
            if float(args.min_tensor_keep_ratio) > 0.0:
                print(
                    '[run_devdna_vit] warning: `min_tensor_keep_ratio` only takes effect in '
                    '`codec_allocation_mode=blockwise_thresholded`; current legacy_global path will ignore it.',
                    flush=True,
                )
            summary = compress_vit_to_archive(
                model=model,
                save_path=args.archive_path,
                model_name=args.model,
                mlp_keep_ratio=float(args.mlp_keep_ratio),
                attn_keep_ratio=float(args.attn_keep_ratio),
                quant_bits=int(args.quant_bits),
                device=device,
                gauge_proj_dim=int(args.gauge_proj_dim),
                gauge_pca_dim=int(args.gauge_pca_dim),
                sensitivity_power_iters=int(args.sensitivity_power_iters),
                enable_attn_rotation_gauge=not args.disable_attn_rotation_gauge,
                verify_gauge=not args.disable_gauge_verify,
                include_head_linear=bool(args.include_head_linear),
            )
        timing_report['compression_seconds'] = float(time.perf_counter() - compression_start_time)
        report['compression'] = summary.to_dict()
        if allocation_report is not None:
            report['allocation'] = allocation_report
            if args.allocation_report_output:
                saved_report = write_allocation_report(allocation_report, args.allocation_report_output)
                report['allocation_report_saved'] = saved_report
                print(f'Saved allocation report to: {saved_report}')
            if args.allocation_csv_output:
                saved_csv = write_allocation_csv(allocation_report.get('blocks', []), args.allocation_csv_output)
                report['allocation_csv_saved'] = saved_csv
                print(f'Saved allocation csv to: {saved_csv}')
        print(json.dumps(report['compression'], indent=2, ensure_ascii=False))
        if args.mode == 'compress':
            timing_report['total_runtime_seconds'] = float(time.perf_counter() - run_start_time)
            report['timing'] = timing_report
            if args.save_report:
                Path(args.save_report).parent.mkdir(parents=True, exist_ok=True)
                with open(args.save_report, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
            return
        decode_archive_start_time = time.perf_counter()
        model, archive_meta = load_gaugedna_archive(
            archive_path=args.archive_path,
            model_builder=lambda model_name: build_timm_model(
                model_name=model_name,
                pretrained=False,
                checkpoint_path='',
                pretrained_weight_path='',
            ),
            device=device,
            decode_fraction=float(args.decode_fraction),
            decode_planes=(None if args.decode_planes <= 0 else int(args.decode_planes)),
        )
        timing_report['archive_decode_seconds'] = float(time.perf_counter() - decode_archive_start_time)

    if archive_meta is not None:
        report['archive'] = _compact_archive_manifest_for_report(archive_meta)
        report['archive_bytes'] = int(archive_meta.get('archive_bytes', 0))
        report['archive_human'] = format_bytes(int(archive_meta.get('archive_bytes', 0)))

    if distribution_start_time is not None:
        e2e_distribution_seconds = float(time.perf_counter() - distribution_start_time)
        timing_report['e2e_distribution_seconds'] = float(e2e_distribution_seconds)
        timing_report['e2e_distribution_ms'] = float(e2e_distribution_seconds * 1000.0)
        timing_report['e2e_distribution_scope'] = (
            'artifact preparation/loading until model is ready for evaluation (network transfer excluded)'
        )
        timing_report['network_transfer_included'] = False
        report['e2e_distribution_seconds'] = float(e2e_distribution_seconds)
        report['e2e_distribution_ms'] = float(e2e_distribution_seconds * 1000.0)

    eval_start_time = time.perf_counter()
    eval_report = evaluate_model(model=model, args=args, device=device)
    timing_report['evaluation_seconds'] = float(time.perf_counter() - eval_start_time)
    report['evaluation'] = eval_report
    if 'e2e_distribution_seconds' in timing_report:
        report['evaluation']['e2e_distribution_seconds'] = float(timing_report['e2e_distribution_seconds'])
        report['evaluation']['e2e_distribution_ms'] = float(timing_report['e2e_distribution_ms'])
    decode_induced_e2e_latency = _build_decode_induced_e2e_latency_report(
        decode_seconds=float(timing_report.get('archive_decode_seconds', 0.0) or 0.0),
        eval_samples=int(report['evaluation'].get('samples', 0)),
    )
    report['evaluation']['decode_induced_e2e_inference_latency'] = decode_induced_e2e_latency
    baseline_fp32_scope_bytes: Optional[int] = None
    baseline_scope_include_head_linear: Optional[bool] = None
    compression_report = report.get('compression', None)
    if isinstance(compression_report, dict):
        value = compression_report.get('compressed_scope_fp32_bytes', compression_report.get('baseline_fp32_scope_bytes', None))
        if value is not None:
            baseline_fp32_scope_bytes = int(value)
            baseline_scope_include_head_linear = bool(args.include_head_linear)
    if baseline_fp32_scope_bytes is None and isinstance(archive_meta, dict):
        value = archive_meta.get('compressed_scope_fp32_bytes', archive_meta.get('baseline_fp32_scope_bytes', None))
        if value is not None:
            baseline_fp32_scope_bytes = int(value)
            if 'include_head_linear' in archive_meta:
                baseline_scope_include_head_linear = bool(archive_meta.get('include_head_linear'))
    if baseline_fp32_scope_bytes is not None:
        baseline_human = format_bytes(int(baseline_fp32_scope_bytes))
        report['baseline_fp32_scope_bytes'] = int(baseline_fp32_scope_bytes)
        report['baseline_fp32_scope_human'] = baseline_human
        report['evaluation']['baseline_fp32_scope_bytes'] = int(baseline_fp32_scope_bytes)
        report['evaluation']['baseline_fp32_scope_human'] = baseline_human
        if baseline_scope_include_head_linear is not None:
            report['baseline_scope_include_head_linear'] = bool(baseline_scope_include_head_linear)
            report['evaluation']['baseline_scope_include_head_linear'] = bool(baseline_scope_include_head_linear)

    saved_checkpoint_bytes = maybe_save_dense_checkpoint(model, args.save_decoded_checkpoint)
    if saved_checkpoint_bytes is not None:
        report['decoded_checkpoint_bytes'] = int(saved_checkpoint_bytes)
        report['decoded_checkpoint_human'] = format_bytes(saved_checkpoint_bytes)

    timing_report['total_runtime_seconds'] = float(time.perf_counter() - run_start_time)
    report['timing'] = timing_report

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.save_report:
        Path(args.save_report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_report, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f'Saved report to: {args.save_report}')


if __name__ == '__main__':
    main()
