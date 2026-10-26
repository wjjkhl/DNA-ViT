"""Activation fake quantization helpers for PTQ-style calibration and eval."""
from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def is_target_linear_layer_name(layer_name: str, include_head: bool = False) -> bool:
    name = str(layer_name)
    is_head = (
        name == "head"
        or name.startswith("head.")
        or name == "head_dist"
        or name.startswith("head_dist.")
    )
    if is_head:
        return bool(include_head)
    return True


@dataclass
class _QuantParams:
    scale: float
    zero_point: float
    qmin: int
    qmax: int


class _TensorObserver:
    def __init__(
        self,
        observer_type: str = "percentile",
        percentile: float = 0.999,
        max_samples: int = 262144,
        max_samples_per_update: int = 8192,
    ) -> None:
        self.observer_type = str(observer_type).strip().lower()
        if self.observer_type not in ("minmax", "percentile"):
            raise ValueError(f"Unsupported observer_type={observer_type!r}")
        self.percentile = float(percentile)
        self.percentile = max(0.5, min(1.0, self.percentile))
        self.max_samples = int(max(1024, max_samples))
        self.max_samples_per_update = int(max(256, max_samples_per_update))
        self.min_val: Optional[float] = None
        self.max_val: Optional[float] = None
        self._samples: List[torch.Tensor] = []
        self._sample_count: int = 0

    def update(self, x: torch.Tensor) -> None:
        if not torch.is_tensor(x) or x.numel() <= 0:
            return
        if not torch.is_floating_point(x):
            x = x.float()

        x_detached = x.detach()
        current_min = float(x_detached.min().item())
        current_max = float(x_detached.max().item())
        if self.min_val is None:
            self.min_val = current_min
            self.max_val = current_max
        else:
            self.min_val = min(float(self.min_val), current_min)
            self.max_val = max(float(self.max_val), current_max)

        if self.observer_type != "percentile":
            return

        flat = x_detached.reshape(-1)
        if flat.numel() > self.max_samples_per_update:
            step = max(1, int(flat.numel()) // int(self.max_samples_per_update))
            sampled = flat[::step][: self.max_samples_per_update]
        else:
            sampled = flat
        sampled_cpu = sampled.to(device="cpu", dtype=torch.float32)
        self._samples.append(sampled_cpu)
        self._sample_count += int(sampled_cpu.numel())
        self._trim_samples_if_needed()

    def _trim_samples_if_needed(self) -> None:
        if self._sample_count <= self.max_samples:
            return
        merged = torch.cat(self._samples, dim=0)
        if merged.numel() > self.max_samples:
            step = max(1, int(merged.numel()) // int(self.max_samples))
            merged = merged[::step][: self.max_samples].contiguous()
        self._samples = [merged]
        self._sample_count = int(merged.numel())

    def _percentile_minmax(self, scheme: str) -> Tuple[float, float]:
        if self._sample_count <= 0:
            if self.min_val is None or self.max_val is None:
                return 0.0, 0.0
            return float(self.min_val), float(self.max_val)

        merged = torch.cat(self._samples, dim=0)
        if scheme == "symmetric":
            max_abs = float(torch.quantile(torch.abs(merged), self.percentile).item())
            return -max_abs, max_abs

        q_hi = self.percentile
        q_lo = max(0.0, 1.0 - self.percentile)
        min_val = float(torch.quantile(merged, q_lo).item())
        max_val = float(torch.quantile(merged, q_hi).item())
        return min_val, max_val

    def build_qparams(self, bit: int, scheme: str, eps: float) -> _QuantParams:
        scheme_norm = str(scheme).strip().lower()
        if scheme_norm not in ("symmetric", "asymmetric"):
            raise ValueError(f"Unsupported scheme={scheme!r}")
        bit = int(bit)
        if bit <= 0:
            raise ValueError(f"bit must be positive, got {bit}")
        eps = float(max(1e-12, eps))

        if self.observer_type == "percentile":
            min_val, max_val = self._percentile_minmax(scheme_norm)
        else:
            if self.min_val is None or self.max_val is None:
                min_val = 0.0
                max_val = 0.0
            else:
                min_val = float(self.min_val)
                max_val = float(self.max_val)

        if scheme_norm == "symmetric":
            qmin = -(2 ** (bit - 1))
            qmax = (2 ** (bit - 1)) - 1
            max_abs = max(abs(float(min_val)), abs(float(max_val)))
            scale = max(max_abs / max(1, qmax), eps)
            zero_point = 0.0
        else:
            qmin = 0
            qmax = (2 ** bit) - 1
            min_scalar = min(float(min_val), float(max_val))
            max_scalar = max(float(min_val), float(max_val))
            if max_scalar - min_scalar < eps:
                max_scalar = min_scalar + eps
            scale = max((max_scalar - min_scalar) / float(max(1, qmax - qmin)), eps)
            zero_point = float(qmin) - float(min_scalar) / float(scale)
            zero_point = float(round(zero_point))
            zero_point = max(float(qmin), min(float(qmax), zero_point))
        return _QuantParams(scale=float(scale), zero_point=float(zero_point), qmin=int(qmin), qmax=int(qmax))


class SmoothQuantController:
    """SmoothQuant over Linear-input path with per-channel scales."""

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.5,
        eps: float = 1e-5,
        include_head_linear_input: bool = False,
        skip_layernorm_outputs: bool = False,
        min_scale: float = 1e-4,
        max_scale: float = 1e4,
    ) -> None:
        self.model = model
        self.alpha = float(alpha)
        self.eps = float(max(1e-12, eps))
        self.include_head_linear_input = bool(include_head_linear_input)
        self.skip_layernorm_outputs = bool(skip_layernorm_outputs)
        self.min_scale = float(max(1e-12, min_scale))
        self.max_scale = float(max(self.min_scale, max_scale))

        self._collecting: bool = False
        self._enabled: bool = False
        self._act_absmax: Dict[str, torch.Tensor] = {}
        self._scales_cpu: Dict[str, torch.Tensor] = {}
        self._scale_cache: Dict[Tuple[str, str, int, torch.dtype], torch.Tensor] = {}
        self._linear_modules: Dict[str, nn.Linear] = {}
        self._layernorm_output_ptrs: set = set()
        self._linear_hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._layernorm_hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._model_pre_hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None

        self._register_layernorm_tracking_hooks()
        self._register_linear_pre_hooks()

    def begin_calibration(self) -> None:
        self._collecting = True
        self._enabled = False
        self._act_absmax.clear()
        self._scales_cpu.clear()
        self._scale_cache.clear()

    def end_calibration(self) -> None:
        self._collecting = False
        self._build_scales()
        self._apply_weight_reparameterization()
        self._enabled = len(self._scales_cpu) > 0
        self._scale_cache.clear()

    def summary(self) -> dict:
        num_scale_params = int(sum(int(scale.numel()) for scale in self._scales_cpu.values()))
        return {
            "enabled": bool(self._enabled),
            "alpha": float(self.alpha),
            "eps": float(self.eps),
            "num_target_linear": int(len(self._linear_modules)),
            "num_smooth_layers": int(len(self._scales_cpu)),
            "num_scale_params": num_scale_params,
            "include_head_linear_input": bool(self.include_head_linear_input),
            "skip_layernorm_outputs": bool(self.skip_layernorm_outputs),
        }

    def _is_layernorm_output(self, x: torch.Tensor) -> bool:
        if not self.skip_layernorm_outputs:
            return False
        if not torch.is_tensor(x):
            return False
        return int(x.data_ptr()) in self._layernorm_output_ptrs

    def _register_layernorm_tracking_hooks(self) -> None:
        if not self.skip_layernorm_outputs:
            return

        def model_pre_hook(_module, _inputs):
            self._layernorm_output_ptrs.clear()

        self._model_pre_hook_handle = self.model.register_forward_pre_hook(model_pre_hook)

        for _name, module in self.model.named_modules():
            if not isinstance(module, nn.LayerNorm):
                continue

            def ln_hook(_module, _inputs, output):
                if torch.is_tensor(output):
                    self._layernorm_output_ptrs.add(int(output.data_ptr()))
                elif isinstance(output, (tuple, list)):
                    for tensor in output:
                        if torch.is_tensor(tensor):
                            self._layernorm_output_ptrs.add(int(tensor.data_ptr()))

            self._layernorm_hook_handles.append(module.register_forward_hook(ln_hook))

    def _update_input_absmax(self, key: str, x: torch.Tensor, expected_channels: int) -> None:
        x_detached = x.detach()
        if not torch.is_floating_point(x_detached):
            x_detached = x_detached.float()
        if x_detached.shape[-1] != expected_channels:
            return
        x_absmax = x_detached.abs().reshape(-1, expected_channels).amax(dim=0)
        x_absmax_cpu = x_absmax.to(device="cpu", dtype=torch.float32)
        prev = self._act_absmax.get(key)
        self._act_absmax[key] = x_absmax_cpu if prev is None else torch.maximum(prev, x_absmax_cpu)

    def _scale_for_input(self, key: str, x: torch.Tensor) -> Optional[torch.Tensor]:
        scale_cpu = self._scales_cpu.get(key)
        if scale_cpu is None:
            return None
        dev_index = -1 if x.device.index is None else int(x.device.index)
        cache_key = (key, x.device.type, dev_index, x.dtype)
        scale = self._scale_cache.get(cache_key)
        if scale is None:
            scale = scale_cpu.to(device=x.device, dtype=x.dtype)
            shape = [1] * int(x.ndim)
            shape[-1] = int(scale.shape[0])
            scale = scale.reshape(shape)
            self._scale_cache[cache_key] = scale
        return scale

    def _register_linear_pre_hooks(self) -> None:
        for module_name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not is_target_linear_layer_name(module_name, include_head=self.include_head_linear_input):
                continue
            self._linear_modules[module_name] = module

            def pre_hook(_module, inputs, key_local=module_name, in_features_local=int(module.in_features)):
                if not inputs:
                    return inputs
                x = inputs[0]
                if not torch.is_tensor(x):
                    return inputs

                is_ln_output = self._is_layernorm_output(x)
                if self._collecting and not is_ln_output:
                    self._update_input_absmax(key_local, x, in_features_local)

                if (not self._enabled) or is_ln_output:
                    return inputs

                scale = self._scale_for_input(key_local, x)
                if scale is None:
                    return inputs
                x_scaled = x / scale
                if len(inputs) == 1:
                    return (x_scaled,)
                return (x_scaled, *inputs[1:])

            self._linear_hook_handles.append(module.register_forward_pre_hook(pre_hook))

    def _build_scales(self) -> None:
        self._scales_cpu.clear()
        alpha = float(max(0.0, min(1.0, self.alpha)))
        one_minus_alpha = 1.0 - alpha
        for layer_name, module in self._linear_modules.items():
            x_absmax = self._act_absmax.get(layer_name)
            if x_absmax is None:
                continue
            w_absmax = module.weight.detach().abs().amax(dim=0).to(device="cpu", dtype=torch.float32)
            x_absmax = torch.clamp(x_absmax, min=self.eps)
            w_absmax = torch.clamp(w_absmax, min=self.eps)
            scale = x_absmax.pow(alpha) / w_absmax.pow(one_minus_alpha)
            scale = torch.clamp(scale, min=self.min_scale, max=self.max_scale)
            self._scales_cpu[layer_name] = scale.contiguous()

    def _apply_weight_reparameterization(self) -> None:
        for layer_name, scale_cpu in self._scales_cpu.items():
            module = self._linear_modules.get(layer_name)
            if module is None:
                continue
            weight = module.weight.data
            scale = scale_cpu.to(device=weight.device, dtype=weight.dtype)
            weight.mul_(scale.view(1, -1))

    def restore(self) -> None:
        if self._enabled:
            for layer_name, scale_cpu in self._scales_cpu.items():
                module = self._linear_modules.get(layer_name)
                if module is None:
                    continue
                weight = module.weight.data
                scale = scale_cpu.to(device=weight.device, dtype=weight.dtype)
                weight.div_(scale.view(1, -1))
        self._enabled = False
        self._collecting = False
        self._scale_cache.clear()
        for handle in self._linear_hook_handles:
            handle.remove()
        self._linear_hook_handles.clear()
        for handle in self._layernorm_hook_handles:
            handle.remove()
        self._layernorm_hook_handles.clear()
        if self._model_pre_hook_handle is not None:
            self._model_pre_hook_handle.remove()
            self._model_pre_hook_handle = None
        self._layernorm_output_ptrs.clear()


class ActivationFakeQuantController:
    """Controller for activation fake quant over target Linear inputs and attention matmul inputs."""

    def __init__(
        self,
        model: nn.Module,
        bit: int = 8,
        scheme: str = "symmetric",
        observer_type: str = "percentile",
        percentile: float = 0.999,
        eps: float = 1e-8,
        linear_inputs: bool = True,
        matmul_qk_inputs: bool = True,
        matmul_av_inputs: bool = True,
        include_head_linear_input: bool = False,
        skip_layernorm_outputs: bool = True,
        max_observer_samples: int = 262144,
        observer_samples_per_update: int = 8192,
    ) -> None:
        self.model = model
        self.bit = int(bit)
        self.scheme = str(scheme).strip().lower()
        self.observer_type = str(observer_type).strip().lower()
        self.percentile = float(percentile)
        self.eps = float(eps)
        self.linear_inputs = bool(linear_inputs)
        self.matmul_qk_inputs = bool(matmul_qk_inputs)
        self.matmul_av_inputs = bool(matmul_av_inputs)
        self.include_head_linear_input = bool(include_head_linear_input)
        self.skip_layernorm_outputs = bool(skip_layernorm_outputs)
        self.max_observer_samples = int(max_observer_samples)
        self.observer_samples_per_update = int(observer_samples_per_update)

        self._observers: Dict[str, _TensorObserver] = {}
        self._qparams: Dict[str, _QuantParams] = {}
        self._collecting: bool = False
        self._quant_enabled: bool = False
        self._layernorm_output_ptrs: set = set()
        self._linear_hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._layernorm_hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._model_pre_hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._patched_attn: List[Tuple[nn.Module, object]] = []

        self._register_layernorm_tracking_hooks()
        self._register_linear_pre_hooks()
        self._patch_attention_forwards()

    def _make_observer(self) -> _TensorObserver:
        return _TensorObserver(
            observer_type=self.observer_type,
            percentile=self.percentile,
            max_samples=self.max_observer_samples,
            max_samples_per_update=self.observer_samples_per_update,
        )

    def _get_observer(self, key: str) -> _TensorObserver:
        obs = self._observers.get(key)
        if obs is None:
            obs = self._make_observer()
            self._observers[key] = obs
        return obs

    def begin_calibration(self) -> None:
        self._collecting = True
        self._quant_enabled = False
        self._qparams.clear()

    def end_calibration(self) -> None:
        qparams = {}
        for key, observer in self._observers.items():
            qparams[key] = observer.build_qparams(self.bit, self.scheme, self.eps)
        self._qparams = qparams
        self._collecting = False
        self._quant_enabled = True

    def disable_quantization(self) -> None:
        self._quant_enabled = False
        self._collecting = False

    def summary(self) -> dict:
        return {
            "bit": int(self.bit),
            "scheme": self.scheme,
            "observer_type": self.observer_type,
            "percentile": float(self.percentile),
            "num_observers": int(len(self._observers)),
            "num_qparams": int(len(self._qparams)),
            "linear_inputs": bool(self.linear_inputs),
            "matmul_qk_inputs": bool(self.matmul_qk_inputs),
            "matmul_av_inputs": bool(self.matmul_av_inputs),
            "skip_layernorm_outputs": bool(self.skip_layernorm_outputs),
        }

    def _is_layernorm_output(self, x: torch.Tensor) -> bool:
        if not self.skip_layernorm_outputs:
            return False
        if not torch.is_tensor(x):
            return False
        return int(x.data_ptr()) in self._layernorm_output_ptrs

    def _fake_quantize(self, x: torch.Tensor, params: _QuantParams) -> torch.Tensor:
        scale = torch.as_tensor(params.scale, device=x.device, dtype=x.dtype)
        zero_point = torch.as_tensor(params.zero_point, device=x.device, dtype=x.dtype)
        q = torch.round(x / scale + zero_point)
        q = torch.clamp(q, params.qmin, params.qmax)
        return (q - zero_point) * scale

    def _observe_and_maybe_quantize(self, key: str, x: torch.Tensor) -> torch.Tensor:
        if self._collecting:
            self._get_observer(key).update(x)
        if not self._quant_enabled:
            return x
        params = self._qparams.get(key)
        if params is None:
            return x
        return self._fake_quantize(x, params)

    def _register_layernorm_tracking_hooks(self) -> None:
        if not self.skip_layernorm_outputs:
            return

        def model_pre_hook(_module, _inputs):
            self._layernorm_output_ptrs.clear()

        self._model_pre_hook_handle = self.model.register_forward_pre_hook(model_pre_hook)

        for _name, module in self.model.named_modules():
            if not isinstance(module, nn.LayerNorm):
                continue

            def ln_hook(_module, _inputs, output):
                if torch.is_tensor(output):
                    self._layernorm_output_ptrs.add(int(output.data_ptr()))
                elif isinstance(output, (tuple, list)):
                    for tensor in output:
                        if torch.is_tensor(tensor):
                            self._layernorm_output_ptrs.add(int(tensor.data_ptr()))

            self._layernorm_hook_handles.append(module.register_forward_hook(ln_hook))

    def _register_linear_pre_hooks(self) -> None:
        if not self.linear_inputs:
            return
        for module_name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not is_target_linear_layer_name(module_name, include_head=self.include_head_linear_input):
                continue
            key = f"linear_input::{module_name}"

            def pre_hook(_module, inputs, key_local=key):
                if not inputs:
                    return inputs
                x = inputs[0]
                if not torch.is_tensor(x):
                    return inputs
                if self._is_layernorm_output(x):
                    return inputs
                x_qdq = self._observe_and_maybe_quantize(key_local, x)
                if len(inputs) == 1:
                    return (x_qdq,)
                return (x_qdq, *inputs[1:])

            self._linear_hook_handles.append(module.register_forward_pre_hook(pre_hook))

    @staticmethod
    def _is_swin_window_attention_module(module: nn.Module) -> bool:
        if not hasattr(module, "window_size"):
            return False
        if not hasattr(module, "qkv") or not isinstance(getattr(module, "qkv"), nn.Linear):
            return False
        if not hasattr(module, "proj") or not isinstance(getattr(module, "proj"), nn.Linear):
            return False
        if not hasattr(module, "num_heads"):
            return False
        cls_name = module.__class__.__name__.lower()
        if "windowattention" in cls_name:
            return True
        return hasattr(module, "relative_position_bias_table") or hasattr(module, "cpb_mlp")

    @staticmethod
    def _is_vit_attention_module(module: nn.Module) -> bool:
        if hasattr(module, "window_size"):
            return False
        if not hasattr(module, "qkv") or not isinstance(getattr(module, "qkv"), nn.Linear):
            return False
        if not hasattr(module, "proj") or not isinstance(getattr(module, "proj"), nn.Linear):
            return False
        if not hasattr(module, "num_heads"):
            return False
        # Keep specialized relative-position modules on their dedicated paths.
        if hasattr(module, "relative_position_bias_table") or hasattr(module, "cpb_mlp"):
            return False
        return True

    @staticmethod
    def _get_swin_relative_position_bias(attn_module: nn.Module, num_tokens: int) -> Optional[torch.Tensor]:
        rel_bias_getter = getattr(attn_module, "_get_rel_pos_bias", None)
        if callable(rel_bias_getter):
            rel_bias = rel_bias_getter()
            if torch.is_tensor(rel_bias):
                return rel_bias

        rel_pos_table = getattr(attn_module, "relative_position_bias_table", None)
        rel_pos_index = getattr(attn_module, "relative_position_index", None)
        num_heads = int(getattr(attn_module, "num_heads", 0))
        if torch.is_tensor(rel_pos_table) and torch.is_tensor(rel_pos_index) and num_heads > 0:
            rel_bias = rel_pos_table[rel_pos_index.view(-1)].view(num_tokens, num_tokens, num_heads)
            rel_bias = rel_bias.permute(2, 0, 1).contiguous()
            return rel_bias.unsqueeze(0)

        cpb_mlp = getattr(attn_module, "cpb_mlp", None)
        rel_coords_table = getattr(attn_module, "relative_coords_table", None)
        if cpb_mlp is not None and torch.is_tensor(rel_coords_table) and torch.is_tensor(rel_pos_index) and num_heads > 0:
            rel_bias_table = cpb_mlp(rel_coords_table).view(-1, num_heads)
            rel_bias = rel_bias_table[rel_pos_index.view(-1)].view(num_tokens, num_tokens, num_heads)
            rel_bias = rel_bias.permute(2, 0, 1).contiguous()
            rel_bias = 16.0 * torch.sigmoid(rel_bias)
            return rel_bias.unsqueeze(0)

        return None

    def _patch_attention_forwards(self) -> None:
        if not (self.matmul_qk_inputs or self.matmul_av_inputs):
            return
        for module_name, module in self.model.named_modules():
            is_swin_attn = self._is_swin_window_attention_module(module)
            if not is_swin_attn and not self._is_vit_attention_module(module):
                continue
            original_forward = module.forward
            self._patched_attn.append((module, original_forward))

            controller = self
            attn_name = str(module_name)

            if is_swin_attn:
                def patched_forward(
                    attn_module,
                    x,
                    *args,
                    attn_name_local=attn_name,
                    orig=original_forward,
                    **kwargs,
                ):
                    mask = kwargs.pop("mask", None)
                    if len(args) > 1:
                        if mask is None:
                            return orig(x, *args, **kwargs)
                        return orig(x, *args, mask=mask, **kwargs)
                    if len(args) == 1:
                        if mask is not None:
                            return orig(x, *args, mask=mask, **kwargs)
                        mask = args[0]
                    if kwargs:
                        if mask is None:
                            return orig(x, **kwargs)
                        return orig(x, mask=mask, **kwargs)
                    if not torch.is_tensor(x) or x.ndim != 3:
                        if mask is None:
                            return orig(x)
                        return orig(x, mask)

                    batch_size, num_tokens, channels = x.shape
                    num_heads = int(attn_module.num_heads)
                    if num_heads <= 0:
                        if mask is None:
                            return orig(x)
                        return orig(x, mask)

                    qkv = None
                    q_bias = getattr(attn_module, "q_bias", None)
                    v_bias = getattr(attn_module, "v_bias", None)
                    if q_bias is not None and v_bias is not None:
                        k_bias = getattr(attn_module, "k_bias", None)
                        if k_bias is None:
                            k_bias = torch.zeros_like(q_bias)
                        qkv_bias = torch.cat((q_bias, k_bias, v_bias))
                        if bool(getattr(attn_module, "qkv_bias_separate", False)):
                            qkv = attn_module.qkv(x)
                            qkv = qkv + qkv_bias
                        else:
                            qkv = F.linear(x, weight=attn_module.qkv.weight, bias=qkv_bias)
                    if qkv is None:
                        qkv = attn_module.qkv(x)

                    qkv = qkv.reshape(batch_size, num_tokens, 3, num_heads, -1)
                    qkv = qkv.permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)

                    q_norm = getattr(attn_module, "q_norm", None)
                    if q_norm is not None:
                        q = q_norm(q)
                    k_norm = getattr(attn_module, "k_norm", None)
                    if k_norm is not None:
                        k = k_norm(k)

                    if hasattr(attn_module, "logit_scale"):
                        q_for_scores = F.normalize(q, dim=-1)
                        k_for_scores = F.normalize(k, dim=-1)
                        if controller.matmul_qk_inputs:
                            if not controller._is_layernorm_output(q_for_scores):
                                q_for_scores = controller._observe_and_maybe_quantize(
                                    f"matmul_q::{attn_name_local}",
                                    q_for_scores,
                                )
                            if not controller._is_layernorm_output(k_for_scores):
                                k_for_scores = controller._observe_and_maybe_quantize(
                                    f"matmul_k::{attn_name_local}",
                                    k_for_scores,
                                )
                        attn_scores = q_for_scores @ k_for_scores.transpose(-2, -1)
                        logit_scale = torch.clamp(attn_module.logit_scale, max=math.log(1.0 / 0.01)).exp()
                        attn_scores = attn_scores * logit_scale
                    else:
                        if controller.matmul_qk_inputs:
                            if not controller._is_layernorm_output(q):
                                q = controller._observe_and_maybe_quantize(f"matmul_q::{attn_name_local}", q)
                            if not controller._is_layernorm_output(k):
                                k = controller._observe_and_maybe_quantize(f"matmul_k::{attn_name_local}", k)
                        scale = getattr(attn_module, "scale", None)
                        if scale is None:
                            scale = float(max(1, int(q.shape[-1])) ** -0.5)
                        q = q * scale
                        attn_scores = q @ k.transpose(-2, -1)

                    rel_pos_bias = controller._get_swin_relative_position_bias(attn_module, num_tokens)
                    if rel_pos_bias is not None:
                        rel_pos_bias = rel_pos_bias.to(device=attn_scores.device, dtype=attn_scores.dtype)
                        attn_scores = attn_scores + rel_pos_bias

                    if mask is not None:
                        if not torch.is_tensor(mask):
                            return orig(x, mask)
                        num_windows = int(mask.shape[0])
                        attn_scores = attn_scores.view(-1, num_windows, num_heads, num_tokens, num_tokens)
                        attn_scores = attn_scores + mask.unsqueeze(1).unsqueeze(0)
                        attn_scores = attn_scores.view(-1, num_heads, num_tokens, num_tokens)

                    attn_probs = attn_scores.float().softmax(dim=-1)
                    if attn_scores.dtype != torch.float32:
                        attn_probs = attn_probs.to(dtype=attn_scores.dtype)
                    attn_drop = getattr(attn_module, "attn_drop", None)
                    if attn_drop is not None:
                        attn_probs = attn_drop(attn_probs)

                    if controller.matmul_av_inputs:
                        # Keep softmax output in floating-point for stability at low bitwidth.
                        if not controller._is_layernorm_output(v):
                            v = controller._observe_and_maybe_quantize(f"matmul_v::{attn_name_local}", v)

                    out = attn_probs @ v
                    out = out.transpose(1, 2).reshape(batch_size, num_tokens, channels)
                    out = attn_module.proj(out)

                    proj_drop = getattr(attn_module, "proj_drop", None)
                    if proj_drop is not None:
                        out = proj_drop(out)
                    return out

                module.forward = types.MethodType(patched_forward, module)
                continue

            def patched_forward(attn_module, x, *args, attn_name_local=attn_name, orig=original_forward, **kwargs):
                if args or kwargs:
                    return orig(x, *args, **kwargs)
                if not torch.is_tensor(x) or x.ndim != 3:
                    return orig(x)

                batch_size, num_tokens, channels = x.shape
                num_heads = int(attn_module.num_heads)
                if num_heads <= 0 or channels % num_heads != 0:
                    return orig(x)

                qkv = attn_module.qkv(x)
                qkv = qkv.reshape(batch_size, num_tokens, 3, num_heads, channels // num_heads)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)

                q_norm = getattr(attn_module, "q_norm", None)
                if q_norm is not None:
                    q = q_norm(q)
                k_norm = getattr(attn_module, "k_norm", None)
                if k_norm is not None:
                    k = k_norm(k)

                if controller.matmul_qk_inputs:
                    if not controller._is_layernorm_output(q):
                        q = controller._observe_and_maybe_quantize(f"matmul_q::{attn_name_local}", q)
                    if not controller._is_layernorm_output(k):
                        k = controller._observe_and_maybe_quantize(f"matmul_k::{attn_name_local}", k)

                scale = getattr(attn_module, "scale", None)
                if scale is None:
                    scale = float((channels // num_heads) ** -0.5)
                q = q * scale

                attn_scores = q @ k.transpose(-2, -1)
                attn_probs = attn_scores.float().softmax(dim=-1)
                if attn_scores.dtype != torch.float32:
                    attn_probs = attn_probs.to(dtype=attn_scores.dtype)

                attn_drop = getattr(attn_module, "attn_drop", None)
                if attn_drop is not None:
                    attn_probs = attn_drop(attn_probs)

                if controller.matmul_av_inputs:
                    # Keep softmax output in floating-point for stability at low bitwidth.
                    if not controller._is_layernorm_output(v):
                        v = controller._observe_and_maybe_quantize(f"matmul_v::{attn_name_local}", v)

                out = attn_probs @ v
                out = out.transpose(1, 2).reshape(batch_size, num_tokens, channels)
                out = attn_module.proj(out)

                proj_drop = getattr(attn_module, "proj_drop", None)
                if proj_drop is not None:
                    out = proj_drop(out)
                return out

            module.forward = types.MethodType(patched_forward, module)

    def restore(self) -> None:
        for handle in self._linear_hook_handles:
            handle.remove()
        self._linear_hook_handles.clear()
        for handle in self._layernorm_hook_handles:
            handle.remove()
        self._layernorm_hook_handles.clear()
        if self._model_pre_hook_handle is not None:
            self._model_pre_hook_handle.remove()
            self._model_pre_hook_handle = None
        self._layernorm_output_ptrs.clear()
        for module, original_forward in self._patched_attn:
            module.forward = original_forward
        self._patched_attn.clear()
