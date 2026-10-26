# DevDNA: Transformer Weight Compression and Streaming Inference

DevDNA implements sensitivity-calibrated weight compression, progressive decoding of compressed archives, and streaming inference by module group. `run_devdna_vit.py` provides the unified command-line interface. Some internal identifiers and archive filenames retain the earlier GaugeDNA name for compatibility.

**Streaming inference is part of the algorithmic framework, rather than a utility for generating paper figures.** The runtime supports on-demand decoding of module groups, loading and releasing weights by group, and decoding the next group concurrently with the current group's forward pass.

## 1. Project Structure

```text
.
├── run_devdna_vit.py               # CLI entry point and workflow orchestration
├── devdna_core.py                  # Normalization and transform primitives
├── devdna_codec.py                 # Serialization and progressive bit-plane coding
├── devdna_model.py                 # Calibration, budget allocation, compression, and decoding
├── devdna_eval.py                  # Local weight loading and ImageNet evaluation
├── devdna_activation_fake_quant.py # Optional evaluation-time activation fake quantization
├── devdna_streaming_runtime.py     # On-demand decoding and streaming inference
├── config.yaml                    # Existing local experiment configuration
├── requirements.txt               # Recorded dependency versions
├── .gitignore                     # Git ignore rules
└── README.md
```

Pretrained weights, datasets, and run outputs are not included; prepare them separately. Generated reports and compressed archives are typically stored under `outputs/`. Weights and generated artifacts are excluded by `.gitignore`.

## 2. Dependencies and Environment

### Reference Environment

The following versions correspond to the local `dcavit_seg` reference environment used for the successfully executed calibration, compression, and full-decoding evaluation workflow with `.pth` weights:

| Component | Recorded Version |
| --- | --- |
| Python | 3.8.20 |
| PyTorch | 2.3.1+cu121 |
| torchvision | 0.18.1+cu121 |
| timm | 0.6.12 |
| NumPy | 1.24.3 |
| PyYAML | 6.0 |
| Pillow | 9.5.0 |
| huggingface-hub | 0.36.2 |

`requirements.txt` pins the PyTorch and torchvision release versions without fixing their CUDA build suffixes. The recorded builds use CUDA 12.1. Select compatible builds for your target machine; arbitrary PyTorch and torchvision combinations are not interchangeable.

`requirements.txt` also records `packaging==24.2` and `tqdm==4.65.0` from the reference environment. Python 3.8 is the reference baseline; other Python versions have not undergone the same validation. Dependency installation has not been revalidated in a fresh environment.

`safetensors` is needed only for loading `.safetensors` weights and is not included in the default dependencies. It was not installed in the reference environment, and the completed `.pth` workflow does not depend on it. To use this format, you can install the optional version below. This version comes from another local environment and has not been validated together with the reference environment above:

```bash
python -m pip install safetensors==0.4.3
```

### Installation

Use an isolated environment with a Python interpreter compatible with the recorded versions. For example, with a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

`requirements.txt` records core dependencies and some transitive dependencies; it is not a complete lockfile for all transitive dependencies. Installing in an isolated environment helps avoid overwriting dependencies used by other projects.

First check module imports and CLI registration. These checks do not load weights or datasets:

```bash
python -c 'from devdna_eval import import_timm; import_timm(); import yaml; print("imports OK")'
python run_devdna_vit.py --help
```

CPU execution can be used for correctness checks. GPU latency, CUDA memory usage, and gains from overlapping decoding with computation must be measured on the target CUDA hardware. The included source code does not depend on SciPy or Matplotlib. You can select visible GPUs before launching Python, for example: `CUDA_VISIBLE_DEVICES=0 python run_devdna_vit.py --help`.

## 3. Datasets and Local Weights

Run commands from the project root. Relative paths are resolved against the process's current working directory.

The ImageNet loader expects directories organized by class:

```text
/path/to/imagenet/
├── train/
│   └── n01440764/
│       └── example.JPEG
└── val/
    └── n01440764/
        └── example.JPEG
```

The training set is not required when evaluating weight compression only on `val`. If activation fake quantization is enabled and its calibration source is set to `train`, the training set is also required. Activation calibration and the sensitivity calibration used for compression are separate workflows.

The model loader uses local weights; `--pretrained` **does not trigger an online download through timm**. Supply weights matching the timm model architecture through `--pretrained-weight-path` or `--checkpoint`; the latter takes precedence. Supported local formats include `.pth`, `.pt`, and `.safetensors`. Inspect loading warnings for missing and unexpected keys; a partially loaded model is not automatically a valid pretrained baseline. Load weights only from trusted sources.

The existing YAML still references weights exported by another project and dataset directories on the original machine. These values have been left unchanged. Even within the Swin architecture family, weights exported from an object detection backbone may not directly substitute for ImageNet classification weights.

## 4. Configuration Precedence and Run Modes

Parameter precedence is: `command-line arguments > YAML configuration > parser defaults`. Loading the current YAML without overriding `--mode` selects `analyze_sensitivity`, rather than the end-to-end compression workflow. Verify that the required local inputs are available before running it.

| Mode | Purpose |
| --- | --- |
| `analyze_sensitivity` | Generate calibration statistics for blockwise budget allocation |
| `compress` | Compress weights and save an archive |
| `eval` | Decode a valid archive and evaluate it; see the fallback behavior below |
| `compress_eval` | Compress weights, decode the resulting archive, and evaluate |
| `inspect` | Read archive metadata without running model inference |
| `eval_pretrained` | Evaluate a locally loaded dense model without archive decoding |
| `eval_streaming_runtime_familywise` | Run streaming inference and its comparison modes |

After verifying the input paths in the existing configuration, use these configuration-based entry points:

```bash
python run_devdna_vit.py --config config.yaml --mode analyze_sensitivity
python run_devdna_vit.py --config config.yaml --mode compress
python run_devdna_vit.py --config config.yaml --mode eval
python run_devdna_vit.py --config config.yaml --mode eval_streaming_runtime_familywise
```

**Note:** `eval` decodes an archive only when `--archive-path` is recognized as a valid GaugeDNA archive. If the archive is missing or invalid, execution may fall back to ordinary model construction. Inspect the archive first, then confirm that the saved evaluation report contains `archive` metadata and `timing.archive_decode_seconds`. A successful process exit alone does not establish that a compressed model was evaluated. The internal `loaded_from_archive` flag is currently not exported to that report.

The current YAML enables activation fake quantization. Pass `--activation-quant-disabled` to evaluate weight compression alone. The streaming inference entry point does not use the activation fake-quantization controller from the regular evaluation path. Comparisons should therefore not be described as using identical weight and activation quantization settings until those settings have been independently verified.

## 5. End-to-End Command Examples

The examples below are **command templates**, not newly validated experimental configurations, and they do not modify `config.yaml`. They omit YAML loading to make all inputs explicit. Confirm the model, calibration parameters, and archive size target for your experiment before reporting results.

Enter the project root, then set the paths and budget in Bash. Weight and dataset paths are placeholders and must be replaced with actual paths. The weights must match the selected model architecture. `36.0` is only an example archive size target in MiB; it is neither guaranteed to be achievable nor a recommendation for accuracy.

The subsequent commands depend on these shell variables. Run them in the same terminal, or set all variables again in a new terminal, including `TARGET_MB` used during compression.

```bash
MODEL=deit_small_patch16_224
WEIGHTS=/path/to/deit_small_patch16_224-cd65a155.pth
IMAGENET=/path/to/imagenet
OUT="outputs/$MODEL"
TARGET_MB=36.0
mkdir -p "$OUT"
```

### Dense Model Baseline

```bash
python run_devdna_vit.py --mode eval_pretrained \
  --model "$MODEL" --pretrained --pretrained-weight-path "$WEIGHTS" \
  --data-root "$IMAGENET" --split val --device cuda \
  --activation-quant-disabled --save-report "$OUT/dense.json"
```

### Sensitivity Calibration

```bash
python run_devdna_vit.py --mode analyze_sensitivity \
  --model "$MODEL" --pretrained --pretrained-weight-path "$WEIGHTS" \
  --data-root "$IMAGENET" --split val --device cuda \
  --sensitivity-calib-size 1024 --sensitivity-keep-ratio 0.75 \
  --sensitivity-quant-bits 8 --sensitivity-allocation equal_keep_ratio \
  --sensitivity-output "$OUT/sensitivity.json" \
  --sensitivity-csv-output "$OUT/sensitivity.csv" \
  --save-report "$OUT/calibration_run.json"
```

This workflow includes reference-model evaluation, local compression and reconstruction, and perturbation statistics. `blockwise_thresholded` requires the report generated here and checks alignment of the model and module blocks. Use the same weights, calibration procedure, and compression scope for calibration and compression. The current alignment checks cannot establish that both stages used identical weights. To compress the classification head, use `--include-head-linear` consistently in the relevant stages.

### Weight Compression

```bash
python run_devdna_vit.py --mode compress \
  --model "$MODEL" --pretrained --pretrained-weight-path "$WEIGHTS" \
  --device cuda --codec-allocation-mode blockwise_thresholded \
  --sensitivity-report "$OUT/sensitivity.json" \
  --target-archive-mb "$TARGET_MB" --quant-bits 8 \
  --archive-path "$OUT/model.zip" --save-report "$OUT/compression.json" \
  --allocation-report-output "$OUT/allocation.json" \
  --allocation-csv-output "$OUT/allocation.csv"
```

`blockwise_thresholded` combines the sensitivity report, coefficient scores, and a lambda search to allocate the budget using the actual encoded archive size. Each unit of `--target-archive-mb` corresponds to `1024 * 1024` bytes. Check the achieved size, tolerance, and convergence status instead of assuming the target was met. The retained `legacy_global` option is a separate path using `--mlp-keep-ratio` and `--attn-keep-ratio`; it does not replace validation of blockwise budget allocation.

### Archive Inspection and Full-Decoding Evaluation

```bash
python run_devdna_vit.py --mode inspect --device cpu \
  --archive-path "$OUT/model.zip" --save-report "$OUT/manifest_report.json"

python run_devdna_vit.py --mode eval \
  --archive-path "$OUT/model.zip" --data-root "$IMAGENET" --split val \
  --device cuda --decode-fraction 1.0 --activation-quant-disabled \
  --save-report "$OUT/eval_full.json"
```

Full archive decoding constructs the model from archive metadata and does not require the original pretrained weights. If you provide the inputs needed for both compression and evaluation, you can use `compress_eval` in place of the two separate calls.

### Progressive-Decoding Evaluation

```bash
python run_devdna_vit.py --mode eval \
  --archive-path "$OUT/model.zip" --data-root "$IMAGENET" --split val \
  --device cuda --decode-planes 4 --activation-quant-disabled \
  --save-report "$OUT/eval_planes4.json"
```

A positive `--decode-planes` value takes precedence over `--decode-fraction`. When no positive bit-plane count is specified, `--decode-fraction 1.0` requests full decoding, while smaller fractions use fewer bit planes. Accuracy with fewer bit planes must be measured; it should not be assumed to match full decoding.

## 6. Streaming Inference

Use the same archive when comparing runtime modes. The runtime reads the model architecture from the archive manifest, and archive decoding does not require the original pretrained weight file.

| Runtime Mode | Behavior |
| --- | --- |
| `serial_full_decode` | Fully decode and load the archive before inference, providing a serial-path reference |
| `lazy_family_decode` | Decode and load module groups on demand, then release their dynamic device weights after execution |
| `lazy_family_decode_overlap` | Decode the next group in a worker thread while the current group executes |
| `oracle_ready_to_run` | Evaluate an already decoded model, excluding initial archive decoding from inference timing |

```bash
python run_devdna_vit.py --mode eval_streaming_runtime_familywise \
  --archive-path "$OUT/model.zip" --data-root "$IMAGENET" --split val \
  --device cuda --decode-fraction 1.0 --decode-planes -1 \
  --streaming-runtime-modes serial_full_decode,lazy_family_decode,lazy_family_decode_overlap,oracle_ready_to_run \
  --streaming-runtime-group-policy block_attn_mlp \
  --streaming-runtime-eval-batch-size 128 --streaming-runtime-eval-samples 1024 \
  --streaming-runtime-warmup-batches 5 --streaming-runtime-repeats 3 \
  --streaming-runtime-cache-decoded-groups \
  --streaming-runtime-reset-decoded-cache-before-timed-eval \
  --streaming-runtime-drop-first-repeat-for-aggregation \
  --streaming-runtime-prefetch-next-group \
  --streaming-runtime-output-json "$OUT/streaming.json" \
  --streaming-runtime-output-csv "$OUT/streaming.csv" \
  --save-report "$OUT/streaming_run.json"
```

Grouping policies include `block_attn_mlp` (separate attention and MLP groups), `block_full`, `super_block` (controlled by `--streaming-runtime-super-block-k`), and `auto` (profile candidate grouping granularities). CLI and YAML defaults may differ: the parser defaults to `auto`, while the current YAML selects `block_attn_mlp`. Record both the requested policy and the policy actually selected.
