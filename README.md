# DevDNA：Transformer 权重压缩与流式推理

DevDNA 实现了基于灵敏度校准的权重压缩、压缩归档的渐进解码，以及按模块组进行的流式推理（streaming inference）。`run_devdna_vit.py` 是统一命令行入口。部分内部标识和归档文件名仍使用早期的 GaugeDNA 名称，以保持兼容性。

**流式推理属于算法框架的组成部分，并非用于论文绘图的辅助工具。** 当前运行时支持按需解码模块组、分组加载与释放权重，以及在执行当前组前向计算时并行解码下一组。

## 1. 项目结构

```text
.
├── run_devdna_vit.py               # 命令行入口与流程调度
├── devdna_core.py                  # 规范化处理与变换基础操作
├── devdna_codec.py                 # 序列化与渐进位平面编码
├── devdna_model.py                 # 校准、预算分配、压缩与解码
├── devdna_eval.py                  # 本地权重加载与 ImageNet 评测
├── devdna_activation_fake_quant.py # 可选的评测阶段激活伪量化
├── devdna_streaming_runtime.py     # 按需解码与流式推理
├── config.yaml                    # 现有本地实验配置
├── requirements.txt               # 依赖版本记录
├── .gitignore                     # Git 忽略规则
└── README.md
```

仓库不附带预训练权重、数据集或运行结果，请自行准备。生成的报告与压缩归档通常放在 `outputs/` 下，权重和运行产物由 `.gitignore` 排除。

## 2. 依赖与运行环境

### 参考运行环境

以下版本对应本地 `dcavit_seg` 参考环境，用于复现已跑通的 `.pth` 权重校准、压缩与完整解码评测流程：

| 组件 | 已记录版本 |
| --- | --- |
| Python | 3.8.20 |
| PyTorch | 2.3.1+cu121 |
| torchvision | 0.18.1+cu121 |
| timm | 0.6.12 |
| NumPy | 1.24.3 |
| PyYAML | 6.0 |
| Pillow | 9.5.0 |
| huggingface-hub | 0.36.2 |

`requirements.txt` 固定 PyTorch 和 torchvision 的发行版本，但未写死 CUDA 构建后缀。记录中的构建版本使用 CUDA 12.1。请根据目标机器选择匹配的构建版本，不要默认任意 PyTorch 与 torchvision 组合均可互换。

`requirements.txt` 同时记录参考环境中的 `packaging==24.2` 和 `tqdm==4.65.0`。Python 3.8 是参考基线，不表示其他 Python 版本已通过相同验证。依赖安装尚未在全新环境中重新验证。

`safetensors` 只在加载 `.safetensors` 权重时需要，不列入默认依赖。参考环境未安装该包，已完成的 `.pth` 流程不依赖它。如需使用该格式，可另行安装下面的可选版本；此版本取自另一套本地环境，尚未在上述参考环境中联合验证：

```bash
python -m pip install safetensors==0.4.3
```

### 安装方式

建议使用独立环境，并选择与记录版本兼容的 Python 解释器。下面以虚拟环境为例：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

`requirements.txt` 记录核心依赖及部分间接依赖，并非锁定全部间接依赖的完整锁文件。建议在独立环境中安装，避免覆盖其他项目的依赖。

可先检查模块导入和命令行注册，此步骤不加载权重或数据集：

```bash
python -c 'from devdna_eval import import_timm; import_timm(); import yaml; print("imports OK")'
python run_devdna_vit.py --help
```

CPU 运行可用于正确性检查；GPU 延迟、CUDA 显存占用及解码与计算重叠的收益，需要在目标 CUDA 硬件上测量。当前保留的源码不依赖 SciPy 或 Matplotlib。可以在启动 Python 前指定可见 GPU，例如 `CUDA_VISIBLE_DEVICES=0 python run_devdna_vit.py --help`。

## 3. 数据集与本地权重

请从项目根目录执行命令。相对路径以进程的当前工作目录为基准。

ImageNet 加载器要求按类别组织目录：

```text
/path/to/imagenet/
├── train/
│   └── n01440764/
│       └── example.JPEG
└── val/
    └── n01440764/
        └── example.JPEG
```

仅在 `val` 上评测权重压缩时，不需要训练集。如果启用了激活伪量化，且其校准数据来源设为 `train`，则还需要准备训练集。激活校准与压缩所需的灵敏度校准是两个独立流程。

模型加载器使用本地权重；`--pretrained` **不会触发 timm 在线下载**。请通过 `--pretrained-weight-path` 或 `--checkpoint` 指定与 timm 模型结构匹配的权重，后者优先。支持的本地格式包括 `.pth`、`.pt` 和 `.safetensors`。请检查缺失键、非预期键等加载警告，不要将部分加载自动视为有效的预训练基线。只加载来自可信来源的权重。

现有 YAML 仍引用另一个项目导出的权重及本机数据集目录，这些值没有修改。即使同属 Swin 架构，目标检测骨干导出的权重也不一定能够直接替代 ImageNet 分类权重。

## 4. 配置优先级与运行模式

参数优先级为：`命令行参数 > YAML 配置 > 解析器默认值`。加载当前 YAML 而不覆盖 `--mode` 时，将进入 `analyze_sensitivity`，而不是端到端压缩流程。执行前请确认其本地输入条件。

| 模式 | 用途 |
| --- | --- |
| `analyze_sensitivity` | 生成分块预算分配所需的校准统计 |
| `compress` | 压缩权重并保存归档 |
| `eval` | 解码有效归档并评测；请注意下方的回退行为 |
| `compress_eval` | 压缩权重、解码生成的归档并评测 |
| `inspect` | 读取归档元数据，不执行模型推理 |
| `eval_pretrained` | 评测本地加载的稠密模型，不执行归档解码 |
| `eval_streaming_runtime_familywise` | 执行流式推理及各对照模式 |

确认现有配置中的输入路径有效后，可使用以下配置文件入口：

```bash
python run_devdna_vit.py --config config.yaml --mode analyze_sensitivity
python run_devdna_vit.py --config config.yaml --mode compress
python run_devdna_vit.py --config config.yaml --mode eval
python run_devdna_vit.py --config config.yaml --mode eval_streaming_runtime_familywise
```

**注意：** 仅当 `--archive-path` 被识别为有效的 GaugeDNA 归档时，`eval` 才会执行解码。归档缺失或无效时，程序可能回退到普通模型构建流程。请先检查归档，再确认保存的评测报告中存在 `archive` 元数据及 `timing.archive_decode_seconds`，不要仅根据进程成功退出就认定已完成压缩模型评测。内部的 `loaded_from_archive` 标记目前不会导出到该报告。

当前 YAML 启用了激活伪量化。如需只评测权重压缩，请传入 `--activation-quant-disabled`。流式推理入口没有使用普通评测路径中的激活伪量化控制器，因此在单独验证相关设置前，不应宣称两者是在相同权重与激活量化条件下进行的对照。

## 5. 完整流程命令示例

以下内容是**命令模板**，不是新验证通过的实验工作点，也不会修改 `config.yaml`。示例不加载 YAML，以明确展示各项输入。正式报告结果前，需要根据目标实验确认模型、校准参数及压缩大小目标。

先进入项目根目录，再在 Bash 中设置路径和预算。权重及数据集路径均为占位符，必须替换为实际路径；权重需与所选模型结构匹配。`36.0` 仅表示示例归档大小目标，单位为 MiB，不保证一定可达到，也不是准确率方面的推荐值。

后续命令依赖这些 Shell 变量，请在同一个终端中执行；新开终端后需重新设置全部变量，包括压缩阶段使用的 `TARGET_MB`。

```bash
MODEL=deit_small_patch16_224
WEIGHTS=/path/to/deit_small_patch16_224-cd65a155.pth
IMAGENET=/path/to/imagenet
OUT="outputs/$MODEL"
TARGET_MB=36.0
mkdir -p "$OUT"
```

### 稠密模型基线

```bash
python run_devdna_vit.py --mode eval_pretrained \
  --model "$MODEL" --pretrained --pretrained-weight-path "$WEIGHTS" \
  --data-root "$IMAGENET" --split val --device cuda \
  --activation-quant-disabled --save-report "$OUT/dense.json"
```

### 灵敏度校准

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

该流程保留参考模型评测、局部压缩与重建，以及扰动统计。`blockwise_thresholded` 需要这里生成的报告，并检查模型及模块块的对齐情况。校准与压缩阶段应使用相同的权重、校准流程及压缩范围；当前对齐检查不能证明两次使用的是同一份权重。如果需要压缩分类头，应在相关阶段一致使用 `--include-head-linear`。

### 权重压缩

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

`blockwise_thresholded` 结合灵敏度报告、系数分数及 lambda 搜索，依据实际编码后的归档大小进行预算分配。`--target-archive-mb` 的每单位对应 `1024 * 1024` 字节。请检查实际大小、容差与收敛状态，不要默认目标一定已经达到。保留的 `legacy_global` 是另一条使用 `--mlp-keep-ratio` 和 `--attn-keep-ratio` 的路径，不能替代对分块预算分配的验证。

### 归档检查与完整解码评测

```bash
python run_devdna_vit.py --mode inspect --device cpu \
  --archive-path "$OUT/model.zip" --save-report "$OUT/manifest_report.json"

python run_devdna_vit.py --mode eval \
  --archive-path "$OUT/model.zip" --data-root "$IMAGENET" --split val \
  --device cuda --decode-fraction 1.0 --activation-quant-disabled \
  --save-report "$OUT/eval_full.json"
```

完整归档解码依据归档元数据构建模型，不需要原始预训练权重。如果同时提供压缩和评测所需的输入，也可以用 `compress_eval` 替代分开的两次调用。

### 渐进解码评测

```bash
python run_devdna_vit.py --mode eval \
  --archive-path "$OUT/model.zip" --data-root "$IMAGENET" --split val \
  --device cuda --decode-planes 4 --activation-quant-disabled \
  --save-report "$OUT/eval_planes4.json"
```

正数 `--decode-planes` 优先于 `--decode-fraction`。未指定正数位平面数量时，`--decode-fraction 1.0` 表示完整解码，较小比例表示使用更少的位平面。减少位平面后的准确率需要实测，不能默认与完整解码一致。

## 6. 流式推理（streaming inference）

各运行模式应使用同一份归档进行对照。运行时从归档清单读取模型结构，归档解码不需要原始预训练权重文件。

| 运行模式 | 行为 |
| --- | --- |
| `serial_full_decode` | 推理前完整解码并加载归档，作为串行路径参考 |
| `lazy_family_decode` | 按需解码和加载模块组，执行后释放其动态设备权重 |
| `lazy_family_decode_overlap` | 当前组执行时，在工作线程中解码下一组 |
| `oracle_ready_to_run` | 评测已解码的模型，推理计时不包含初始归档解码 |

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

分组策略包括 `block_attn_mlp`（注意力与 MLP 分组）、`block_full`、`super_block`（由 `--streaming-runtime-super-block-k` 控制）和 `auto`（对候选粒度进行性能探测）。命令行与 YAML 默认值可能不同：解析器默认 `auto`，当前 YAML 选择 `block_attn_mlp`。请记录请求的策略与最终实际选用的策略。
