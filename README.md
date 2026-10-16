# ONNX Edge Quantizer

项目地址：[zhzhzh-ai/onnx-edge-quantizer](https://github.com/zhzhzh-ai/onnx-edge-quantizer)。

将 **FP32 ONNX** 模型转换成使用 **Min-Max 校准**的 TFLite 模型。默认使用 **INT8 对称权重 + INT8 非对称激活**，逐张量量化。提供命令行入口 `onnx-edge-quantizer` 和 `python -m onnx_edge_quantizer`。

权重量化粒度通过 `--weight-granularity per-tensor|per-channel`（别名 `--granularity`）选择，默认 `per-tensor`，原生模式与参考模式都支持。**该参数仅控制权重，激活始终按 per-tensor 量化。**

## 支持范围：原生整数与参考模拟

**TFLite/Google AI Edge Quantizer 无法原生执行任意权重、激活精度组合。** 本工具不把更改文件后缀、浮点模型或 8 位存储冒充 4 位整数模型，也不会静默修改用户选择的精度。

- `--mode native`（默认）：调用 Google AI Edge Quantizer，真正量化支持的算子。权重支持 `int4` / `int8`，要求对称；内部激活支持 `int8` / `int16`，其中 `int16` 要求显式选择对称量化。INT4 算子覆盖取决于所安装的 AEQ/LiteRT 版本；不支持的算子默认报错。上游官方支持 Linux/macOS；本项目在 Windows + Python 3.12 上也实测通过了当前 nightly 的 W8A8、W8A16、W4A8 导出推理，Windows 支持仍应视为实验性质。
- `--mode reference`：权重和激活均接受 `int4 / uint4 / int8 / uint8 / int16`，支持对称和非对称。导出的 `.tflite` 使用浮点算子显式计算除法、舍入、截断、反量化；权重保存为量化后重建的 FP32 值。**这是精度参考模拟，不提供整数压缩、4 位打包或整数加速。** 报告和 TFLite description 均明确标记此行为。
- `uint8` 模型输入/输出与 `uint8` 内部激活不是一回事。本工具的精度参数指计算图内部精度；原生模式不把 UINT8 I/O 包装视为 UINT8 内部量化。
- 此工具不能保证转换任意 ONNX 算子。浮点 ONNX→TFLite 转换由 onnx2tf 负责；不支持的算子、控制流、多子图和不匹配的输入布局会明确失败。当前模型运行时输入必须全部是 FP32 张量。

## 安装（Conda，推荐）

先安装 Miniconda、Anaconda 或 Miniforge，然后在支持 `conda` 的终端中执行。以下步骤适用于 Windows、Linux 和 macOS：

```bash
git clone https://github.com/zhzhzh-ai/onnx-edge-quantizer.git
cd onnx-edge-quantizer
conda env create -f environment.yml
conda activate onnx-edge-quantizer
python scripts/install.py
```

`environment.yml` 创建名为 `onnx-edge-quantizer` 的 Python 3.12 环境，并安装 pip。随后安装脚本按经过验证的顺序安装 onnx2tf、Google AI Edge Quantizer 和 LiteRT；这些转换依赖仍通过当前 Conda 环境里的 pip 安装。

也可以不使用环境文件，手动创建相同环境：

```bash
conda create -n onnx-edge-quantizer -c conda-forge --override-channels python=3.12 pip -y
conda activate onnx-edge-quantizer
python scripts/install.py
```

检查环境并生成 W4A8 模型：

```bash
python -m onnx_edge_quantizer --doctor
onnx-edge-quantizer model.onnx -o model_w4a8.tflite --weight-dtype int4 --activation-dtype int8 --calibration calibration.npz
```

Windows PowerShell 激活 Conda 环境后也可以使用：

```powershell
.\quantize.ps1 model.onnx -o model_w4a8.tflite --weight-dtype int4 --activation-dtype int8 --calibration calibration.npz
```

Windows 启动脚本优先使用当前激活的 Conda 环境；没有激活 Conda 时，仍兼容激活的 venv 和项目原有的 `.venv312` / `.venv`。安装脚本会检查实际运行的 Python 是否属于所激活的 Conda 环境，且不往 Conda `base` 环境安装。以后重新打开终端，只需先执行 `conda activate onnx-edge-quantizer`。

`onnx2tf>=2.6.9` 的直接 FlatBuffer 转换后端要求 Python 3.12+，不需要 TensorFlow。onnx2tf 依赖 stable LiteRT，而 AEQ 依赖 nightly LiteRT，两者使用相同 Python 模块名；`scripts/install.py` 保证最后安装本项目验证的 nightly 运行时。请不要把所有依赖无序地并入 `environment.yml` 的 pip 部分。

如需手动安装后端，请在激活的 Conda 环境执行：

```bash
python -m pip install -e '.[convert,test]' onnx2tf==2.6.9
python -m pip install ai-edge-quantizer-nightly==0.10.0.dev20261016
python -m pip install --force-reinstall --no-deps ai-edge-litert-nightly==2.4.0.dev20261014
```

`requirements-windows-tested.txt` 记录最初在 Windows venv 中验证的依赖版本；Conda 环境需按上述顺序安装后端。GitHub Actions 现在使用同一份 `environment.yml` 创建 Conda 环境后运行测试。若已有等价的浮点 TFLite，可用 `--float-tflite float.tflite` 跳过 onnx2tf；核心模块支持 Python 3.11+，仍会校验与原始 ONNX 的输出一致性。


## 常用命令

默认：不提供校准数据时，按模型输入形状生成 **1 组** `[-1, 1)` 均匀分布 FP32 输入，种子为 0，实际运行模型、收集中间层范围：

```bash
onnx-edge-quantizer model.onnx -o model_int8.tflite
```

使用外部校准数据和独立验证数据：

```bash
onnx-edge-quantizer model.onnx -o model_int8.tflite \
  --calibration calibration.npz --validation-data validation.npz
```

按输出通道量化权重，每个通道独立计算 Min-Max、scale 和 zero-point：

```powershell
.\quantize.ps1 model.onnx -o model_per_channel.tflite --weight-granularity per-channel --calibration calibration.npz
```

逐张量量化可显式使用 `--weight-granularity per-tensor`。两种粒度均可搭配 `--weight-dtype` 和 `--mode reference`。通道轴按转换后的 TFLite 权重布局确定：Conv2D、FullyConnected、TransposeConv 为第 0 维，DepthwiseConv2D 为第 3 维；BatchMatMul 按左右矩阵和转置标志选择输出通道轴。共享权重若被以不兼容的通道轴使用，会明确报错。原生后端若没有按所选粒度导出，也会拒绝发布模型。

原生 INT4 权重 + INT8 激活（需要对应算子的内核支持）：

```bash
onnx-edge-quantizer model.onnx -o model_w4a8.tflite \
  --weight-dtype int4 --activation-dtype int8 --calibration calibration.npz
```

这会请求真正的 W4A8 原生整数模型，而非参考模拟。当前验证过全连接算子的两种权重粒度。实际模型是否能全部使用 W4A8，取决于 AEQ/LiteRT 的算子覆盖：当前安装版本的 W4 静态策略包含 FullyConnected 和 Conv2D，DepthwiseConv2D 等算子并不在该 W4 策略中。若模型含不支持的算子，默认严格报错；`--allow-float-ops` 导出的混合模型不能视为全模型 W4A8，也不保证加速器支持。

原生 INT8 权重 + INT16 激活，显式使用对称激活：

```bash
onnx-edge-quantizer model.onnx -o model_w8a16.tflite \
  --activation-dtype int16 --activation-scheme symmetric \
  --calibration calibration.npz
```

UINT4 权重 + INT4 非对称激活的**参考模拟**：

```bash
onnx-edge-quantizer model.onnx -o model_reference.tflite \
  --mode reference --weight-dtype uint4 --activation-dtype int4 \
  --calibration calibration.npz
```

动态输入尺寸必须显式指定，数据沿用 ONNX 的输入布局（例如 NCHW），工具不猜测或自动转置外部数据：

```bash
onnx-edge-quantizer model.onnx -o model_int8.tflite \
  --input-shape 'images=1,3,224,224' --calibration calibration.npz
```

多输入模型可以重复指定 `--input-shape`。转换后的输入/输出若更名，可显式提供 `--input-map 'images=converted_images'`、`--output-map 'scores=converted_scores'`。不根据同形状或顺序猜测多输入/输出映射。

其他选项：

- `--synthetic-samples 32 --seed 42 --random-min 0 --random-max 1`：配置合成校准输入。
- `--weight-scheme symmetric|asymmetric`、`--activation-scheme symmetric|asymmetric`：选择方案，受原生模式约束。
- `--weight-granularity per-tensor|per-channel`：选择权重量化粒度，默认 per-tensor；`--granularity` 是同一参数的别名。激活保持 per-tensor。
- `--allow-float-ops`：显式允许原生导出中不支持的算子保留浮点；报告列出残留浮点算子。默认禁止。
- `--keep-workdir artifacts/debug-run`：保留临时 ONNX、转换日志和中间模型；该目录必须尚不存在。
- `--overwrite`：允许覆盖输出模型及其报告。
- `--float-atol`、`--float-rtol`：配置 ONNX 与浮点 TFLite 输出一致性检查的容差；默认分别为 `1e-4`、`1e-3`。

**合成的单组随机输入可验证流程，不代表真实任务分布。** 实际部署应使用经过正确预处理、有代表性的输入校准，并使用独立验证集检查精度；分类准确率、检测 mAP 等任务指标需要在应用侧评估。

## 校准数据格式

**校准输入不要求归一化到 0～1，工具也不会自动归一化。** 应输入模型正常推理时真正接收到的数据：如果预处理位于模型外，先完成相同的归一化、均值方差变换、通道顺序与 resize；如果预处理已包含在 ONNX 内，则提供该模型接口要求的原始浮点输入。不同输入可以有完全不同的数值范围；例如一个输入为 `[-1,1]`，另一个输入为 `[0,255]`，均可使用。当前接口要求浮点张量，原始 uint8 图像需按模型接口转换为 FP32，而非未经确认就除以 255。

单输入 `.npy` 或按 ONNX 输入名保存的 `.npz`。支持：

- 一组输入，形状完全等于模型输入形状；
- 多组输入，额外增加最前面的 `samples` 维；
- 一个目录，包含多个 `.npz` 文件，按文件名字典序读取；
- 多输入 `.npz`，键必须与 ONNX 的运行时输入名称一致，每个输入的样本数相同。

例如，模型输入 `images` 形状为 `[1,3,224,224]`：

```python
import numpy as np

# 单组：保留模型的 batch 维。
np.save('one.npy', np.random.rand(1, 3, 224, 224).astype(np.float32))

# 32 组：增加 samples 维，不能把它与模型的 batch 维混淆。
np.savez('calibration.npz',
         images=np.random.rand(32, 1, 3, 224, 224).astype(np.float32))

# 多输入，例如模型输入 left/right 均为 [1,4]。
np.savez('multi.npz',
         left=np.random.rand(32, 1, 4).astype(np.float32),
         right=np.random.rand(32, 1, 4).astype(np.float32))
```

不接受整数校准数据、NaN/Inf、错位布局或不匹配的尺寸；不加载 pickle。

可直接运行两个输入、不同数值范围的完整示例：

```powershell
python examples/make_multi_input_demo.py
.\quantize.ps1 artifacts/multi-input/multi_input.onnx -o artifacts/multi-input/quantized.tflite --calibration artifacts/multi-input/calibration.npz --validation-data artifacts/multi-input/validation.npz
```

该示例的 `left` 输入范围为 `[-5,5]`，`right` 为 `[10,100]`；这只是展示数据格式，不是推荐所有模型使用这些范围。

## 输出和验证

成功输出：

- 用户指定的 `.tflite`；
- 同名 `.quantization.json`：权重/激活粒度配置、激活校准 Min/Max、原生 recipe 或参考量化参数、输出张量实际类型/scale/zero-point/quantized_dimension、残留浮点算子、逐样本输出 MSE/最大绝对误差/余弦相似度。参考模式逐通道权重记录 axis、各通道范围、scale 和 zero-point。

输出前会实际运行 ONNX、FP32 TFLite、量化 TFLite。FP32 转换不一致、量化模型不可运行或严格原生模式仍含浮点计算时，命令失败，不发布候选模型。量化误差会记录在报告里，不内置一个适用于所有任务的精度阈值。所有校准样本默认也用于验证，可用 `--validation-data` 提供独立数据。

Min-Max 统计针对**转换后的浮点 TFLite 计算图**，并非保证与 ONNX 节点一一对应（转换器可能融合算子）。Bias、shape 和 index 常量不作为普通权重量化；原生 bias 类型由 AEQ 按算子规则推导。参考模式量化 Conv2D、DepthwiseConv2D、FullyConnected、BatchMatMul 和 TransposeConv 的浮点常量权重，所有可观测浮点运行时激活边使用固定校准范围；其他浮点常量保持原值。所有激活的校准范围都来自原始浮点图，不会逐层重新估算量化后范围。

参考模式的对称有符号量化使用窄范围（如 INT8 `[-127,127]`）。无符号对称量化采用中点零点（如 UINT8 `zero_point=128`，有效范围 `[1,255]`），不将负权重强行截断为零。非对称量化确保浮点零可精确表示。全零张量使用 `scale=1`；零点按对应方案确定。

## Docker（原生模式）

```bash
docker build -t onnx-edge-quantizer .
docker run --rm -v "$PWD:/work" onnx-edge-quantizer \
  model.onnx -o model_int8.tflite --calibration calibration.npz
```

Windows PowerShell 挂载语法：

```powershell
docker run --rm -v "${PWD}:/work" onnx-edge-quantizer model.onnx -o model_int8.tflite
```

## 示例和测试

```bash
python examples/make_demo.py
onnx-edge-quantizer artifacts/demo/demo.onnx -o artifacts/demo/demo_int8.tflite \
  --calibration artifacts/demo/calibration.npz \
  --validation-data artifacts/demo/validation.npz

# Windows 本机可验证完整的 ONNX -> 浮点 TFLite -> 参考量化流程。
onnx-edge-quantizer artifacts/demo/demo.onnx -o artifacts/demo/demo_reference.tflite \
  --mode reference --calibration artifacts/demo/calibration.npz

python -m pytest -q
```

测试包含：两种权重粒度下各 5×5 共 25 个参考精度组合的实际 LiteRT 推理与独立 NumPy 结果比较、Min-Max 数学边界和全零通道、多输入与动态尺寸、外部和合成校准、模型转换一致性、禁止静默覆盖及不支持的原生精度拒绝。完整 onnx2tf 和 AEQ 集成测试见 `tests/test_integration.py`，相应依赖可用时执行。

2026-10-16 本机验证：原有 105 项测试与新增 4 项集成测试均通过，共 109 项。集成覆盖 ONNX 转换、W8A8 / W8A16 / W4A8 的逐张量和逐通道原生整数推理、NCHW 普通卷积/深度卷积网络、实际 scale 数量和通道轴检查、两个输入且范围超出 `[0,1]` 的校准、从 ONNX 开始的完整 W4A8 导出，以及 Windows 临时文件清理；实际 PowerShell 命令也已生成 `artifacts/demo/demo_int8.tflite`、`demo_per_channel_int8.tflite`、`demo_w4a8.tflite` 与 UINT4/INT4 参考模型。此验证使用生成的小模型，未宣称覆盖所有 ONNX 模型或加速器。Docker 配置已提供，未在本机运行 Docker 构建测试。

## 上游资料

- [Google AI Edge Quantizer（支持精度与算子覆盖）](https://github.com/google-ai-edge/ai-edge-quantizer)
- [AEQ 静态量化 recipe 约束](https://github.com/google-ai-edge/ai-edge-quantizer/blob/main/ai_edge_quantizer/recipe_manager.py)
- [onnx2tf（ONNX 到 TFLite 的转换后端）](https://github.com/PINTO0309/onnx2tf)
- [LiteRT INT8 量化规范](https://ai.google.dev/edge/litert/models/quantization_spec)
