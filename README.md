# ONNX Edge Quantizer

将 FP32 ONNX 模型转换为 Min-Max 校准的量化 TFLite，默认导出 W8A8，支持逐张量和逐通道权重量化。

## 安装

使用 Conda 创建 Python 3.12 环境：

```bash
git clone https://github.com/zhzhzh-ai/onnx-edge-quantizer.git
cd onnx-edge-quantizer
conda env create -f environment.yml
conda activate onnx-edge-quantizer
export PYTHONNOUSERSITE=1
python scripts/install.py
python -m onnx_edge_quantizer --doctor
```

Windows PowerShell 将 `export PYTHONNOUSERSITE=1` 换为 `$env:PYTHONNOUSERSITE = "1"`。后续使用时，也需激活环境并设置该变量。安装脚本会按顺序安装 onnx2tf、AI Edge Quantizer 和 LiteRT。

## ONNX 转 TFLite

**W8A8（默认）**：

```bash
onnx-edge-quantizer model.onnx -o model_w8a8.tflite \
  --calibration calibration.npz
```

**W4A8**：

```bash
onnx-edge-quantizer model.onnx -o model_w4a8.tflite \
  --weight-dtype int4 --activation-dtype int8 \
  --weight-granularity per-channel --calibration calibration.npz
```

**W8A16**：

```bash
onnx-edge-quantizer model.onnx -o model_w8a16.tflite \
  --activation-dtype int16 --activation-scheme symmetric \
  --calibration calibration.npz
```

`calibration.npz` 按 ONNX 输入名保存经过预处理的 FP32 数据，沿用 ONNX 输入布局。例如输入 `images` 为 `[1,3,224,224]`，32 组校准数据的形状为 `[32,1,3,224,224]`；单输入也支持 `.npy`。不提供校准数据时，默认生成一组随机输入，仅用于验证转换流程。

常用选项：

| 选项 | 用途 |
| --- | --- |
| `--weight-granularity per-tensor\|per-channel` | 权重量化粒度，默认 `per-tensor`；激活始终为 `per-tensor` |
| `--input-shape 'images=1,3,224,224'` | 为动态输入指定固定形状，多输入可重复传入 |
| `--validation-data validation.npz` | 使用独立数据验证输出误差 |
| `--structure-reference baseline_uint8.tflite` | 要求导出模型与已有 TFLite 的算子、连接、形状和控制常量一致 |
| `--keep-onnx-input-layout` | 保留 ONNX 输入布局；默认可将输入接口调整为 NHWC |
| `--overwrite` | 覆盖已有输出 |

成功后生成 `.tflite` 和同名 `.quantization.json` 报告，包含量化配置及输出误差。导出前会检查 ONNX/FP32 TFLite 输出一致性，并实际运行量化模型。完整参数见 `onnx-edge-quantizer --help`。

## 支持的精度类型

W 表示权重位宽，A 表示内部激活位宽；模型输入/输出类型另行区分。

| 模式 | 精度 | 说明 |
| --- | --- | --- |
| `native`（默认） | W8A8：INT8 权重 + INT8 激活 | 已验证，默认配置 |
| `native` | W4A8：INT4 权重 + INT8 激活 | 已验证，使用静态 INT4 打包权重，算子范围见下文 |
| `native` | W8A16：INT8 权重 + INT16 激活 | 已验证，需 `--activation-scheme symmetric` |
| `native` | W4A16：INT4 权重 + INT16 激活 | 接受配置，依赖 AEQ/LiteRT 算子覆盖，尚未列入已验证组合 |
| `reference` | 权重、激活均可选 `int4 / uint4 / int8 / uint8 / int16` | 浮点参考模拟，支持对称/非对称量化，不提供整数压缩或加速 |

原生模式要求权重对称量化；INT8 激活默认非对称，INT16 激活要求对称。具体模型能否转换还取决于后端算子覆盖，默认遇到残留浮点计算会报错。

参考模拟示例：

```bash
onnx-edge-quantizer model.onnx -o model_reference.tflite \
  --mode reference --weight-dtype uint4 --activation-dtype int4 \
  --calibration calibration.npz
```

## W4A8 支持的算子

目前支持以下 TFLite 算子的 **INT4 权重 + INT8 激活** 导出：

| TFLite 算子 | 用途 | 权重量化粒度 |
| --- | --- | --- |
| `CONV_2D` | 普通二维卷积 | `per-tensor` / `per-channel` |
| `DEPTHWISE_CONV_2D` | 深度卷积 | `per-tensor` / `per-channel` |
| `FULLY_CONNECTED` | 全连接 | `per-tensor` / `per-channel` |

权重必须是稠密常量，Bias 使用 INT32。`BATCH_MATMUL` 和 `TRANSPOSE_CONV`（反卷积）目前不支持 W4A8 导出。

ReLU、LeakyReLU、Sigmoid、Softmax、池化、Add/Mul、Concat、Reshape、Transpose、Slice 等无 INT4 权重的算子沿用 INT8 或形状/索引内核，具体参数组合仍受 AEQ/LiteRT 限制；含上述算子不代表整个模型一定可转换。详细模型验证记录见 [W4A8 调试记录](docs/W4A8_DEBUG_30.md)。
