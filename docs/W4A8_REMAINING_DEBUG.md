# W4A8 剩余模型结构调试（2026-10-30）

后续已按用户指定允许输入尺寸特化，补齐 yolox_nano/tiny：**44 对现均通过结构检查并部署，42 对直接复用，2 对采用 224×224 输入新生成**。见 [44 对模型部署](DEPLOYMENT_44.md)及[文件与 SHA256 清单](deployment_44.json)。下文保留输入尺寸特化之前的历史结果。

本轮按结构一致性目标继续处理此前未通过的 13 对模型。比较范围仍包括算子、连接、输入输出形状、控制常量与算子选项；不比较权重数值、量化 scale/zero point、INT8/UINT8 signedness 和 opcode 版本。

## 结果与验证口径

最终 44 对中，**42 对结构通过，2 对因输入 shape 不一致跳过**。其中 41 对匹配原 W8A8/UINT8 基准，NanoDet 匹配另行生成的全 INT8 基准副本；原 NanoDet 基准保留不变。不能把这个结果写成“42 对均与原基准相同”或“42 对全部通过原 ONNX 数值验证”。

- 保持 ONNX padding 语义、使用原基准时：36 对结构通过，2 对 shape 跳过，6 对仍有差异。
- 继续处理这 6 对时：显式采用基准池化 padding 后，5 对与原基准结构一致；NanoDet 完成布局修复后，与修正的全 INT8 基准结构一致。
- 单元及集成测试：**222 passed**。
- 本轮结构模式仍实际执行 FP32 校准，检查真实 INT4 打包、INT32 bias 和内部 A8 精度；报告明确记录 `runtime_validation=not_requested`、`onnx_parity_validation=not_requested`。

最终批量报告与模型位于：

- [44 对最终结构报告](../output/debug_remaining_20261030/final_integer_structure/validation_summary.json)
- `output/debug_remaining_20261030/final_integer_structure/<原基准相对目录>/<模型名>_w4a8.tflite`
- 每个模型同名的 `.quantization.json` 包含结构比较、精度检查、padding 策略和具体改写。
- [基准对应关系](../output/debug_remaining_20261030/reference_manifest.json) 记录唯一替换的 NanoDet 基准副本。

`output/` 为本地生成产物，未纳入 Git；上述链接指向运行本轮验证的本地工作区。

## 此前剩余的 6 对

| 模型 | 本轮处理 | 最终结构基准 |
| --- | --- | --- |
| inception_v2-9 | SAME AvgPool 改为基准显式零 PAD + VALID AvgPool，明确采用基准边界分母 | 原基准，0 差异 |
| inception_v4_Opset18 | 同上 | 原基准，0 差异 |
| resnetv2_50_Opset18 | 采用基准零填充，规范化等价的通道常量前导单例维 | 原基准，0 差异 |
| resnetv2_101_Opset18 | 同上 | 原基准，0 差异 |
| yolov9e-seg | 恢复 Concat 原始轴；保留基准的恒等采样 Gather；共享重复 PAD；交换等价 PAD/Transpose；融合反卷积 bias；采用基准池化 padding | 原基准，0 差异 |
| nanodet-plus-m-1.5x_320 | 证明并展开通道 shuffle/分区，恢复检测头逐分支的通道优先 flatten 和最终转置；采用基准 padding；单独量化原基准中的浮点 DWConv 分支 | 全 INT8 基准副本，0 差异；原混合精度基准仍有差异 |

### Padding 策略

默认 `--reference-padding-policy preserve` 保留源模型语义。只有显式传入 `--reference-padding-policy baseline --validation-mode structure`，才采用基准池化的零填充规则。

这是结构目标下的语义选择：负输入 MaxPool 的零填充与负无穷填充可能给出不同结果；AvgPool 显式补零时边界分母也可能不同。改写记录带有 `semantics_changed`，报告不宣称与原 ONNX 数值等价。原生整数精度检查和严格结构比较保持启用。

### NanoDet 原基准并非全整数

原文件包含以下真实类型链：

```text
UINT8 -> DEQUANTIZE -> FLOAT32 PAD
      -> FLOAT32 DEPTHWISE_CONV_2D（FLOAT32 权重、bias）
      -> QUANTIZE -> UINT8 LEAKY_RELU
```

完整保留这条链，与要求该分支为 INT4 权重、INT8 激活冲突。本轮创建新的全 INT8 基准，保留原文件：

```bash
python scripts/make_integer_reference.py /path/to/nanodet-plus-m-1.5x_320_uint8.tflite \
  -o output/integer_reference/nanodet-plus-m-1.5x_320_uint8.tflite
```

脚本只处理独立的 `DQ -> PAD -> 浮点 DWConv -> Q` 分支，沿用已有输入和输出的 A8 编码，从原始 FP32 常量重新计算 W8 权重和 INT32 bias，删除该分支的 DQ/Q；生成文件必须能由所装 LiteRT 加载。新副本会引入不同的量化误差，附带 `.integerization.json` 记录。W4A8 权重仍从原 ONNX 转换得到的 FP32 常量计算，没有复制基准 W8 码值。

[CPU 检查记录](../output/debug_remaining_20261030/nanodet_cpu_verification.json) 中，原混合基准、全整数副本、真实打包的 W4A8 三个文件均完成实际推理。记录了输出误差；这不代表数据集上的任务精度已经验证。

## INT4 反卷积

导出器已支持 `TRANSPOSE_CONV` 的 `[output_shape, filter, input, bias]` 输入槽位、逐张量/逐通道 INT4 权重与 INT32 bias，并将独立的通道 Add/激活融合回反卷积。保留原算子结构，不将它改写成普通卷积。

centerface、FastSAM-s、FastSAM-x、yolov8m-seg 已通过结构检查；本轮修复的 yolov9e-seg 也包含反卷积。

**所测 LiteRT CPU 运行时仍缺少 INT4 反卷积内核**，实际准备阶段报错：

```text
tflite/kernels/transpose_conv.cc:372 weights->type != kTfLiteInt8 (INT4 != INT8)
```

默认完整验证继续拒绝发布这类不可由当前运行时执行的模型。结构模式可以导出真实 INT4 文件，实际部署须补充 INT4 内核。

额外的 [反卷积 CPU 检查](../output/debug_remaining_20261030/transpose_conv_cpu_verification.json) 在独立临时副本中，将反卷积已导出的 INT4 码值无损展开到 INT8，保持 scale、bias 和计算图不变。centerface、FastSAM-s/x、yolov8m-seg 均通过 FP32/ONNX 一致性及副本推理；这证明参数导出可用，不代表原 INT4 文件获得了 stock LiteRT 内核支持。

Yolo-v7 同时修复了标量与转置顺序、INT8 ReduceMax/ArgMax、整数索引输出 Cast 和量化边界；它已通过默认完整数值/运行验证，见 [完整验证报告](../output/debug_remaining_20261030/yolov7_full/validation_summary.json)。

## 复现

```bash
python -m pytest -q

# 默认保留 ONNX padding，先核对原基准。
python scripts/validate_model_zoo.py \
  --onnx-root /path/to/model_zoo/onnx \
  --reference-root /path/to/tflite_uint8 \
  --output-dir output/structure_preserve --workers 2 --timeout 300 \
  --validation-mode structure --skip-input-shape-mismatch

# 对结构目标显式采用基准 padding；NanoDet 使用另行生成的整数基准副本。
python scripts/validate_model_zoo.py \
  --onnx-root /path/to/model_zoo/onnx \
  --reference-root /path/to/effective_integer_references \
  --output-dir output/structure_baseline --workers 2 --timeout 300 \
  --validation-mode structure --reference-padding-policy baseline \
  --skip-input-shape-mismatch
```

本轮保留原 44 个基准，并用本地文件链接组成 `effective_integer_references`，仅 NanoDet 指向新副本。两份 YOLOX 源输入为 416×416，基准为 224×224，按要求记录为 `skipped_input_shape`，不计为通过。
