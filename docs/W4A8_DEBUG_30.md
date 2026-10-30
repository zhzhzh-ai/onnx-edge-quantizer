# 剩余 30 对模型的 W4A8 调试记录（2026-10-30）

后续结构专项调试已补充 INT4 反卷积导出、显式基准 padding 策略及 NanoDet 混合精度基准修正，见 [剩余模型调试](W4A8_REMAINING_DEBUG.md)。本文保留此前完整推理验证的历史结果。

本次继续处理上次失败的 30 对。最终代码对全部 44 对重新转换验证：**31 对通过，13 对未通过**；原先通过的 14 对保持通过，原先失败的 30 对中新增 **17 对通过**。

最终产物、每个模型的日志、候选文件、完整结构差异和数值指标位于：

- [完整批量报告](../output/debug_30_20261030/final_verified/validation_summary.json)
- 成功模型：`output/debug_30_20261030/final_verified/<原 UINT8 相对目录>/<模型名>_w4a8.tflite`
- 同目录的 `*.quantization.json` 包含精度检查、ONNX/浮点输出误差、W4A8 输出误差和具体改写记录。
- [上次 14/44 的历史报告](../output/validation_20261029/zoo_all/validation_summary.json) 保留原状。

## 新增通过的 17 对

下表每一行都完成了重新转换、严格结构比较、INT4 权重/A8 激活检查、ONNX 与 FP32 TFLite 输出校验和实际 W4A8 LiteRT 推理。两份 yolov3-tinyu 参考分别计数。

| 模型 | UINT8 分组 | 算子数 | INT4 权重数 |
| --- | --- | ---: | ---: |
| DeepLabV3 | onnx1 | 71 | 55 |
| FaceBlur | onnx2/FaceBlur | 122 | 88 |
| FaceReid | onnx2/FaceReid | 87 | 60 |
| HumanSegV2 | onnx3/HumanSegV2 | 131 | 73 |
| inception_v3_Opset18 | onnx3/inception_v3_Opset18 | 129 | 95 |
| RetinaFace | onnx5/RetinaFace | 140 | 82 |
| shufflenet_v2_x2_0_Opset17 | onnx5/shufflenet_v2_x2_0_Opset17 | 190 | 57 |
| yolo-fastestv2 | onnx8/yolo-fastestv2 | 227 | 79 |
| yolov3-tinyu | onnx8/yolov3-tinyu | 98 | 24 |
| yolov3-tinyu | onnx8 | 98 | 24 |
| yolov3u | onnx8/yolov3u | 323 | 91 |
| yolov5nu | onnx8/yolov5nu | 279 | 76 |
| yolov5su | onnx8/yolov5su | 279 | 76 |
| yolov8m | onnx9/yolov8m | 317 | 84 |
| yolov8m-cls | onnx9/yolov8m-cls | 142 | 39 |
| yolov8m-pose | onnx9/yolov8m-pose | 354 | 93 |
| yolov9t | onnx9/yolov9t | 675 | 186 |

## 修复内容

### 静态形状与原始 ONNX 控制

`static_folding.py` 根据固定输入和算子语义传播可信形状，折叠 Shape、Gather、Slice、Floor、Fill、Range、广播等静态计算，支持 Split/SplitV 的输出形状传播。不会用校准样本数值替代动态张量，也不会直接相信转换器的旧输出尺寸标注。

FaceReid 的 ONNX Reshape 控制原本是 `[-1, 1]` 等，直接转换时被旧的 640 尺寸标注替换成 `[12800, 1]`，而本次部署输入是 320。`onnx_compat.py` 恢复原始 `-1`，再推导实际维度。只有原始显式维度和转换器控制一致时才恢复推导槽，避免覆盖转换器已有的布局调整。FaceReid 不再出现 `3200 != 12800`。

DeepLabV3 的末尾确实有从 33 到 257 的 Resize。原来的差异来自它前面多出的 **33 到 33 恒等 Resize**，并非 ONNX 缺少 257 的输出配置。本次删除恒等 Resize 后通过；修正了历史排查中对这一项的判断。

### 布局和结构等价改写

- 在切分前规范化单位步长 StridedSlice，传播通道布局并修正切分轴、控制向量和广播常量。
- 根据明确的参考契约将完整 Slice 分区合并为整数 SplitV，或将 Split 降为参考中的完整 Slice 分区。
- 将单元素向量 Gather 后仅移除索引维的 Reshape 合并为标量 Gather。
- 在逐点运算中恢复被合并的连续维度；对非标量广播常量证明维度分组后才调整其形状。
- 根据参考中的显式转换保留 Softmax 行布局、Reduction 输入布局及输出接口转换，维护验证所需的逆置换和公开输出名称。
- ShuffleNet 的交织通道 Gather 只有在索引与参考的 Reshape/Transpose/Reshape 排列完全相同时才展开。
- 消除零 batch padding 辅助 Concat/Slice/零 Add；只在实际填充量严格相等且参考采用 SAME Conv 时吸收显式零 PAD。
- FaceReid 将秩无关的 Sigmoid 移到 Reshape 前，与参考顺序一致。

### A8 约束与后端兼容

- 恢复 AEQ 导出的 `DQ -> LeakyRelu -> Q` 为原生 INT8 LeakyRelu，处理共享 DQ 和多个 Q 消费者。
- 统一布局、拼接和池化相关的 A8 编码，保留 Sigmoid/Softmax/Tanh 的固定编码约束；冲突时保留必要的重定量化。
- 恢复参考 Quantize 边界前，必须证明去掉边界后的**完整核心图相同**。边界使用候选的真实编码，允许恒等 Quantize；不复制 UINT8 基准的权重或校准参数。
- 广播常量只调整前导单例维，且两种形状都必须广播到同一输出；不会将 `[2,1]` 和 `[1,2]` 当作等价。
- AEQ 无法读取委托消除的中间张量时，使用它公开的 profiler 校准模式重新收集 min/max。
- `aeq_compat.py` 仅在导出期间修正单个 scale/zero point 跨秩复用的数组形状，保留原有数值和 AEQ 校验；完成或异常时恢复钩子，不修改安装环境中的库。
- 为确认为标量的张量显式保存空 shape 向量，避免 AEQ 插入 DQ 时处理 `None` 形状失败。

最终 INT4 权重和 INT32 bias 仍从原始 FP32 常量与实际 A8 scale 重新计算，保持低 nibble 在前的打包格式。新增变换没有改变这一数值协议。

### ONNX 验证基准

YOLOv3-tinyu 在 ONNX Runtime 开启 Basic/Extended/All 图优化时，与原始计算图存在明显输出偏差。实测同一输入：开启优化最大偏差约 `216.79`；关闭优化后与原始 FP32 TFLite 的最大偏差约 `9e-4`，相对误差检查通过。额外暴露中间张量、阻止相关融合时也与原图一致。

验证现使用 `ORT_DISABLE_ALL` 执行原始 ONNX，并在报告中记录 `onnx_reference_graph_optimization=disabled`。没有放宽 `float-atol/float-rtol`。

## 仍未通过的 13 对

| 模型 | 当前阻塞及后续工作 |
| --- | --- |
| centerface | 包含 TransposeConv；当前 LiteRT 的 A8 反卷积内核拒绝 INT4 权重。 |
| FastSAM-s | 动态形状与空 shape 校准问题已解决；现在阻塞于 INT4 TransposeConv。 |
| FastSAM-x | 同上。 |
| yolov8m-seg | 静态形状与校准已前进到权重导出；阻塞于 INT4 TransposeConv。 |
| yolov9e-seg | 浮点图 Concat 仍有 `7 != 14` 的空间尺寸问题；此外还包含 TransposeConv。 |
| inception_v2-9 | 参考为显式零 PAD + VALID AvgPool；源图为不计边界 padding 的平均池化。边界分母不同，不能直接当作等价。 |
| inception_v4_Opset18 | 同上；需要先确定采用哪一方的边界语义。 |
| resnetv2_50_Opset18 | 参考 PAD 和源图负值 PADV2 不同；还有通道广播常量的布局差异。不能在未证明输入非负时替换为零填充。 |
| resnetv2_101_Opset18 | 同上。 |
| nanodet-plus-m-1.5x_320 | AEQ 秩问题已解决；负值 MaxPool padding、通道布局和检测头结构仍不同。 |
| Yolo-v7 | LeakyRelu、Slice 和连续维度合并已修复；仍有逐点运算与 Transpose 的位置差异、ReduceMax/ArgMax 浮点岛，以及类别输出的 Cast/Quantize 链差异。 |
| yolox_nano | 源 ONNX 输入固定为 416，参考部署输入为 224；输出网格和结构随之不同。需要匹配尺寸的源模型或明确的输入特化方案。 |
| yolox_tiny | 同上。 |

INT4 反卷积不是仅由 Python 导出器的限制推测得出。使用最小的 A8/INT4 TransposeConv 模型，当前 LiteRT 实际报错：

```text
tflite/kernels/transpose_conv.cc:372 weights->type != kTfLiteInt8 (INT4 != INT8)
Node number 0 (TRANSPOSE_CONV) failed to prepare.
```

见 [运行时能力检查](../output/debug_30_20261030/capabilities/runtime_capabilities.json)。在保持相同 TransposeConv 结构、权重全部 INT4 和当前 LiteRT 实际推理三个要求下，需要补充内核支持；本次没有将这类模型静默改为 W8 或浮点。

## 验证和复现

项目测试：**193 passed**。覆盖真实 LiteRT 数值、原始 ONNX Reshape 恢复及布局保护、Split 静态形状、切分表示互换、广播方向保护、通道重排、padding 辅助图消除、恒等 Resize 和异常时兼容钩子的恢复。

```bash
cd /data/Teammember/huan.zhang/modelconvert/onnx-edge-quantizer
/data/miniforge3/envs/onnx-edge-quantizer/bin/python -m pytest -q

/data/miniforge3/envs/onnx-edge-quantizer/bin/python scripts/validate_model_zoo.py \
  --onnx-root /data/Teammember/huan.zhang/AI-accelerator/rtl_compile/model/model_zoo/onnx \
  --reference-root /data/Teammember/huan.zhang/AI-accelerator/rtl_compile/model/tflite_uint8 \
  --output-dir output/revalidate_w4a8 --workers 2 --timeout 240
```

仅重试原来失败的 30 对，可追加：

```bash
--failed-from output/validation_20261029/zoo_all/validation_summary.json
```

本次使用 seed=0 的一组合成输入。通过表示结构一致、FP32 转换输出通过既定误差检查、W4/A8 精度类型与打包正确并能实际执行；报告记录了 W4A8 数值误差，但不代表真实数据集上的分类精度或检测 mAP 已验证。
