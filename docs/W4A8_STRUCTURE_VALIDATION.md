# W4A8 结构对齐验证（2026-10-29）

后续调试已将通过数从 14/44 提升至 **31/44**，见 [2026-10-30 调试记录](W4A8_DEBUG_30.md)。下文保留首次检查的历史结果；DeepLabV3 的原始 ONNX 实际包含输出到 257 的 Resize，其差异已确认为额外恒等 Resize 并修复。

已修复并重新生成 `output/googlenet_w4a8.tflite`。原模型备份位于 `output/backups/20261029_pre_structure_fix/`。

## 实现与协议

- 转换后、校准前折叠通道仿射、融合激活，合并连续 reshape，并在证明窗口位置和填充值等价时吸收显式 padding。GoogleNet 将三路通道的标量 MUL/ADD 移到 concat 后，形成通道向量广播。
- W4A8 使用 AEQ 静态 A8 校准图；INT4 权重从原始 FP32 常量重新量化，码值范围 `[-7,7]`，低 nibble 在前、二进制补码、末尾奇数元素的高 nibble 为 0。权重量化尺度和 INT32 bias 均重新计算。
- Conv2D/DepthwiseConv2D/FC 分别使用输出通道轴 0/3/0。支持 per-tensor/per-channel；激活保持 per-tensor。没有动态 blockwise FC，也没有 W8 码值截断。
- bias 码值 `round(fp32_bias / float32(input_scale * weight_scale))`。量化尺度过小时提高权重 scale，为 bias 和点积保留 INT32 范围；不使用截断、溢出绕回或假定为 0。
- `--structure-reference` 比较算子、边、形状、控制常量和实际 padding；忽略名称、权重码值、scale/zp、signedness 和 opcode 版本。只有 SAME 与 VALID 的实际 padding 均为零等确证等价情况才规范化。
- 结构不同、精度错误、浮点计算残留或 ONNX/FP32 数值不一致时，拒绝发布，不覆盖已有模型。候选仅保留在显式的 debug workdir。

## GoogleNet

旧产物 103 个算子，新产物和 UINT8 基准均为 92 个。新产物有 57 个卷积和 1 个 FC，58 份权重全部为 affine INT4，内部 feature map 为 INT8。无 FLOAT32 计算残留；结构比较为 0 个差异。

原先多出的量化、padding 和 reshape 已消除；输入改为 `[1,224,224,3]`，通道预处理和三路 Gather/Reshape/Concat 与基准相同。

详细参数、结构对比和数值指标见 [GoogleNet 报告](../output/googlenet_w4a8.quantization.json)。

## 全量配对检查

从指定两个目录匹配出 **44 对** ONNX/`*_uint8.tflite`，其中 **14 对通过，30 对失败**。EfficientNet-Lite2 在 onnx1/onnx2 各有一对，按两对记录。未把所有模型声明为已支持。

成功必须同时通过结构检查、INT4/INT8 类型及权重打包检查、ONNX/FP32 输出校验和最终 LiteRT 推理。

| 模型 | 组 | 算子数 | INT4 权重数 |
| --- | --- | ---: | ---: |
| Alexnet | onnx1 | 15 | 8 |
| EfficientNet-Lite2 | onnx1 | 82 | 65 |
| EfficientNet-Lite2 | onnx2 | 82 | 65 |
| EfficientNet-Lite4 | onnx2 | 118 | 92 |
| GoogleNet | onnx3 | 92 | 58 |
| MobilenetV1_1.0_224 | onnx4 | 32 | 28 |
| MobilenetV2_1.0_224 | onnx4 | 67 | 53 |
| SqueezeNet | onnx5 | 41 | 26 |
| resnet101_Opset18 | onnx4 | 145 | 105 |
| resnet152_Opset18 | onnx5 | 213 | 156 |
| resnet50_Opset18 | onnx4 | 77 | 54 |
| resnet50_Opset18_299 | onnx4 | 77 | 54 |
| vgg16_Opset18 | onnx7 | 24 | 16 |
| vgg19_Opset18 | onnx7 | 27 | 19 |

失败项（完整差异和日志见 JSON）：

| 模型 | 原因 |
| --- | --- |
| DeepLabV3 | 结构不同（15 处）；subgraphs[0].operators[69].inputs[1].value[0]: 33 / 基准 257 |
| FaceBlur | 结构不同（890 处）；subgraphs[0].operators.length: 126 / 基准 122 |
| FaceReid | ./tflite/kernels/internal/reshape_utils.h num_input_elements != num_output_elements (3200 != 12800)Node number 101 (RESHAPE) failed to prepare. |
| FastSAM-s | KeyError: 'wa/model.22/Range_output_0_range_scalar_0_reshape_shape' |
| FastSAM-x | KeyError: 'wa/model.22/Range_output_0_range_scalar_0_reshape_shape' |
| HumanSegV2 | 结构不同（388 处）；subgraphs[0].operators.length: 133 / 基准 131 |
| RetinaFace | 结构不同（1176 处）；subgraphs[0].operators.length: 292 / 基准 140 |
| Yolo-v7 | 结构不同（2533 处）；subgraphs[0].operators.length: 394 / 基准 231 |
| centerface | Static W4A8 supports Conv2D, DepthwiseConv2D and FullyConnected weights; unsupported matrix/deconvolution operation |
| inception_v2-9 | 结构不同（1104 处）；subgraphs[0].operators.length: 101 / 基准 106 |
| inception_v3_Opset18 | Ranks of scales (5) and zps (5) must be the same as the tensor rank (1). |
| inception_v4_Opset18 | 结构不同（1778 处）；subgraphs[0].operators.length: 188 / 基准 202 |
| nanodet-plus-m-1.5x_320 | Ranks of scales (5) and zps (5) must be the same as the tensor rank (1). |
| resnetv2_101_Opset18 | Ranks of scales (5) and zps (5) must be the same as the tensor rank (1). |
| resnetv2_50_Opset18 | Ranks of scales (5) and zps (5) must be the same as the tensor rank (1). |
| shufflenet_v2_x2_0_Opset17 | tflite/kernels/slice.cc:157 output_shape_vector[idx] != SizeOfDimension(output, idx) (28 != 1)Node number 40 (SLICE) failed to invoke. |
| yolo-fastestv2 | 结构不同（2947 处）；subgraphs[0].operators.length: 266 / 基准 227 |
| yolov3-tinyu | KeyError: 'wa/model.20/Range_2_output_0_range_scalar_0_reshape_shape' |
| yolov3-tinyu | KeyError: 'wa/model.20/Range_2_output_0_range_scalar_0_reshape_shape' |
| yolov3u | KeyError: 'wa/model.28/Range_4_output_0_range_scalar_0_reshape_shape' |
| yolov5nu | KeyError: 'wa/model.24/Unsqueeze_6_output_0_unsqueeze_suffix_2' |
| yolov5su | KeyError: 'wa/model.24/Range_output_0_range_scalar_0_reshape_shape' |
| yolov8m | KeyError: 'wa/model.22/Range_output_0_range_scalar_0_reshape_shape' |
| yolov8m-cls | 结构不同（1508 处）；subgraphs[0].operators.length: 192 / 基准 142 |
| yolov8m-pose | KeyError: 'wa/model.22/Range_output_0_range_scalar_0_reshape_shape' |
| yolov8m-seg | KeyError: 'wa/model.22/Range_output_0_range_scalar_0_reshape_shape' |
| yolov9e-seg | tflite/kernels/concatenation.cc:213 t->dims->data[d] != t0->dims->data[d] (7 != 14)Node number 750 (CONCATENATION) failed to prepare. |
| yolov9t | KeyError: 'wa/model.22/Range_output_0_range_scalar_0_reshape_shape' |
| yolox_nano | 结构不同（4350 处）；subgraphs[0].inputs[0][1]: 416 / 基准 224 |
| yolox_tiny | 结构不同（3361 处）；subgraphs[0].inputs[0][1]: 416 / 基准 224 |

所有动态尺寸模型已按基准部署轴顺序补全后复查。FaceReid 在浮点 Reshape 阶段失败；RetinaFace 仍有浮点区域和拓扑差异；多份 YOLO 在 AEQ 的空张量/shape 常量校准或浮点图连接处失败。标量 Reshape 的空 shape 兼容已修复，但未将其等同于覆盖所有 shape 子图。

InceptionV2/V4 的基准包含显式零 PAD + VALID AvgPool，当前 ONNX 转换为 SAME AvgPool。两种边界分母语义需要独立核对；不能为了凑齐算子数量直接互换。DeepLabV3、部分 YOLO 则存在原 ONNX 固定尺寸/Resize 常量与基准不一致的情况。

全部成功模型与各失败模型的 candidate/log/结构差异位于 [批量报告](../output/validation_20261029/zoo_all/validation_summary.json)。

## 复现

```bash
/data/miniforge3/envs/onnx-edge-quantizer/bin/python -m onnx_edge_quantizer \
  /data/Teammember/huan.zhang/AI-accelerator/rtl_compile/model/model_zoo/onnx/onnx3/GoogleNet.onnx \
  -o output/googlenet_w4a8.tflite --weight-dtype int4 \
  --weight-granularity per-channel --overwrite \
  --structure-reference /data/Teammember/huan.zhang/AI-accelerator/rtl_compile/model/tflite_uint8/onnx3/GoogleNet/GoogleNet_uint8.tflite

python scripts/validate_model_zoo.py \
  --onnx-root /data/Teammember/huan.zhang/AI-accelerator/rtl_compile/model/model_zoo/onnx \
  --reference-root /data/Teammember/huan.zhang/AI-accelerator/rtl_compile/model/tflite_uint8 \
  --output-dir output/new_zoo_run --workers 2 --timeout 180
```

本次使用 seed=0 的一组合成输入，只验证结构与推理流程及记录量化误差；不能据此推断分类精度、检测 mAP 或真实数据分布上的量化效果。部署精度需要代表性校准及独立验证集。
