# 44 对模型的 W4A8 部署

最终 **44/44 对通过结构及实际 W4A8 精度检查**，并已放入本机目录：

```text
/data/Teammember/huan.zhang/AI-accelerator/rtl_compile/model/tflite_w4a8
```

其中 42 对直接复用之前完成的产物，未重新转换；仅新生成 yolox_nano、yolox_tiny 两个模型。原有根目录 `Alexnet_w4a8.tflite` 已检查结构和 INT4 打包后直接复用，其权重为 per-tensor；其余 43 个文件为 per-channel。

输出保留基准目录层级以区分同名模型，每个 `.tflite` 都有对应的 `.quantization.json`。根目录 `validation_summary.json` 记录全部 44 对、结构检查、输入尺寸、来源和 SHA256。仓库中的[部署清单](deployment_44.json)保留文件相对路径和校验值，不包含模型权重。

## 两个 YOLOX 的输入尺寸

原始 ONNX 的输入为 `[1,3,416,416]`；为与已有 W8A8 基准严格对齐，在副本中采用 `[1,3,224,224]`，导出 TFLite 输入为 `[1,224,224,3]`。

副本不修改任何算子、算子常量或权重，只更新输入尺寸并重新推导形状。导出阶段额外规范化等价的 Focus StridedSlice 布局、共享重复切片、对齐末端转置及量化边界。最终 yolox_nano 与基准均为 386 个算子，yolox_tiny 均为 296 个算子，结构差异均为 0。原始 ONNX 文件保持不变。

## 复用与补齐

已有之前的 `validation_summary.json` 时，可使用：

```bash
bash run_w4a8.sh --batch \
  --validated-summary /path/to/previous/validation_summary.json \
  --reference-root /path/to/effective_integer_references \
  --output-dir /path/to/tflite_w4a8 \
  --work-dir output/deploy_44 \
  --specialize-to-reference
```

脚本优先验证并复用目标文件，再复用旧报告中的已验证模型；仅缺少产物时调用 `run_w4a8.sh` 转换。`--specialize-to-reference` 为固定输入尺寸不一致的模型创建 ONNX 副本，再做严格结构比较；它不把 shape 不同直接判成通过。

## 验证范围

本次按结构目标验收，包括算子、数据连接、形状、控制常量、算子选项、真实 INT4 权重打包和 A8 激活。没有把 44 对声明为全部完成原 ONNX 数值等价或 stock LiteRT 推理验证。

- 6 对模型显式采用 W8A8 基准的池化 padding，可能改变原 ONNX 的边界数值。
- NanoDet 使用单独生成的全整数 W8A8 基准副本；原混合精度基准保留不变。
- INT4 反卷积可导出且结构一致，但所测 LiteRT CPU 仍需要补充 INT4 反卷积内核。

批量扩大范围时生成的额外产物已移出目标目录；交付目录只保留这 44 个 W4A8 模型。详细历史调试见[剩余模型记录](W4A8_REMAINING_DEBUG.md)。
