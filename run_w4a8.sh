#!/usr/bin/env bash
# ONNX -> native W4A8 TFLite. Extra arguments are passed to the Python CLI.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${W4A8_PYTHON:-/data/miniforge3/envs/onnx-edge-quantizer/bin/python}"
DEFAULT_MODEL="/data/Teammember/huan.zhang/AI-accelerator/rtl_compile/model/model_zoo/onnx/onnx1/Alexnet.onnx"
DEFAULT_REFERENCE="/data/Teammember/huan.zhang/AI-accelerator/rtl_compile/model/tflite_uint8/onnx1/Alexnet_uint8.tflite"

usage() {
    cat <<'EOF'
用法：
  bash run_w4a8.sh [模型.onnx] [选项]

不指定模型时转换 Alexnet，并使用现有 Alexnet UINT8 模型检查结构。
默认输出：本项目 output/<模型名>_w4a8.tflite 及 .quantization.json。
默认权重 INT4、激活 INT8、权重按输出通道量化。

常用选项：
  -o, --output PATH          指定输出 TFLite
  --calibration PATH         FP32 校准 .npy/.npz 或 .npz 目录（ONNX 原始布局）
  --structure-reference PATH 指定用于结构对齐的 UINT8 TFLite
  --input-shape NAME=SHAPE  指定动态输入形状，如 images=1,3,320,320
  --overwrite               允许覆盖已有输出；默认不覆盖
  --keep-workdir NEW_DIR    保留转换日志和中间文件，目录必须尚不存在
  --dry-run                 只打印转换命令
  -h, --help                显示本帮助

其它选项原样传给 python -m onnx_edge_quantizer，覆盖对应默认值。
未提供 --calibration 时使用合成输入，只用于流程验证。
其他模型不会自动套用 Alexnet 的结构参考；需要时显式指定。
可通过 W4A8_PYTHON=/path/to/python 切换 Python 环境。

示例：
  bash run_w4a8.sh --overwrite
  bash run_w4a8.sh /path/to/model.onnx --calibration /path/to/data.npy
  bash run_w4a8.sh /path/to/model.onnx -o /path/to/model_w4a8.tflite \
    --structure-reference /path/to/model_uint8.tflite --overwrite
EOF
}

MODEL="$DEFAULT_MODEL"
if [[ $# -gt 0 && "$1" != -* ]]; then
    MODEL="$1"
    shift
fi

DRY_RUN=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --dry-run) DRY_RUN=1 ;;
        *) EXTRA_ARGS+=("$1") ;;
    esac
    shift
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    printf '找不到 Python：%s；请设置 W4A8_PYTHON。\n' "$PYTHON_BIN" >&2
    exit 2
fi
if [[ ! -f "$MODEL" ]]; then
    printf '找不到 ONNX 模型：%s\n' "$MODEL" >&2
    exit 2
fi

MODEL_FILE="${MODEL##*/}"
OUTPUT="$SCRIPT_DIR/output/${MODEL_FILE%.*}_w4a8.tflite"
COMMAND=("$PYTHON_BIN" -m onnx_edge_quantizer "$MODEL" -o "$OUTPUT"
         --mode native --weight-dtype int4 --activation-dtype int8
         --weight-granularity per-tensor)
if [[ "$MODEL" -ef "$DEFAULT_MODEL" ]]; then
    COMMAND+=(--structure-reference "$DEFAULT_REFERENCE")
fi
COMMAND+=("${EXTRA_ARGS[@]}")

# Keep relative model/data/output paths relative to the caller's directory,
# while importing this checkout even when invoked from another directory.
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
printf '执行命令：\n'
printf '%q ' "${COMMAND[@]}"
printf '\n'
if [[ "$DRY_RUN" -eq 1 ]]; then
    exit 0
fi
exec "${COMMAND[@]}"
