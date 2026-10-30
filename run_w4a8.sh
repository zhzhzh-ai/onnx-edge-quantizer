#!/usr/bin/env bash
# ONNX -> native W4A8 TFLite. Extra arguments are passed to the Python CLI.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${W4A8_PYTHON:-/data/miniforge3/envs/onnx-edge-quantizer/bin/python}"
DEFAULT_MODEL="/data/Teammember/huan.zhang/AI-accelerator/rtl_compile/model/model_zoo/onnx/onnx1/Alexnet.onnx"
DEFAULT_REFERENCE="/data/Teammember/huan.zhang/AI-accelerator/rtl_compile/model/tflite_uint8/onnx1/Alexnet_uint8.tflite"

# A canceled batch can leave queued child launches. Honor a marker in the
# explicitly supplied work directory's ancestors before starting conversion.
W4A8_ARGS=("$@")
for ((W4A8_ARG_INDEX=0; W4A8_ARG_INDEX+1<${#W4A8_ARGS[@]}; W4A8_ARG_INDEX++)); do
    if [[ "${W4A8_ARGS[W4A8_ARG_INDEX]}" == "--keep-workdir" || "${W4A8_ARGS[W4A8_ARG_INDEX]}" == "--work-dir" ]]; then
        W4A8_CHECK_DIR="${W4A8_ARGS[W4A8_ARG_INDEX+1]}"
        while [[ "$W4A8_CHECK_DIR" != "/" && "$W4A8_CHECK_DIR" != "." && -n "$W4A8_CHECK_DIR" ]]; do
            if [[ -f "$W4A8_CHECK_DIR/.w4a8-cancel" ]]; then
                printf '批量任务已取消：%s\n' "$W4A8_CHECK_DIR" >&2
                exit 130
            fi
            W4A8_CHECK_DIR="$(dirname -- "$W4A8_CHECK_DIR")"
        done
    fi
done

usage() {
    cat <<'EOF'
用法：
  bash run_w4a8.sh [模型.onnx] [选项]
  bash run_w4a8.sh --batch --onnx-root DIR [--onnx-root DIR ...] --output-dir DIR [批量选项]
  bash run_w4a8.sh --batch --validated-summary REPORT --reference-root DIR --output-dir DIR --work-dir DIR

不指定模型时转换 Alexnet，并使用现有 Alexnet UINT8 模型检查结构。
默认输出：本项目 output/<模型名>_w4a8.tflite 及 .quantization.json。
单模型默认权重 INT4、激活 INT8、权重逐张量量化。

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

批量选项：
  --validated-summary FILE  仅部署报告中的配对模型，复用已验证产物，不重复转换
  --specialize-to-reference  配合上述报告，为尺寸不一致的输入创建基准尺寸副本
  --reference-root DIR       W8A8 基准目录，可重复；按模型名及目录匹配
  --work-dir DIR             转换日志、中间文件和修正基准目录
  --workers N --timeout N    并发数与每个模型超时秒数
  --reference-padding-policy baseline  显式采用基准池化 padding 语义
  --integerize-reference-islands       为混合精度基准创建全 INT8 副本
  --resume                  复用源模型及产物 SHA256 均未变化的成功结果
  --fallback-without-reference  结构不匹配时尝试独立的完整数值/推理验证
批量模式保留来源目录层级，权重默认 per-channel；有基准时检查结构，
无基准时执行完整转换/推理验证。失败及跳过项写入 export_summary.json。

示例：
  bash run_w4a8.sh --overwrite
  bash run_w4a8.sh /path/to/model.onnx --calibration /path/to/data.npy
  bash run_w4a8.sh /path/to/model.onnx -o /path/to/model_w4a8.tflite \
    --structure-reference /path/to/model_uint8.tflite --overwrite
EOF
}

if [[ "${1:-}" == "--batch" ]]; then
    shift
    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        printf '找不到 Python：%s；请设置 W4A8_PYTHON。\n' "$PYTHON_BIN" >&2
        exit 2
    fi
    export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
    for W4A8_BATCH_ARG in "$@"; do
        if [[ "$W4A8_BATCH_ARG" == "--validated-summary" ]]; then
            exec "$PYTHON_BIN" "$SCRIPT_DIR/scripts/deploy_validated_w4a8.py" "$@"
        fi
    done
    exec "$PYTHON_BIN" "$SCRIPT_DIR/scripts/export_all_w4a8.py" "$@"
fi

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
