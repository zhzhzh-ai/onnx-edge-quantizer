"""Command-line workflow; fail closed on unsupported native configurations."""

import argparse
from dataclasses import dataclass
import importlib.util
import json
import logging
import os
from pathlib import Path
import platform
import sys
import tempfile

from . import __version__
from .quantization import DTYPES, validate_native

LOG = logging.getLogger("onnx-edge-quantizer")


@dataclass(frozen=True)
class Config:
    weight_dtype: str = "int8"
    activation_dtype: str = "int8"
    weight_scheme: str = "symmetric"
    activation_scheme: str = "asymmetric"
    weight_granularity: str = "per-tensor"


def pairs(items, shape=False):
    result = {}
    for item in items or []:
        name, separator, value = item.partition("=")
        if not separator or not name or not value or name in result:
            raise ValueError(f"Expected unique NAME=VALUE, got {item!r}")
        result[name] = [int(d) for d in value.split(",")] if shape else value
    return result


def parser():
    p = argparse.ArgumentParser(description="FP32 ONNX -> min-max TFLite quantizer. Native mode validates LiteRT constraints; reference mode explicitly simulates all precision combinations in FP32.")
    p.add_argument("model", type=Path, nargs="?", help="Input FP32 ONNX model")
    p.add_argument("-o", "--output", type=Path, help="Output .tflite file")
    p.add_argument("--mode", choices=["native", "reference"], default="native")
    p.add_argument("--weight-dtype", choices=DTYPES, default="int8")
    p.add_argument("--activation-dtype", choices=DTYPES, default="int8")
    p.add_argument("--weight-scheme", choices=["symmetric", "asymmetric"], default="symmetric")
    p.add_argument("--activation-scheme", choices=["symmetric", "asymmetric"], default="asymmetric")
    p.add_argument("--weight-granularity", "--granularity", choices=["per-tensor", "per-channel"],
                   default="per-tensor", help="Weight quantization granularity; activations remain per-tensor")
    p.add_argument("--calibration", type=Path, help=".npy / named .npz / directory of .npz files in ONNX input layout")
    p.add_argument("--input-shape", action="append", metavar="NAME=1,3,224,224")
    p.add_argument("--input-map", action="append", metavar="ONNX_NAME=TFLITE_NAME")
    p.add_argument("--output-map", action="append", metavar="ONNX_NAME=TFLITE_NAME")
    p.add_argument("--synthetic-samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--random-min", type=float, default=-1.0)
    p.add_argument("--random-max", type=float, default=1.0)
    p.add_argument("--float-tflite", type=Path, help="Reuse an existing equivalent FP32 TFLite, bypassing onnx2tf; parity is still verified")
    p.add_argument("--validation-data", type=Path, help="Separate .npy/.npz validation inputs; otherwise use all calibration samples")
    p.add_argument("--float-atol", type=float, default=1e-4)
    p.add_argument("--float-rtol", type=float, default=1e-3)
    p.add_argument("--allow-float-ops", action="store_true", help="Permit native backend to leave unsupported operations in float; report lists these operations")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep-workdir", type=Path, help="Keep conversion logs and intermediate files in a NEW directory")
    p.add_argument("--doctor", action="store_true", help="Print dependency/platform diagnostics and exit")
    p.add_argument("--version", action="version", version=__version__)
    return p


def doctor():
    dependencies = {name: importlib.util.find_spec(name) is not None for name in
                    ["numpy", "onnx", "onnxruntime", "flatbuffers", "ai_edge_litert", "onnx2tf", "ai_edge_quantizer"]}
    print(json.dumps({"python": sys.version.split()[0], "platform": platform.system(),
                      "dependencies": dependencies,
                      "native_note": "Google AI Edge Quantizer officially supports Linux/macOS; current nightly is also tested here on Windows. onnx2tf 2.6.9 requires Python >=3.12"}, indent=2))
    return 0


def _execute(args, workdir):
    from .data import input_specs, load_samples
    from .converter import convert_onnx
    from .runtime import calibrate, inventory, check_native_precision
    from .backends import native_export, reference_export
    from .validation import validate
    model, specs = input_specs(args.model, pairs(args.input_shape, shape=True))
    samples = load_samples(args.calibration, specs, args.synthetic_samples, args.seed, args.random_min, args.random_max)
    if args.calibration is None:
        LOG.warning("Using %d synthetic sample(s); this verifies the workflow but is not representative accuracy calibration", len(samples))
    config = Config(args.weight_dtype, args.activation_dtype, args.weight_scheme, args.activation_scheme, args.weight_granularity)
    LOG.info("Converting ONNX to FP32 TFLite")
    float_path = args.float_tflite.resolve() if args.float_tflite else convert_onnx(model, specs, workdir)
    LOG.info("Running FP32 inference and collecting intermediate min/max ranges")
    ranges, mapping = calibrate(float_path, specs, samples, pairs(args.input_map))
    if not ranges:
        raise ValueError("No floating-point activations found for calibration")
    candidate = workdir / "candidate.tflite"
    LOG.info("Exporting %s model (weights=%s, activations=%s)", args.mode, args.weight_dtype, args.activation_dtype)
    if args.mode == "native":
        details = native_export(float_path, candidate, specs, samples, mapping, config)
    else:
        LOG.warning("Reference export uses FP32 storage/computation with explicit rounding and clipping; it provides no integer compression or integer acceleration")
        details = reference_export(float_path, candidate, ranges, config)
    model_inventory = inventory(candidate)
    if args.mode == "native":
        quantized = [t for t in model_inventory["tensors"] if t["scales"] and t["type"] in ("INT4", "INT8", "INT16")]
        if not quantized:
            raise ValueError("Backend produced no quantized tensors; refusing to publish a floating-point model as quantized")
        if model_inventory["float_compute_ops"] and not args.allow_float_ops:
            raise ValueError(f"Native export left float compute operations: {model_inventory['float_compute_ops']}. Use --allow-float-ops to explicitly accept a mixed model")
        details["precision_verification"] = check_native_precision(candidate, args.weight_dtype, args.activation_dtype, args.weight_granularity)
    validation_samples = load_samples(args.validation_data, specs) if args.validation_data else samples
    LOG.info("Checking ONNX/FP32 TFLite parity and running quantized model on %d sample(s)", len(validation_samples))
    comparisons = validate(args.model, float_path, candidate, specs, validation_samples, pairs(args.input_map),
                           pairs(args.output_map), args.float_atol, args.float_rtol)
    report = {"tool_version": __version__, "mode": args.mode, "algorithm": "min-max", "granularity": config.weight_granularity,
              "weight_granularity": config.weight_granularity, "activation_granularity": "per-tensor",
              "requested": config.__dict__, "inputs": specs, "synthetic_calibration": args.calibration is None,
              "calibration_samples": len(samples), "validation_samples": len(validation_samples),
              "seed": args.seed if args.calibration is None else None,
              "activation_ranges": {str(i): {"min": lo, "max": hi} for i, (lo, hi) in ranges.items()},
              "backend_details": details, "output_inventory": model_inventory, "validation": comparisons,
              "input_onnx": str(args.model.resolve()), "output_tflite": str(args.output.resolve()),
              "output_bytes": candidate.stat().st_size,
              "warnings": (["Synthetic data does not establish real task accuracy"] if args.calibration is None else [])
                           + (["Reference simulation: FP32 arithmetic/storage, no integer speed or size benefits"] if args.mode == "reference" else [])}
    report_path = args.output.with_suffix(".quantization.json")
    staged_report = workdir / "report.json"
    staged_report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    # Only publish after all inference and validation checks pass.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for source, destination in [(candidate, args.output), (staged_report, report_path)]:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False, suffix=".tmp") as stream:
            temp_path = Path(stream.name)
            stream.write(source.read_bytes())
        try:
            os.replace(temp_path, destination)
        finally:
            temp_path.unlink(missing_ok=True)
    LOG.info("Wrote %s", args.output)
    LOG.info("Wrote %s", report_path)
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.doctor:
        return doctor()
    try:
        if args.model is None or args.output is None:
            raise ValueError("MODEL and --output are required")
        if not args.model.is_file() or args.model.suffix.lower() != ".onnx":
            raise ValueError("Input must be an existing .onnx model")
        if args.output.suffix.lower() != ".tflite":
            raise ValueError("Output must have the .tflite extension")
        if args.model.resolve() == args.output.resolve() or (args.float_tflite and args.float_tflite.resolve() == args.output.resolve()):
            raise ValueError("Output must not replace a source model")
        for destination in [args.output, args.output.with_suffix(".quantization.json")]:
            if destination.exists() and not args.overwrite:
                raise ValueError(f"Output already exists: {destination}; use --overwrite")
        if not np_finite_nonnegative(args.float_atol, args.float_rtol):
            raise ValueError("Float comparison tolerances must be finite and nonnegative")
        if args.mode == "native":
            validate_native(args.weight_dtype, args.activation_dtype, args.weight_scheme == "symmetric", args.activation_scheme == "symmetric")
            if importlib.util.find_spec("ai_edge_quantizer") is None:
                raise ValueError("Native mode requires ai-edge-quantizer-nightly and its matching LiteRT nightly. See README.md for the tested installation order or use the Dockerfile")
        if args.keep_workdir:
            args.keep_workdir.mkdir(parents=True, exist_ok=False)
            return _execute(args, args.keep_workdir.resolve())
        with tempfile.TemporaryDirectory(prefix="onnx-edge-quant-") as directory:
            return _execute(args, Path(directory))
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        LOG.error("%s", error)
        return 2


def np_finite_nonnegative(*values):
    import math
    return all(math.isfinite(v) and v >= 0 for v in values)
