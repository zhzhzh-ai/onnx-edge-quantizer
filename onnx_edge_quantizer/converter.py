"""Run onnx2tf in an isolated process so backend exits cannot kill the CLI."""

import json
from pathlib import Path
import subprocess
import sys


def convert_onnx(model, specs, workdir):
    import onnx
    # Work on a copy loaded by the caller. External weights are embedded here.
    inputs = {value.name: value for value in model.graph.input}
    for spec in specs:
        dims = inputs[spec["name"]].type.tensor_type.shape.dim
        for dim, size in zip(dims, spec["shape"]):
            dim.ClearField("dim_param")
            dim.dim_value = size
    fixed_path = workdir / "source.onnx"
    onnx.save_model(model, str(fixed_path))
    options = {"input_onnx_file_path": str(fixed_path), "output_folder_path": str(workdir / "converted"),
               "output_signaturedefs": True, "copy_onnx_input_output_names_to_tflite": True,
               "keep_shape_absolutely_input_names": [s["name"] for s in specs],
               "not_use_onnxsim": True, "not_use_opname_auto_generate": True,
               "verbosity": "error", "tflite_backend": "flatbuffer_direct"}
    config_path = workdir / "converter.json"
    config_path.write_text(json.dumps(options), encoding="utf-8")
    log = workdir / "conversion.log"
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.run([sys.executable, "-m", "onnx_edge_quantizer.converter", str(config_path)],
                                 stdout=output, stderr=subprocess.STDOUT, check=False)
    if process.returncode:
        tail = log.read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeError(f"ONNX -> FP32 TFLite failed (exit {process.returncode}).\n{tail}")
    candidates = sorted((workdir / "converted").glob("*_float32.tflite"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one float32 TFLite output; found {candidates}. See {log}")
    return candidates[0]


def _worker(path):
    import inspect
    # The package-level lazy wrapper exposes only (*args, **kwargs).
    from onnx2tf.onnx2tf import convert
    options = json.loads(Path(path).read_text(encoding="utf-8"))
    supported = inspect.signature(convert).parameters
    if "tflite_backend" not in supported:
        raise RuntimeError("This tool requires onnx2tf>=2.6.9 and Python>=3.12 (flatbuffer_direct backend)")
    convert(**options)


if __name__ == "__main__":
    _worker(sys.argv[1])
