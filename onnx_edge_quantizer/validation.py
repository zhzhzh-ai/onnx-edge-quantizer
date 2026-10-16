"""Compare ONNX/FP32-TFLite/quantized outputs by explicit tensor names."""

import numpy as np
from .runtime import make_interpreter, map_details, run, _normalized


def metrics(reference, actual):
    reference, actual = np.asarray(reference), np.asarray(actual)
    if reference.shape != actual.shape:
        raise ValueError(f"Output shape/layout mismatch: {reference.shape} vs {actual.shape}")
    if not reference.size or not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError("Output arrays must be nonempty and finite")
    a, b = reference.astype(np.float64).ravel(), actual.astype(np.float64).ravel()
    error = a - b
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return {"mse": float(np.mean(error ** 2)), "max_abs_error": float(np.max(np.abs(error))),
            "cosine_similarity": float(np.dot(a, b) / denominator) if denominator else (1.0 if np.array_equal(a, b) else 0.0)}


def validate(onnx_path, float_path, quant_path, specs, samples, input_map, output_map, atol, rtol,
             output_layouts=None, input_layouts=None):
    import onnxruntime as ort
    original = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    fp = make_interpreter(float_path)
    q = make_interpreter(quant_path)
    fm = map_details(specs, fp.get_input_details(), input_map, input_layouts)
    qm = map_details(specs, q.get_input_details(), input_map, input_layouts)
    onnx_names = [d.name for d in original.get_outputs()]
    mappings = {}
    for name in onnx_names:
        requested = output_map.get(name, name)
        available = [d["name"] for d in fp.get_output_details()]
        matches = [v for v in available if v == requested]
        if not matches and name not in output_map:
            matches = [v for v in available if _normalized(v) == _normalized(requested)]
        if len(matches) != 1:
            raise ValueError(f"Cannot map ONNX output {name!r} to TFLite outputs {available}; pass --output-map ONNX_NAME=TFLITE_NAME")
        mappings[name] = matches[0]
    if len(set(mappings.values())) != len(mappings) or len(mappings) != len(fp.get_output_details()):
        raise ValueError("ONNX and TFLite outputs must have a one-to-one name mapping")
    if set(output_map) - set(onnx_names):
        raise ValueError("--output-map contains unknown ONNX output names")
    output_layouts = output_layouts or {}
    if set(output_layouts) - set(onnx_names):
        raise ValueError("Output layout metadata contains unknown ONNX output names")
    comparisons = []
    for index, sample in enumerate(samples):
        baseline = dict(zip(onnx_names, original.run(onnx_names, sample)))
        float_outputs, quant_outputs = run(fp, sample, fm), run(q, sample, qm)
        for name, tfl_name in mappings.items():
            if tfl_name not in quant_outputs:
                raise ValueError(f"Quantized output name changed: {tfl_name}")
            float_value, quant_value = float_outputs[tfl_name], quant_outputs[tfl_name]
            raw_shape = list(float_value.shape)
            if quant_value.shape != float_value.shape:
                raise ValueError(f"FP32/quantized output shape mismatch for {tfl_name!r}: "
                                 f"{float_value.shape} vs {quant_value.shape}")
            layout = output_layouts.get(name)
            if layout:
                if layout["tflite_name"] != tfl_name:
                    raise ValueError(f"Converted output layout name mismatch for {name!r}")
                permutation = layout["tflite_to_onnx_permutation"]
                if sorted(permutation) != list(range(float_value.ndim)):
                    raise ValueError(f"Invalid converted output permutation for {name!r}: {permutation}")
                float_value = np.transpose(float_value, permutation)
                quant_value = np.transpose(quant_value, permutation)
            float_metrics = metrics(baseline[name], float_value)
            if not np.allclose(baseline[name], float_value, atol=atol, rtol=rtol):
                raise ValueError(f"FP32 conversion parity failed for output {name!r}: {float_metrics}. Check layout/operator conversion or configure --float-atol/--float-rtol")
            comparisons.append({"sample": index, "output": name, "onnx_vs_float_tflite": float_metrics,
                                "onnx_vs_quantized": metrics(baseline[name], quant_value),
                                "float_vs_quantized": metrics(float_value, quant_value),
                                "output_shapes": {"onnx": list(baseline[name].shape), "tflite": raw_shape}})
    return comparisons
