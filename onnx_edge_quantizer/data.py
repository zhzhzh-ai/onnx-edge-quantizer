"""Load calibration samples in ONNX input names and layout, without guessing axes."""

from pathlib import Path
import numpy as np


def input_specs(model_path, shapes):
    import onnx
    from onnx import helper
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)
    initializers = {x.name for x in model.graph.initializer}
    specs = []
    for value in model.graph.input:
        if value.name in initializers:
            continue
        tensor = value.type.tensor_type
        dtype = np.dtype(helper.tensor_dtype_to_np_dtype(tensor.elem_type))
        if dtype != np.dtype("float32"):
            raise ValueError(f"Input {value.name!r} is {dtype}; this tool currently requires FP32 tensor inputs")
        dims = [int(d.dim_value) if d.HasField("dim_value") else None for d in tensor.shape.dim]
        shape = list(shapes.get(value.name, dims))
        if len(shape) != len(dims) or any(d is None or d <= 0 for d in shape):
            raise ValueError(f"Input {value.name!r} has dynamic/invalid shape {dims}; pass --input-shape '{value.name}=1,3,224,224'")
        if any(d is not None and d != a for d, a in zip(dims, shape)):
            raise ValueError(f"Shape override {shape} contradicts fixed dimensions {dims} for {value.name}")
        specs.append({"name": value.name, "shape": shape, "dtype": "float32"})
    unknown = set(shapes) - {s["name"] for s in specs}
    if unknown:
        raise ValueError(f"Unknown --input-shape names: {sorted(unknown)}")
    if not specs:
        raise ValueError("Model has no runtime tensor inputs")
    return model, specs


def _sample_arrays(arrays, specs):
    expected = {s["name"] for s in specs}
    if set(arrays) != expected:
        raise ValueError(f"Calibration keys must be {sorted(expected)}, got {sorted(arrays)}")
    counts = set()
    for spec in specs:
        a = arrays[spec["name"]]
        shape = tuple(spec["shape"])
        if a.dtype.kind != "f" or not np.isfinite(a).all():
            raise ValueError(f"Calibration {spec['name']} must contain finite floating point values")
        if a.shape == shape:
            counts.add(1)
        elif a.ndim == len(shape) + 1 and a.shape[1:] == shape:
            counts.add(a.shape[0])
        else:
            raise ValueError(f"Calibration {spec['name']} shape {a.shape}: expected {shape} or (samples, {shape})")
    if len(counts) != 1 or next(iter(counts)) < 1:
        raise ValueError("All model inputs must have the same nonzero calibration sample count")
    count = counts.pop()
    return [{s["name"]: np.ascontiguousarray(
        arrays[s["name"]] if arrays[s["name"]].shape == tuple(s["shape"]) else arrays[s["name"]][i],
        dtype=np.float32) for s in specs} for i in range(count)]


def load_samples(path, specs, count=1, seed=0, low=-1.0, high=1.0):
    if path is None:
        if count < 1 or not np.isfinite([low, high]).all() or low >= high:
            raise ValueError("Synthetic count must be positive and finite low < high")
        rng = np.random.default_rng(seed)
        return [{s["name"]: rng.uniform(low, high, s["shape"]).astype(np.float32) for s in specs} for _ in range(count)]
    path = Path(path)
    paths = sorted(path.glob("*.npz")) if path.is_dir() else [path]
    if not paths:
        raise ValueError(f"No .npz calibration files in {path}")
    samples = []
    for file in paths:
        if file.suffix.lower() == ".npy":
            if len(specs) != 1:
                raise ValueError(".npy calibration supports one input; use named .npz for multiple inputs")
            arrays = {specs[0]["name"]: np.load(file, allow_pickle=False)}
        elif file.suffix.lower() == ".npz":
            with np.load(file, allow_pickle=False) as data:
                arrays = {k: data[k] for k in data.files}
        else:
            raise ValueError("Calibration must be .npy, .npz, or a directory of .npz files")
        samples.extend(_sample_arrays(arrays, specs))
    return samples
