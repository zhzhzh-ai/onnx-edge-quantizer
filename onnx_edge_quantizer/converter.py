"""Run onnx2tf in an isolated process so backend exits cannot kill the CLI."""

import json
from pathlib import Path
import subprocess
import sys


def expose_channel_last_inputs(float_path, specs):
    """Absorb proven NC* -> N*C boundary adapters into the TFLite interface.

    Only a sole input consumer with an explicit transpose or an equivalent
    singleton-dimension reshape is eligible. Model computations and ONNX
    calibration input conventions stay unchanged.
    """
    import numpy as np
    from .runtime import modules, read_model, write_model

    model = read_model(float_path)
    if len(model.subgraphs) != 1:
        return {}
    _, schema = modules()
    graph = model.subgraphs[0]
    specs_by_name = {spec["name"]: spec for spec in specs}
    removed, layouts = set(), {}
    for input_index in list(map(int, graph.inputs)):
        tensor = graph.tensors[input_index]
        name = tensor.name.decode() if isinstance(tensor.name, bytes) else tensor.name
        spec = specs_by_name.get(name)
        if spec is None or len(spec["shape"]) not in (3, 4, 5):
            continue
        if input_index in graph.outputs or tensor.type != schema.TensorType.FLOAT32:
            continue
        consumers = [(index, op) for index, op in enumerate(graph.operators)
                     if input_index in op.inputs]
        if len(consumers) != 1:
            continue
        index, op = consumers[0]
        if len(op.outputs) != 1 or int(op.inputs[0]) != input_index:
            continue
        target_index = int(op.outputs[0])
        if target_index in graph.outputs or target_index in graph.inputs:
            continue
        target = graph.tensors[target_index]
        rank = len(spec["shape"])
        permutation = [0, *range(2, rank), 1]
        shape = list(map(int, tensor.shape))
        expected_shape = [shape[axis] for axis in permutation]
        if shape != spec["shape"] or list(target.shape) != expected_shape:
            continue
        if target.type != tensor.type:
            continue
        code = model.operatorCodes[op.opcodeIndex]
        builtin = max(code.builtinCode, code.deprecatedBuiltinCode)
        if builtin == schema.BuiltinOperator.TRANSPOSE:
            if len(op.inputs) != 2 or int(op.inputs[1]) < 0:
                continue
            perm_tensor = graph.tensors[int(op.inputs[1])]
            data = model.buffers[perm_tensor.buffer].data
            if perm_tensor.type != schema.TensorType.INT32 or data is None:
                continue
            if np.frombuffer(bytes(data), dtype="<i4").tolist() != permutation:
                continue
            adapter = "TRANSPOSE"
        elif builtin == schema.BuiltinOperator.RESHAPE:
            # A reshape is a layout adapter only when the permutation leaves
            # every non-singleton axis in its original order (e.g. C == 1).
            if ([axis for axis in permutation if shape[axis] != 1]
                    != [axis for axis in range(rank) if shape[axis] != 1]):
                continue
            adapter = "RESHAPE"
        else:
            continue

        graph.inputs = np.array([target_index if int(i) == input_index else int(i)
                                 for i in graph.inputs], dtype=np.int32)
        for signature in model.signatureDefs or []:
            if signature.subgraphIndex == 0:
                for entry in signature.inputs or []:
                    if entry.tensorIndex == input_index:
                        entry.tensorIndex = target_index
        target.name = tensor.name
        tensor.name = (name + "/unused_onnx_layout").encode()
        removed.add(index)
        layouts[name] = {"tflite_name": name, "onnx_shape": shape,
                         "tflite_shape": expected_shape,
                         "onnx_to_tflite_permutation": permutation,
                         "removed_adapter": adapter}
    if removed:
        graph.operators = [op for index, op in enumerate(graph.operators) if index not in removed]
        # AEQ reads every tensor during calibration, including unused ones.
        # Remove the old input and adapter constants and remap all references.
        used = set(map(int, graph.inputs)) | set(map(int, graph.outputs))
        for op in graph.operators:
            for refs in (op.inputs, op.outputs, op.intermediates):
                if refs is not None:
                    used.update(int(i) for i in refs if i >= 0)
        signatures = [s for s in model.signatureDefs or [] if s.subgraphIndex == 0]
        for signature in signatures:
            used.update(int(entry.tensorIndex) for entry in [*(signature.inputs or []),
                                                            *(signature.outputs or [])])
        remap = {old: new for new, old in enumerate(sorted(used))}

        def remap_indices(indices):
            return np.array([remap[int(i)] if i >= 0 else -1 for i in indices], dtype=np.int32)

        graph.tensors = [graph.tensors[old] for old in sorted(used)]
        graph.inputs, graph.outputs = remap_indices(graph.inputs), remap_indices(graph.outputs)
        for op in graph.operators:
            op.inputs, op.outputs = remap_indices(op.inputs), remap_indices(op.outputs)
            if op.intermediates is not None:
                op.intermediates = remap_indices(op.intermediates)
        for signature in signatures:
            for entry in [*(signature.inputs or []), *(signature.outputs or [])]:
                entry.tensorIndex = remap[int(entry.tensorIndex)]
        write_model(model, float_path)
    return layouts


def converted_output_layouts(model, float_path):
    """Read named output layout changes recorded by the direct converter.

    Do not infer a permutation from matching dimensions or output values.
    The exported TFLite keeps its native layout; this metadata is for parity
    checks against the original ONNX interface only.
    """
    report_path = float_path.with_name(
        float_path.stem.removesuffix("_float32") + "_tensor_correspondence_report.json"
    )
    if not report_path.is_file():
        return {}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1:
        raise ValueError("Unsupported onnx2tf tensor correspondence report version")
    outputs = {value.name: len(value.type.tensor_type.shape.dim) for value in model.graph.output}
    layouts = {}
    for record in report.get("records", []):
        name = record.get("onnx_output_name")
        rank = outputs.get(name, 0)
        # The converter can lose the channel-last hint when it renames a
        # terminal Conv/Pool tensor to the public output name. These matched
        # operator contracts still establish NCHW (ONNX) vs NHWC (TFLite).
        operator_layout_change = rank == 4 and (
            record.get("onnx_op_type"), record.get("producer_op_type")
        ) in {("Conv", "CONV_2D"), ("Conv", "DEPTHWISE_CONV_2D"),
              ("ConvTranspose", "TRANSPOSE_CONV"),
              ("AveragePool", "AVERAGE_POOL_2D"), ("MaxPool", "MAX_POOL_2D")}
        if (rank not in (3, 4, 5) or not record.get("is_graph_output")
                or not record.get("exists_in_final_model")
                or not (record.get("assume_channel_last_layout") is True or operator_layout_change)):
            continue
        layout = {
            "tflite_name": record["resolved_tflite_tensor_name"],
            "tflite_to_onnx_permutation": [0, rank - 1, *range(1, rank - 1)],
            "source": "onnx2tf_tensor_correspondence_report",
            "evidence": "operator_layout_contract" if operator_layout_change else "channel_last_metadata",
        }
        if name in layouts and layouts[name] != layout:
            raise ValueError(f"Conflicting converted output layouts for {name!r}")
        layouts[name] = layout
    return layouts


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
