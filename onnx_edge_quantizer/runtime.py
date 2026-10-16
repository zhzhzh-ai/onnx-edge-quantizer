"""LiteRT schema, deterministic input mapping, calibration and execution."""

import re
import numpy as np


def modules():
    from ai_edge_litert import interpreter, schema_py_generated
    return interpreter, schema_py_generated


def read_model(path):
    _, schema = modules()
    # Object API arrays are zero-copy views; bytearray permits graph rewrites.
    return schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(bytearray(path.read_bytes()), 0))


def write_model(model, path):
    import flatbuffers
    builder = flatbuffers.Builder(1024)
    root = model.Pack(builder)
    builder.Finish(root, file_identifier=b"TFL3")
    path.write_bytes(bytes(builder.Output()))


def make_interpreter(path, preserve=False):
    module, _ = modules()
    instance = module.Interpreter(
        # LiteRT's native filename loader can fail on non-ASCII Windows paths.
        model_content=path.read_bytes(), num_threads=1,
        experimental_preserve_all_tensors=preserve,
        experimental_op_resolver_type=module.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    instance.allocate_tensors()
    return instance


def _normalized(name):
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    return re.sub(r"[^A-Za-z0-9]", "_", re.sub(r":\d+$", "", name))


def map_details(specs, details, explicit=None):
    explicit = explicit or {}
    result, used = {}, set()
    for spec in specs:
        requested = explicit.get(spec["name"], spec["name"])
        candidates = [d for d in details if d["name"] == requested]
        if not candidates and spec["name"] not in explicit:
            candidates = [d for d in details if _normalized(d["name"]) == _normalized(requested)]
        if len(candidates) != 1 or candidates[0]["index"] in used:
            raise ValueError(f"Cannot uniquely map ONNX input {spec['name']!r} to TFLite names {[d['name'] for d in details]}; use --input-map ONNX_NAME=TFLITE_NAME")
        detail = candidates[0]
        if tuple(detail["shape"]) != tuple(spec["shape"]):
            raise ValueError(f"Input layout/shape mismatch for {spec['name']}: ONNX {spec['shape']}, TFLite {detail['shape'].tolist()}; a layout-preserving conversion is required")
        result[spec["name"]] = detail
        used.add(detail["index"])
    if len(used) != len(details):
        raise ValueError("ONNX and TFLite input counts differ")
    if set(explicit) - set(result):
        raise ValueError("--input-map contains unknown ONNX input names")
    return result


def run(instance, sample, mapping):
    for name, detail in mapping.items():
        value = sample[name]
        if np.issubdtype(detail["dtype"], np.integer):
            scales = detail["quantization_parameters"]["scales"]
            zeros = detail["quantization_parameters"]["zero_points"]
            if len(scales) != 1 or scales[0] <= 0:
                raise ValueError(f"Invalid input quantization parameters for {name}")
            limits = np.iinfo(detail["dtype"])
            value = np.clip(np.rint(value / scales[0]) + zeros[0], limits.min, limits.max)
        instance.set_tensor(detail["index"], value.astype(detail["dtype"]))
    instance.invoke()
    outputs = {}
    for detail in instance.get_output_details():
        value = instance.get_tensor(detail["index"])
        if np.issubdtype(value.dtype, np.integer):
            qp = detail["quantization_parameters"]
            if len(qp["scales"]) == 1:
                value = (value.astype(np.float32) - qp["zero_points"][0]) * qp["scales"][0]
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite model output: {detail['name']}")
        outputs[detail["name"]] = value
    return outputs


def calibrate(path, specs, samples, input_map=None):
    instance = make_interpreter(path, preserve=True)
    mapping = map_details(specs, instance.get_input_details(), input_map)
    model = read_model(path)
    if len(model.subgraphs) != 1:
        raise ValueError("Calibration currently supports one TFLite subgraph; control-flow models need a specialized backend")
    _, schema = modules()
    graph = model.subgraphs[0]
    produced = {int(i) for op in graph.operators for i in op.outputs if i >= 0}
    targets = produced | set(map(int, graph.inputs))
    ranges = {}
    for sample in samples:
        run(instance, sample, mapping)
        for i in sorted(targets):
            tensor = graph.tensors[i]
            if tensor.type != schema.TensorType.FLOAT32:
                continue
            value = instance.get_tensor(i)
            if value.size == 0 or not np.isfinite(value).all():
                raise ValueError(f"Empty/non-finite intermediate activation: {tensor.name!r}")
            lo, hi = float(value.min()), float(value.max())
            old = ranges.get(i, (lo, hi))
            ranges[i] = (min(lo, old[0]), max(hi, old[1]))
    return ranges, mapping


def signature_samples(model, samples, mapping):
    if len(model.signatureDefs or []) != 1:
        raise ValueError("AI Edge Quantizer requires exactly one exported signature; convert with output_signaturedefs=True")
    signature = model.signatureDefs[0]
    reverse = {int(d["index"]): name for name, d in mapping.items()}
    signature_mapping = {}
    for tensor_map in signature.inputs:
        if int(tensor_map.tensorIndex) not in reverse:
            raise ValueError("Signature input does not match model inputs")
        name = tensor_map.name.decode() if isinstance(tensor_map.name, bytes) else tensor_map.name
        signature_mapping[name] = reverse[int(tensor_map.tensorIndex)]
    key = signature.signatureKey.decode() if isinstance(signature.signatureKey, bytes) else signature.signatureKey
    return {key: [{name: sample[source] for name, source in signature_mapping.items()} for sample in samples]}


def inventory(path):
    _, schema = modules()
    model = read_model(path)
    type_names = {value: key for key, value in vars(schema.TensorType).items() if isinstance(value, int)}
    op_names = {value: key for key, value in vars(schema.BuiltinOperator).items() if isinstance(value, int)}
    tensors, float_ops = [], []
    for gi, graph in enumerate(model.subgraphs):
        for ti, tensor in enumerate(graph.tensors):
            qp = tensor.quantization
            tensors.append({"subgraph": gi, "index": ti, "name": tensor.name.decode() if isinstance(tensor.name, bytes) else tensor.name,
                            "type": type_names.get(tensor.type, str(tensor.type)),
                            "quantized_dimension": int(qp.quantizedDimension) if qp is not None else None,
                            "scales": qp.scale.tolist() if qp is not None and qp.scale is not None else [],
                            "zero_points": qp.zeroPoint.tolist() if qp is not None and qp.zeroPoint is not None else []})
        for op in graph.operators:
            code = model.operatorCodes[op.opcodeIndex]
            builtin = max(code.builtinCode, code.deprecatedBuiltinCode)
            if builtin in (schema.BuiltinOperator.QUANTIZE, schema.BuiltinOperator.DEQUANTIZE):
                continue
            if any(i >= 0 and graph.tensors[i].type == schema.TensorType.FLOAT32 for i in list(op.inputs) + list(op.outputs)):
                float_ops.append({"subgraph": gi, "op": op_names.get(builtin, str(builtin)),
                                  "outputs": [int(i) for i in op.outputs]})
    return {"tensors": tensors, "float_compute_ops": float_ops}


def weight_channel_axis(builtin, position, rank, options=None):
    """Output-channel axis in TFLite's weight layout, not ONNX's input layout."""
    _, schema = modules()
    if rank < 1:
        raise ValueError("A weight channel axis requires a non-scalar weight")
    if builtin == schema.BuiltinOperator.DEPTHWISE_CONV_2D:
        axis = 3  # [1, H, W, output_channels]
    elif builtin == schema.BuiltinOperator.BATCH_MATMUL:
        if rank < 2:
            raise ValueError("BatchMatMul weights require rank >= 2")
        transposed = bool(getattr(options, "adjX" if position == 0 else "adjY", False))
        axis = (rank - 1 if transposed else rank - 2) if position == 0 else (rank - 2 if transposed else rank - 1)
    else:
        axis = 0  # Conv2D, FullyConnected, TransposeConv
    if axis >= rank:
        raise ValueError(f"Invalid output-channel axis {axis} for weight rank {rank}")
    return axis


def check_native_precision(path, weight_dtype, activation_dtype, weight_granularity="per-tensor"):
    """Check materialized tensor types, in addition to checking the requested recipe."""
    _, schema = modules()
    model = read_model(path)
    expected_weight = getattr(schema.TensorType, weight_dtype.upper())
    expected_activation = getattr(schema.TensorType, activation_dtype.upper())
    roles = {schema.BuiltinOperator.CONV_2D: ([1], [2]),
             schema.BuiltinOperator.DEPTHWISE_CONV_2D: ([1], [2]),
             schema.BuiltinOperator.FULLY_CONNECTED: ([1], [2]),
             schema.BuiltinOperator.BATCH_MATMUL: ([0, 1], []),
             schema.BuiltinOperator.TRANSPOSE_CONV: ([1], [3])}
    checked_weights, checked_activations = 0, 0
    for graph in model.subgraphs:
        weights, biases = {}, set()
        for op in graph.operators:
            code = model.operatorCodes[op.opcodeIndex]
            positions, bias_positions = roles.get(max(code.builtinCode, code.deprecatedBuiltinCode), ([], []))
            for pos in positions:
                if pos < len(op.inputs) and op.inputs[pos] >= 0:
                    i = int(op.inputs[pos])
                    tensor = graph.tensors[i]
                    data = model.buffers[tensor.buffer].data
                    if data is not None and len(data):
                        axis = weight_channel_axis(max(code.builtinCode, code.deprecatedBuiltinCode), pos, len(tensor.shape), op.builtinOptions)
                        if i in weights and weights[i] != axis and weight_granularity == "per-channel":
                            raise ValueError(f"Shared weight {tensor.name!r} has incompatible channel axes")
                        weights[i] = axis
            biases.update(int(op.inputs[pos]) for pos in bias_positions if pos < len(op.inputs) and op.inputs[pos] >= 0)
        for i, tensor in enumerate(graph.tensors):
            qp = tensor.quantization
            if qp is None or qp.scale is None or not len(qp.scale) or i in biases:
                continue
            if not np.isfinite(qp.scale).all() or np.any(qp.scale <= 0):
                raise ValueError(f"Invalid exported quantization scale on {tensor.name!r}")
            if tensor.type not in (schema.TensorType.INT4, schema.TensorType.INT8, schema.TensorType.INT16):
                continue
            expected = expected_weight if i in weights else expected_activation
            if tensor.type != expected:
                raise ValueError(f"Native backend changed requested {'weight' if i in weights else 'activation'} precision on {tensor.name!r}: actual type id {tensor.type}, expected {expected}")
            if i in weights:
                checked_weights += 1
                if qp.zeroPoint is not None and np.any(qp.zeroPoint != 0):
                    raise ValueError(f"Native weight {tensor.name!r} is not symmetric")
                axis = weights[i]
                expected_scales = int(tensor.shape[axis]) if weight_granularity == "per-channel" else 1
                if len(qp.scale) != expected_scales or (weight_granularity == "per-channel" and qp.quantizedDimension != axis):
                    raise ValueError(f"Native backend changed requested {weight_granularity} weight granularity on {tensor.name!r}: {len(qp.scale)} scales on axis {qp.quantizedDimension}; expected {expected_scales} scales on axis {axis}")
            else:
                checked_activations += 1
                if len(qp.scale) != 1:
                    raise ValueError(f"Native activation {tensor.name!r} must remain per-tensor")
    if not checked_activations:
        raise ValueError("Native backend produced no activations in the requested integer precision")
    return {"checked_weight_tensors": checked_weights, "checked_activation_tensors": checked_activations,
            "weight_granularity": weight_granularity, "activation_granularity": "per-tensor"}
