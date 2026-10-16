"""Native Google AI Edge Quantizer and explicitly labeled float reference export."""

from copy import deepcopy
import numpy as np
from .quantization import minmax_params, reconstruct_weight
from .runtime import modules, read_model, write_model, signature_samples, weight_channel_axis


def native_export(source, destination, specs, samples, mapping, config):
    from ai_edge_quantizer import quantizer
    recipe = [{"regex": ".*", "operation": "*", "algorithm_key": "min_max_uniform_quantize",
               "op_config": {
                   "weight_tensor_config": {"num_bits": int(config.weight_dtype.removeprefix("int")),
                                            "symmetric": True, "granularity": "CHANNELWISE" if config.weight_granularity == "per-channel" else "TENSORWISE", "dtype": "INT"},
                   "activation_tensor_config": {"num_bits": int(config.activation_dtype.removeprefix("int")),
                                                "symmetric": config.activation_scheme == "symmetric",
                                                "granularity": "TENSORWISE", "dtype": "INT"},
                   "compute_precision": "INTEGER", "explicit_dequantize": False, "skip_checks": False,
                   "min_weight_elements": 0}}]
    # Passing a filename makes AEQ retain an mmap, which can lock the temporary
    # FP32 model during cleanup on Windows. An owned buffer has no file handle.
    qt = quantizer.Quantizer(bytearray(source.read_bytes()))
    qt.load_quantization_recipe(recipe)
    data = signature_samples(read_model(source), samples, mapping)
    calibration = qt.calibrate(data, num_threads=1)
    qt.quantize(calibration_result=calibration, enable_progress_report=False).export_model(str(destination))
    return {"backend": "google-ai-edge-quantizer", "recipe": recipe}


def reference_export(source, destination, ranges, config):
    """Insert DIV/ROUND/ADD/MAX/MIN/SUB/MUL with fixed min-max parameters.

    Every float runtime edge is rounded/clipped; weight constants are replaced by
    reconstructed FP32 values. This is numerical simulation, NOT integer storage
    or integer arithmetic. Biases and shape/index constants are preserved.
    """
    _, schema = modules()
    model = read_model(source)
    graph = model.subgraphs[0]
    weights = {}
    weight_inputs = {schema.BuiltinOperator.CONV_2D: [1], schema.BuiltinOperator.DEPTHWISE_CONV_2D: [1],
                     schema.BuiltinOperator.FULLY_CONNECTED: [1], schema.BuiltinOperator.BATCH_MATMUL: [0, 1],
                     schema.BuiltinOperator.TRANSPOSE_CONV: [1]}
    constant_indices = set(range(len(graph.tensors))) - set(ranges)
    for op in graph.operators:
        code = model.operatorCodes[op.opcodeIndex]
        for pos in weight_inputs.get(max(code.builtinCode, code.deprecatedBuiltinCode), []):
            if pos >= len(op.inputs):
                continue
            i = int(op.inputs[pos])
            if i < 0 or i not in constant_indices:
                continue
            tensor = graph.tensors[i]
            axis = weight_channel_axis(max(code.builtinCode, code.deprecatedBuiltinCode), pos, len(tensor.shape), op.builtinOptions)
            if i in weights:
                if config.weight_granularity == "per-channel" and weights[i]["axis"] != axis:
                    raise ValueError(f"Shared weight {tensor.name!r} has incompatible channel axes")
                continue
            buffer = model.buffers[tensor.buffer]
            if tensor.type != schema.TensorType.FLOAT32 or buffer.data is None or not len(buffer.data):
                continue
            values = np.frombuffer(bytes(buffer.data), dtype="<f4").reshape(tensor.shape)
            restored, params = reconstruct_weight(values, config.weight_dtype, config.weight_scheme == "symmetric",
                                                  config.weight_granularity, axis)
            # Use a private buffer: different constants may share the original.
            new_buffer = schema.BufferT()
            new_buffer.data = np.frombuffer(restored.astype("<f4").tobytes(), dtype=np.uint8)
            model.buffers.append(new_buffer)
            tensor.buffer = len(model.buffers) - 1
            weights[i] = params

    codes = {}
    def opcode(builtin):
        if builtin not in codes:
            entry = schema.OperatorCodeT()
            entry.builtinCode = builtin
            entry.deprecatedBuiltinCode = min(builtin, 127)
            entry.version = 1
            model.operatorCodes.append(entry)
            codes[builtin] = len(model.operatorCodes) - 1
        return codes[builtin]

    def clone(index, name):
        tensor = deepcopy(graph.tensors[index])
        tensor.name = name.encode()
        tensor.buffer = 0
        tensor.quantization = None
        graph.tensors.append(tensor)
        return len(graph.tensors) - 1

    def scalar(value, name):
        buffer = schema.BufferT()
        buffer.data = np.frombuffer(np.float32(value).tobytes(), dtype=np.uint8)
        model.buffers.append(buffer)
        tensor = schema.TensorT()
        tensor.name, tensor.shape, tensor.type, tensor.buffer = name.encode(), np.array([], dtype=np.int32), schema.TensorType.FLOAT32, len(model.buffers) - 1
        graph.tensors.append(tensor)
        return len(graph.tensors) - 1

    def op(builtin, inputs, output):
        result = schema.OperatorT()
        result.opcodeIndex = opcode(builtin)
        result.inputs, result.outputs = np.array(inputs, dtype=np.int32), np.array([output], dtype=np.int32)
        options = {schema.BuiltinOperator.DIV: (schema.BuiltinOptions.DivOptions, schema.DivOptionsT),
                   schema.BuiltinOperator.ADD: (schema.BuiltinOptions.AddOptions, schema.AddOptionsT),
                   schema.BuiltinOperator.SUB: (schema.BuiltinOptions.SubOptions, schema.SubOptionsT),
                   schema.BuiltinOperator.MUL: (schema.BuiltinOptions.MulOptions, schema.MulOptionsT),
                   schema.BuiltinOperator.MAXIMUM: (schema.BuiltinOptions.MaximumMinimumOptions, schema.MaximumMinimumOptionsT),
                   schema.BuiltinOperator.MINIMUM: (schema.BuiltinOptions.MaximumMinimumOptions, schema.MaximumMinimumOptionsT)}
        if builtin in options:
            result.builtinOptionsType, klass = options[builtin]
            result.builtinOptions = klass()
        return result

    def qdq(index, raw, target, params):
        prefix = f"edge_quant/{index}"
        scale = scalar(params.scale, prefix + "/scale")
        zero = scalar(params.zero_point, prefix + "/zero")
        low, high = scalar(params.qmin, prefix + "/min"), scalar(params.qmax, prefix + "/max")
        steps = [(schema.BuiltinOperator.DIV, scale), (schema.BuiltinOperator.ROUND, None),
                 (schema.BuiltinOperator.ADD, zero), (schema.BuiltinOperator.MAXIMUM, low),
                 (schema.BuiltinOperator.MINIMUM, high), (schema.BuiltinOperator.SUB, zero),
                 (schema.BuiltinOperator.MUL, scale)]
        operators, current = [], raw
        for step, (builtin, arg) in enumerate(steps):
            output = target if step == len(steps) - 1 else clone(index, f"{prefix}/{step}")
            operators.append(op(builtin, [current] if arg is None else [current, arg], output))
            current = output
        return operators

    original_ops = list(graph.operators)
    original_names = {i: graph.tensors[i].name for i in ranges}
    params = {i: minmax_params(*extent, config.activation_dtype, config.activation_scheme == "symmetric") for i, extent in ranges.items()}
    input_remap, operators = {}, []
    for i in map(int, graph.inputs):
        if i in params:
            target = clone(i, f"edge_quant/input/{i}")
            input_remap[i] = target
            operators.extend(qdq(i, i, target, params[i]))
    for original in original_ops:
        original.inputs = np.array([input_remap.get(int(i), int(i)) for i in original.inputs], dtype=np.int32)
        following = []
        for position, index in enumerate(original.outputs):
            i = int(index)
            if i in params:
                raw = clone(i, f"edge_quant/raw/{i}")
                original.outputs[position] = raw
                following.extend(qdq(i, raw, i, params[i]))
        operators.append(original)
        operators.extend(following)
    graph.operators = operators
    # A graph that directly returns an input must return its simulated value.
    graph.outputs = np.array([input_remap.get(int(i), int(i)) for i in graph.outputs], dtype=np.int32)
    for signature in model.signatureDefs or []:
        for item in signature.outputs:
            item.tensorIndex = input_remap.get(int(item.tensorIndex), int(item.tensorIndex))
    model.description = b"REFERENCE MIN-MAX SIMULATION: FP32 arithmetic and storage; not native integer quantization"
    write_model(model, destination)
    return {"backend": "float-reference-simulation", "integer_arithmetic": False, "integer_weight_storage": False,
            "weights": {str(i): p for i, p in weights.items()},
            "activations": {str(i): {"name": original_names[i].decode() if isinstance(original_names[i], bytes) else original_names[i], **p.to_dict()} for i, p in params.items()}}
