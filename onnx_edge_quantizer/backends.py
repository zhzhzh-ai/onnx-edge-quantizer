"""Native Google AI Edge Quantizer and explicitly labeled float reference export."""

from copy import deepcopy
from dataclasses import replace
import numpy as np
from .quantization import minmax_params, reconstruct_weight
from .runtime import modules, read_model, write_model, signature_samples, weight_channel_axis


def native_export(source, destination, specs, samples, mapping, config, *, structure_reference=None):
    if config.weight_dtype == 'int4' and config.activation_dtype == 'int8':
        return static_w4a8_export(source, destination, specs, samples, mapping, config,
                                  structure_reference=structure_reference)
    return _aeq_export(source, destination, specs, samples, mapping, config)


def _aeq_export(source, destination, specs, samples, mapping, config):
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
    calibration_mode = 'preserve_all_tensors'
    try:
        try:
            calibration = qt.calibrate(data, num_threads=1)
        except ValueError as error:
            if 'Tensor data is null' not in str(error):
                raise
            # XNNPACK may elide a tensor despite preserve_all_tensors. AEQ's
            # public profiler mode disables fusion and collects min/max in C++.
            from ai_edge_quantizer.calibrator import CalibrationMode
            calibration = qt.calibrate(data, num_threads=1,
                                       mode=CalibrationMode.CALIBRATION_PROFILER_BASED)
            calibration_mode = 'profiler_based_after_null_tensor'
        from .aeq_compat import scalar_quantization_ranks
        with scalar_quantization_ranks() as rank_fixes:
            qt.quantize(calibration_result=calibration, enable_progress_report=False).export_model(str(destination))
    except (KeyError, IndexError) as error:
        raise RuntimeError(f"AEQ calibration/quantization failed ({type(error).__name__}): {error}") from error
    return {"backend": "google-ai-edge-quantizer", "recipe": recipe,
            "calibration_mode": calibration_mode, **rank_fixes}


def static_w4a8_export(source, destination, specs, samples, mapping, config, *, structure_reference=None):
    """Calibrate an integer graph, then quantize original FP32 weights to W4.

    AEQ's W4 recipe can silently leave DepthwiseConv/FC in float or use dynamic
    blockwise FC. Its W8 static graph provides A8 calibration and connectivity;
    final W4 weights/biases are computed directly from the original FP32 data.
    No W8 code values are truncated and no W8 scales are reused for weights.
    """
    details = _aeq_export(source, destination, specs, samples, mapping,
                          replace(config, weight_dtype='int8'))
    from .quantized_optimizations import satisfy_quantization_constraints, restore_integer_leaky_relu
    from .layout_normalization import coalesce_slice_partitions
    concat_rewrites = restore_integer_leaky_relu(destination)
    concat_rewrites.extend(coalesce_slice_partitions(destination, structure_reference))
    concat_rewrites.extend(satisfy_quantization_constraints(
        destination, symmetric=config.activation_scheme == 'symmetric'))
    _, s = modules()
    original, model = read_model(source), read_model(destination)
    if len(original.subgraphs) != 1 or len(model.subgraphs) != 1:
        raise ValueError('Static W4A8 requires a single subgraph')
    float_graph, graph = original.subgraphs[0], model.subgraphs[0]
    by_name = {}
    for tensor in float_graph.tensors:
        by_name.setdefault(tensor.name, []).append(tensor)

    def float_constant(tensor):
        candidates = by_name.get(tensor.name, [])
        if len(candidates) != 1:
            raise ValueError(f'Cannot uniquely recover original FP32 tensor {tensor.name!r}')
        source_tensor = candidates[0]
        buffer = original.buffers[source_tensor.buffer].data
        if (source_tensor.type != s.TensorType.FLOAT32 or buffer is None
                or list(source_tensor.shape) != list(tensor.shape)):
            raise ValueError(f'Static W4A8 requires an original FP32 constant for {tensor.name!r}')
        values = np.frombuffer(bytes(buffer), dtype='<f4').reshape(tensor.shape)
        if not values.size or not np.isfinite(values).all():
            raise ValueError(f'Invalid FP32 weights/biases on {tensor.name!r}')
        return values

    def store(tensor, values):
        buf = s.BufferT()
        buf.data = np.frombuffer(np.ascontiguousarray(values).tobytes(), dtype=np.uint8)
        model.buffers.append(buf)
        tensor.buffer = len(model.buffers) - 1

    def affine(tensor, scales, axis):
        tensor.quantization = s.QuantizationParametersT()
        tensor.quantization.scale = np.asarray(scales, dtype=np.float32).reshape(-1)
        tensor.quantization.zeroPoint = np.zeros(len(tensor.quantization.scale), dtype=np.int64)
        tensor.quantization.quantizedDimension = axis

    weights, converted, bias_encodings = {}, [], {}
    supported = {s.BuiltinOperator.CONV_2D: (0, 7),
                 s.BuiltinOperator.DEPTHWISE_CONV_2D: (3, 7),
                 s.BuiltinOperator.FULLY_CONNECTED: (0, 13)}
    minimum_scales = {}
    for op in graph.operators:
        code = model.operatorCodes[op.opcodeIndex]
        if max(code.builtinCode, code.deprecatedBuiltinCode) not in supported:
            continue
        if len(op.inputs) < 3 or op.inputs[2] < 0:
            continue
        activation = graph.tensors[int(op.inputs[0])]
        qp = activation.quantization
        if (qp is None or qp.scale is None or len(qp.scale) != 1
                or not np.isfinite(qp.scale[0]) or qp.scale[0] <= 0):
            raise ValueError('Static W4A8 requires positive scalar activation scales')
        bias_values = float_constant(graph.tensors[int(op.inputs[2])]).astype(np.float64)
        # Reserve half the int32 range for the dot product. Calibrating a
        # nearly constant edge can otherwise make even an ordinary bias
        # overflow. Enlarge weight scale rather than wrap/clamp bias codes.
        lower = np.abs(bias_values) / (float(1 << 30) * float(qp.scale[0]))
        if config.weight_granularity == 'per-tensor':
            lower = np.array([lower.max()])
        index = int(op.inputs[1])
        minimum_scales[index] = np.maximum(minimum_scales.get(index, 0), lower)
    for op in graph.operators:
        code = model.operatorCodes[op.opcodeIndex]
        builtin = max(code.builtinCode, code.deprecatedBuiltinCode)
        if builtin not in supported:
            if builtin in (s.BuiltinOperator.BATCH_MATMUL, s.BuiltinOperator.TRANSPOSE_CONV):
                raise ValueError('Static W4A8 supports Conv2D, DepthwiseConv2D and FullyConnected weights; unsupported matrix/deconvolution operation')
            continue
        input_tensor = graph.tensors[int(op.inputs[0])]
        output_tensor = graph.tensors[int(op.outputs[0])]
        if input_tensor.type != s.TensorType.INT8 or output_tensor.type != s.TensorType.INT8:
            raise ValueError('Static W4A8 requires INT8 input/output on every weighted operation')
        axis, version = supported[builtin]
        index = int(op.inputs[1])
        tensor = graph.tensors[index]
        if tensor.sparsity is not None or tensor.isVariable:
            raise ValueError('Static W4A8 requires dense constant weights')
        if index in weights:
            saved_axis, scales = weights[index]
            if saved_axis != axis:
                raise ValueError('Shared W4 weights have incompatible channel axes')
        else:
            values = float_constant(tensor)
            extent = np.max(np.abs(values), axis=tuple(i for i in range(values.ndim) if i != axis)) if config.weight_granularity == 'per-channel' else np.array([np.max(np.abs(values))])
            scales = np.asarray(np.maximum(np.where(extent == 0, 1.0, extent / 7.0),
                                           minimum_scales.get(index, 0)), dtype=np.float32).reshape(-1)
            if not np.isfinite(scales).all() or np.any(scales <= 0):
                raise ValueError('INT4 weight scales are outside finite float32 range')
            broadcast = [1] * values.ndim
            broadcast[axis] = len(scales)
            codes = np.clip(np.rint(values / scales.reshape(broadcast)), -7, 7).astype(np.int8).reshape(-1)
            packed = np.zeros((codes.size + 1) // 2, dtype=np.uint8)
            packed[:] = codes[::2].view(np.uint8) & 15
            packed[:codes[1::2].size] |= (codes[1::2].view(np.uint8) & 15) << 4
            store(tensor, packed)
            tensor.type = s.TensorType.INT4
            affine(tensor, scales, axis if config.weight_granularity == 'per-channel' else 0)
            weights[index] = (axis, scales)
            converted.append({'name': tensor.name.decode(), 'axis': axis,
                              'scales': scales.tolist(), 'elements': int(codes.size),
                              'packed_bytes': int(packed.size)})
        qp = input_tensor.quantization
        if qp is None or qp.scale is None or len(qp.scale) != 1:
            raise ValueError('Static W4A8 requires scalar input activation scale')
        if len(op.inputs) > 2 and op.inputs[2] >= 0:
            bias_index = int(op.inputs[2])
            bias = graph.tensors[bias_index]
            bias_scales = (np.float64(qp.scale[0]) * scales.astype(np.float64)).astype(np.float32)
            if not np.isfinite(bias_scales).all() or np.any(bias_scales <= 0):
                raise ValueError('INT4 accumulator scales are outside finite positive float32 range')
            if bias_index in bias_encodings:
                if not np.array_equal(bias_encodings[bias_index], bias_scales):
                    raise ValueError('Shared W4 bias has incompatible accumulator scales')
            else:
                values = float_constant(bias)
                rounded = np.rint(values.astype(np.float64) / bias_scales.astype(np.float64))
                if np.any(rounded < -(1 << 31)) or np.any(rounded > (1 << 31) - 1):
                    raise ValueError('INT4 accumulator bias overflows INT32')
                store(bias, rounded.astype('<i4'))
                bias.type = s.TensorType.INT32
                affine(bias, bias_scales, 0)
                bias_encodings[bias_index] = bias_scales
        code.version = max(code.version, version)
    if not converted:
        raise ValueError('No supported constant weights were converted to INT4')
    # Discard old W8 weight buffers so the file actually gains compression.
    used = {0} | {int(t.buffer) for g in model.subgraphs for t in g.tensors}
    used.update(int(m.buffer) for m in model.metadata or [])
    if model.metadataBuffer is not None:
        used.update(int(i) for i in model.metadataBuffer)
    remap = {old: new for new, old in enumerate(sorted(used))}
    model.buffers = [model.buffers[i] for i in sorted(used)]
    for g in model.subgraphs:
        for tensor in g.tensors:
            tensor.buffer = remap[int(tensor.buffer)]
    for metadata in model.metadata or []:
        metadata.buffer = remap[int(metadata.buffer)]
    if model.metadataBuffer is not None:
        model.metadataBuffer = np.array([remap[int(i)] for i in model.metadataBuffer], dtype=np.int32)
    write_model(model, destination)
    return {'backend': 'static-affine-w4a8', 'activation_calibration': details,
            'constraint_rewrites': concat_rewrites,
            'weight_source': 'original_fp32_tflite_constants',
            'weight_range': [-7, 7], 'packing': 'low_nibble_first_twos_complement',
            'weights': converted}


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
