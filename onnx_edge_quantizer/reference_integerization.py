"""Make a new integer reference from isolated float depthwise-convolution islands."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile

import numpy as np

from .runtime import inventory, make_interpreter, modules, read_model, write_model
from .tflite_optimizations import compact_model


def integerize_depthwise_reference(source, destination):
    """Quantize DQ -> zero PAD -> float DWConv -> Q in a separate reference.

    Reuse its public activation encodings; derive new W8 weights and INT32
    biases from its original FP32 constants. This changes quantization error,
    never the original file. The new reference must load in stock LiteRT.
    """
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError('The integer reference must not replace its mixed source')
    if destination.exists():
        raise ValueError('The integer reference destination already exists')
    model = read_model(source)
    _, s = modules()
    if len(model.subgraphs) != 1:
        raise ValueError('Integer reference normalization requires one subgraph')
    graph = model.subgraphs[0]
    producers = {int(i): op for op in graph.operators for i in op.outputs}
    dynamic = set(producers) | set(map(int, graph.inputs))
    consumers = {}
    for op in graph.operators:
        for i in op.inputs:
            consumers.setdefault(int(i), []).append(op)

    def kind(op):
        c = model.operatorCodes[op.opcodeIndex]
        return max(c.builtinCode, c.deprecatedBuiltinCode)

    def scalar_encoding(tensor):
        p = tensor.quantization
        if (p is None or p.scale is None or len(p.scale) != 1 or p.zeroPoint is None
                or len(p.zeroPoint) != 1 or not np.isfinite(p.scale[0]) or p.scale[0] <= 0):
            raise ValueError('Reference island requires a finite per-tensor affine activation encoding')
        return float(p.scale[0]), int(p.zeroPoint[0])

    def float_values(index):
        tensor = graph.tensors[index]
        raw = model.buffers[tensor.buffer].data
        if index in dynamic or tensor.type != s.TensorType.FLOAT32 or raw is None:
            raise ValueError('Reference island requires constant FP32 weights and bias')
        values = np.frombuffer(bytes(raw), '<f4')
        if not values.size or not np.isfinite(values).all():
            raise ValueError('Invalid constant in reference island')
        return values

    def store(tensor, values):
        buffer = s.BufferT()
        buffer.data = np.frombuffer(np.ascontiguousarray(values).tobytes(), np.uint8)
        model.buffers.append(buffer)
        tensor.buffer = len(model.buffers) - 1

    def affine(tensor, scale, zero):
        p = s.QuantizationParametersT()
        p.scale, p.zeroPoint = np.array([scale], np.float32), np.array([zero], np.int64)
        p.quantizedDimension = 0
        tensor.quantization = p

    removed, candidates, changes = set(), {}, []
    for op in graph.operators:
        if kind(op) != s.BuiltinOperator.DEPTHWISE_CONV_2D or len(op.inputs) != 3:
            continue
        pad = producers.get(int(op.inputs[0]))
        if pad is None or kind(pad) != s.BuiltinOperator.PAD:
            continue
        dq = producers.get(int(pad.inputs[0]))
        users = consumers.get(int(op.outputs[0]), [])
        if (dq is None or kind(dq) != s.BuiltinOperator.DEQUANTIZE or len(users) != 1
                or kind(users[0]) != s.BuiltinOperator.QUANTIZE
                or int(pad.outputs[0]) in graph.outputs or int(op.outputs[0]) in graph.outputs
                or len(consumers.get(int(pad.outputs[0]), [])) != 1):
            continue
        q = users[0]
        input_id, output_id = int(dq.inputs[0]), int(q.outputs[0])
        input_tensor, output_tensor = graph.tensors[input_id], graph.tensors[output_id]
        if input_tensor.type not in (s.TensorType.INT8, s.TensorType.UINT8) or output_tensor.type != input_tensor.type:
            continue
        input_scale, _ = scalar_encoding(input_tensor)
        scalar_encoding(output_tensor)
        weights = float_values(int(op.inputs[1]))
        bias = float_values(int(op.inputs[2]))
        weight_scale = np.float32(max(float(np.max(np.abs(weights))) / 127,
                                      float(np.max(np.abs(bias))) / (float(1 << 30) * input_scale)) or 1.0)
        bias_scale = np.float32(input_scale * float(weight_scale))
        if not np.isfinite([weight_scale, bias_scale]).all() or min(weight_scale, bias_scale) <= 0:
            raise ValueError('Reference island quantization scales are not representable')
        codes = np.clip(np.rint(weights / weight_scale), -127, 127).astype(np.int16)
        zero = 128 if input_tensor.type == s.TensorType.UINT8 else 0
        dtype = np.uint8 if zero else np.int8
        rounded_bias = np.rint(bias.astype(np.float64) / float(bias_scale))
        if np.any(rounded_bias < -(1 << 31)) or np.any(rounded_bias > (1 << 31) - 1):
            raise ValueError('Reference island bias exceeds INT32')
        # Give weights/bias private tensors as constants may be shared.
        for slot, values, target_type, scale, zp in [
                (1, (codes + zero).astype(dtype), input_tensor.type, weight_scale, zero),
                (2, rounded_bias.astype('<i4'), s.TensorType.INT32, bias_scale, 0)]:
            tensor = deepcopy(graph.tensors[int(op.inputs[slot])])
            tensor.name += b'/integer_reference'
            tensor.type = target_type
            affine(tensor, scale, zp)
            store(tensor, values)
            graph.tensors.append(tensor)
            op.inputs[slot] = len(graph.tensors) - 1
        pad.inputs[0] = input_id
        padded = graph.tensors[int(pad.outputs[0])]
        padded.type, padded.quantization = input_tensor.type, deepcopy(input_tensor.quantization)
        op.outputs = np.array([output_id], np.int32)
        model.operatorCodes[op.opcodeIndex].version = max(model.operatorCodes[op.opcodeIndex].version, 3)
        removed.add(id(q))
        candidates[id(dq)] = dq
        changes.append({'rewrite': 'integerize_float_depthwise_reference_island',
                        'weight_scale': float(weight_scale), 'bias_scale': float(bias_scale),
                        'weight_type': 'UINT8' if zero else 'INT8',
                        'calibration': 'preserved_reference_input_and_output_activation_encodings'})
    if not changes:
        raise ValueError('No isolated float depthwise reference island found')
    used = {int(i) for op in graph.operators if id(op) not in removed for i in op.inputs}
    removed.update(key for key, dq in candidates.items()
                   if int(dq.outputs[0]) not in used and int(dq.outputs[0]) not in graph.outputs)
    graph.operators = [op for op in graph.operators if id(op) not in removed]
    compact_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='integer-reference-') as directory:
        staged = Path(directory) / 'reference.tflite'
        write_model(model, staged)
        make_interpreter(staged)
        if inventory(staged)['float_compute_ops']:
            raise ValueError('The normalized reference still contains float compute operations')
        destination.write_bytes(staged.read_bytes())
    report = {'source': str(source.resolve()), 'destination': str(destination.resolve()),
              'source_preserved': True, 'changes': changes, 'stock_litert_prepare': 'passed',
              'note': 'This is a new W8A8 reference with different quantization error; it is not the original mixed-precision baseline.'}
    destination.with_suffix('.integerization.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
