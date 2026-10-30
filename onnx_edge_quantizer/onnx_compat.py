"""Recover native operators when the direct converter emits a custom fallback."""
from copy import deepcopy
import json

import numpy as np

from .runtime import modules, read_model, write_model
from .tflite_optimizations import compact_model


def restore_inferred_reshapes(path, onnx_model):
    """Recover original -1 shape controls replaced by stale ONNX annotations."""
    import onnx
    report_path = path.with_name(path.stem.removesuffix('_float32') + '_tensor_correspondence_report.json')
    if not report_path.is_file():
        return []
    records = json.loads(report_path.read_text())['records']
    constants = {t.name: onnx.numpy_helper.to_array(t) for t in onnx_model.graph.initializer}
    for node in onnx_model.graph.node:
        if node.op_type == 'Constant' and len(node.output) == 1:
            value = next((a.t for a in node.attribute if a.name == 'value'), None)
            if value is not None:
                constants[node.output[0]] = onnx.numpy_helper.to_array(value)
    nodes = {name: node for node in onnx_model.graph.node if node.op_type == 'Reshape' for name in node.output}
    controls = {}
    for row in records:
        if (row.get('onnx_op_type') != 'Reshape' or not row.get('exists_in_final_model')
                or row.get('assume_channel_last_layout') is not False):
            continue
        name = row['onnx_output_name']
        node = nodes.get(name) or nodes.get(name[2:] if name.startswith('wa/') else name)
        value = constants.get(node.input[1]) if node is not None and len(node.input) == 2 else None
        if (value is not None and value.ndim == 1 and np.count_nonzero(value == -1) == 1
                and np.all((value == -1) | (value > 0)) and np.all(value <= np.iinfo(np.int32).max)):
            controls[row['resolved_tflite_tensor_name']] = value.astype('<i4')
    model = read_model(path)
    _, s = modules()
    count = 0
    for graph in model.subgraphs:
        for op in graph.operators:
            code = model.operatorCodes[op.opcodeIndex]
            if max(code.builtinCode, code.deprecatedBuiltinCode) != s.BuiltinOperator.RESHAPE or len(op.inputs) != 2:
                continue
            value = controls.get(graph.tensors[int(op.outputs[0])].name.decode())
            tensor = graph.tensors[int(op.inputs[1])]
            raw = model.buffers[tensor.buffer].data
            if (value is None or raw is None or tensor.type != s.TensorType.INT32
                    or np.array_equal(np.frombuffer(bytes(raw), '<i4'), value)):
                continue
            current = np.frombuffer(bytes(raw), '<i4')
            # An explicit dimension moved by the converter is layout evidence,
            # not an inferred value. Only restore the original -1 slot.
            if current.shape != value.shape or np.any(current[value != -1] != value[value != -1]):
                continue
            buffer = s.BufferT()
            buffer.data = np.frombuffer(value.tobytes(), np.uint8)
            model.buffers.append(buffer)
            control = deepcopy(tensor)
            control.name = (tensor.name or b'shape') + f'/onnx_inferred_{count}'.encode()
            control.buffer = len(model.buffers) - 1
            control.shape = np.array([len(value)], np.int32)
            graph.tensors.append(control)
            op.inputs[1] = len(graph.tensors) - 1
            count += 1
    if count:
        compact_model(model)
        write_model(model, path)
    return [{'rewrite': 'restore_onnx_inferred_reshape', 'count': count}] if count else []


def lower_custom_gemm(path, onnx_model):
    import onnx
    model = read_model(path)
    _, s = modules()
    if len(model.subgraphs) != 1:
        return []
    graph = model.subgraphs[0]
    gemms = [node for node in onnx_model.graph.node if node.op_type == 'Gemm']
    count = 0
    for op in graph.operators:
        code = model.operatorCodes[op.opcodeIndex]
        if code.customCode != b'ONNX_GEMM' or len(op.inputs) not in (2, 3) or len(op.outputs) != 1:
            continue
        output_name = graph.tensors[int(op.outputs[0])].name.decode()
        candidates = [node for node in gemms if node.output[0] == output_name]
        if len(candidates) != 1:
            continue
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in candidates[0].attribute}
        if attrs.get('transA', 0) != 0 or attrs.get('alpha', 1.0) != 1.0 or attrs.get('beta', 1.0) != 1.0:
            continue
        data, weight = (graph.tensors[int(i)] for i in op.inputs[:2])
        raw = model.buffers[weight.buffer].data
        if data.shape is None or len(data.shape) != 2 or weight.type != s.TensorType.FLOAT32 or raw is None or len(weight.shape) != 2:
            continue
        values = np.frombuffer(bytes(raw), '<f4').reshape(weight.shape)
        if not attrs.get('transB', 0):
            values = values.T
        if values.shape[1] != data.shape[-1]:
            continue
        if len(op.inputs) == 3:
            bias = graph.tensors[int(op.inputs[2])]
            if bias.shape is None or list(bias.shape) != [values.shape[0]]:
                continue
        buffer = s.BufferT()
        buffer.data = np.frombuffer(np.ascontiguousarray(values).tobytes(), np.uint8)
        model.buffers.append(buffer)
        new_weight = deepcopy(weight)
        new_weight.name = weight.name + f'/native_gemm_{count}'.encode()
        new_weight.shape, new_weight.shapeSignature = np.array(values.shape, np.int32), None
        new_weight.buffer = len(model.buffers) - 1
        graph.tensors.append(new_weight)
        op.inputs[1] = len(graph.tensors) - 1
        entry = s.OperatorCodeT()
        entry.builtinCode = entry.deprecatedBuiltinCode = s.BuiltinOperator.FULLY_CONNECTED
        entry.version = 1
        model.operatorCodes.append(entry)
        op.opcodeIndex = len(model.operatorCodes) - 1
        op.customOptions = None
        op.builtinOptionsType, op.builtinOptions = s.BuiltinOptions.FullyConnectedOptions, s.FullyConnectedOptionsT()
        count += 1
    if count:
        compact_model(model)
        write_model(model, path)
    return [{'rewrite': 'custom_gemm_to_fully_connected', 'count': count}] if count else []
