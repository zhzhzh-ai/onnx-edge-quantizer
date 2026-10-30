"""Fuse constant channel bias into the fourth TransposeConv input."""
from copy import deepcopy

import numpy as np

from .runtime import modules, read_model, write_model
from .tflite_optimizations import compact_model


def fuse_transpose_conv_bias(path):
    model = read_model(path)
    _, s = modules()
    if len(model.subgraphs) != 1:
        return []
    graph = model.subgraphs[0]
    producers = {int(i): op for op in graph.operators for i in op.outputs}
    dynamic = set(producers) | set(map(int, graph.inputs))
    users = {}
    for op in graph.operators:
        for i in op.inputs:
            users.setdefault(int(i), []).append(op)

    def kind(op):
        code = model.operatorCodes[op.opcodeIndex]
        return max(code.builtinCode, code.deprecatedBuiltinCode)

    def constant(index, channels):
        tensor = graph.tensors[index]
        raw = model.buffers[tensor.buffer].data
        shape = list(tensor.shape) if tensor.shape is not None else []
        if (index in dynamic or tensor.type != s.TensorType.FLOAT32 or raw is None
                or not shape or shape[-1] != channels or np.prod(shape[:-1]) != 1):
            return None
        values = np.frombuffer(bytes(raw), '<f4')
        return values if values.size == channels and np.isfinite(values).all() else None

    changes, removed = [], set()
    for add in graph.operators:
        if kind(add) != s.BuiltinOperator.ADD or len(add.inputs) != 2:
            continue
        for slot in (0, 1):
            intermediate = int(add.inputs[slot])
            parent = producers.get(intermediate)
            if (parent is None or kind(parent) != s.BuiltinOperator.TRANSPOSE_CONV
                    or len(parent.inputs) not in (3, 4) or len(parent.outputs) != 1
                    or intermediate in graph.outputs or len(users.get(intermediate, [])) != 1
                    or parent.builtinOptions.fusedActivationFunction != s.ActivationFunctionType.NONE):
                continue
            shape = graph.tensors[intermediate].shape
            if (shape is None or len(shape) != 4
                    or not np.array_equal(shape, graph.tensors[int(add.outputs[0])].shape)):
                continue
            channels = int(shape[-1])
            values = constant(int(add.inputs[1 - slot]), channels)
            if values is None:
                continue
            if len(parent.inputs) == 4 and parent.inputs[3] >= 0:
                old_bias = constant(int(parent.inputs[3]), channels)
                if old_bias is None:
                    continue
                values = values + old_bias
            if not np.isfinite(values).all():
                continue
            tensor = deepcopy(graph.tensors[int(add.inputs[1 - slot])])
            tensor.name = (tensor.name or b'bias') + b'/transpose_conv_bias'
            tensor.shape, tensor.shapeSignature = np.array([channels], np.int32), None
            buffer = s.BufferT()
            buffer.data = np.frombuffer(values.astype('<f4').tobytes(), np.uint8)
            model.buffers.append(buffer)
            tensor.buffer = len(model.buffers) - 1
            graph.tensors.append(tensor)
            parent.inputs = np.array([*parent.inputs[:3], len(graph.tensors) - 1], np.int32)
            parent.outputs = add.outputs.copy()
            parent.builtinOptions.fusedActivationFunction = add.builtinOptions.fusedActivationFunction
            removed.add(id(add))
            changes.append({'rewrite': 'transpose_conv_channel_bias', 'channels': channels,
                            'fused_activation': int(parent.builtinOptions.fusedActivationFunction)})
            break
    if changes:
        graph.operators = [op for op in graph.operators if id(op) not in removed]
        compact_model(model)
        write_model(model, path)
    return changes
