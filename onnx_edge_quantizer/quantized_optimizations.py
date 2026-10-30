"""Quantization constraints for branches joining at an integer concatenation."""
from copy import deepcopy

import numpy as np

from .runtime import modules, read_model, write_model
from .tflite_optimizations import compact_model


def restore_integer_reductions(path):
    """Reduce monotone affine INT8 codes directly, without a DQ/float/Q island."""
    model = read_model(path)
    _, s = modules()
    if len(model.subgraphs) != 1:
        return []
    graph = model.subgraphs[0]
    producers = {int(i): op for op in graph.operators for i in op.outputs}
    users = {}
    for op in graph.operators:
        for i in op.inputs:
            users.setdefault(int(i), []).append(op)

    def kind(op):
        c = model.operatorCodes[op.opcodeIndex]
        return max(c.builtinCode, c.deprecatedBuiltinCode)

    removed, candidates, changes = set(), {}, []
    for op in graph.operators:
        code = kind(op)
        if code not in (s.BuiltinOperator.ARG_MAX, s.BuiltinOperator.ARG_MIN,
                        s.BuiltinOperator.REDUCE_MAX, s.BuiltinOperator.REDUCE_MIN):
            continue
        original = int(op.inputs[0])
        dq = producers.get(original)
        if dq is None or kind(dq) != s.BuiltinOperator.DEQUANTIZE:
            continue
        source = int(dq.inputs[0])
        tensor = graph.tensors[source]
        qp = tensor.quantization
        if (tensor.type != s.TensorType.INT8 or qp is None or qp.scale is None
                or len(qp.scale) != 1 or qp.zeroPoint is None or len(qp.zeroPoint) != 1
                or not np.isfinite(qp.scale[0]) or qp.scale[0] <= 0):
            continue
        if code in (s.BuiltinOperator.REDUCE_MAX, s.BuiltinOperator.REDUCE_MIN):
            output = int(op.outputs[0])
            consumers = users.get(output, [])
            if (output in graph.outputs or not consumers
                    or any(kind(q) != s.BuiltinOperator.QUANTIZE for q in consumers)):
                continue
            q = consumers[0]
            target = int(q.outputs[0])
            encoding = graph.tensors[target].quantization
            if (graph.tensors[target].type != s.TensorType.INT8 or encoding is None
                    or not np.array_equal(qp.scale, encoding.scale)
                    or not np.array_equal(qp.zeroPoint, encoding.zeroPoint)):
                continue
            op.outputs = np.array([target], np.int32)
            removed.add(id(q))
            for other in consumers[1:]:
                other.inputs = np.array([target], np.int32)
        op.inputs[0] = source
        model.operatorCodes[op.opcodeIndex].version = max(model.operatorCodes[op.opcodeIndex].version, 2)
        candidates[id(dq)] = dq
        changes.append({'rewrite': 'native_int8_order_reduction', 'operator': int(code),
                        'encoding_preserved': True})
    if changes:
        referenced = {int(i) for op in graph.operators if id(op) not in removed for i in op.inputs}
        removed.update(key for key, dq in candidates.items()
                       if int(dq.outputs[0]) not in referenced and int(dq.outputs[0]) not in graph.outputs)
        graph.operators = [op for op in graph.operators if id(op) not in removed]
        compact_model(model)
        write_model(model, path)
    return changes


def restore_integer_leaky_relu(path):
    """Use LiteRT's native INT8 LeakyRelu instead of a float DQ/op/Q island."""
    model = read_model(path)
    _, s = modules()
    if len(model.subgraphs) != 1:
        return []
    g = model.subgraphs[0]
    producers = {int(i): op for op in g.operators for i in op.outputs}
    consumers = {}
    for op in g.operators:
        for i in op.inputs:
            consumers.setdefault(int(i), []).append(op)

    def code(op):
        c = model.operatorCodes[op.opcodeIndex]
        return max(c.builtinCode, c.deprecatedBuiltinCode)

    removed, dq_candidates, count = set(), {}, 0
    for op in g.operators:
        if code(op) != s.BuiltinOperator.LEAKY_RELU:
            continue
        data, output = int(op.inputs[0]), int(op.outputs[0])
        dq = producers.get(data)
        users = consumers.get(output, [])
        if (dq is None or code(dq) != s.BuiltinOperator.DEQUANTIZE
                or not users or any(code(user) != s.BuiltinOperator.QUANTIZE for user in users)
                or data in g.outputs or output in g.outputs):
            continue
        q = users[0]
        source, target = int(dq.inputs[0]), int(q.outputs[0])
        if any(g.tensors[i].type != s.TensorType.INT8 for i in (source, target)):
            continue
        op.inputs, op.outputs = np.array([source], np.int32), np.array([target], np.int32)
        model.operatorCodes[op.opcodeIndex].version = max(model.operatorCodes[op.opcodeIndex].version, 2)
        removed.add(id(q))
        dq_candidates[id(dq)] = dq
        for other in users[1:]:
            other.inputs = np.array([target], np.int32)
        count += 1
    if count:
        referenced = {int(i) for op in g.operators if id(op) not in removed for i in op.inputs}
        removed.update(key for key, dq in dq_candidates.items() if int(dq.outputs[0]) not in referenced)
        g.operators = [op for op in g.operators if id(op) not in removed]
        compact_model(model)
        write_model(model, path)
    return [{'rewrite': 'native_int8_leaky_relu', 'count': count}] if count else []


def satisfy_quantization_constraints(path, *, symmetric=False):
    """Share A8 encodings across layout/concat edges and absorb redundant Q.

    Weighted operators and arithmetic consume their actual input/output scales;
    W4 bias values are regenerated after this pass. Hard sigmoid/softmax/tanh
    encodings are anchors. Conflicting anchors retain their requantization.
    """
    model = read_model(path)
    _, s = modules()
    if len(model.subgraphs) != 1:
        return []
    g = model.subgraphs[0]
    parents, anchors = {}, {}
    feature_ids = set()
    for i, tensor in enumerate(g.tensors):
        qp = tensor.quantization
        if (tensor.type == s.TensorType.INT8 and qp is not None
                and qp.detailsType == 0 and qp.scale is not None and len(qp.scale) == 1):
            parents[i] = i
            feature_ids.add(i)

    def code(op):
        entry = model.operatorCodes[op.opcodeIndex]
        return max(entry.builtinCode, entry.deprecatedBuiltinCode)

    def find(i):
        if parents[i] != i:
            parents[i] = find(parents[i])
        return parents[i]

    def union(a, b):
        a, b = find(a), find(b)
        if a == b:
            return True
        if a in anchors and b in anchors and anchors[a] != anchors[b]:
            return False
        parents[b] = a
        if b in anchors:
            anchors[a] = anchors.pop(b)
        return True

    hard_ops = {s.BuiltinOperator.LOGISTIC, s.BuiltinOperator.SOFTMAX, s.BuiltinOperator.TANH}
    for op in g.operators:
        if code(op) in hard_ops:
            for i in op.outputs:
                i = int(i)
                if i in feature_ids:
                    qp = g.tensors[i].quantization
                    anchors[i] = (float(qp.scale[0]), int(qp.zeroPoint[0]))
    preserving = {s.BuiltinOperator.RESHAPE, s.BuiltinOperator.TRANSPOSE,
                  s.BuiltinOperator.SLICE, s.BuiltinOperator.STRIDED_SLICE,
                  s.BuiltinOperator.SPLIT, s.BuiltinOperator.SPLIT_V,
                  s.BuiltinOperator.GATHER, s.BuiltinOperator.PAD,
                  s.BuiltinOperator.MAX_POOL_2D, s.BuiltinOperator.AVERAGE_POOL_2D,
                  s.BuiltinOperator.RESIZE_BILINEAR, s.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR,
                  s.BuiltinOperator.CONCATENATION, s.BuiltinOperator.PACK, s.BuiltinOperator.UNPACK}
    for op in g.operators:
        if code(op) not in preserving:
            continue
        members = [int(i) for i in [*op.inputs, *op.outputs] if int(i) in feature_ids]
        for i in members[1:]:
            if not union(members[0], i):
                raise ValueError('Incompatible fixed quantization encodings in a scale-preserving operation')
    removed, aliases, merged = set(), {}, set()
    for op in g.operators:
        if code(op) != s.BuiltinOperator.QUANTIZE:
            continue
        source, target = int(op.inputs[0]), int(op.outputs[0])
        if source not in feature_ids or target not in feature_ids or target in g.outputs:
            continue
        if union(source, target):
            merged.add(source)
            aliases[target] = source
            removed.add(id(op))
    if not removed:
        return []
    changed_roots = {find(i) for i in merged}
    components = {}
    for i in feature_ids:
        root = find(i)
        if root in changed_roots:
            components.setdefault(root, []).append(i)
    for root, members in components.items():
        if root in anchors:
            scale, zero = anchors[root]
        else:
            low, high = 0.0, 0.0
            for i in members:
                qp = g.tensors[i].quantization
                low = min(low, (-128 - int(qp.zeroPoint[0])) * float(qp.scale[0]))
                high = max(high, (127 - int(qp.zeroPoint[0])) * float(qp.scale[0]))
            if symmetric:
                scale, zero = np.float32(max(abs(low), abs(high)) / 127 or 1.0), 0
            else:
                scale = np.float32((high - low) / 255 or 1.0)
                zero = int(np.clip(np.rint(-128 - low / scale), -128, 127))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Invalid joint A8 quantization scale')
        for i in members:
            tensor = g.tensors[i]
            data = model.buffers[tensor.buffer].data
            old = tensor.quantization
            if data is not None and len(data):
                # Constant feature operands are part of a shared constraint,
                # e.g. a static concat branch; weights are not unioned here.
                values = (np.frombuffer(bytes(data), np.int8).astype(np.float64)
                          - int(old.zeroPoint[0])) * float(old.scale[0])
                codes = np.clip(np.rint(values / scale) + zero, -128, 127).astype(np.int8)
                buffer = s.BufferT()
                buffer.data = codes.view(np.uint8)
                model.buffers.append(buffer)
                tensor.buffer = len(model.buffers) - 1
            old.scale = np.array([scale], np.float32)
            old.zeroPoint = np.array([zero], np.int64)

    def resolve(i):
        while i in aliases:
            i = aliases[i]
        return i

    for op in g.operators:
        op.inputs = np.array([resolve(int(i)) for i in op.inputs], np.int32)
    g.operators = [op for op in g.operators if id(op) not in removed]
    compact_model(model)
    write_model(model, path)
    return [{'rewrite': 'joint_a8_encoding_constraints', 'removed_quantize': len(removed),
             'components': len(components)}]


def align_concat_encodings(path):
    model = read_model(path)
    _, s = modules()
    if len(model.subgraphs) != 1:
        return []
    graph = model.subgraphs[0]
    producers = {int(i): op for op in graph.operators for i in op.outputs}
    consumers = {}
    for op in graph.operators:
        for i in op.inputs:
            consumers.setdefault(int(i), []).append(op)

    def code(op):
        entry = model.operatorCodes[op.opcodeIndex]
        return max(entry.builtinCode, entry.deprecatedBuiltinCode)

    preserving = {s.BuiltinOperator.RESHAPE, s.BuiltinOperator.TRANSPOSE, s.BuiltinOperator.SLICE}
    adjustable = {s.BuiltinOperator.CONV_2D, s.BuiltinOperator.DEPTHWISE_CONV_2D,
                  s.BuiltinOperator.FULLY_CONNECTED, s.BuiltinOperator.ADD,
                  s.BuiltinOperator.MUL, s.BuiltinOperator.SUB}
    removed, aliases = set(), {}
    for op in graph.operators:
        if code(op) != s.BuiltinOperator.QUANTIZE or len(op.inputs) != 1 or len(op.outputs) != 1:
            continue
        source, target = int(op.inputs[0]), int(op.outputs[0])
        users = consumers.get(target, [])
        target_tensor = graph.tensors[target]
        qp = target_tensor.quantization
        if (not users or any(code(user) != s.BuiltinOperator.CONCATENATION for user in users)
                or target in graph.outputs or target_tensor.type != s.TensorType.INT8
                or qp is None or qp.scale is None or len(qp.scale) != 1 or qp.detailsType != 0):
            continue
        chain, current, valid = [], source, True
        while True:
            tensor = graph.tensors[current]
            parent = producers.get(current)
            if (current in graph.inputs or current in graph.outputs or parent is None
                    or len(consumers.get(current, [])) != 1 or tensor.type != target_tensor.type
                    or len(parent.outputs) != 1):
                valid = False
                break
            chain.append(current)
            kind = code(parent)
            if kind in adjustable:
                break
            if kind not in preserving:
                valid = False
                break
            current = int(parent.inputs[0])
        if not valid:
            continue
        for index in chain:
            graph.tensors[index].quantization = deepcopy(qp)
        aliases[target] = source
        removed.add(id(op))
    if removed:
        for op in graph.operators:
            op.inputs = np.array([aliases.get(int(i), int(i)) for i in op.inputs], np.int32)
        graph.operators = [op for op in graph.operators if id(op) not in removed]
        compact_model(model)
        write_model(model, path)
    return [{'rewrite': 'align_concat_branch_encoding', 'count': len(removed)}] if removed else []
