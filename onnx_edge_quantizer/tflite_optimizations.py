"""Semantics-preserving compatibility rewrites before quantization."""

from functools import lru_cache
from copy import deepcopy
import numpy as np
from .runtime import modules, read_model, write_model


def normalize_deployment_graph(path, reference=None):
    """Canonical float graph for static integer deployment, before calibration.

    Commute independent scalar channel affine transforms through concatenation,
    fold unobserved trailing MaxPool padding into SAME, and let FC flatten its
    input using its native contract. Each rewrite checks shapes and sole uses.
    """
    model = read_model(path)
    _, s = modules()
    if len(model.subgraphs) != 1:
        return []
    g = model.subgraphs[0]
    changes, removed = [], set()
    identity_resize_shapes = set()
    if reference is not None:
        baseline = read_model(reference)
        ref = baseline.subgraphs[0]
        producers = {int(i): op for op in ref.operators for i in op.outputs}
        for second in ref.operators:
            first = producers.get(int(second.inputs[0])) if len(second.inputs) else None
            pair = [first, second]
            if first is None:
                continue
            valid = True
            for axis, gather in zip((1, 2), pair):
                code = baseline.operatorCodes[gather.opcodeIndex]
                if (max(code.builtinCode, code.deprecatedBuiltinCode) != s.BuiltinOperator.GATHER
                        or gather.builtinOptions.axis != axis or gather.builtinOptions.batchDims != 0):
                    valid = False
                    break
                tensor = ref.tensors[int(gather.inputs[1])]
                raw = baseline.buffers[tensor.buffer].data
                shape = ref.tensors[int(gather.inputs[0])].shape
                if (shape is None or len(shape) != 4 or tensor.type != s.TensorType.INT32 or raw is None
                        or not np.array_equal(np.frombuffer(bytes(raw), '<i4'), np.arange(shape[axis]))):
                    valid = False
                    break
            if valid:
                identity_resize_shapes.add(tuple(ref.tensors[int(first.inputs[0])].shape))

    pad_code = next((i for i, entry in enumerate(model.operatorCodes)
                     if max(entry.builtinCode, entry.deprecatedBuiltinCode) == s.BuiltinOperator.PAD), None)
    for op in g.operators:
        entry = model.operatorCodes[op.opcodeIndex]
        if max(entry.builtinCode, entry.deprecatedBuiltinCode) != s.BuiltinOperator.PADV2:
            continue
        fill = g.tensors[int(op.inputs[2])]
        data = model.buffers[fill.buffer].data
        if fill.type != s.TensorType.FLOAT32 or data is None:
            continue
        value = np.frombuffer(bytes(data), '<f4')
        if value.size != 1 or value.item() != 0:
            continue
        if pad_code is None:
            entry = s.OperatorCodeT()
            entry.builtinCode = entry.deprecatedBuiltinCode = s.BuiltinOperator.PAD
            entry.version = 1
            model.operatorCodes.append(entry)
            pad_code = len(model.operatorCodes) - 1
        op.opcodeIndex = pad_code
        op.inputs = op.inputs[:2].copy()
        op.builtinOptionsType, op.builtinOptions = s.BuiltinOptions.PadOptions, s.PadOptionsT()
        changes.append({'rewrite': 'zero_padv2_to_pad'})

    produced = {int(i) for op in g.operators for i in op.outputs}
    for op in g.operators:
        entry = model.operatorCodes[op.opcodeIndex]
        if max(entry.builtinCode, entry.deprecatedBuiltinCode) not in (
                s.BuiltinOperator.RESIZE_BILINEAR, s.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR):
            continue
        if (max(entry.builtinCode, entry.deprecatedBuiltinCode) == s.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR
                and g.tensors[int(op.inputs[0])].shape is not None
                and tuple(g.tensors[int(op.inputs[0])].shape) in identity_resize_shapes):
            continue
        source, target = int(op.inputs[0]), int(op.outputs[0])
        size = g.tensors[int(op.inputs[1])]
        data = model.buffers[size.buffer].data
        shape = g.tensors[source].shape
        if (target in g.outputs or shape is None or len(shape) != 4
                or int(op.inputs[1]) in produced or int(op.inputs[1]) in g.inputs
                or size.type != s.TensorType.INT32 or data is None
                or not np.array_equal(np.frombuffer(bytes(data), '<i4'), shape[1:3])
                or not np.array_equal(shape, g.tensors[target].shape)):
            continue
        for user in g.operators:
            user.inputs = np.array([source if int(i) == target else int(i) for i in user.inputs], np.int32)
        removed.add(id(op))
        changes.append({'rewrite': 'remove_identity_resize'})
    g.operators = [op for op in g.operators if id(op) not in removed]
    for op in g.operators:
        c = model.operatorCodes[op.opcodeIndex]
        if (max(c.builtinCode, c.deprecatedBuiltinCode) != s.BuiltinOperator.CONCATENATION
                or op.builtinOptions.axis != 0):
            continue
        kept = []
        for index in op.inputs:
            tensor = g.tensors[int(index)]
            empty_constant = (
                int(index) not in produced and int(index) not in g.inputs
                and tensor.type in (s.TensorType.INT32, s.TensorType.INT64)
                and tensor.shape is not None and list(tensor.shape) == [0]
            )
            if not empty_constant:
                kept.append(int(index))
        if kept and len(kept) != len(op.inputs):
            op.inputs = np.array(kept, dtype=np.int32)
            changes.append({'rewrite': 'remove_empty_shape_concat_inputs'})

    # A scalar reshape uses an empty shape-vector constant. Its zero-byte
    # buffer is valid TFLite, but AEQ mistakes it for a runtime activation.
    # Fold proven constant reshapes by aliasing their data (including any
    # ordinary constant weight reshapes); no inference sample is consulted.
    for op in g.operators:
        c = model.operatorCodes[op.opcodeIndex]
        if (max(c.builtinCode, c.deprecatedBuiltinCode) != s.BuiltinOperator.RESHAPE
                or len(op.outputs) != 1 or op.inputs[0] < 0):
            continue
        source = g.tensors[int(op.inputs[0])]
        target = g.tensors[int(op.outputs[0])]
        data = model.buffers[source.buffer].data
        if len(op.inputs) == 2:
            shape_tensor = g.tensors[int(op.inputs[1])]
            if (shape_tensor.type == s.TensorType.INT32 and shape_tensor.shape is not None
                    and list(shape_tensor.shape) == [0]
                    and (target.shape is None or len(target.shape) == 0)):
                op.inputs = np.array([op.inputs[0]], dtype=np.int32)
                op.builtinOptionsType, op.builtinOptions = s.BuiltinOptions.ReshapeOptions, s.ReshapeOptionsT()
                op.builtinOptions.newShape = np.array([], dtype=np.int32)
                changes.append({'rewrite': 'scalar_reshape_options'})
        if (data is not None and len(data) and source.type == target.type
                and np.prod(source.shape if source.shape is not None else [])
                == np.prod(target.shape if target.shape is not None else [])):
            target.buffer = source.buffer
            if target.shape is None:
                target.shape = np.array([], dtype=np.int32)
            removed.add(id(op))
            changes.append({'rewrite': 'fold_constant_reshape'})
    g.operators = [op for op in g.operators if id(op) not in removed]
    producers = {int(i): op for op in g.operators for i in op.outputs}
    consumers = {}
    for op in g.operators:
        for i in op.inputs:
            consumers.setdefault(int(i), []).append(op)

    def code(op):
        c = model.operatorCodes[op.opcodeIndex]
        return max(c.builtinCode, c.deprecatedBuiltinCode)

    def values(index, dtype):
        t = g.tensors[int(index)]
        data = model.buffers[t.buffer].data
        return None if data is None else np.frombuffer(bytes(data), dtype=dtype).reshape(t.shape)

    def single_use(op):
        return all(int(i) not in g.outputs and len(consumers.get(int(i), [])) == 1 for i in op.outputs)

    @lru_cache(None)
    def nonnegative(index):
        op = producers.get(index)
        if op is None:
            return False
        if getattr(op.builtinOptions, 'fusedActivationFunction', None) in (
                s.ActivationFunctionType.RELU, s.ActivationFunctionType.RELU6):
            return True
        builtin = code(op)
        if builtin in (s.BuiltinOperator.RELU, s.BuiltinOperator.RELU6, s.BuiltinOperator.ABS):
            return True
        if builtin == s.BuiltinOperator.CONCATENATION:
            return all(nonnegative(int(i)) for i in op.inputs if i >= 0)
        if builtin in (s.BuiltinOperator.MAX_POOL_2D, s.BuiltinOperator.AVERAGE_POOL_2D,
                       s.BuiltinOperator.RESHAPE, s.BuiltinOperator.TRANSPOSE,
                       s.BuiltinOperator.SQUEEZE, s.BuiltinOperator.MEAN):
            return nonnegative(int(op.inputs[0]))
        return False

    def const(value, name):
        data = np.ascontiguousarray(value)
        buf = s.BufferT()
        buf.data = np.frombuffer(data.tobytes(), dtype=np.uint8)
        model.buffers.append(buf)
        t = s.TensorT()
        t.name, t.shape, t.type, t.buffer = name, np.array(data.shape, dtype=np.int32), s.TensorType.FLOAT32, len(model.buffers) - 1
        g.tensors.append(t)
        return len(g.tensors) - 1

    # onnx2tf's direct backend can leave exported BatchNorm/Scale as
    # Conv -> Mul(channel scale) -> Add(channel bias). Fold the affine
    # transform into private FP32 weights before any range calibration.
    for add in g.operators:
        if code(add) != s.BuiltinOperator.ADD or len(add.inputs) != 2:
            continue
        mul = producers.get(int(add.inputs[0]))
        if mul is None or code(mul) != s.BuiltinOperator.MUL or len(mul.inputs) != 2:
            continue
        conv = producers.get(int(mul.inputs[0]))
        if (conv is None or code(conv) not in (s.BuiltinOperator.CONV_2D, s.BuiltinOperator.DEPTHWISE_CONV_2D, s.BuiltinOperator.FULLY_CONNECTED)
                or not single_use(conv) or not single_use(mul)
                or conv.builtinOptions.fusedActivationFunction != 0
                or mul.builtinOptions.fusedActivationFunction != 0):
            continue
        axis = 3 if code(conv) == s.BuiltinOperator.DEPTHWISE_CONV_2D else 0
        weight = g.tensors[int(conv.inputs[1])]
        channels = int(weight.shape[axis])
        if any(g.tensors[int(i)].type != s.TensorType.FLOAT32 for i in (conv.inputs[1], mul.inputs[1], add.inputs[1])):
            continue
        scale, offset = values(mul.inputs[1], '<f4'), values(add.inputs[1], '<f4')
        if any(v is None or v.size not in (1, channels) or not np.isfinite(v).all()
               or (v.size > 1 and (v.shape[-1] != channels or np.prod(v.shape[:-1]) != 1))
               for v in (scale, offset)):
            continue
        w = values(conv.inputs[1], '<f4')
        if w is None:
            continue
        bias = np.zeros(channels, dtype=np.float32)
        if len(conv.inputs) > 2 and conv.inputs[2] >= 0:
            if g.tensors[int(conv.inputs[2])].type != s.TensorType.FLOAT32:
                continue
            bias = values(conv.inputs[2], '<f4')
            if bias is None or bias.shape != (channels,):
                continue
        scale, offset = scale.reshape(-1), offset.reshape(-1)
        broadcast = [1] * w.ndim
        broadcast[axis] = scale.size
        new_w = (w * scale.reshape(broadcast)).astype(np.float32)
        new_bias = (bias * scale + offset).astype(np.float32)
        if not np.isfinite(new_w).all() or not np.isfinite(new_bias).all():
            raise ValueError('Folded channel affine constants overflow FP32')
        conv.inputs = np.array([int(conv.inputs[0]), const(new_w, weight.name + b'/affine_folded'),
                                const(new_bias, weight.name + b'/affine_bias')], dtype=np.int32)
        conv.outputs = add.outputs.copy()
        conv.builtinOptions.fusedActivationFunction = add.builtinOptions.fusedActivationFunction
        removed.update((id(mul), id(add)))
        following = consumers.get(int(add.outputs[0]), [])
        if len(following) == 1 and single_use(add) and code(following[0]) in (s.BuiltinOperator.RELU, s.BuiltinOperator.RELU6):
            activation = following[0]
            if conv.builtinOptions.fusedActivationFunction == 0:
                conv.builtinOptions.fusedActivationFunction = s.ActivationFunctionType.RELU if code(activation) == s.BuiltinOperator.RELU else s.ActivationFunctionType.RELU6
                conv.outputs = activation.outputs.copy()
                removed.add(id(activation))
        changes.append({'rewrite': 'fold_weighted_channel_affine', 'channels': channels})

    insert_after = {}
    for concat in list(g.operators):
        if code(concat) != s.BuiltinOperator.CONCATENATION or concat.builtinOptions.axis != 3:
            continue
        branches = []
        for index in concat.inputs:
            reshape = producers.get(int(index))
            add = producers.get(int(reshape.inputs[0])) if reshape is not None else None
            mul = producers.get(int(add.inputs[0])) if add is not None else None
            gather = producers.get(int(mul.inputs[0])) if mul is not None else None
            chain = [reshape, add, mul, gather]
            if any(op is None for op in chain) or [code(op) for op in chain] != [s.BuiltinOperator.RESHAPE, s.BuiltinOperator.ADD, s.BuiltinOperator.MUL, s.BuiltinOperator.GATHER]:
                break
            if (not all(single_use(op) for op in chain)
                    or any(getattr(op.builtinOptions, 'fusedActivationFunction', 0) != 0 for op in chain)
                    or gather.builtinOptions.axis != 1 or gather.builtinOptions.batchDims != 0):
                break
            scale, bias = values(mul.inputs[1], '<f4'), values(add.inputs[1], '<f4')
            source_shape = list(g.tensors[int(gather.inputs[0])].shape)
            reshaped = list(g.tensors[int(reshape.outputs[0])].shape)
            if (len(source_shape) != 4 or reshaped != [source_shape[0], source_shape[2], source_shape[3], 1]
                    or list(g.tensors[int(gather.outputs[0])].shape) != [source_shape[0], 1, *source_shape[2:]]
                    or scale is None or bias is None or scale.size != 1 or bias.size != 1
                    or g.tensors[int(gather.inputs[1])].type != s.TensorType.INT32
                    or values(gather.inputs[1], '<i4').size != 1):
                break
            branches.append((reshape, add, mul, gather, float(scale.item()), float(bias.item())))
        if len(branches) != len(concat.inputs) or len(branches) < 2:
            continue
        if len({int(branch[3].inputs[0]) for branch in branches}) != 1:
            continue
        # Clone the index tensor: its buffer may be shared by other constants.
        for reshape, add, mul, gather, _, _ in branches:
            indices = deepcopy(g.tensors[int(gather.inputs[1])])
            indices.shape = np.array([], dtype=np.int32)
            g.tensors.append(indices)
            gather.inputs[1] = len(g.tensors) - 1
            source_shape = list(g.tensors[int(gather.inputs[0])].shape)
            g.tensors[int(gather.outputs[0])].shape = np.array([source_shape[0], *source_shape[2:]], dtype=np.int32)
            reshape.inputs[0] = gather.outputs[0]
            removed.update((id(add), id(mul)))
        target = int(concat.outputs[0])
        raw, scaled = deepcopy(g.tensors[target]), deepcopy(g.tensors[target])
        raw.name, scaled.name = raw.name + b'/channel_concat', scaled.name + b'/channel_scaled'
        g.tensors.extend([raw, scaled])
        raw_id, scaled_id = len(g.tensors) - 2, len(g.tensors) - 1
        concat.outputs[0] = raw_id
        mul, add = deepcopy(branches[0][2]), deepcopy(branches[0][1])
        mul.inputs = np.array([raw_id, const(np.array([b[4] for b in branches], dtype='<f4'), b'channel_scales')], dtype=np.int32)
        mul.outputs = np.array([scaled_id], dtype=np.int32)
        add.inputs = np.array([scaled_id, const(np.array([b[5] for b in branches], dtype='<f4'), b'channel_biases')], dtype=np.int32)
        add.outputs = np.array([target], dtype=np.int32)
        insert_after[id(concat)] = [mul, add]
        changes.append({'rewrite': 'channel_affine_after_concat', 'channels': len(branches)})

    for op in g.operators:
        if id(op) in removed:
            continue
        if code(op) in (s.BuiltinOperator.RELU, s.BuiltinOperator.RELU6):
            parent = producers.get(int(op.inputs[0]))
            if (parent is not None and code(parent) in (s.BuiltinOperator.FULLY_CONNECTED, s.BuiltinOperator.CONV_2D, s.BuiltinOperator.DEPTHWISE_CONV_2D,
                                                       s.BuiltinOperator.ADD, s.BuiltinOperator.MUL)
                    and single_use(parent) and parent.builtinOptions.fusedActivationFunction == s.ActivationFunctionType.NONE):
                parent.builtinOptions.fusedActivationFunction = (s.ActivationFunctionType.RELU if code(op) == s.BuiltinOperator.RELU else s.ActivationFunctionType.RELU6)
                parent.outputs = op.outputs.copy()
                removed.add(id(op))
                changes.append({'rewrite': 'fuse_weighted_activation'})
        elif code(op) == s.BuiltinOperator.RESHAPE:
            inner = producers.get(int(op.inputs[0]))
            if inner is not None and code(inner) == s.BuiltinOperator.RESHAPE and single_use(inner):
                op.inputs[0] = inner.inputs[0]
                removed.add(id(inner))
                changes.append({'rewrite': 'collapse_reshape_chain'})
        elif code(op) in (s.BuiltinOperator.CONV_2D, s.BuiltinOperator.DEPTHWISE_CONV_2D):
            pad = producers.get(int(op.inputs[0]))
            if pad is None or code(pad) not in (s.BuiltinOperator.PAD, s.BuiltinOperator.PADV2) or not single_use(pad):
                continue
            if code(pad) == s.BuiltinOperator.PADV2:
                fill = values(pad.inputs[2], '<f4')
                if fill is None or fill.size != 1 or fill.item() != 0:
                    continue
            source = int(pad.inputs[0])
            shape = list(g.tensors[source].shape)
            target = list(g.tensors[int(op.outputs[0])].shape)
            pads = values(pad.inputs[1], '<i4')
            opt = op.builtinOptions
            if len(shape) != 4 or pads is None or pads.shape != (4, 2) or np.any(pads[[0, 3]]) or opt.padding != s.Padding.VALID:
                continue
            kh, kw = g.tensors[int(op.inputs[1])].shape[1:3]
            equivalent = True
            for axis, kernel, stride, dilation in [(1, kh, opt.strideH, opt.dilationHFactor), (2, kw, opt.strideW, opt.dilationWFactor)]:
                count = (shape[axis] + stride - 1) // stride
                total = max((count - 1) * stride + (int(kernel) - 1) * dilation + 1 - shape[axis], 0)
                equivalent &= (count == target[axis] and pads[axis, 0] == total // 2
                               and pads[axis, 1] >= total - total // 2)
            if equivalent:
                op.inputs[0], opt.padding = source, s.Padding.SAME
                removed.add(id(pad))
                changes.append({'rewrite': 'conv_pad_to_same'})
        elif code(op) == s.BuiltinOperator.MAX_POOL_2D:
            pad = producers.get(int(op.inputs[0]))
            if pad is None or code(pad) not in (s.BuiltinOperator.PAD, s.BuiltinOperator.PADV2) or not single_use(pad):
                continue
            # PADV2(-inf) has already been proven equivalent to zero PAD by
            # normalize_maxpool_padding. Other fills must not be discarded.
            if code(pad) != s.BuiltinOperator.PAD:
                continue
            source = int(pad.inputs[0])
            if not nonnegative(source):
                continue
            shape = list(g.tensors[source].shape)
            target_shape = list(g.tensors[int(op.outputs[0])].shape)
            pads = values(pad.inputs[1], '<i4')
            opt = op.builtinOptions
            if len(shape) != 4 or pads is None or pads.shape != (4, 2) or np.any(pads[[0, 3]]) or opt.padding != s.Padding.VALID:
                continue
            equivalent = True
            for axis, kernel, stride in [(1, opt.filterHeight, opt.strideH), (2, opt.filterWidth, opt.strideW)]:
                count = (shape[axis] + stride - 1) // stride
                total = max((count - 1) * stride + kernel - shape[axis], 0)
                # Extra trailing padding is safe only if it never creates a
                # new output window. Before-padding fixes window alignment.
                equivalent &= (count == target_shape[axis] and pads[axis, 0] == total // 2
                               and pads[axis, 1] >= total - total // 2)
            if equivalent:
                op.inputs[0], opt.padding = source, s.Padding.SAME
                removed.add(id(pad))
                changes.append({'rewrite': 'maxpool_pad_to_same'})
        elif code(op) == s.BuiltinOperator.FULLY_CONNECTED and not op.builtinOptions.keepNumDims:
            reshape = producers.get(int(op.inputs[0]))
            if reshape is None or code(reshape) != s.BuiltinOperator.RESHAPE or not single_use(reshape):
                continue
            source = int(reshape.inputs[0])
            old_shape = g.tensors[int(op.inputs[0])].shape
            weight_shape = g.tensors[int(op.inputs[1])].shape
            source_shape = list(g.tensors[source].shape)
            if (len(old_shape) == 2 and old_shape[-1] == weight_shape[1]
                    and len(source_shape) == 4 and source_shape[1:3] == [1, 1]
                    and source_shape[-1] == weight_shape[1]
                    and np.prod(source_shape) == np.prod(old_shape)):
                op.inputs[0] = source
                removed.add(id(reshape))
                changes.append({'rewrite': 'fc_native_flatten'})
    g.operators = [item for op in g.operators if id(op) not in removed
                   for item in [op, *insert_after.get(id(op), [])]]
    if changes:
        compact_model(model)
        write_model(model, path)
    return changes


def compact_model(model):
    """Drop unreferenced tensors and deterministically order by data edges."""
    for gi, graph in enumerate(model.subgraphs):
        producers = {int(i): op for op in graph.operators for i in op.outputs}
        ordered, visited, active = [], set(), set()

        def visit(op):
            key = id(op)
            if key in visited:
                return
            if key in active:
                raise ValueError('Cyclic TFLite graph')
            active.add(key)
            for i in op.inputs:
                parent = producers.get(int(i))
                if parent is not None:
                    visit(parent)
            active.remove(key)
            visited.add(key)
            ordered.append(op)

        for i in graph.outputs:
            if int(i) in producers:
                visit(producers[int(i)])
        # Keep side-effect operators; don't silently drop disconnected work.
        for op in graph.operators:
            visit(op)
        graph.operators = ordered
        used = set(map(int, graph.inputs)) | set(map(int, graph.outputs))
        for op in ordered:
            for refs in (op.inputs, op.outputs, op.intermediates):
                if refs is not None:
                    used.update(int(i) for i in refs if i >= 0)
        signatures = [sig for sig in model.signatureDefs or [] if sig.subgraphIndex == gi]
        for sig in signatures:
            used.update(int(t.tensorIndex) for t in [*(sig.inputs or []), *(sig.outputs or [])])
        mapping = {old: new for new, old in enumerate(sorted(used))}

        def remap(refs):
            return np.array([mapping[int(i)] if i >= 0 else -1 for i in refs], dtype=np.int32)

        graph.tensors = [graph.tensors[i] for i in sorted(used)]
        graph.inputs, graph.outputs = remap(graph.inputs), remap(graph.outputs)
        for op in graph.operators:
            op.inputs, op.outputs = remap(op.inputs), remap(op.outputs)
            if op.intermediates is not None:
                op.intermediates = remap(op.intermediates)
        for sig in signatures:
            for t in [*(sig.inputs or []), *(sig.outputs or [])]:
                t.tensorIndex = mapping[int(t.tensorIndex)]


def commute_unary_before_reshape(path, reference):
    """Apply a reference's rank-independent activation before flattening."""
    if reference is None:
        return []
    model, baseline = read_model(path), read_model(reference)
    _, s = modules()
    unary = {s.BuiltinOperator.LOGISTIC, s.BuiltinOperator.RELU, s.BuiltinOperator.RELU6,
             s.BuiltinOperator.LEAKY_RELU, s.BuiltinOperator.TANH, s.BuiltinOperator.EXP}
    def code(m, op):
        c = m.operatorCodes[op.opcodeIndex]
        return max(c.builtinCode, c.deprecatedBuiltinCode)
    ref = baseline.subgraphs[0]
    signatures = {(code(baseline, op), tuple(ref.tensors[int(op.inputs[0])].shape))
                  for op in ref.operators if code(baseline, op) in unary}
    count = 0
    for g in model.subgraphs:
        producers = {int(i): op for op in g.operators for i in op.outputs}
        users = {}
        for op in g.operators:
            for i in op.inputs:
                users.setdefault(int(i), []).append(op)
        for op in g.operators:
            if code(model, op) not in unary or len(op.inputs) != 1 or len(op.outputs) != 1:
                continue
            intermediate, output = int(op.inputs[0]), int(op.outputs[0])
            reshape = producers.get(intermediate)
            if (reshape is None or code(model, reshape) != s.BuiltinOperator.RESHAPE
                    or intermediate in g.outputs or len(users[intermediate]) != 1):
                continue
            source = int(reshape.inputs[0])
            shape = g.tensors[source].shape
            if (shape is None or np.array_equal(shape, g.tensors[intermediate].shape)
                    or (code(model, op), tuple(shape)) not in signatures):
                continue
            g.tensors[intermediate].name += b'/activation_before_reshape'
            g.tensors[intermediate].shape, g.tensors[intermediate].shapeSignature = shape.copy(), None
            op.inputs, op.outputs = np.array([source], np.int32), np.array([intermediate], np.int32)
            reshape.inputs[0], reshape.outputs = intermediate, np.array([output], np.int32)
            count += 1
    if count:
        compact_model(model)
        write_model(model, path)
    return [{'rewrite': 'reference_unary_before_reshape', 'count': count}] if count else []


def restore_pointwise_rank(path):
    """Lift pointwise chains through proven contiguous axis coalescing."""
    model = read_model(path)
    _, s = modules()
    unary = {s.BuiltinOperator.LOGISTIC, s.BuiltinOperator.RELU, s.BuiltinOperator.RELU6,
             s.BuiltinOperator.LEAKY_RELU, s.BuiltinOperator.TANH, s.BuiltinOperator.EXP}
    binary = {s.BuiltinOperator.ADD, s.BuiltinOperator.SUB, s.BuiltinOperator.MUL,
              s.BuiltinOperator.DIV, s.BuiltinOperator.MAXIMUM, s.BuiltinOperator.MINIMUM}
    count = 0
    for g in model.subgraphs:
        def code(op):
            c = model.operatorCodes[op.opcodeIndex]
            return max(c.builtinCode, c.deprecatedBuiltinCode)
        produced = {int(i) for op in g.operators for i in op.outputs}
        users = {}
        for op in g.operators:
            for i in op.inputs:
                users.setdefault(int(i), []).append(op)
        removed = set()
        for first in g.operators:
            if code(first) != s.BuiltinOperator.RESHAPE or id(first) in removed:
                continue
            source = int(first.inputs[0])
            shape = g.tensors[source].shape
            current = int(first.outputs[0])
            flat_shape = g.tensors[current].shape
            if (shape is None or flat_shape is None or len(shape) < 3
                    or len(flat_shape) >= len(shape) or np.any(shape <= 0)
                    or np.prod(shape) != np.prod(flat_shape)):
                continue
            chain, lifted = [], []

            def lift_constant(tensor):
                old = [] if tensor.shape is None else list(tensor.shape)
                if len(old) > len(flat_shape):
                    return None
                padded = [1] * (len(flat_shape) - len(old)) + old
                if any(c not in (1, int(d)) for c, d in zip(padded, flat_shape)):
                    return None

                def partition(start, axis):
                    if axis == len(flat_shape):
                        return [] if start == len(shape) else None
                    product = 1
                    for end in range(start + 1, len(shape) + 1):
                        product *= int(shape[end - 1])
                        if product == int(flat_shape[axis]):
                            tail = partition(end, axis + 1)
                            if tail is not None:
                                dims = list(shape[start:end]) if padded[axis] != 1 else [1] * (end-start)
                                return dims + tail
                        if product > int(flat_shape[axis]):
                            break
                    return None
                desired = partition(0, 0)
                return desired if desired is not None and np.prod(desired) == np.prod(old) else None

            while current not in g.outputs and len(users.get(current, [])) == 1:
                op = users[current][0]
                kind = code(op)
                if kind == s.BuiltinOperator.RESHAPE:
                    target = g.tensors[int(op.outputs[0])]
                    if chain and target.shape is not None:
                        for item, slot, index, desired in lifted:
                            clone = deepcopy(g.tensors[index])
                            clone.name = (clone.name or b'constant') + f'/lifted_{len(g.tensors)}'.encode()
                            clone.shape, clone.shapeSignature = np.array(desired, np.int32), None
                            g.tensors.append(clone)
                            item.inputs[slot] = len(g.tensors) - 1
                        previous = source
                        for item, slot in chain:
                            item.inputs[slot] = previous
                            previous = int(item.outputs[0])
                            g.tensors[previous].shape = shape.copy()
                            g.tensors[previous].shapeSignature = None
                        removed.add(id(first))
                        if list(target.shape) == list(shape):
                            chain[-1][0].outputs = op.outputs.copy()
                            removed.add(id(op))
                        count += 1
                    break
                slots = [j for j, i in enumerate(op.inputs) if int(i) == current]
                if len(slots) != 1 or len(op.outputs) != 1:
                    break
                slot = slots[0]
                if kind in binary and len(op.inputs) == 2:
                    other_id = int(op.inputs[1-slot])
                    other = g.tensors[other_id]
                    data = model.buffers[other.buffer].data
                    desired = lift_constant(other)
                    if (other_id in produced or other_id in g.inputs or data is None or desired is None
                            or other.type not in (s.TensorType.FLOAT32, s.TensorType.INT8, s.TensorType.UINT8)):
                        break
                    lifted.append((op, 1-slot, other_id, desired))
                elif kind not in unary or len(op.inputs) != 1:
                    break
                chain.append((op, slot))
                current = int(op.outputs[0])
        g.operators = [op for op in g.operators if id(op) not in removed]
    if count:
        compact_model(model)
        write_model(model, path)
    return [{'rewrite': 'pointwise_rank_restored', 'count': count}] if count else []


def fuse_scalar_gather_reshapes(path):
    """Fold length-one Gather plus removal of its index dimension into scalar Gather."""
    model = read_model(path)
    _, s = modules()
    count = 0
    for g in model.subgraphs:
        produced = {int(i) for op in g.operators for i in op.outputs}
        users = {}
        for op in g.operators:
            for i in op.inputs:
                users.setdefault(int(i), []).append(op)

        def kind(op):
            code = model.operatorCodes[op.opcodeIndex]
            return max(code.builtinCode, code.deprecatedBuiltinCode)

        removed = set()
        for op in g.operators:
            if kind(op) != s.BuiltinOperator.GATHER or op.builtinOptions.batchDims != 0:
                continue
            source, indices = (g.tensors[int(i)] for i in op.inputs)
            output = int(op.outputs[0])
            consumers = users.get(output, [])
            if (output in g.outputs or len(consumers) != 1 or indices.shape is None
                    or list(indices.shape) != [1] or int(op.inputs[1]) in produced
                    or int(op.inputs[1]) in g.inputs):
                continue
            reshape = consumers[0]
            if kind(reshape) != s.BuiltinOperator.RESHAPE or source.shape is None:
                continue
            axis = op.builtinOptions.axis % len(source.shape)
            expected = list(source.shape[:axis]) + list(source.shape[axis + 1:])
            target = g.tensors[int(reshape.outputs[0])]
            current = g.tensors[output]
            if (target.shape is None or list(target.shape) != expected or current.shape is None
                    or list(current.shape) != expected[:axis] + [1] + expected[axis:]):
                continue
            scalar = deepcopy(indices)
            scalar.name = (indices.name or b'indices') + f'/scalar_{len(g.tensors)}'.encode()
            scalar.shape, scalar.shapeSignature = np.array([], np.int32), None
            g.tensors.append(scalar)
            op.inputs[1], op.outputs = len(g.tensors) - 1, reshape.outputs.copy()
            removed.add(id(reshape))
            count += 1
        g.operators = [op for op in g.operators if id(op) not in removed]
    if count:
        compact_model(model)
        write_model(model, path)
    return [{'rewrite': 'scalar_gather_reshape', 'count': count}] if count else []


def normalize_elementwise_constants(path):
    """Canonicalize scalar broadcasts without changing output shape or polarity."""
    model = read_model(path)
    _, s = modules()
    changes = []
    commutative = {s.BuiltinOperator.ADD, s.BuiltinOperator.MUL,
                   s.BuiltinOperator.MAXIMUM, s.BuiltinOperator.MINIMUM}
    allowed = commutative | {s.BuiltinOperator.SUB, s.BuiltinOperator.DIV}
    for g in model.subgraphs:
        dynamic = set(map(int, g.inputs)) | {int(i) for op in g.operators for i in op.outputs}
        for op in g.operators:
            code = model.operatorCodes[op.opcodeIndex]
            kind = max(code.builtinCode, code.deprecatedBuiltinCode)
            if kind not in allowed or len(op.inputs) != 2:
                continue
            slots = [slot for slot, index in enumerate(op.inputs)
                     if int(index) not in dynamic and int(index) >= 0
                     and model.buffers[g.tensors[int(index)].buffer].data is not None
                     and len(model.buffers[g.tensors[int(index)].buffer].data)]
            if len(slots) != 1:
                continue
            slot = slots[0]
            constant = g.tensors[int(op.inputs[slot])]
            other = g.tensors[int(op.inputs[1 - slot])]
            output = g.tensors[int(op.outputs[0])]
            shape = [] if constant.shape is None else list(constant.shape)
            other_shape = [] if other.shape is None else list(other.shape)
            if (shape and np.prod(shape) == 1 and len(shape) <= len(other_shape)
                    and output.shape is not None and list(output.shape) == other_shape):
                scalar = deepcopy(constant)
                scalar.name = (scalar.name or b'constant') + f'/scalar_{len(g.tensors)}'.encode()
                scalar.shape, scalar.shapeSignature = np.array([], np.int32), None
                g.tensors.append(scalar)
                op.inputs[slot] = len(g.tensors) - 1
                changes.append({'rewrite': 'scalar_broadcast', 'tensor': scalar.name.decode()})
            if kind in commutative and slot == 0:
                op.inputs = op.inputs[::-1].copy()
                changes.append({'rewrite': 'dynamic_first_commutative'})
    if changes:
        compact_model(model)
        write_model(model, path)
    return changes


def normalize_maxpool_padding(path):
    """Use zero PAD for negative PADV2 feeding MaxPool on proven nonnegative data.

    Every pooling window must contain real input cells, and the padded tensor
    must have no other observable use. Negative inputs and ordinary PADV2
    computations are kept unchanged.
    """
    model = read_model(path)
    if len(model.subgraphs) != 1:
        return []
    _, schema = modules()
    graph = model.subgraphs[0]

    def opcode(op):
        code = model.operatorCodes[op.opcodeIndex]
        return max(code.builtinCode, code.deprecatedBuiltinCode)

    producers = {int(i): op for op in graph.operators for i in op.outputs if i >= 0}
    consumers = {}
    for op in graph.operators:
        for i in op.inputs:
            if i >= 0:
                consumers.setdefault(int(i), []).append(op)

    @lru_cache(None)
    def nonnegative(index):
        op = producers.get(index)
        if op is None:
            return False
        if getattr(op.builtinOptions, "fusedActivationFunction", None) in (
                schema.ActivationFunctionType.RELU, schema.ActivationFunctionType.RELU6):
            return True
        code = opcode(op)
        if code in (schema.BuiltinOperator.RELU, schema.BuiltinOperator.RELU6, schema.BuiltinOperator.ABS):
            return True
        if code == schema.BuiltinOperator.CONCATENATION:
            return all(nonnegative(int(i)) for i in op.inputs if i >= 0)
        if code in (schema.BuiltinOperator.MAX_POOL_2D, schema.BuiltinOperator.AVERAGE_POOL_2D,
                    schema.BuiltinOperator.RESHAPE, schema.BuiltinOperator.TRANSPOSE,
                    schema.BuiltinOperator.SQUEEZE, schema.BuiltinOperator.EXPAND_DIMS,
                    schema.BuiltinOperator.MEAN, schema.BuiltinOperator.GATHER,
                    schema.BuiltinOperator.PAD):
            return nonnegative(int(op.inputs[0]))
        return False

    rewrites = []
    pad_code = next((i for i, code in enumerate(model.operatorCodes)
                     if max(code.builtinCode, code.deprecatedBuiltinCode) == schema.BuiltinOperator.PAD), None)
    for op in graph.operators:
        if opcode(op) != schema.BuiltinOperator.PADV2 or len(op.inputs) != 3 or len(op.outputs) != 1:
            continue
        if any(i < 0 for i in op.inputs):
            continue
        source, padding_index, value_index = map(int, op.inputs)
        target = int(op.outputs[0])
        pools = consumers.get(target, [])
        if (target in graph.outputs or not pools or not nonnegative(source)
                or any(opcode(pool) != schema.BuiltinOperator.MAX_POOL_2D for pool in pools)):
            continue
        tensor, padding, value = (graph.tensors[i] for i in (source, padding_index, value_index))
        if tensor.type != schema.TensorType.FLOAT32 or value.type != schema.TensorType.FLOAT32:
            continue
        shape = list(map(int, tensor.shape))
        padding_data = model.buffers[padding.buffer].data
        value_data = model.buffers[value.buffer].data
        if len(shape) != 4 or min(shape) <= 0 or padding_data is None or value_data is None:
            continue
        if padding.type not in (schema.TensorType.INT32, schema.TensorType.INT64):
            continue
        pads = np.frombuffer(bytes(padding_data), dtype="<i4" if padding.type == schema.TensorType.INT32 else "<i8")
        fill = np.frombuffer(bytes(value_data), dtype="<f4")
        if pads.size != 8 or fill.size != 1 or np.isnan(fill[0]) or fill[0] > 0:
            continue
        pads = pads.reshape(4, 2)
        if np.any(pads < 0) or np.any(pads[[0, 3]]):
            continue
        if list(graph.tensors[target].shape) != [d + int(p.sum()) for d, p in zip(shape, pads)]:
            continue
        safe = True
        for pool in pools:
            options = pool.builtinOptions
            if options.padding != schema.Padding.VALID:
                safe = False
                break
            output_shape = graph.tensors[int(pool.outputs[0])].shape
            for axis, kernel, stride in [(1, options.filterHeight, options.strideH),
                                         (2, options.filterWidth, options.strideW)]:
                before = int(pads[axis, 0])
                count = int(output_shape[axis])
                if (count <= 0 or kernel <= before or stride <= 0
                        or (count - 1) * stride >= before + shape[axis]):
                    safe = False
        if not safe:
            continue
        if pad_code is None:
            entry = schema.OperatorCodeT()
            entry.builtinCode = entry.deprecatedBuiltinCode = schema.BuiltinOperator.PAD
            entry.version = 1
            model.operatorCodes.append(entry)
            pad_code = len(model.operatorCodes) - 1
        op.opcodeIndex = pad_code
        op.inputs = np.array([source, padding_index], dtype=np.int32)
        op.builtinOptionsType, op.builtinOptions = schema.BuiltinOptions.PadOptions, schema.PadOptionsT()
        rewrites.append({"input": tensor.name.decode(), "output": graph.tensors[target].name.decode(),
                         "original_pad_value": "-inf" if np.isneginf(fill[0]) else float(fill[0]),
                         "rewrite": "PADV2_to_PAD"})
    if rewrites:
        write_model(model, path)
    return rewrites
