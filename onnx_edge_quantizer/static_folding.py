"""Fold statically proven shape computations without consulting sample values."""
from collections import Counter

import numpy as np

from .runtime import modules, read_model, write_model
from .tflite_optimizations import compact_model


def fold_static_tensors(path):
    model = read_model(path)
    _, s = modules()
    if len(model.subgraphs) != 1:
        return []
    graph = model.subgraphs[0]
    dtype_map = {s.TensorType.FLOAT32: '<f4', s.TensorType.FLOAT16: '<f2',
                 s.TensorType.INT32: '<i4', s.TensorType.INT64: '<i8',
                 s.TensorType.INT8: 'i1', s.TensorType.UINT8: 'u1',
                 s.TensorType.INT16: '<i2', s.TensorType.BOOL: '?'}
    produced = {int(i) for op in graph.operators for i in op.outputs}
    producers = {int(i): op for op in graph.operators for i in op.outputs}
    known = {}
    for i, tensor in enumerate(graph.tensors):
        if i in produced or i in graph.inputs or tensor.type not in dtype_map:
            continue
        shape = [] if tensor.shape is None else list(tensor.shape)
        data = model.buffers[tensor.buffer].data
        if data is not None and len(data):
            known[i] = np.frombuffer(bytes(data), dtype_map[tensor.type]).reshape(shape)
        elif 0 in shape:
            known[i] = np.empty(shape, dtype_map[tensor.type])
    names = {v: k for k, v in vars(s.BuiltinOperator).items() if isinstance(v, int)}
    removed, folded = set(), Counter()
    trusted_shapes = {int(i): tuple(graph.tensors[int(i)].shape) for i in graph.inputs
                      if graph.tensors[int(i)].shape is not None
                      and np.all(graph.tensors[int(i)].shape > 0)}
    trusted_shapes.update({i: value.shape for i, value in known.items()})

    def infer_shapes(op, code):
        ids = [int(i) for i in op.inputs]
        options = op.builtinOptions
        if code in ('SPLIT', 'SPLIT_V'):
            data = ids[1] if code == 'SPLIT' else ids[0]
            shape = trusted_shapes.get(data)
            axis_id = ids[0] if code == 'SPLIT' else ids[2]
            if shape is None or axis_id not in known:
                return None
            axis = int(known[axis_id].item()) % len(shape)
            if code == 'SPLIT':
                count = int(options.numSplits)
                if count <= 0 or shape[axis] % count:
                    return None
                sizes = [shape[axis] // count] * count
            else:
                if ids[1] not in known:
                    return None
                sizes = list(map(int, known[ids[1]].reshape(-1)))
                if sizes.count(-1) == 1:
                    sizes[sizes.index(-1)] = shape[axis] - sum(i for i in sizes if i >= 0)
                if any(i < 0 for i in sizes) or sum(sizes) != shape[axis]:
                    return None
            return [shape[:axis] + (size,) + shape[axis + 1:] for size in sizes]
        shape = trusted_shapes.get(ids[0]) if ids else None
        if shape is None:
            return None
        if code in ('CONV_2D', 'DEPTHWISE_CONV_2D', 'MAX_POOL_2D', 'AVERAGE_POOL_2D'):
            if len(shape) != 4:
                return None
            if code in ('CONV_2D', 'DEPTHWISE_CONV_2D'):
                weight_shape = trusted_shapes.get(ids[1])
                if weight_shape is None:
                    return None
                kh, kw = weight_shape[1:3]
                channels = weight_shape[0 if code == 'CONV_2D' else 3]
            else:
                kh, kw, channels = options.filterHeight, options.filterWidth, shape[-1]
            spatial = []
            for dim, kernel, stride, dilation in [
                (shape[1], kh, options.strideH, getattr(options, 'dilationHFactor', 1)),
                (shape[2], kw, options.strideW, getattr(options, 'dilationWFactor', 1)),
            ]:
                spatial.append((dim + stride - 1) // stride if options.padding == s.Padding.SAME
                               else (dim - (kernel - 1) * dilation - 1) // stride + 1)
            return [(shape[0], *spatial, channels)]
        if code in ('LOGISTIC', 'RELU', 'RELU6', 'LEAKY_RELU', 'SOFTMAX', 'CAST', 'ABS', 'NEG', 'EXP', 'LOG', 'SQRT', 'RSQRT', 'TANH', 'FLOOR'):
            return [shape]
        if code in ('ADD', 'SUB', 'MUL', 'DIV', 'MAXIMUM', 'MINIMUM'):
            if all(i in trusted_shapes for i in ids):
                return [np.broadcast_shapes(*(trusted_shapes[i] for i in ids))]
        if code == 'TRANSPOSE' and ids[1] in known:
            return [tuple(shape[int(i)] for i in known[ids[1]].reshape(-1))]
        if code == 'RESHAPE':
            requested = (known[ids[1]].reshape(-1).tolist() if len(ids) > 1 and ids[1] in known
                         else list(options.newShape) if len(ids) == 1 and options.newShape is not None else None)
            if requested is not None:
                total = int(np.prod(shape))
                if requested.count(-1) == 1:
                    product = -int(np.prod(requested))
                    if not product or total % product:
                        return None
                    requested[requested.index(-1)] = total // product
                if np.prod(requested) == total and all(i >= 0 for i in requested):
                    return [tuple(requested)]
        if code == 'CONCATENATION' and all(i in trusted_shapes for i in ids):
            result = list(shape)
            axis = options.axis % len(shape)
            if all(len(trusted_shapes[i]) == len(shape) and all(
                    trusted_shapes[i][j] == shape[j] for j in range(len(shape)) if j != axis) for i in ids):
                result[axis] = sum(trusted_shapes[i][axis] for i in ids)
                return [tuple(result)]
        if code in ('PAD', 'PADV2') and ids[1] in known:
            pads = known[ids[1]]
            return [tuple(int(d + p.sum()) for d, p in zip(shape, pads))]
        if code in ('SLICE', 'STRIDED_SLICE') and all(i in known for i in ids[1:]):
            dummy = np.lib.stride_tricks.as_strided(np.zeros(1, np.int8), shape=shape, strides=(0,) * len(shape))
            values = evaluate(op, code, [dummy, *(known[i] for i in ids[1:])])
            return [value.shape for value in values] if values is not None else None
        if code == 'GATHER' and ids[1] in known and options.batchDims == 0:
            axis = options.axis % len(shape)
            return [shape[:axis] + known[ids[1]].shape + shape[axis + 1:]]
        if code in ('MEAN', 'SUM', 'REDUCE_MAX') and ids[1] in known:
            axes = {int(i) % len(shape) for i in known[ids[1]].reshape(-1)}
            return [tuple(1 if i in axes else d for i, d in enumerate(shape)) if options.keepDims
                    else tuple(d for i, d in enumerate(shape) if i not in axes)]
        if code in ('RESIZE_BILINEAR', 'RESIZE_NEAREST_NEIGHBOR') and ids[1] in known and len(shape) == 4:
            return [(shape[0], *map(int, known[ids[1]].reshape(-1)), shape[-1])]
        if code == 'EXPAND_DIMS' and ids[1] in known:
            axis = int(known[ids[1]].item()) % (len(shape) + 1)
            return [shape[:axis] + (1,) + shape[axis:]]
        if code == 'FILL' and ids[0] in known:
            return [tuple(int(i) for i in known[ids[0]].reshape(-1))]
        return None

    def evaluate(op, code, args):
        options = op.builtinOptions
        if code == 'RESHAPE':
            shape = args[1].reshape(-1).tolist() if len(args) > 1 else (list(options.newShape) if options.newShape is not None else [])
            return [args[0].reshape(shape)]
        if code == 'TRANSPOSE':
            return [np.transpose(args[0], args[1].reshape(-1))]
        if code == 'CONCATENATION':
            return [np.concatenate(args, axis=options.axis)]
        if code == 'PACK':
            return [np.stack(args, axis=options.axis)]
        if code == 'UNPACK':
            return list(np.moveaxis(args[0], options.axis, 0))
        if code == 'GATHER' and options.batchDims == 0:
            return [np.take(args[0], args[1], axis=options.axis)]
        if code == 'SLICE':
            start, size = args[1].reshape(-1), args[2].reshape(-1)
            return [args[0][tuple(slice(int(b), None if n == -1 else int(b + n))
                                 for b, n in zip(start, size))]]
        if code == 'STRIDED_SLICE' and options.ellipsisMask == 0 and options.newAxisMask == 0:
            slices = []
            for axis, (begin, end, stride) in enumerate(zip(*(a.reshape(-1) for a in args[1:]))):
                if options.shrinkAxisMask & (1 << axis):
                    slices.append(int(begin))
                else:
                    slices.append(slice(None if options.beginMask & (1 << axis) else int(begin),
                                        None if options.endMask & (1 << axis) else int(end), int(stride)))
            return [args[0][tuple(slices)]]
        if code == 'CAST':
            return [args[0].astype(dtype_map[graph.tensors[int(op.outputs[0])].type])]
        if code == 'FLOOR':
            return [np.floor(args[0])]
        if code == 'FILL':
            dims = tuple(int(i) for i in args[0].reshape(-1))
            if any(i < 0 for i in dims) or np.prod(dims) > 1_000_000:
                return None
            return [np.full(dims, args[1].item(), dtype=args[1].dtype)]
        if code in ('TILE', 'BROADCAST_TO'):
            dims = tuple(int(i) for i in args[1].reshape(-1))
            count = np.prod(dims) * (args[0].size if code == 'TILE' else 1)
            if any(i < 0 for i in dims) or count > 1_000_000:
                return None
            return [np.tile(args[0], dims) if code == 'TILE' else np.broadcast_to(args[0], dims)]
        if code == 'EXPAND_DIMS':
            return [np.expand_dims(args[0], int(args[1].item()))]
        if code == 'SQUEEZE':
            axes = tuple(int(i) for i in options.squeezeDims) if options.squeezeDims is not None else None
            return [np.squeeze(args[0], axis=axes or None)]
        if code in ('ADD', 'SUB', 'MUL', 'FLOOR_DIV', 'MAXIMUM', 'MINIMUM'):
            if getattr(options, 'fusedActivationFunction', 0) != 0:
                return None
            function = {'ADD': np.add, 'SUB': np.subtract, 'MUL': np.multiply,
                        'FLOOR_DIV': np.floor_divide, 'MAXIMUM': np.maximum, 'MINIMUM': np.minimum}[code]
            return [function(*args)]
        if code in ('EQUAL', 'NOT_EQUAL', 'LESS', 'LESS_EQUAL', 'GREATER', 'GREATER_EQUAL'):
            function = {'EQUAL': np.equal, 'NOT_EQUAL': np.not_equal, 'LESS': np.less,
                        'LESS_EQUAL': np.less_equal, 'GREATER': np.greater,
                        'GREATER_EQUAL': np.greater_equal}[code]
            return [function(*args)]
        if code in ('SELECT', 'SELECT_V2'):
            return [np.where(*args)]
        if code == 'DIV' and np.all(args[1] != 0) and getattr(options, 'fusedActivationFunction', 0) == 0:
            result = args[0] / args[1]
            return [np.trunc(result) if np.issubdtype(args[0].dtype, np.integer) else result]
        if code == 'RANGE':
            start, stop, step = (a.item() for a in args)
            if not step or not np.isfinite([start, stop, step]).all() or abs((stop - start) / step) > 1_000_000:
                return None
            # Integer-valued float ranges (anchor grids) are exact below 2^24.
            if any(float(v) != int(v) or abs(v) >= 1 << 24 for v in (start, stop, step)):
                return None
            return [np.arange(start, stop, step, dtype=args[0].dtype)]
        return None

    for op in graph.operators:
        entry = model.operatorCodes[op.opcodeIndex]
        code = names.get(max(entry.builtinCode, entry.deprecatedBuiltinCode), '')
        outputs = None
        if code == 'SHAPE':
            shape = trusted_shapes.get(int(op.inputs[0]))
            if shape is not None:
                outputs = [np.asarray(shape, dtype=dtype_map[graph.tensors[int(op.outputs[0])].type])]
        elif code == 'SLICE' and all(int(i) in known for i in op.inputs[1:]):
            begin, size = (known[int(i)].reshape(-1) for i in op.inputs[1:])
            parent = producers.get(int(op.inputs[0]))
            # Shape values may remain dynamic, but the length of a Shape
            # vector is the rank of its input. Its empty suffix is constant.
            if parent is not None:
                pc = model.operatorCodes[parent.opcodeIndex]
                if max(pc.builtinCode, pc.deprecatedBuiltinCode) == s.BuiltinOperator.SHAPE:
                    tensor = graph.tensors[int(parent.inputs[0])]
                    rank = len(tensor.shape) if tensor.shape is not None else None
                    if (rank is not None and begin.size == size.size == 1
                            and size.item() == 0 and 0 <= begin.item() <= rank):
                        outputs = [np.empty((0,), dtype=dtype_map[graph.tensors[int(op.outputs[0])].type])]
            if outputs is None and all(int(i) in known for i in op.inputs):
                try:
                    outputs = evaluate(op, code, [known[int(i)] for i in op.inputs])
                except (ValueError, IndexError, KeyError, TypeError):
                    outputs = None
        elif all(int(i) in known for i in op.inputs):
            try:
                with np.errstate(over='raise', invalid='raise', divide='raise'):
                    outputs = evaluate(op, code, [known[int(i)] for i in op.inputs])
            except (ValueError, IndexError, KeyError, TypeError, FloatingPointError):
                outputs = None
        if outputs is None or len(outputs) != len(op.outputs):
            try:
                inferred = infer_shapes(op, code)
            except (ValueError, IndexError, TypeError, ZeroDivisionError):
                inferred = None
            if inferred is not None:
                for index, shape in zip(op.outputs, inferred):
                    trusted_shapes[int(index)] = tuple(int(i) for i in shape)
                    tensor = graph.tensors[int(index)]
                    previous = () if tensor.shape is None else tuple(tensor.shape)
                    if (tensor.shape is None or previous != tuple(shape)) and all(i >= 0 for i in shape):
                        tensor.shape = np.array(shape, dtype=np.int32)
                        folded['STATIC_SHAPE_ANNOTATION'] += 1
            continue
        if any(np.asarray(value).size > 1_000_000 for value in outputs):
            continue
        for index, value in zip(op.outputs, outputs):
            tensor = graph.tensors[int(index)]
            if tensor.type not in dtype_map:
                raise ValueError(f'Unsupported constant-fold output dtype on {tensor.name!r}')
            value = np.asarray(value, dtype=dtype_map[tensor.type])
            buffer = s.BufferT()
            buffer.data = np.frombuffer(value.tobytes(), dtype=np.uint8)
            model.buffers.append(buffer)
            tensor.buffer = len(model.buffers) - 1
            tensor.shape = np.array(value.shape, dtype=np.int32)
            known[int(index)] = value
            trusted_shapes[int(index)] = value.shape
        removed.add(id(op))
        folded[code] += 1
    # Empty vectors can also be suffixes of a dynamic shape concat. Removing
    # them is exact even when the other operands cannot be folded.
    for op in graph.operators:
        entry = model.operatorCodes[op.opcodeIndex]
        if id(op) in removed or max(entry.builtinCode, entry.deprecatedBuiltinCode) != s.BuiltinOperator.CONCATENATION:
            continue
        if op.builtinOptions.axis != 0:
            continue
        kept = [int(i) for i in op.inputs if not (int(i) in known and known[int(i)].shape == (0,))]
        if kept and len(kept) < len(op.inputs):
            op.inputs = np.array(kept, dtype=np.int32)
            folded['EMPTY_CONCAT_INPUT'] += 1
    if folded:
        graph.operators = [op for op in graph.operators if id(op) not in removed]
        compact_model(model)
        write_model(model, path)
    return [{'rewrite': 'fold_static_tensor', 'op': name, 'count': count}
            for name, count in sorted(folded.items())]
